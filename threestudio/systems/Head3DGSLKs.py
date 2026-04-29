import io
import math
import numpy as np
from plyfile import PlyData, PlyElement
from dataclasses import dataclass, field
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

import threestudio
# from threestudio.utils.poser import Skeleton
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.reference_sheet import load_reference_sheet
from threestudio.utils.typing import *

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussiansplatting.scene.cameras import Camera, MiniCam
from gaussiansplatting.scene.gaussian_flame_model import GaussianFlameModel


@threestudio.register("head-3dgs-lks-rig-system")
class Head3DGSLKsRig(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        texture_structure_joint: bool = False
        controlnet: bool = False
        flame_path: str = "/path/to/flame/model"
        flame_gender: str = 'generic'
        pts_num: int = 100000

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        densify_min_opacity: float = 0.05
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_only_min_opacity: float = 0.005
        prune_size_threshold: float = 0.008

        apose: bool = True
        bg_white: bool = False

        area_relax: bool = False
        shape_update_end_step: int = 12000
        training_w_animation: bool = True
        reference_fidelity: dict = field(default_factory=dict)
        opacity_coverage: dict = field(default_factory=dict)
        rear_opacity: dict = field(default_factory=dict)
        prune_region_guard: dict = field(default_factory=dict)

        # area scaling factor
        # area_scaling_factor: float = 1

    cfg: Config

    def configure(self) -> None:
        self.radius = self.cfg.radius
        # self.gaussian = GaussianModel(sh_degree=0)
        self.gaussian = GaussianFlameModel(sh_degree=0, gender=self.cfg.flame_gender, model_folder=self.cfg.flame_path)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32,
                                              device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0],
                                                                                                    dtype=torch.float32,
                                                                                                    device="cuda")

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.controlnet = self.cfg.controlnet

        self.cameras_extent = 4.0

        self.cfg.loss.lambda_position = 0.01 * self.cfg.loss.lambda_position
        self.cfg.loss.lambda_scaling = 0.01 * self.cfg.loss.lambda_scaling
        if self.cfg.area_relax:
            reduction = 'none'
        else:
            reduction = 'mean'
        self.smoothl1_position = torch.nn.SmoothL1Loss(beta=1.0, reduction=reduction)
        self.l1_scaling = torch.nn.L1Loss(reduction=reduction)
        self.reference_sheet = None
        self.reference_targets = None
        ref_cfg = self.cfg.get("reference_fidelity", None)
        if ref_cfg is not None and ref_cfg.get("enabled", False):
            self.reference_sheet = load_reference_sheet(ref_cfg.metadata_path)
            self.reference_targets = self._build_reference_targets(self.reference_sheet)

    def post_configure(self) -> None:
        if self.cfg.weights is None or self.gaussian.get_faces.numel() > 0:
            return
        weights_path = Path(self.cfg.weights)
        run_dir = weights_path.parent.parent if weights_path.parent.name == "ckpts" else weights_path.parent
        ply_path = run_dir / "save" / "last.ply"
        if ply_path.exists():
            threestudio.info(f"Loading FLAME Gaussian bindings from {ply_path}")
            self.gaussian.load_ply(str(ply_path))

    def _read_reference_image(self, image_path):
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read reference image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).float().to(self.device) / 255.0

    def _crop_reference_image(self, image, crop):
        x0, y0, x1, y1 = crop
        height, width = image.shape[:2]
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        return image[y0:y1, x0:x1]

    def _reference_region_stats(self, crop):
        return {
            "mean": crop.mean(dim=(0, 1)),
            "std": crop.std(dim=(0, 1)),
        }

    def _build_reference_targets(self, reference_sheet):
        weighted = {
            "face_mean": [],
            "face_std": [],
            "person_mean": [],
            "person_std": [],
            "weights": [],
        }
        for ref in reference_sheet.references:
            image = self._read_reference_image(ref.image_path)
            face_stats = self._reference_region_stats(self._crop_reference_image(image, ref.face_crop))
            person_stats = self._reference_region_stats(self._crop_reference_image(image, ref.person_crop))
            weighted["face_mean"].append(face_stats["mean"])
            weighted["face_std"].append(face_stats["std"])
            weighted["person_mean"].append(person_stats["mean"])
            weighted["person_std"].append(person_stats["std"])
            weighted["weights"].append(float(ref.weight))

        weights = torch.tensor(weighted["weights"], dtype=torch.float32, device=self.device)
        weights = weights / weights.sum().clamp_min(1.0e-6)

        def combine(values):
            return (torch.stack(values, dim=0) * weights[:, None]).sum(dim=0)

        return {
            "face_mean": combine(weighted["face_mean"]).detach(),
            "face_std": combine(weighted["face_std"]).detach(),
            "person_mean": combine(weighted["person_mean"]).detach(),
            "person_std": combine(weighted["person_std"]).detach(),
        }

    def save_gif_to_file(self, images, output_file):
        with io.BytesIO() as writer:
            images[0].save(
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
            )
            writer.seek(0)
            with open(output_file, 'wb') as file:
                file.write(writer.read())

    def get_c2w(self, dist, elev, azim):
        elev = elev * math.pi / 180
        azim = azim * math.pi / 180
        batch_size = dist.shape[0]
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                dist * torch.cos(elev) * torch.cos(azim),
                dist * torch.cos(elev) * torch.sin(azim),
                dist * torch.sin(elev),
            ],
            dim=-1,
        )
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions, device=self.device)
        up: Float[Tensor, "B 3"] = torch.as_tensor(
            [0, 0, 1], dtype=torch.float32, device=self.device)[None, :].repeat(batch_size, 1)
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1], device=self.device)], dim=1
        )
        c2w[:, 3, 3] = 1.0
        return c2w

    def set_pose(self, expression, jaw_pose, leye_pose, reye_pose, neck_pose=None):
        self.gaussian._expression = expression.detach()
        self.gaussian._jaw_pose = jaw_pose.detach()
        # self.gaussian._leye_pose = leye_pose.detach()
        # self.gaussian._reye_pose = reye_pose.detach()
        if neck_pose is not None:
            self.gaussian._neck_pose = neck_pose.detach()

    def forward(self, batch: Dict[str, Any], renderbackground=None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        alphas = []
        self.viewspace_point_list = []

        if self.cfg.training_w_animation:
            self.set_pose(batch['expression'], batch['jaw_pose'], batch['leye_pose'], batch['reye_pose'])

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w=batch['c2w'][id], FoVy=batch['fovy'][id], height=batch['height'],
                                   width=batch['width'])

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg[
                "radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            alpha = render_pkg["alpha_3dgs"]

            depth = depth.permute(1, 2, 0)
            alpha = alpha.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        alphas = torch.stack(alphas, 0)
        # depth_min = torch.amin(depths, dim=[1, 2, 3], keepdim=True)
        # depth_max = torch.amax(depths, dim=[1, 2, 3], keepdim=True)
        # depths = (depths - depth_min) / (depth_max - depth_min + 1e-10)
        # depths = depths.repeat(1, 1, 1, 3)

        self.visibility_filter = self.radii > 0.0

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = alphas

        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def compute_opacity_coverage_loss(self, out):
        cfg = self.cfg.get("opacity_coverage", {})
        if not cfg.get("enabled", False):
            return out["opacity"].new_tensor(0.0)
        opacity = out["opacity"]
        min_alpha = float(cfg.get("min_alpha", 0.85))
        return F.relu(opacity.new_tensor(min_alpha) - opacity).mean()

    def compute_rear_opacity_loss(self):
        cfg = self.cfg.get("rear_opacity", {})
        if not cfg.get("enabled", False):
            return self.gaussian.get_opacity.new_tensor(0.0)
        labels = self.gaussian.get_gaussian_region_labels()
        if not labels:
            return self.gaussian.get_opacity.new_tensor(0.0)
        rear_mask = torch.tensor(
            [label == "rear" for label in labels],
            dtype=torch.bool,
            device=self.gaussian.get_opacity.device,
        )
        if not rear_mask.any():
            return self.gaussian.get_opacity.new_tensor(0.0)
        min_mean_opacity = float(cfg.get("min_mean_opacity", 0.35))
        rear_mean = self.gaussian.get_opacity.squeeze()[rear_mask].mean()
        return F.relu(rear_mean.new_tensor(min_mean_opacity) - rear_mean)

    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)

        if self.true_global_step > self.cfg.half_scheduler_max_step:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch)

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        flame_conds = batch["flame_conds"]

        if isinstance(flame_conds, dict):
            control_images = [
                flame_conds['pose'].permute(0, 3, 1, 2),
                flame_conds['depth'].permute(0, 3, 1, 2),
            ]
        else:
            control_images = flame_conds.permute(0, 3, 1, 2)

        guidance_eval = False

        guidance_out = self.guidance(
            images.permute(0, 3, 1, 2), control_images, prompt_utils,
            **batch, rgb_as_latents=False,
        )

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])

        # scaling = self.gaussian.get_scaling.max(dim=1).values
        scaling = self.gaussian.get_scaling
        tris_scaling = self.gaussian.get_tris_scaling.max(dim=1).values
        big_points_ws = scaling > (0.5 * tris_scaling).unsqueeze(-1)
        loss_scaling = self.l1_scaling(scaling[big_points_ws], torch.zeros_like(scaling[big_points_ws]))
        if self.cfg.area_relax:
            T, R, S = self.gaussian.get_trans_matrix()
            loss_scaling = (loss_scaling / (
                    S.unsqueeze(-1).repeat(1, 3)[big_points_ws] + 1e-10)).mean()
        self.log("train/loss_scaling", loss_scaling)
        loss += loss_scaling * self.C(self.cfg.loss.lambda_scaling)

        if self.true_global_step >= self.cfg.prune_only_start_step:
            position_threshold = 0.5 * tris_scaling
            T, R, S = self.gaussian.get_trans_matrix()
            xyz = self.gaussian.get_xyz - T
            position = torch.norm(xyz, dim=1)
            mask = position > position_threshold
            loss_position = self.smoothl1_position(position[mask], torch.zeros_like(position[mask]))
            if self.cfg.area_relax:
                loss_position = (loss_position / (S[mask] + 1e-10)).mean()
            self.log("train/loss_position", loss_position)
            loss += loss_position * self.C(self.cfg.loss.lambda_position)

        loss_shape = torch.norm(self.gaussian._shape)
        self.log("train/loss_shape", loss_shape)
        loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        lambda_anchor = self.cfg.loss.get("lambda_anchor", 0.0)
        if lambda_anchor > 0.0:
            anchor_xyz = self.gaussian.get_anchor_world_xyz()
            current_xyz = self.gaussian.get_xyz
            loss_anchor = F.smooth_l1_loss(current_xyz, anchor_xyz.detach(), reduction="mean")
            self.log("train/loss_anchor", loss_anchor)
            loss += loss_anchor * self.C(lambda_anchor)

        lambda_temporal_xyz = self.cfg.loss.get("lambda_temporal_xyz", 0.0)
        if lambda_temporal_xyz > 0.0 and batch.get("adjacent_expression", None) is not None:
            current_xyz = self.gaussian.get_xyz
            original_expression = self.gaussian._expression
            original_jaw_pose = self.gaussian._jaw_pose
            original_neck_pose = self.gaussian._neck_pose

            self.gaussian._expression = batch["adjacent_expression"].detach()
            self.gaussian._jaw_pose = batch["adjacent_jaw_pose"].detach()
            self.gaussian._neck_pose = batch["adjacent_neck_pose"].detach()
            adjacent_xyz = self.gaussian.get_xyz

            self.gaussian._expression = original_expression
            self.gaussian._jaw_pose = original_jaw_pose
            self.gaussian._neck_pose = original_neck_pose

            loss_temporal_xyz = (adjacent_xyz - current_xyz).norm(dim=1).mean()
            self.log("train/loss_temporal_xyz", loss_temporal_xyz)
            loss += loss_temporal_xyz * self.C(lambda_temporal_xyz)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        lambda_opacity_coverage = self.cfg.loss.get("lambda_opacity_coverage", 0.0)
        if lambda_opacity_coverage > 0.0:
            loss_opacity_coverage = self.compute_opacity_coverage_loss(out)
            self.log("train/loss_opacity_coverage", loss_opacity_coverage)
            loss += loss_opacity_coverage * self.C(lambda_opacity_coverage)

        lambda_rear_opacity = self.cfg.loss.get("lambda_rear_opacity", 0.0)
        if lambda_rear_opacity > 0.0:
            loss_rear_opacity = self.compute_rear_opacity_loss()
            self.log("train/loss_rear_opacity", loss_rear_opacity)
            loss += loss_rear_opacity * self.C(lambda_rear_opacity)

        if self.reference_sheet is not None:
            ref_losses = self.compute_reference_fidelity_losses(out, batch)
            self.log("train/loss_ref_person", ref_losses["loss_ref_person"])
            self.log("train/loss_ref_face", ref_losses["loss_ref_face"])
            self.log("train/loss_ref_temporal_face", ref_losses["loss_ref_temporal_face"])
            loss += ref_losses["loss_ref_person"] * self.C(self.cfg.loss.lambda_ref_person)
            loss += ref_losses["loss_ref_face"] * self.C(self.cfg.loss.lambda_ref_face)
            loss += ref_losses["loss_ref_temporal_face"] * self.C(self.cfg.loss.lambda_ref_temporal_face)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def _relative_crop(self, images, box):
        _, height, width, _ = images.shape
        x0 = int(width * box[0])
        y0 = int(height * box[1])
        x1 = max(x0 + 1, int(width * box[2]))
        y1 = max(y0 + 1, int(height * box[3]))
        return images[:, y0:y1, x0:x1, :]

    def _render_region_stats(self, crop):
        return {
            "mean": crop.mean(dim=(0, 1, 2)),
            "std": crop.std(dim=(0, 1, 2)),
        }

    def compute_reference_fidelity_losses(self, out, batch):
        images = out["comp_rgb"]
        ref_cfg = self.cfg.reference_fidelity
        start_step = int(ref_cfg.start_step)
        end_step = int(ref_cfg.end_step)
        if self.true_global_step < start_step or self.true_global_step > end_step:
            zero = images.new_tensor(0.0)
            return {
                "loss_ref_person": zero,
                "loss_ref_face": zero,
                "loss_ref_temporal_face": zero,
            }

        face_crop = self._relative_crop(images, (0.25, 0.08, 0.75, 0.68))
        person_crop = self._relative_crop(images, (0.15, 0.05, 0.85, 0.90))
        face_stats = self._render_region_stats(face_crop)
        person_stats = self._render_region_stats(person_crop)
        reference_targets = {
            name: target.to(device=images.device, dtype=images.dtype)
            for name, target in self.reference_targets.items()
        }

        loss_ref_face = F.smooth_l1_loss(face_stats["mean"], reference_targets["face_mean"])
        loss_ref_face += F.smooth_l1_loss(face_stats["std"], reference_targets["face_std"])
        loss_ref_person = F.smooth_l1_loss(person_stats["mean"], reference_targets["person_mean"])
        loss_ref_person += F.smooth_l1_loss(person_stats["std"], reference_targets["person_std"])

        if face_crop.shape[0] > 1:
            loss_ref_temporal_face = (face_crop[1:] - face_crop[:-1]).abs().mean()
        else:
            loss_ref_temporal_face = images.new_tensor(0.0)

        loss_ref_face = loss_ref_face * float(ref_cfg.get("face_weight", 1.0))
        loss_ref_person = loss_ref_person * float(ref_cfg.get("person_weight", 0.25))
        loss_ref_temporal_face = loss_ref_temporal_face * float(ref_cfg.get("temporal_face_weight", 0.1))

        return {
            "loss_ref_person": loss_ref_person,
            "loss_ref_face": loss_ref_face,
            "loss_ref_temporal_face": loss_ref_temporal_face,
        }

    def build_region_min_opacity(self, base_min_opacity):
        cfg = self.cfg.get("prune_region_guard", {})
        if not cfg.get("enabled", False):
            return None
        rear_scale = float(cfg.get("rear_min_opacity_scale", 0.5))
        return {
            "front": float(base_min_opacity),
            "side": float(base_min_opacity),
            "rear": float(base_min_opacity) * rear_scale,
        }

    def on_before_optimizer_step(self, optimizer):

        # return

        with torch.no_grad():

            if self.true_global_step < self.cfg.densify_prune_end_step:  # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:  # 500 100
                    size_threshold = self.cfg.size_threshold if self.true_global_step > self.cfg.size_threshold_fix_step else None  # 3000
                    region_min_opacity = self.build_region_min_opacity(self.cfg.densify_min_opacity)
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.densify_min_opacity,
                        self.cameras_extent,
                        size_threshold,
                        region_min_opacity=region_min_opacity,
                    )

                    # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step % self.cfg.prune_only_interval == 0:
                    region_min_opacity = self.build_region_min_opacity(self.cfg.prune_only_min_opacity)
                    self.gaussian.prune_only(
                        min_opacity=self.cfg.prune_only_min_opacity,
                        extent=self.cameras_extent,
                        region_min_opacity=region_min_opacity,
                    )

            if self.true_global_step > self.cfg.shape_update_end_step:
                for param_group in self.gaussian.optimizer.param_groups:
                    if param_group['name'] == 'flame_shape':
                        param_group['lr'] = 1e-10

    def on_after_backward(self):
        self.dataset.skel.betas = self.gaussian.get_shape.detach()
        # pass

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        self.gaussian.create_from_flame(self.cameras_extent, -10, N=self.cfg.pts_num)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )
