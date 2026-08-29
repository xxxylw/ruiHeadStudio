import copy
import io
import math
import numpy as np
from plyfile import PlyData, PlyElement
from dataclasses import dataclass, field
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

import threestudio
# from threestudio.utils.poser import Skeleton
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.models.clip_alignment import (
    CLIPAlignment,
    clip_alignment_weight,
    clip_decay_weight,
    frequency_quality_loss,
    normalized_parameter_drift,
    quality_ramp_weight,
    reference_statistics_loss,
    rendered_reference_loss,
)

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
        gaussian_init_ply: Optional[str] = None
        gaussian_init_step: int = 0

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_size_threshold: float = 0.008

        apose: bool = True
        bg_white: bool = False

        area_relax: bool = False
        shape_update_end_step: int = 12000
        surface_constraint_start_step: int = 2400
        temporal_loss_start_step: int = 2400
        scale_ratio_threshold: float = 0.5
        training_w_animation: bool = True
        clip_model_name: str = "ViT-L/14"
        clip_start_step: int = 2000
        clip_foreground_only: bool = False
        clip_use_view_prompt: bool = False
        clip_global_weight: float = 0.0
        clip_foreground_weight: float = 0.0
        clip_view_weight: float = 0.0
        clip_decay_start_step: int = 0
        clip_decay_end_step: int = 0
        lambda_trust: float = 0.0
        trust_xyz_weight: float = 1.0
        trust_scaling_weight: float = 1.0
        trust_opacity_weight: float = 1.0
        trust_feature_weight: float = 0.25
        quality_start_step: int = 0
        quality_ramp_end_step: int = 0
        lambda_frequency_quality: float = 0.0
        lambda_rendered_reference: float = 0.0
        lambda_reference_statistics: float = 0.0
        use_eye_pose: bool = False
        use_neck_pose: bool = False

        # area scaling factor
        # area_scaling_factor: float = 1

    cfg: Config

    @property
    def true_global_step(self):
        return super().true_global_step + self.cfg.gaussian_init_step

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
        self.cfg.loss.lambda_local_position = 0.01 * self.cfg.loss.lambda_local_position
        self.cfg.loss.lambda_scaling = 0.01 * self.cfg.loss.lambda_scaling
        self.cfg.loss.lambda_barycentric_inside = 0.01 * self.cfg.loss.lambda_barycentric_inside
        self.cfg.loss.lambda_normal_offset = 0.01 * self.cfg.loss.lambda_normal_offset
        self.cfg.loss.lambda_scale_ratio = 0.01 * self.cfg.loss.lambda_scale_ratio
        self.cfg.loss.lambda_temporal_motion = 0.01 * self.cfg.loss.lambda_temporal_motion
        self.cfg.loss.lambda_temporal_scale_ratio = 0.01 * self.cfg.loss.lambda_temporal_scale_ratio
        self.cfg.loss.lambda_temporal_local_offset = 0.01 * self.cfg.loss.lambda_temporal_local_offset
        self.cfg.loss.lambda_temporal_local_offset_accel = 0.01 * self.cfg.loss.lambda_temporal_local_offset_accel
        self.cfg.loss.lambda_temporal_scale_ratio_accel = 0.01 * self.cfg.loss.lambda_temporal_scale_ratio_accel
        if self.cfg.area_relax:
            reduction = 'none'
        else:
            reduction = 'mean'
        self.smoothl1_position = torch.nn.SmoothL1Loss(beta=1.0, reduction=reduction)
        self.l1_scaling = torch.nn.L1Loss(reduction=reduction)

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

    def set_pose(self, expression, jaw_pose, leye_pose, reye_pose, neck_pose=None, gaussian_model=None):
        gaussian_model = self.gaussian if gaussian_model is None else gaussian_model
        gaussian_model._expression = expression.detach()
        gaussian_model._jaw_pose = jaw_pose.detach()
        if self.cfg.use_eye_pose:
            gaussian_model._leye_pose = leye_pose.detach()
            gaussian_model._reye_pose = reye_pose.detach()
        if self.cfg.use_neck_pose and neck_pose is not None:
            gaussian_model._neck_pose = neck_pose.detach()

    def forward(
        self,
        batch: Dict[str, Any],
        renderbackground=None,
        gaussian_model=None,
        track_stats=True,
    ) -> Dict[str, Any]:

        gaussian_model = self.gaussian if gaussian_model is None else gaussian_model

        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        opacities = []
        if track_stats:
            self.viewspace_point_list = []

        if self.cfg.training_w_animation:
            self.set_pose(
                batch['expression'],
                batch['jaw_pose'],
                batch['leye_pose'],
                batch['reye_pose'],
                batch.get('neck_pose', None),
                gaussian_model=gaussian_model,
            )

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam = Camera(c2w=batch['c2w'][id], FoVy=batch['fovy'][id], height=batch['height'],
                                   width=batch['width'])

            render_pkg = render(viewpoint_cam, gaussian_model, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg[
                "radii"]
            if track_stats:
                self.viewspace_point_list.append(viewspace_point_tensor)

            if track_stats:
                if id == 0:
                    self.radii = radii
                else:
                    self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]

            alpha = render_pkg.get("alpha_3dgs")
            if alpha is not None:
                opacity = alpha.permute(1, 2, 0)
            else:
                opacity = depth / (depth.max() + 1e-5)
            opacities.append(opacity)

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        opacities = torch.stack(opacities, 0)
        # depth_min = torch.amin(depths, dim=[1, 2, 3], keepdim=True)
        # depth_max = torch.amax(depths, dim=[1, 2, 3], keepdim=True)
        # depths = (depths - depth_min) / (depth_max - depth_min + 1e-10)
        # depths = depths.repeat(1, 1, 1, 3)

        if track_stats:
            self.visibility_filter = self.radii > 0.0

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = opacities

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
        self.clip_alignment = None
        if self.C(self.cfg.loss.lambda_clip) > 0.0:
            self.clip_alignment = CLIPAlignment(
                self.cfg.clip_model_name, self.cfg.prompt_processor.prompt, self.device
            )

    def compute_temporal_losses(
            self,
            batch,
            compute_motion=True,
            compute_scale_ratio=False,
            compute_local_offset_loss=False,
            compute_local_offset_accel=False,
            compute_scale_ratio_accel=False,
            include_local_offset=False,
    ):
        states = self.gaussian.get_temporal_surface_states(
            batch["temporal_expression"],
            batch["temporal_jaw_pose"],
            batch["temporal_leye_pose"] if self.cfg.use_eye_pose else None,
            batch["temporal_reye_pose"] if self.cfg.use_eye_pose else None,
            batch.get("temporal_neck_pose", None) if self.cfg.use_neck_pose else None,
            include_local_offset=include_local_offset,
        )

        if len(states) < 2:
            zero = states[0]["xyz"].new_tensor(0.0)
            return {
                "motion": zero,
                "scale_ratio": zero,
                "local_offset": zero,
                "local_offset_accel": zero,
                "scale_ratio_accel": zero,
            }

        motion_losses = []
        scale_ratio_losses = []
        local_offset_losses = []
        if compute_motion or compute_scale_ratio or compute_local_offset_loss:
            for index in range(len(states) - 1):
                current = states[index]
                next_state = states[index + 1]

                if compute_motion:
                    gaussian_motion = next_state["xyz"] - current["xyz"]
                    triangle_motion = next_state["triangle_centroid"] - current["triangle_centroid"]
                    motion_losses.append(torch.norm(gaussian_motion - triangle_motion, dim=-1).mean())
                if compute_scale_ratio:
                    scale_ratio_losses.append(torch.abs(next_state["scale_ratio"] - current["scale_ratio"]).mean())
                if compute_local_offset_loss:
                    local_offset_losses.append(
                        torch.norm(next_state["local_offset"] - current["local_offset"], dim=-1).mean()
                    )

        local_offset_accel_losses = []
        scale_ratio_accel_losses = []
        if compute_local_offset_accel or compute_scale_ratio_accel:
            for index in range(len(states) - 2):
                current = states[index]
                next_state = states[index + 1]
                next_next_state = states[index + 2]

                if compute_local_offset_accel:
                    local_offset_accel = next_next_state["local_offset"] - 2 * next_state["local_offset"] + current[
                        "local_offset"]
                    local_offset_accel_losses.append(torch.norm(local_offset_accel, dim=-1).mean())
                if compute_scale_ratio_accel:
                    scale_ratio_accel = (
                            next_next_state["scale_ratio"] - 2 * next_state["scale_ratio"] + current["scale_ratio"]
                    )
                    scale_ratio_accel_losses.append(torch.abs(scale_ratio_accel).mean())

        zero = states[0]["xyz"].new_tensor(0.0)

        return {
            "motion": torch.stack(motion_losses).mean() if len(motion_losses) > 0 else zero,
            "scale_ratio": torch.stack(scale_ratio_losses).mean() if len(scale_ratio_losses) > 0 else zero,
            "local_offset": torch.stack(local_offset_losses).mean() if len(local_offset_losses) > 0 else zero,
            "local_offset_accel": torch.stack(local_offset_accel_losses).mean() if len(
                local_offset_accel_losses) > 0 else zero,
            "scale_ratio_accel": torch.stack(scale_ratio_accel_losses).mean() if len(
                scale_ratio_accel_losses) > 0 else zero,
        }

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

        clip_weight = clip_alignment_weight(
            self.C(self.cfg.loss.lambda_clip), self.true_global_step, self.cfg.clip_start_step
        )
        if clip_weight > 0.0 and self.cfg.clip_decay_end_step > self.cfg.clip_decay_start_step:
            clip_weight = clip_decay_weight(
                clip_weight,
                self.true_global_step,
                self.cfg.clip_decay_start_step,
                self.cfg.clip_decay_end_step,
            )
        loss_clip = torch.zeros((), device=images.device)
        loss_clip_global = torch.zeros((), device=images.device)
        loss_clip_foreground = torch.zeros((), device=images.device)
        loss_clip_view = torch.zeros((), device=images.device)
        if clip_weight > 0.0:
            if self.clip_alignment is None:
                raise RuntimeError("CLIP alignment was not initialized while lambda_clip is positive")
            image_chw = images.permute(0, 3, 1, 2)
            component_weights = {
                "global": float(self.cfg.clip_global_weight),
                "foreground": float(self.cfg.clip_foreground_weight),
                "view": float(self.cfg.clip_view_weight),
            }
            component_total = sum(weight for weight in component_weights.values() if weight > 0.0)
            if component_total > 0.0:
                if component_weights["global"] > 0.0:
                    loss_clip_global = self.clip_alignment(image_chw)
                    loss_clip = loss_clip + loss_clip_global * component_weights["global"]
                if component_weights["foreground"] > 0.0:
                    loss_clip_foreground = self.clip_alignment(
                        image_chw, opacity=out["opacity"], foreground_only=True
                    )
                    loss_clip = loss_clip + loss_clip_foreground * component_weights["foreground"]
                if component_weights["view"] > 0.0:
                    loss_clip_view = self.clip_alignment(
                        image_chw,
                        opacity=out["opacity"],
                        azimuth=batch["azimuth"],
                        foreground_only=True,
                        view_dependent=True,
                    )
                    loss_clip = loss_clip + loss_clip_view * component_weights["view"]
                loss_clip = loss_clip / component_total
            else:
                loss_clip = self.clip_alignment(
                    image_chw,
                    opacity=out["opacity"],
                    azimuth=batch["azimuth"],
                    foreground_only=self.cfg.clip_foreground_only,
                    view_dependent=self.cfg.clip_use_view_prompt,
                )
            loss = loss + loss_clip * clip_weight
        self.log("train/loss_clip", loss_clip)
        self.log("train/loss_clip_global", loss_clip_global)
        self.log("train/loss_clip_foreground", loss_clip_foreground)
        self.log("train/loss_clip_view", loss_clip_view)

        quality_weight = quality_ramp_weight(
            self.C(self.cfg.lambda_frequency_quality),
            self.true_global_step,
            self.cfg.quality_start_step,
            self.cfg.quality_ramp_end_step,
        )
        loss_frequency_quality = torch.zeros((), device=images.device)
        if quality_weight > 0.0:
            loss_frequency_quality = frequency_quality_loss(
                images.permute(0, 3, 1, 2), out["opacity"]
            )
            loss = loss + loss_frequency_quality * quality_weight
        self.log("train/loss_frequency_quality", loss_frequency_quality)

        reference_weight = quality_ramp_weight(
            self.C(self.cfg.lambda_rendered_reference),
            self.true_global_step,
            self.cfg.quality_start_step,
            self.cfg.quality_ramp_end_step,
        )
        loss_rendered_reference = torch.zeros((), device=images.device)
        statistics_weight = quality_ramp_weight(
            self.C(self.cfg.lambda_reference_statistics),
            self.true_global_step,
            self.cfg.quality_start_step,
            self.cfg.quality_ramp_end_step,
        )
        reference_out = None
        if reference_weight > 0.0 or statistics_weight > 0.0:
            if self.reference_gaussian is None:
                raise RuntimeError("rendered reference loss is enabled without a reference Gaussian")
            with torch.no_grad():
                reference_out = self(
                    batch,
                    gaussian_model=self.reference_gaussian,
                    track_stats=False,
                )
        if reference_weight > 0.0:
            loss_rendered_reference = rendered_reference_loss(
                images.permute(0, 3, 1, 2),
                reference_out["comp_rgb"].permute(0, 3, 1, 2),
                out["opacity"],
            )
            loss = loss + loss_rendered_reference * reference_weight
        self.log("train/loss_rendered_reference", loss_rendered_reference)

        loss_reference_statistics = torch.zeros((), device=images.device)
        if statistics_weight > 0.0:
            loss_reference_statistics = reference_statistics_loss(
                images.permute(0, 3, 1, 2),
                reference_out["comp_rgb"].permute(0, 3, 1, 2),
                out["opacity"],
            )
            loss = loss + loss_reference_statistics * statistics_weight
        self.log("train/loss_reference_statistics", loss_reference_statistics)

        loss_trust_xyz = torch.zeros((), device=images.device)
        loss_trust_scaling = torch.zeros((), device=images.device)
        loss_trust_opacity = torch.zeros((), device=images.device)
        loss_trust_feature = torch.zeros((), device=images.device)
        lambda_trust = self.C(self.cfg.lambda_trust)
        if lambda_trust > 0.0 and self.trust_region_anchor is not None and clip_weight > 0.0:
            anchor = self.trust_region_anchor
            loss_trust_xyz = normalized_parameter_drift(
                self.gaussian.get_xyz, anchor["xyz"], anchor["scaling"]
            )
            loss_trust_scaling = normalized_parameter_drift(
                self.gaussian.get_scaling, anchor["scaling"], anchor["scaling"]
            )
            loss_trust_opacity = normalized_parameter_drift(
                self.gaussian.get_opacity, anchor["opacity"]
            )
            loss_trust_feature = normalized_parameter_drift(
                self.gaussian._features_dc, anchor["features_dc"]
            )
            loss_trust = (
                loss_trust_xyz * self.cfg.trust_xyz_weight
                + loss_trust_scaling * self.cfg.trust_scaling_weight
                + loss_trust_opacity * self.cfg.trust_opacity_weight
                + loss_trust_feature * self.cfg.trust_feature_weight
            )
            self.log("train/loss_trust", loss_trust)
            self.log("train/loss_trust_xyz", loss_trust_xyz)
            self.log("train/loss_trust_scaling", loss_trust_scaling)
            self.log("train/loss_trust_opacity", loss_trust_opacity)
            self.log("train/loss_trust_feature", loss_trust_feature)
            loss = loss + loss_trust * lambda_trust

        lambda_scaling = self.C(self.cfg.loss.lambda_scaling)
        lambda_scale_ratio = self.C(self.cfg.loss.lambda_scale_ratio)
        lambda_position = self.C(self.cfg.loss.lambda_position)
        lambda_local_position = self.C(self.cfg.loss.lambda_local_position)
        position_loss_active = self.true_global_step >= self.cfg.prune_only_start_step and (
                lambda_position > 0.0 or lambda_local_position > 0.0
        )
        scale_loss_active = lambda_scaling > 0.0 or lambda_scale_ratio > 0.0

        if scale_loss_active or position_loss_active:
            scaling = self.gaussian.get_scaling
            tris_scaling = self.gaussian.get_tris_scaling.max(dim=1).values

        if lambda_scaling > 0.0:
            big_points_ws = scaling > (0.5 * tris_scaling).unsqueeze(-1)
            loss_scaling = self.l1_scaling(scaling[big_points_ws], torch.zeros_like(scaling[big_points_ws]))
            if self.cfg.area_relax:
                T, R, S = self.gaussian.get_trans_matrix()
                loss_scaling = (loss_scaling / (
                        S.unsqueeze(-1).repeat(1, 3)[big_points_ws] + 1e-10)).mean()
            self.log("train/loss_scaling", loss_scaling)
            loss += loss_scaling * lambda_scaling

        if lambda_scale_ratio > 0.0:
            scale_ratio = scaling / (tris_scaling.unsqueeze(-1) + 1e-10)
            scale_ratio_excess = F.relu(scale_ratio - self.cfg.scale_ratio_threshold)
            loss_scale_ratio = (scale_ratio_excess ** 2).mean()
            # Opacity weighting is intentionally left out for the first ratio-loss experiment.
            # If visible outliers remain, try weighting this penalty by detached opacity.
            self.log("train/loss_scale_ratio", loss_scale_ratio)
            loss += loss_scale_ratio * lambda_scale_ratio

        if position_loss_active:
            position_threshold = 0.5 * tris_scaling
            T, R, S = self.gaussian.get_trans_matrix()
            xyz = self.gaussian.get_xyz - T
            position = torch.norm(xyz, dim=1)
            mask = position > position_threshold
            loss_position = self.smoothl1_position(position[mask], torch.zeros_like(position[mask]))
            if self.cfg.area_relax:
                loss_position = (loss_position / (S[mask] + 1e-10)).mean()
            if lambda_position > 0.0:
                self.log("train/loss_position", loss_position)
                loss += loss_position * lambda_position
            if lambda_local_position > 0.0:
                self.log("train/loss_local_position", loss_position)
                loss += loss_position * lambda_local_position

        lambda_barycentric_inside = self.C(self.cfg.loss.lambda_barycentric_inside)
        lambda_normal_offset = self.C(self.cfg.loss.lambda_normal_offset)
        if self.true_global_step >= self.cfg.surface_constraint_start_step and (
                lambda_barycentric_inside > 0.0 or lambda_normal_offset > 0.0):
            barycentric, normal_offset = self.gaussian.get_surface_constraint_terms()
            loss_barycentric_inside = F.relu(-barycentric).mean()
            loss_normal_offset = torch.abs(normal_offset).mean()
            self.log("train/loss_barycentric_inside", loss_barycentric_inside)
            self.log("train/loss_normal_offset", loss_normal_offset)
            loss += loss_barycentric_inside * lambda_barycentric_inside
            loss += loss_normal_offset * lambda_normal_offset

        lambda_temporal_motion = self.C(self.cfg.loss.lambda_temporal_motion)
        lambda_temporal_scale_ratio = self.C(self.cfg.loss.lambda_temporal_scale_ratio)
        lambda_temporal_local_offset = self.C(self.cfg.loss.lambda_temporal_local_offset)
        lambda_temporal_local_offset_accel = self.C(self.cfg.loss.lambda_temporal_local_offset_accel)
        lambda_temporal_scale_ratio_accel = self.C(self.cfg.loss.lambda_temporal_scale_ratio_accel)
        if batch.get("temporal_enabled", False) and self.true_global_step >= self.cfg.temporal_loss_start_step and (
                lambda_temporal_motion > 0.0
                or lambda_temporal_scale_ratio > 0.0
                or lambda_temporal_local_offset > 0.0
                or lambda_temporal_local_offset_accel > 0.0
                or lambda_temporal_scale_ratio_accel > 0.0):
            include_local_offset = lambda_temporal_local_offset > 0.0 or lambda_temporal_local_offset_accel > 0.0
            temporal_losses = self.compute_temporal_losses(
                batch,
                compute_motion=lambda_temporal_motion > 0.0,
                compute_scale_ratio=lambda_temporal_scale_ratio > 0.0,
                compute_local_offset_loss=lambda_temporal_local_offset > 0.0,
                compute_local_offset_accel=lambda_temporal_local_offset_accel > 0.0,
                compute_scale_ratio_accel=lambda_temporal_scale_ratio_accel > 0.0,
                include_local_offset=include_local_offset,
            )
            if lambda_temporal_motion > 0.0:
                self.log("train/loss_temporal_motion", temporal_losses["motion"])
                loss += temporal_losses["motion"] * lambda_temporal_motion
            if lambda_temporal_scale_ratio > 0.0:
                self.log("train/loss_temporal_scale_ratio", temporal_losses["scale_ratio"])
                loss += temporal_losses["scale_ratio"] * lambda_temporal_scale_ratio
            if lambda_temporal_local_offset > 0.0:
                self.log("train/loss_temporal_local_offset", temporal_losses["local_offset"])
                loss += temporal_losses["local_offset"] * lambda_temporal_local_offset
            if lambda_temporal_local_offset_accel > 0.0:
                self.log("train/loss_temporal_local_offset_accel", temporal_losses["local_offset_accel"])
                loss += temporal_losses["local_offset_accel"] * lambda_temporal_local_offset_accel
            if lambda_temporal_scale_ratio_accel > 0.0:
                self.log("train/loss_temporal_scale_ratio_accel", temporal_losses["scale_ratio_accel"])
                loss += temporal_losses["scale_ratio_accel"] * lambda_temporal_scale_ratio_accel

        loss_shape = torch.norm(self.gaussian._shape)
        self.log("train/loss_shape", loss_shape)
        loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

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
                    self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, size_threshold)

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
                    self.gaussian.prune_only(extent=self.cameras_extent)

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
        if self.cfg.gaussian_init_ply is not None:
            threestudio.info(f"Initializing Gaussian state from PLY: {self.cfg.gaussian_init_ply}")
            self.gaussian.load_ply(self.cfg.gaussian_init_ply)
        self.gaussian.training_setup(opt)
        self.reference_gaussian = None
        if self.C(self.cfg.lambda_rendered_reference) > 0.0 or self.C(self.cfg.lambda_reference_statistics) > 0.0:
            self.reference_gaussian = copy.deepcopy(self.gaussian)
            self.reference_gaussian.model.eval()
            for parameter in self.reference_gaussian.model.parameters():
                parameter.requires_grad_(False)
        self.trust_region_anchor = None
        if self.C(self.cfg.lambda_trust) > 0.0:
            self.trust_region_anchor = {
                "xyz": self.gaussian.get_xyz.detach().clone(),
                "scaling": self.gaussian.get_scaling.detach().clone(),
                "opacity": self.gaussian.get_opacity.detach().clone(),
                "features_dc": self.gaussian._features_dc.detach().clone(),
            }

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
