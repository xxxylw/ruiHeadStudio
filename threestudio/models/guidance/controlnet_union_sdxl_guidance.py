"""SDXL Union ControlNet guidance for RuiHeadStudio."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.guidance.controlnet_union_sdxl_contract import resolve_control_modes
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@threestudio.register("controlnet-union-sdxl-guidance")
class ControlNetUnionSDXLGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix"
        controlnet_model_name_or_path: str = "xinsir/controlnet-union-sdxl-1.0"
        control_modes: list[str] = None  # type: ignore[assignment]
        guidance_resolution: int = 512
        guidance_scale: float = 3.0
        condition_scale_pose: float = 1.0
        condition_scale_depth: float = 0.8
        control_guidance_start: list[float] = None  # type: ignore[assignment]
        control_guidance_end: list[float] = None  # type: ignore[assignment]
        half_precision_weights: bool = True
        min_step_percent: float = 0.05
        max_step_percent: float = 0.8
        diffusion_steps: int = 20
        enable_model_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        local_files_only: bool = False

    cfg: Config

    def configure(self) -> None:
        self.control_modes = self.cfg.control_modes or ["openpose", "depth"]
        self.control_mode_ids = resolve_control_modes(self.control_modes)
        self.control_guidance_start = self.cfg.control_guidance_start or [0.0] * len(
            self.control_mode_ids
        )
        self.control_guidance_end = self.cfg.control_guidance_end or [1.0] * len(
            self.control_mode_ids
        )
        self.conditioning_scale = [self.cfg.condition_scale_pose, self.cfg.condition_scale_depth]
        if len(self.conditioning_scale) != len(self.control_mode_ids):
            raise ValueError(
                "SDXL Union guidance currently expects pose and depth conditioning scales "
                "to match control_modes."
            )

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        from diffusers import (
            AutoencoderKL,
            ControlNetUnionModel,
            EulerAncestralDiscreteScheduler,
            StableDiffusionXLControlNetUnionPipeline,
        )

        threestudio.info("Loading SDXL Union ControlNet guidance ...")
        self.controlnet = ControlNetUnionModel.from_pretrained(
            self.cfg.controlnet_model_name_or_path,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
            local_files_only=self.cfg.local_files_only,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.cfg.vae_model_name_or_path,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
            local_files_only=self.cfg.local_files_only,
        )
        self.pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
            local_files_only=self.cfg.local_files_only,
        )
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps, device=self.device)

        if self.cfg.enable_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()
        for module in [self.vae, self.unet, self.controlnet]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps(self.cfg.min_step_percent, self.cfg.max_step_percent)
        threestudio.info("Loaded SDXL Union ControlNet guidance.")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.05, max_step_percent=0.8):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        num_inference_steps = len(self.scheduler.timesteps)
        self.min_step_index = max(0, int(num_inference_steps * min_step_percent))
        self.max_step_index = min(
            num_inference_steps - 1,
            max(self.min_step_index, int(num_inference_steps * max_step_percent)),
        )

    def sample_timesteps(self, batch_size: int) -> Tensor:
        timestep_indices = torch.randint(
            self.min_step_index,
            self.max_step_index + 1,
            [1],
            dtype=torch.long,
            device=self.device,
        )
        timestep = self.scheduler.timesteps.to(self.device)[timestep_indices]
        return timestep.repeat(batch_size)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B 4 h w"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    def make_add_time_ids(self, batch_size: int, dtype, device) -> Float[Tensor, "B 6"]:
        resolution = self.cfg.guidance_resolution
        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            dtype=dtype,
            device=device,
        )
        return add_time_ids.repeat(batch_size, 1)

    def make_control_type(self, batch_size: int, dtype, device) -> Float[Tensor, "B N"]:
        num_control_type = int(self.controlnet.config.num_control_type)
        control_type = torch.zeros(
            (batch_size, num_control_type),
            dtype=dtype,
            device=device,
        )
        for control_mode_id in self.control_mode_ids:
            control_type[:, control_mode_id] = 1
        return control_type

    def prepare_image_cond(self, control_image, batch_repeat: int):
        resolution = self.cfg.guidance_resolution
        if not isinstance(control_image, list):
            raise TypeError("SDXL Union guidance expects independent pose and depth Control Conditions")
        return [
            F.interpolate(img, (resolution, resolution), mode="bilinear", align_corners=False)
            .to(self.weights_dtype)
            .repeat(batch_repeat, 1, 1, 1)
            for img in control_image
        ]

    def __call__(
        self,
        rgb: Float[Tensor, "B C H W"],
        control_image,
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        **kwargs,
    ):
        if not hasattr(prompt_utils, "get_sdxl_text_embeddings"):
            raise TypeError(
                "SDXL Union guidance requires stable-diffusion-xl-prompt-processor output"
            )

        batch_size = rgb.shape[0]
        resolution = self.cfg.guidance_resolution
        if rgb_as_latents:
            latents = F.interpolate(
                rgb, (resolution // 8, resolution // 8), mode="bilinear", align_corners=False
            )
        else:
            latents = self.encode_images(
                F.interpolate(rgb, (resolution, resolution), mode="bilinear", align_corners=False)
            )

        text = prompt_utils.get_sdxl_text_embeddings(
            elevation, azimuth, camera_distances, True
        )
        t = self.sample_timesteps(batch_size)
        image_cond = self.prepare_image_cond(control_image, batch_repeat=2)
        grad = self.compute_grad_sds(text, latents, image_cond, t)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {"loss_sds": loss_sds, "grad_norm": grad.norm()}

    def compute_grad_sds(self, text, latents, image_cond, t):
        batch_size = latents.shape[0]
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_input = torch.cat([t] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t[0])

            prompt_embeds = torch.cat(
                [
                    text["negative_prompt_embeds"],
                    text["prompt_embeds"],
                ],
                dim=0,
            ).to(self.weights_dtype)
            add_text_embeds = torch.cat(
                [
                    text["negative_pooled_prompt_embeds"],
                    text["pooled_prompt_embeds"],
                ],
                dim=0,
            ).to(self.weights_dtype)
            add_time_ids = self.make_add_time_ids(
                batch_size * 2,
                dtype=prompt_embeds.dtype,
                device=latents.device,
            )
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            control_type = self.make_control_type(
                batch_size * 2,
                dtype=prompt_embeds.dtype,
                device=latents.device,
            )

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input.to(self.weights_dtype),
                t_input,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=image_cond,
                control_type=control_type,
                control_type_idx=self.control_mode_ids,
                conditioning_scale=self.conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            noise_pred = self.unet(
                latent_model_input.to(self.weights_dtype),
                t_input,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
        grad = noise_pred - noise
        return grad
