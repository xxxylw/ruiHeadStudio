import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def build_flux_view_prompt(prompt: str, azimuth: float, elevation: float) -> str:
    """Make camera metadata explicit because the FLUX reference has no ControlNet depth input."""
    return (
        f"{prompt}, a clean studio portrait, camera azimuth {azimuth:.1f} degrees, "
        f"camera elevation {elevation:.1f} degrees, consistent identity and facial structure"
    )


def flux_reference_loss(
    image: torch.Tensor,
    reference: torch.Tensor,
    opacity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Opacity-weighted Charbonnier plus edge matching for a frozen FLUX reference."""
    if image.ndim != 4 or reference.ndim != 4:
        raise ValueError("image and reference must be BCHW")
    if image.shape != reference.shape:
        raise ValueError("image and reference must have the same shape")
    diff = torch.sqrt((image - reference.detach()).square() + 1.0e-6) - 1.0e-3
    diff = diff.clamp_min(0.0)
    if opacity is not None:
        if opacity.ndim == 4 and opacity.shape[-1] == 1:
            opacity = opacity.permute(0, 3, 1, 2)
        if opacity.ndim != 4 or opacity.shape[0] != image.shape[0]:
            raise ValueError("opacity must be BCHW or BHWC")
        opacity = F.interpolate(opacity.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
        weight = opacity.clamp_min(0.05)
        diff = diff * weight
        normalizer = weight.mean().clamp_min(1.0e-6)
    else:
        normalizer = diff.new_tensor(1.0)
    image_dx = image[..., :, 1:] - image[..., :, :-1]
    reference_dx = reference.detach()[..., :, 1:] - reference.detach()[..., :, :-1]
    image_dy = image[..., 1:, :] - image[..., :-1, :]
    reference_dy = reference.detach()[..., 1:, :] - reference.detach()[..., :-1, :]
    edge_loss = (image_dx - reference_dx).abs().mean() + (image_dy - reference_dy).abs().mean()
    return diff.mean() / normalizer + 0.1 * edge_loss


def reference_cache_key(
    prompt: str, azimuth: float, elevation: float, angle_bin: float = 30.0
) -> str:
    """Quantize camera angles so random 3DGS views reuse a finite FLUX bank."""
    if angle_bin <= 0:
        raise ValueError("angle_bin must be positive")
    quantized_azimuth = round(azimuth / angle_bin) * angle_bin
    quantized_elevation = round(elevation / angle_bin) * angle_bin
    value = f"{prompt}|{quantized_azimuth:.2f}|{quantized_elevation:.2f}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


class FluxReferenceBackend:
    """Lazy FLUX image teacher. The optional diffusers dependency is imported only when enabled."""

    def __init__(
        self,
        model_name: str,
        device: str,
        cache_dir: Optional[str] = None,
        single_file_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.single_file_path = Path(single_file_path).expanduser() if single_file_path else None
        self.pipe = None

    def _load_strategy(self) -> str:
        """Choose the weight source: "single_file" for the monolithic checkpoint, else the diffusers pipeline."""
        if self.single_file_path is not None and self.single_file_path.is_file():
            return "single_file"
        return "pipeline"

    def _load(self):
        if self._load_strategy() == "single_file":
            self.pipe = self._build_single_file_pipe()
        else:
            self.pipe = self._build_pipeline()
        if hasattr(self.pipe, "enable_sequential_cpu_offload"):
            # FLUX transformer + T5 exceed a 24 GiB card when whole modules
            # are moved at once; sequential offload keeps the smoke/training
            # path usable on the available RTX 3090 GPUs.
            self.pipe.enable_sequential_cpu_offload()
        elif hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def _build_pipeline(self):
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX guidance requires a recent diffusers installation; "
                "the legacy SD-only environment is not sufficient"
            ) from exc
        return DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_files_only=True,
        )

    def _build_single_file_pipe(self):
        """Build the FLUX pipeline from the monolithic transformer checkpoint plus a slim local diffusers repo."""
        try:
            from diffusers import (
                AutoencoderKL,
                FlowMatchEulerDiscreteScheduler,
                FluxPipeline,
                FluxTransformer2DModel,
            )
            from transformers import (
                CLIPTextModel,
                CLIPTokenizer,
                T5EncoderModel,
                T5TokenizerFast,
            )
        except ImportError as exc:
            raise RuntimeError(
                "FLUX guidance requires a recent diffusers installation; "
                "the legacy SD-only environment is not sufficient"
            ) from exc
        transformer = FluxTransformer2DModel.from_single_file(
            str(self.single_file_path),
            config=str(self.model_name),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        return FluxPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_name, subfolder="scheduler", local_files_only=True
            ),
            text_encoder=CLIPTextModel.from_pretrained(
                self.model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
            ),
            tokenizer=CLIPTokenizer.from_pretrained(
                self.model_name, subfolder="tokenizer", local_files_only=True
            ),
            text_encoder_2=T5EncoderModel.from_pretrained(
                self.model_name, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, local_files_only=True
            ),
            tokenizer_2=T5TokenizerFast.from_pretrained(
                self.model_name,
                subfolder="tokenizer_2",
                local_files_only=True,
                # FLUX.1-schnell ships only tokenizer.json (no spiece.model);
                # the config's add_prefix_space=True would force a slow
                # conversion (from_slow) that then fails. The fast tokenizer
                # already encodes the prefix space (Metaspace prepend_scheme).
                add_prefix_space=None,
            ),
            vae=AutoencoderKL.from_pretrained(
                self.model_name, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
            ),
            transformer=transformer,
        )

    @torch.no_grad()
    def generate(self, prompt: str, height: int = 512, width: int = 512, steps: int = 4) -> torch.Tensor:
        if self.pipe is None:
            self._load()
        result = self.pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps)
        image = result.images[0]
        tensor = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)


try:
    if os.environ.get("FLUX_SKIP_THREESTUDIO") == "1":
        # Standalone tests/smoke never need the threestudio registration; skip
        # the heavy plugin-tree import (diffusers/transformers/scipy) there.
        raise ImportError("threestudio skipped by FLUX_SKIP_THREESTUDIO=1")
    import threestudio
    from threestudio.utils.base import BaseObject
    from threestudio.utils.typing import *

    @threestudio.register("flux-reference-guidance")
    class FluxReferenceGuidance(BaseObject):
        """FLUX replaces SD as a frozen multi-view reference teacher."""

        @dataclass
        class Config(BaseObject.Config):
            model_name_or_path: str = "black-forest-labs/FLUX.1-schnell"
            single_file_path: Optional[str] = None
            prompt: str = "a realistic studio portrait of a 3D head avatar"
            cache_dir: Optional[str] = None
            reference_cache_dir: str = "./outputs/flux_references"
            reference_weight: float = 1.0
            edge_weight: float = 0.1
            reference_resolution: int = 512
            num_inference_steps: int = 4
            reference_angle_bin: float = 30.0
            min_step_percent: float = 0.0
            max_step_percent: float = 1.0

        cfg: Config

        def configure(self) -> None:
            self.backend = FluxReferenceBackend(
                self.cfg.model_name_or_path,
                str(self.device),
                self.cfg.cache_dir,
                self.cfg.single_file_path,
            )
            self.reference_cache = Path(self.cfg.reference_cache_dir).expanduser()
            self.reference_cache.mkdir(parents=True, exist_ok=True)

        def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
            self.min_step_percent = float(min_step_percent)
            self.max_step_percent = float(max_step_percent)

        def _reference(self, prompt: str, azimuth: float, elevation: float, height: int, width: int):
            key = reference_cache_key(
                prompt, azimuth, elevation, self.cfg.reference_angle_bin
            )
            path = self.reference_cache / f"{key}.pt"
            if path.exists():
                return torch.load(path, map_location=self.device, weights_only=True)
            reference = self.backend.generate(
                build_flux_view_prompt(prompt, azimuth, elevation),
                height=height,
                width=width,
                steps=self.cfg.num_inference_steps,
            ).cpu()
            torch.save(reference, path)
            return reference.to(self.device)

        def __call__(self, rgb, control_image, prompt_utils, elevation, azimuth, **kwargs):
            del control_image, prompt_utils
            refs = [
                self._reference(
                    self.cfg.prompt,
                    float(azi.detach().cpu()),
                    float(ele.detach().cpu()),
                    self.cfg.reference_resolution,
                    self.cfg.reference_resolution,
                )
                for ele, azi in zip(elevation, azimuth)
            ]
            reference = torch.cat(refs, dim=0).to(device=rgb.device, dtype=rgb.dtype)
            if reference.shape[-2:] != rgb.shape[-2:]:
                reference = F.interpolate(reference, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            loss = flux_reference_loss(rgb, reference, kwargs.get("opacity"))
            return {"loss_sds": loss * self.cfg.reference_weight, "grad_norm": loss.detach().sqrt()}
except ImportError:
    # Keep the pure helper functions importable for offline tests.
    FluxReferenceGuidance = None
