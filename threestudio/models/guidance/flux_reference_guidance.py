import hashlib
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
    diff = torch.sqrt((image - reference.detach()).square() + 1.0e-6)
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


def reference_cache_key(prompt: str, azimuth: float, elevation: float) -> str:
    value = f"{prompt}|{azimuth:.2f}|{elevation:.2f}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


class FluxReferenceBackend:
    """Lazy FLUX image teacher. The optional diffusers dependency is imported only when enabled."""

    def __init__(self, model_name: str, device: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.pipe = None

    def _load(self):
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX guidance requires a recent diffusers installation; "
                "the legacy SD-only environment is not sufficient"
            ) from exc
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str, height: int = 512, width: int = 512, steps: int = 4) -> torch.Tensor:
        if self.pipe is None:
            self._load()
        result = self.pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps)
        image = result.images[0]
        tensor = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)


try:
    import threestudio
    from threestudio.utils.base import BaseObject
    from threestudio.utils.typing import *

    @threestudio.register("flux-reference-guidance")
    class FluxReferenceGuidance(BaseObject):
        """FLUX replaces SD as a frozen multi-view reference teacher."""

        @dataclass
        class Config(BaseObject.Config):
            model_name_or_path: str = "black-forest-labs/FLUX.1-schnell"
            prompt: str = "a realistic studio portrait of a 3D head avatar"
            cache_dir: Optional[str] = None
            reference_cache_dir: str = "./outputs/flux_references"
            reference_weight: float = 1.0
            edge_weight: float = 0.1
            num_inference_steps: int = 4
            min_step_percent: float = 0.0
            max_step_percent: float = 1.0

        cfg: Config

        def configure(self) -> None:
            self.backend = FluxReferenceBackend(
                self.cfg.model_name_or_path, str(self.device), self.cfg.cache_dir
            )
            self.reference_cache = Path(self.cfg.reference_cache_dir).expanduser()
            self.reference_cache.mkdir(parents=True, exist_ok=True)

        def _reference(self, prompt: str, azimuth: float, elevation: float, height: int, width: int):
            key = reference_cache_key(prompt, azimuth, elevation)
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
                    rgb.shape[-2],
                    rgb.shape[-1],
                )
                for ele, azi in zip(elevation, azimuth)
            ]
            reference = torch.cat(refs, dim=0).to(device=rgb.device, dtype=rgb.dtype)
            loss = flux_reference_loss(rgb, reference, kwargs.get("opacity"))
            return {"loss_sds": loss * self.cfg.reference_weight, "grad_norm": loss.detach().sqrt()}
except ImportError:
    # Keep the pure helper functions importable for offline tests.
    FluxReferenceGuidance = None
