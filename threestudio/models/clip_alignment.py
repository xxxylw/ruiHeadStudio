"""Small differentiable helpers used by the optional CLIP alignment loss."""

import os
from typing import Optional
import torch
import torch.nn.functional as F


def clip_alignment_weight(base_weight: float, global_step: int, start_step: int) -> float:
    """Keep CLIP gradients disabled until SDS has formed a stable coarse head."""
    return 0.0 if global_step < start_step else base_weight


def cosine_alignment_loss(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """Return mean cosine distance between normalized image and text features."""
    if text_features.ndim == 1:
        text_features = text_features.unsqueeze(0)
    return (1.0 - F.cosine_similarity(image_features, text_features, dim=-1)).mean()


def foreground_crop(images: torch.Tensor, opacity: torch.Tensor, padding: float = 0.12) -> torch.Tensor:
    """Crop each render to its visible foreground before CLIP resizing.

    Bounding boxes are deliberately detached: crop coordinates do not need gradients,
    while the selected image pixels retain their normal differentiable path.
    """
    if opacity.ndim == 4 and opacity.shape[-1] == 1:
        opacity = opacity.permute(0, 3, 1, 2)
    elif opacity.ndim == 4 and opacity.shape[1] == 1:
        pass
    elif opacity.ndim == 3:
        opacity = opacity.unsqueeze(1)
    else:
        raise ValueError("opacity must be [B,H,W] or [B,1,H,W]")
    if images.shape[0] != opacity.shape[0] or images.shape[-2:] != opacity.shape[-2:]:
        raise ValueError("images and opacity must share batch and spatial dimensions")

    _, _, height, width = images.shape
    crops = []
    for image, alpha in zip(images, opacity.detach()):
        foreground = alpha[0] > 0.1
        ys, xs = foreground.nonzero(as_tuple=True)
        if ys.numel() == 0:
            crop = image
        else:
            pad_y = int(height * padding)
            pad_x = int(width * padding)
            y0 = max(0, int(ys.min()) - pad_y)
            y1 = min(height, int(ys.max()) + 1 + pad_y)
            x0 = max(0, int(xs.min()) - pad_x)
            x1 = min(width, int(xs.max()) + 1 + pad_x)
            crop = image[:, y0:y1, x0:x1]
        crops.append(F.interpolate(crop.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False))
    return torch.cat(crops, dim=0)


class CLIPAlignment:
    """Frozen CLIP image encoder with a differentiable image preprocessing path."""

    def __init__(self, model_name: str, prompt: str, device: torch.device) -> None:
        import clip

        self.model, _ = clip.load(
            model_name, device=device, download_root=os.path.expanduser("~/.cache/clip")
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.prompts = {
            "base": prompt,
            "side": f"side view of {prompt}",
            "front": f"front view of {prompt}",
            "back": f"backside view of {prompt}",
        }
        with torch.no_grad():
            text_features = self.model.encode_text(
                clip.tokenize(list(self.prompts.values())).to(device)
            ).float()
        self.text_features = F.normalize(text_features, dim=-1)
        self.mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

    def text_features_for_azimuth(self, azimuth: torch.Tensor) -> torch.Tensor:
        """Match the prompt processor's front/back thresholds in degrees."""
        azimuth = (azimuth + 180.0) % 360.0 - 180.0
        indices = torch.ones_like(azimuth, dtype=torch.long)  # side
        indices[(azimuth > 45.0) & (azimuth < 135.0)] = 2  # front
        indices[(azimuth > -135.0) & (azimuth < -45.0)] = 3  # back
        return self.text_features[indices]

    def __call__(
        self,
        images: torch.Tensor,
        opacity: Optional[torch.Tensor] = None,
        azimuth: Optional[torch.Tensor] = None,
        foreground_only: bool = False,
        view_dependent: bool = False,
    ) -> torch.Tensor:
        if foreground_only and opacity is not None:
            images = foreground_crop(images, opacity)
        else:
            images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        image_features = self.model.encode_image((images - self.mean) / self.std).float()
        text_features = self.text_features[0]
        if view_dependent and azimuth is not None:
            text_features = self.text_features_for_azimuth(azimuth)
        return cosine_alignment_loss(F.normalize(image_features, dim=-1), text_features)
