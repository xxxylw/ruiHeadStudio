"""Small differentiable helpers used by the optional CLIP alignment loss."""

import os
from typing import Optional
import torch
import torch.nn.functional as F


def clip_alignment_weight(base_weight: float, global_step: int, start_step: int) -> float:
    """Keep CLIP gradients disabled until SDS has formed a stable coarse head."""
    return 0.0 if global_step < start_step else base_weight


def clip_decay_weight(
    base_weight: float,
    global_step: int,
    decay_start_step: int,
    decay_end_step: int,
) -> float:
    """Linearly decay an active CLIP weight over a late refinement window."""
    if base_weight <= 0.0 or decay_end_step <= decay_start_step:
        return max(0.0, base_weight)
    if global_step <= decay_start_step:
        return base_weight
    if global_step >= decay_end_step:
        return 0.0
    progress = (global_step - decay_start_step) / float(decay_end_step - decay_start_step)
    return base_weight * (1.0 - progress)


def quality_ramp_weight(
    base_weight: float,
    global_step: int,
    ramp_start_step: int,
    ramp_end_step: int,
) -> float:
    """Linearly ramp a quality penalty after a semantic checkpoint."""
    if base_weight <= 0.0 or ramp_end_step <= ramp_start_step:
        return max(0.0, base_weight) if global_step >= ramp_start_step else 0.0
    if global_step <= ramp_start_step:
        return 0.0
    if global_step >= ramp_end_step:
        return base_weight
    progress = (global_step - ramp_start_step) / float(ramp_end_step - ramp_start_step)
    return base_weight * progress


def normalized_parameter_drift(
    current: torch.Tensor,
    anchor: torch.Tensor,
    normalizer: Optional[torch.Tensor] = None,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Measure mean absolute drift while keeping the anchor outside autograd."""
    if current.shape != anchor.shape:
        raise ValueError("current and anchor must have the same shape")
    if normalizer is None:
        normalizer = torch.ones_like(anchor)
    if normalizer.shape != current.shape:
        raise ValueError("normalizer must have the same shape as current")
    scale = normalizer.detach().abs().clamp_min(epsilon)
    return ((current - anchor.detach()).abs() / scale).mean()


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


def frequency_quality_loss(
    images: torch.Tensor,
    opacity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize normalized high-frequency energy in rendered RGB images."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("images must be at least 2x2")

    weight = torch.ones(
        (images.shape[0], 1, images.shape[-2], images.shape[-1]),
        dtype=images.dtype,
        device=images.device,
    )
    if opacity is not None:
        if opacity.ndim == 4 and opacity.shape[-1] == 1:
            opacity = opacity.permute(0, 3, 1, 2)
        elif opacity.ndim == 4 and opacity.shape[1] == 1:
            pass
        elif opacity.ndim == 3:
            opacity = opacity.unsqueeze(1)
        else:
            raise ValueError("opacity must be [B,H,W] or [B,1,H,W]")
        if opacity.shape[0] != images.shape[0] or opacity.shape[-2:] != images.shape[-2:]:
            raise ValueError("images and opacity must share batch and spatial dimensions")
        weight = 0.25 + 0.75 * opacity.detach().to(dtype=images.dtype)

    def scale_loss(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = rgb * mask
        intensity = weighted.abs().mean(dim=(1, 2, 3), keepdim=True).detach().clamp_min(1.0e-3)
        tv_x = (weighted[..., :, 1:] - weighted[..., :, :-1]).abs().mean(dim=(1, 2, 3))
        tv_y = (weighted[..., 1:, :] - weighted[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=rgb.dtype,
            device=rgb.device,
        ).view(1, 1, 3, 3).repeat(rgb.shape[1], 1, 1, 1)
        padded = F.pad(weighted, (1, 1, 1, 1), mode="replicate")
        laplacian = F.conv2d(padded, kernel, groups=rgb.shape[1])
        lap = laplacian.abs().mean(dim=(1, 2, 3))
        return ((tv_x + tv_y + lap) / intensity.flatten()).mean()

    full = scale_loss(images, weight)
    half_images = F.avg_pool2d(images, kernel_size=2, stride=2)
    half_weight = F.avg_pool2d(weight, kernel_size=2, stride=2)
    half = scale_loss(half_images, half_weight)
    return 0.5 * (full + half)


def rendered_reference_loss(
    images: torch.Tensor,
    references: torch.Tensor,
    opacity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep a rendered image close to a detached teacher view and its edges."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if references.shape != images.shape:
        raise ValueError("images and references must have the same shape")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("images must be at least 2x2")

    weight = torch.ones(
        (images.shape[0], 1, images.shape[-2], images.shape[-1]),
        dtype=images.dtype,
        device=images.device,
    )
    if opacity is not None:
        if opacity.ndim == 4 and opacity.shape[-1] == 1:
            opacity = opacity.permute(0, 3, 1, 2)
        elif opacity.ndim == 4 and opacity.shape[1] == 1:
            pass
        elif opacity.ndim == 3:
            opacity = opacity.unsqueeze(1)
        else:
            raise ValueError("opacity must be [B,H,W] or [B,1,H,W]")
        if opacity.shape[0] != images.shape[0] or opacity.shape[-2:] != images.shape[-2:]:
            raise ValueError("images and opacity must share batch and spatial dimensions")
        weight = 0.25 + 0.75 * opacity.detach().to(dtype=images.dtype)

    reference = references.detach()
    difference = images - reference
    color = (torch.sqrt(difference.square() + 1.0e-6) - 1.0e-3)
    color = (color * weight).mean() / weight.mean().clamp_min(1.0e-6)
    current_dx = images[..., :, 1:] - images[..., :, :-1]
    reference_dx = reference[..., :, 1:] - reference[..., :, :-1]
    current_dy = images[..., 1:, :] - images[..., :-1, :]
    reference_dy = reference[..., 1:, :] - reference[..., :-1, :]
    edge_x = (current_dx - reference_dx).abs().mean()
    edge_y = (current_dy - reference_dy).abs().mean()
    return 0.7 * color + 0.3 * (edge_x + edge_y)


def reference_statistics_loss(
    images: torch.Tensor,
    references: torch.Tensor,
    opacity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match local contrast and edge magnitude without copying teacher pixels."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if references.shape != images.shape:
        raise ValueError("images and references must have the same shape")
    if images.shape[-2] < 5 or images.shape[-1] < 5:
        raise ValueError("images must be at least 5x5")

    weight = torch.ones(
        (images.shape[0], 1, images.shape[-2], images.shape[-1]),
        dtype=images.dtype,
        device=images.device,
    )
    if opacity is not None:
        if opacity.ndim == 4 and opacity.shape[-1] == 1:
            opacity = opacity.permute(0, 3, 1, 2)
        elif opacity.ndim == 4 and opacity.shape[1] == 1:
            pass
        elif opacity.ndim == 3:
            opacity = opacity.unsqueeze(1)
        else:
            raise ValueError("opacity must be [B,H,W] or [B,1,H,W]")
        if opacity.shape[0] != images.shape[0] or opacity.shape[-2:] != images.shape[-2:]:
            raise ValueError("images and opacity must share batch and spatial dimensions")
        weight = 0.25 + 0.75 * opacity.detach().to(dtype=images.dtype)

    def statistics(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_mean = F.avg_pool2d(rgb, kernel_size=5, stride=1, padding=2)
        local_second = F.avg_pool2d(rgb.square(), kernel_size=5, stride=1, padding=2)
        contrast = (local_second - local_mean.square()).clamp_min(0.0).sqrt()
        dx = rgb[..., :, 1:] - rgb[..., :, :-1]
        dy = rgb[..., 1:, :] - rgb[..., :-1, :]
        edge = torch.zeros_like(rgb)
        edge[..., :, 1:] += dx.abs()
        edge[..., 1:, :] += dy.abs()
        return contrast, edge

    current_contrast, current_edge = statistics(images * weight)
    with torch.no_grad():
        reference_contrast, reference_edge = statistics(references.detach() * weight)
    contrast_loss = ((current_contrast - reference_contrast).abs() * weight).mean()
    edge_loss = ((current_edge - reference_edge).abs() * weight).mean()
    return 0.6 * contrast_loss + 0.4 * edge_loss


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
