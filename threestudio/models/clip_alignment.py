"""Small differentiable helpers used by the optional CLIP alignment loss."""

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


class CLIPAlignment:
    """Frozen CLIP image encoder with a differentiable image preprocessing path."""

    def __init__(self, model_name: str, prompt: str, device: torch.device) -> None:
        import clip

        self.model, _ = clip.load(model_name, device=device, download_root="~/.cache/clip")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize([prompt]).to(device)).float()
        self.text_features = F.normalize(text_features, dim=-1)
        self.mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        image_features = self.model.encode_image((images - self.mean) / self.std).float()
        return cosine_alignment_loss(F.normalize(image_features, dim=-1), self.text_features)
