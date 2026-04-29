from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


def _to_1d_float_tensor(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    return values.detach().float().reshape(-1).cpu()


def summarize_tensor(values: torch.Tensor) -> Dict[str, Optional[float]]:
    tensor = _to_1d_float_tensor(values)
    if tensor.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }

    quantiles = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def classify_by_face_normals(
    face_normals: torch.Tensor,
    front_threshold: float = 0.35,
    rear_threshold: float = -0.35,
) -> List[str]:
    normals = face_normals.detach().float().cpu()
    if normals.ndim != 2 or normals.shape[-1] != 3:
        raise ValueError("face_normals must have shape [N, 3]")

    labels: List[str] = []
    for normal in normals:
        x = float(normal[0].item())
        if x >= front_threshold:
            labels.append("front")
        elif x <= rear_threshold:
            labels.append("rear")
        else:
            labels.append("side")
    return labels


def _mask_for_labels(labels: Iterable[str], target: str) -> torch.Tensor:
    return torch.tensor([label == target for label in labels], dtype=torch.bool)


def summarize_gaussian_regions(
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    labels: List[str],
) -> Dict[str, object]:
    opacity_flat = _to_1d_float_tensor(opacity)
    scaling_tensor = scaling.detach().float().cpu()
    if scaling_tensor.ndim == 2:
        scaling_max = scaling_tensor.max(dim=1).values
    else:
        scaling_max = scaling_tensor.reshape(-1)

    if opacity_flat.numel() != len(labels) or scaling_max.numel() != len(labels):
        raise ValueError(
            "opacity, scaling, and labels must describe the same number of gaussians"
        )

    regions: Dict[str, object] = {}
    for region in ("front", "side", "rear"):
        mask = _mask_for_labels(labels, region)
        regions[region] = {
            "count": int(mask.sum().item()),
            "opacity": summarize_tensor(opacity_flat[mask]),
            "scaling_max": summarize_tensor(scaling_max[mask]),
        }

    return {
        "total": {
            "count": int(opacity_flat.numel()),
            "opacity": summarize_tensor(opacity_flat),
            "scaling_max": summarize_tensor(scaling_max),
        },
        "regions": regions,
    }
