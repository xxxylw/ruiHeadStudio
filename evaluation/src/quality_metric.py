from __future__ import annotations

from pathlib import Path


def score_piqe(examples, device: str) -> list[dict]:
    """Compute PIQE, a no-reference distortion score (lower is better)."""
    import pyiqa

    metric = pyiqa.create_metric("piqe", device=device)
    rows: list[dict] = []
    for example in examples:
        rows.append(
            {
                "model": "PIQE",
                "run_name": example.run_name,
                "prompt": example.prompt,
                "image_path": str(example.image_path),
                "view_index": example.view_index,
                "score": float(metric(str(Path(example.image_path)))),
            }
        )
    return rows


def score_musiq(examples, device: str) -> list[dict]:
    """Compute MUSIQ, a no-reference perceptual image-quality score (higher is better)."""
    import math
    import pyiqa
    import torch

    metric = pyiqa.create_metric("musiq", device=device)
    rows: list[dict] = []
    with torch.inference_mode():
        for example in examples:
            score = float(metric(str(Path(example.image_path))).item())
            if not math.isfinite(score):
                raise ValueError(f"MUSIQ returned a non-finite value for {example.image_path}")
            rows.append(
                {
                    "model": "MUSIQ",
                    "run_name": example.run_name,
                    "prompt": example.prompt,
                    "image_path": str(example.image_path),
                    "view_index": example.view_index,
                    "score": score,
                }
            )
    return rows
