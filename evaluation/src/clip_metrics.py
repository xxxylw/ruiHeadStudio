from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .dataset import Example


CLIP_MODELS = ("ViT-L/14", "ViT-B/16", "ViT-B/32")


def rank_for_target(scores: Sequence[float], target_index: int) -> int:
    """Return the one-based optimistic rank for a target prompt score."""
    if not 0 <= target_index < len(scores):
        raise IndexError("target index is outside the score vector")
    target_score = scores[target_index]
    return 1 + sum(score > target_score for score in scores)


def _unique_prompts(examples: Sequence[Example]) -> list[str]:
    prompts = list(dict.fromkeys(example.prompt for example in examples))
    if not prompts:
        raise ValueError("examples must not be empty")
    return prompts


def score_clip(
    examples: Sequence[Example], model_name: str, device: str, batch_size: int
) -> list[dict[str, Any]]:
    """Evaluate matched CLIP cosine and 26-way prompt rank for every image."""
    if model_name not in CLIP_MODELS:
        raise ValueError(f"unsupported CLIP model: {model_name}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    import clip
    import torch

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    prompts = _unique_prompts(examples)
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}

    with torch.inference_mode():
        text_features = model.encode_text(clip.tokenize(prompts).to(device)).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        rows: list[dict[str, Any]] = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            images = torch.stack(
                [preprocess(Image.open(example.image_path).convert("RGB")) for example in batch]
            ).to(device)
            image_features = model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).cpu().tolist()
            for example, scores in zip(batch, similarities):
                target_index = prompt_index[example.prompt]
                rows.append(
                    {
                        "model": model_name,
                        "run_name": example.run_name,
                        "prompt": example.prompt,
                        "image_path": str(Path(example.image_path)),
                        "view_index": example.view_index,
                        "score": float(scores[target_index]),
                        "rank": rank_for_target(scores, target_index),
                    }
                )
    return rows
