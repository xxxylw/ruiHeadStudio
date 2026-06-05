#!/usr/bin/env python
"""Check whether required SDXL Union model snapshots are present in cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_OUTPUT = Path(
    "outputs/sdxl_union_controlnet_probe/01_environment_validation/check_cache.json"
)

REQUIRED_REPOS = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer_config.json",
        "scheduler/scheduler_config.json",
    ],
    "madebyollin/sdxl-vae-fp16-fix": [
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ],
    "xinsir/controlnet-union-sdxl-1.0": [
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ],
}


def inspect_repo(repo: str, required_files: list[str]) -> dict[str, object]:
    try:
        snapshot = Path(snapshot_download(repo, local_files_only=True))
    except Exception as exc:
        return {
            "ok": False,
            "repo": repo,
            "snapshot": None,
            "missing": required_files,
            "error": repr(exc),
        }

    missing = [file for file in required_files if not (snapshot / file).exists()]
    return {
        "ok": not missing,
        "repo": repo,
        "snapshot": str(snapshot),
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repos = [
        inspect_repo(repo, files)
        for repo, files in REQUIRED_REPOS.items()
    ]
    report = {
        "environment": "ruiheadstudio-sdxl-union-controlnet",
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae_model": "madebyollin/sdxl-vae-fp16-fix",
        "controlnet_model": "xinsir/controlnet-union-sdxl-1.0",
        "repos": repos,
        "ok": all(repo["ok"] for repo in repos),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
