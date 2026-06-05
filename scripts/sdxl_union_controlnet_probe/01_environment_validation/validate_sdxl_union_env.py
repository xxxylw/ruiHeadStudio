#!/usr/bin/env python
"""Validate the ruiheadstudio-sdxl-union-controlnet environment."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = Path(
    "outputs/sdxl_union_controlnet_probe/01_environment_validation/validate_env.json"
)


def check_import(module_name: str, symbol: str | None = None) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, symbol) if symbol else module
        return {
            "ok": True,
            "module": module_name,
            "symbol": symbol,
            "version": getattr(module, "__version__", None),
            "repr": repr(value),
        }
    except Exception as exc:
        return {
            "ok": False,
            "module": module_name,
            "symbol": symbol,
            "error": repr(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checks = [
        check_import("torch"),
        check_import("diffusers"),
        check_import("transformers"),
        check_import("diffusers", "ControlNetUnionModel"),
        check_import("diffusers", "StableDiffusionXLControlNetUnionPipeline"),
        check_import("diffusers", "EulerAncestralDiscreteScheduler"),
        check_import("diffusers", "AutoencoderKL"),
        check_import("threestudio.utils.head_v2", "FlamePointswRandomExp"),
    ]
    report = {
        "environment": "ruiheadstudio-sdxl-union-controlnet",
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae_model": "madebyollin/sdxl-vae-fp16-fix",
        "controlnet_model": "xinsir/controlnet-union-sdxl-1.0",
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
