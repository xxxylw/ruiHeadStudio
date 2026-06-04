#!/usr/bin/env python
"""Probe ordinary FLUX ControlNet generation with RuiHeadStudio FLAME controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threestudio.utils.head_v2 import FlamePointswRandomExp


DEFAULT_PROMPT = (
    "high fidelity realistic portrait of Cristiano Ronaldo, athletic Portuguese male face, "
    "short dark hair, defined jawline, strong cheekbones, natural skin texture, realistic eyes, "
    "coherent head shape, studio lighting, ultra detailed, DSLR photo"
)


def tensor_to_pil(image: torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"expected batch size 1 image, got shape {image.shape}")
        image = image[0]
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def image_stats(image: torch.Tensor | np.ndarray) -> dict[str, object]:
    if isinstance(image, torch.Tensor):
        arr = image.detach().float().cpu().numpy()
    else:
        arr = image.astype(np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "nonzero_fraction": float((arr > 0).mean()),
    }


def make_flame_conditions(args: argparse.Namespace) -> tuple[Image.Image, Image.Image, dict[str, object]]:
    device = torch.device(args.device)
    skel = FlamePointswRandomExp(
        args.flame_path,
        gender=args.gender,
        device=args.device,
        batch_size=1,
        image_size=args.height,
        flame_scale=args.flame_scale,
    )

    at = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    dist = torch.tensor([args.distance], dtype=torch.float32, device=device)
    elev = torch.tensor([args.elevation], dtype=torch.float32, device=device)
    azim = torch.tensor([args.azimuth], dtype=torch.float32, device=device)
    fov = torch.tensor([args.fov], dtype=torch.float32, device=device)

    conds = skel.get_cond_pose_depth(dist=dist, elev=elev, azim=azim, at=at, up=up, fov=fov)
    pose = conds["pose"]
    depth = conds["depth"]

    if pose.shape != depth.shape:
        raise ValueError(f"pose/depth shape mismatch: pose={pose.shape}, depth={depth.shape}")
    if pose.shape[-3:] != (args.height, args.width, 3):
        raise ValueError(f"unexpected condition shape {pose.shape}; expected [1,{args.height},{args.width},3]")

    pose_pil = tensor_to_pil(pose)
    depth_pil = tensor_to_pil(depth)
    metadata = {
        "camera": {
            "distance": args.distance,
            "elevation": args.elevation,
            "azimuth": args.azimuth,
            "fov": args.fov,
        },
        "pose": image_stats(pose),
        "depth": image_stats(depth),
    }
    return pose_pil, depth_pil, metadata


def run_generation(args: argparse.Namespace, pose_pil: Image.Image, depth_pil: Image.Image) -> Image.Image:
    from diffusers import FluxControlNetModel, FluxControlNetPipeline
    from diffusers.models import FluxMultiControlNetModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    controlnet = FluxControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    controlnet = FluxMultiControlNetModel([controlnet])
    pipe = FluxControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    else:
        pipe.to(args.device)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    result = pipe(
        args.prompt,
        control_image=[pose_pil, depth_pil],
        width=args.width,
        height=args.height,
        controlnet_conditioning_scale=[args.pose_scale, args.depth_scale],
        control_guidance_end=[args.pose_guidance_end, args.depth_guidance_end],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=1.0,
        generator=generator,
    ).images[0]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flame-path", default="./ckpts/FLAME-2000")
    parser.add_argument("--gender", default="generic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--flame-scale", type=float, default=-10.0)
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--controlnet-model", default="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--condition-only", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--pose-scale", type=float, default=0.9)
    parser.add_argument("--depth-scale", type=float, default=0.8)
    parser.add_argument("--pose-guidance-end", type=float, default=0.65)
    parser.add_argument("--depth-guidance-end", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "probe_report.json"

    report: dict[str, object] = {
        "ok": False,
        "condition_only": args.condition_only,
        "prompt": args.prompt,
        "models": {
            "base": args.base_model,
            "controlnet": args.controlnet_model,
        },
        "controls": {
            "control_image_order": ["pose", "depth"],
            "controlnet_conditioning_scale": [args.pose_scale, args.depth_scale],
            "control_guidance_end": [args.pose_guidance_end, args.depth_guidance_end],
            "true_cfg_scale": 1.0,
            "guidance_scale": args.guidance_scale,
        },
    }

    try:
        pose_pil, depth_pil, cond_metadata = make_flame_conditions(args)
        pose_pil.save(args.output_dir / "flame_pose.png")
        depth_pil.save(args.output_dir / "flame_depth.png")
        report["conditions"] = cond_metadata

        if not args.condition_only:
            result = run_generation(args, pose_pil, depth_pil)
            result.save(args.output_dir / "flux_controlnet_pose_depth.png")
            report["generated_image"] = "flux_controlnet_pose_depth.png"

        report["ok"] = True
    except Exception as exc:  # pragma: no cover - probe script
        report["error"] = f"{type(exc).__name__}: {exc}"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
