#!/usr/bin/env python
"""Probe ordinary SDXL Union ControlNet generation with RuiHeadStudio FLAME controls."""

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

from threestudio.models.guidance.controlnet_union_sdxl_contract import resolve_control_modes
from threestudio.utils.head_v2 import FlamePointswRandomExp


DEFAULT_PROMPT = (
    "a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, "
    "ultra-HD, next generation"
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

    return tensor_to_pil(pose), tensor_to_pil(depth), {
        "camera": {
            "distance": args.distance,
            "elevation": args.elevation,
            "azimuth": args.azimuth,
            "fov": args.fov,
        },
        "pose": image_stats(pose),
        "depth": image_stats(depth),
    }


def run_generation(args: argparse.Namespace, pose_pil: Image.Image, depth_pil: Image.Image) -> Image.Image:
    from diffusers import (
        AutoencoderKL,
        ControlNetUnionModel,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLControlNetUnionPipeline,
    )

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    controlnet = ControlNetUnionModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        generator_device = args.device
    else:
        pipe = pipe.to(args.device)
        generator_device = args.device

    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    control_mode_ids = resolve_control_modes(args.control_modes)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        control_image=[pose_pil, depth_pil],
        control_mode=control_mode_ids,
        controlnet_conditioning_scale=[args.pose_scale, args.depth_scale],
        control_guidance_start=[args.pose_guidance_start, args.depth_guidance_start],
        control_guidance_end=[args.pose_guidance_end, args.depth_guidance_end],
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation"))
    parser.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae-model", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--controlnet-model", default="xinsir/controlnet-union-sdxl-1.0")
    parser.add_argument("--control-modes", nargs="+", default=["openpose", "depth"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--flame-path", default="./ckpts/FLAME-2000")
    parser.add_argument("--gender", default="generic")
    parser.add_argument("--flame-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--distance", type=float, default=1.6)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--fov", type=float, default=20.0)
    parser.add_argument("--pose-scale", type=float, default=1.0)
    parser.add_argument("--depth-scale", type=float, default=0.8)
    parser.add_argument("--pose-guidance-start", type=float, default=0.0)
    parser.add_argument("--depth-guidance-start", type=float, default=0.0)
    parser.add_argument("--pose-guidance-end", type=float, default=0.65)
    parser.add_argument("--depth-guidance-end", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": "ruiheadstudio-sdxl-union-controlnet",
        "base_model": args.base_model,
        "vae_model": args.vae_model,
        "controlnet_model": args.controlnet_model,
        "control_modes": args.control_modes,
        "control_mode_ids": resolve_control_modes(args.control_modes),
        "prompt": args.prompt,
        "status": "started",
    }
    try:
        pose_pil, depth_pil, condition_metadata = make_flame_conditions(args)
        pose_pil.save(args.output_dir / "flame_pose.png")
        depth_pil.save(args.output_dir / "flame_depth.png")
        report["conditions"] = condition_metadata
        result = run_generation(args, pose_pil, depth_pil)
        result.save(args.output_dir / "sdxl_union_pose_depth.png")
        report["generated_image"] = "sdxl_union_pose_depth.png"
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        (args.output_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
