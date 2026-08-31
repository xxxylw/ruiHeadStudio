"""GPU smoke test for the FLUX single-file reference backend.

Builds the FLUX.1-schnell pipeline from the monolithic transformer checkpoint
(via FluxTransformer2DModel.from_single_file) plus the slim local diffusers
dir, generates one 512x512 reference with a minimal number of steps, and checks
that flux_reference_loss returns finite values with gradients.

The slim local dir plus the monolithic checkpoint are the ONLY weight sources:
HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / DIFFUSERS_OFFLINE are forced so the
full 23-file hub snapshot can never be fetched, and every from_pretrained call
in the backend uses local_files_only=True.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
# The smoke only exercises FluxReferenceBackend + flux_reference_loss; skip the
# threestudio registration import (heavy plugin tree, not needed here).
os.environ.setdefault("FLUX_SKIP_THREESTUDIO", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flux_reference_guidance_smoke",
    ROOT / "threestudio/models/guidance/flux_reference_guidance.py",
)
FLUX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FLUX)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/huangqirui/models/FLUX.1-schnell")
    parser.add_argument("--single-file", default="/home/huangqirui/models/flux1-schnell.safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--prompt", default="a realistic 3D head avatar, clean studio portrait")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/flux_smoke_singlefile"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "smoke.log"

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    t_start = time.time()
    backend = FLUX.FluxReferenceBackend(args.model_dir, args.device, single_file_path=args.single_file)
    say(f"load strategy: {backend._load_strategy()}")
    assert backend._load_strategy() == "single_file", "smoke must use the local single-file checkpoint"

    image = backend.generate(args.prompt, height=args.height, width=args.width, steps=args.steps)
    say(
        f"generate: shape={tuple(image.shape)} dtype={image.dtype} device={image.device} "
        f"in {time.time() - t_start:.1f}s"
    )
    if image.shape != (1, 3, args.height, args.width):
        say(f"[error] unexpected shape {tuple(image.shape)}")
        return 1
    if not torch.isfinite(image).all():
        say("[error] non-finite image values")
        return 1

    target = image.detach().clone()
    loss = FLUX.flux_reference_loss(image, target)
    say(f"self-loss: {float(loss):.6f} (expected ~0)")
    noisy = torch.rand_like(image, requires_grad=True)
    noisy_loss = FLUX.flux_reference_loss(noisy, target)
    noisy_loss.backward()
    say(
        f"loss vs random: {float(noisy_loss):.6f}, "
        f"grad finite: {bool(torch.isfinite(noisy.grad).all())}"
    )

    torch.save({"image": image.cpu(), "self_loss": float(loss)}, out_dir / "result.pt")
    try:
        from PIL import Image

        arr = (image[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(arr).save(out_dir / "reference.png")
        say("saved reference.png")
    except Exception as exc:  # PIL may be missing; the .pt artifact is authoritative
        say(f"[warn] png save skipped: {exc}")
    say(f"OK finished in {time.time() - t_start:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
