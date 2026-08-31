"""Arrange the slim local diffusers model dir for FLUX.1-schnell.

The transformer weights live only in the monolithic checkpoint
/home/huangqirui/models/flux1-schnell.safetensors (loaded via
FluxTransformer2DModel.from_single_file), so this repo dir only needs the
transformer config plus the CLIP/T5/VAE auxiliary weights.

Download policy (resumable + retry-safe):
* huggingface.co is unreachable from this host and hf-mirror.com currently
  answers 403 on large-file GETs, so the primary mirror is the ModelScope
  copy of FLUX.1-schnell (identical LFS objects, verified by sha256).
* Each weight streams to a ``.part`` sibling and is resumed with an HTTP
  Range request when interrupted; the part is atomically renamed into place
  only after the sha256 matches the LFS hash from the ModelScope listing.
* Verified files are recorded in ``.flux_slim_state.json`` next to the model
  dir so re-runs skip straight to "already present".

No hub auth token is required for these public weights.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

DEST = Path("/home/huangqirui/models/FLUX.1-schnell")

# Files already fully cached in the hub snapshot (configs, tokenizers, CLIP weights).
CACHED_FILES: List[str] = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/special_tokens_map.json",
    "vae/config.json",
    "transformer/config.json",
]

# rel -> (expected size, sha256 from the ModelScope LFS listing).
# The LFS objects are identical to the Hugging Face ones (same sha256 as the
# hub metadata already stored next to model-00001-of-00002.safetensors).
MANIFEST: Dict[str, Tuple[int, str]] = {
    "text_encoder_2/model-00001-of-00002.safetensors": (
        4_994_582_224,
        "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e",
    ),
    "text_encoder_2/model-00002-of-00002.safetensors": (
        4_530_066_360,
        "a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00",
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        167_666_902,
        "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
    ),
}

# Mirror bases tried in order; ModelScope is the one that currently progresses.
MIRRORS: List[str] = [
    "https://modelscope.cn/models/AI-ModelScope/FLUX.1-schnell/resolve/master/{rel}",
    "https://hf-mirror.com/black-forest-labs/FLUX.1-schnell/resolve/main/{rel}",
]

CHUNK = 1 << 20  # 1 MiB
BACKOFF_SECONDS = 8
STATE_FILE: Path = DEST.parent / ".flux_slim_state.json"
MAX_CONSECUTIVE_FAILURES = 12


class _RestartDownload(Exception):
    """The mirror ignored our Range request (416 or full 200): start over."""


def load_state() -> Dict[str, dict]:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: Dict[str, dict]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_dir() -> Path:
    snapshots = sorted(
        Path.home().joinpath(
            ".cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots"
        ).glob("*")
    )
    if not snapshots:
        raise SystemExit("no FLUX.1-schnell snapshot in the hub cache; log in first")
    return snapshots[0]


def _stream(url: str, part: Path, size: int, rel: str) -> None:
    """Append the rest of ``url`` onto ``part`` starting at its current size."""
    start = part.stat().st_size if part.is_file() else 0
    if start >= size:
        return
    headers = {"Range": f"bytes={start}-"}
    with part.open("ab") as fh:
        with requests.get(
            url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True
        ) as resp:
            if resp.status_code == 416:
                raise _RestartDownload()
            if resp.status_code == 200 and start > 0:
                # The mirror ignored Range; a full 200 would corrupt the part.
                raise _RestartDownload()
            if resp.status_code != 206 and resp.status_code != 200:
                resp.raise_for_status()
            last_report = time.time()
            for block in resp.iter_content(chunk_size=CHUNK):
                if not block:
                    continue
                fh.write(block)
                now = time.time()
                if now - last_report >= 30:
                    done = fh.tell()
                    pct = done * 100.0 / size
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {rel} {done}/{size} "
                        f"bytes ({pct:.1f}%)",
                        flush=True,
                    )
                    last_report = now


def download_resumable(rel: str, size: int, sha256: str, state: Dict[str, dict]) -> bool:
    dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")

    if dst.is_file() and dst.stat().st_size == size:
        if state.get(rel, {}).get("sha256") == sha256:
            print(f"already present {rel} ({size} bytes, verified)", flush=True)
            return True
        print(f"hashing {rel} ({size} bytes)", flush=True)
        if sha256_file(dst) == sha256:
            state[rel] = {"size": size, "sha256": sha256}
            save_state(state)
            print(f"verified {rel}", flush=True)
            return True
        print(f"[warn] {rel} size ok but sha256 mismatch; re-downloading", flush=True)
        dst.unlink()

    failures = 0
    while failures < MAX_CONSECUTIVE_FAILURES:
        if part.is_file() and part.stat().st_size > size:
            part.unlink()
        start = part.stat().st_size if part.is_file() else 0
        if start < size:
            ok = False
            for url_tpl in MIRRORS:
                url = url_tpl.format(rel=rel)
                try:
                    _stream(url, part, size, rel)
                    ok = True
                    break
                except _RestartDownload:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] [restart] {rel}: mirror "
                        f"ignored range; starting fresh",
                        flush=True,
                    )
                    part.unlink()
                    failures += 1
                    break
                except Exception as exc:  # network blips are expected mid-transfer
                    print(
                        f"[{time.strftime('%H:%M:%S')}] [retry] {rel} via "
                        f"{url_tpl.split('/')[2]}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        return False
                    time.sleep(min(BACKOFF_SECONDS * (1 << min(failures, 4)), 120))
            if not ok:
                continue
        if part.is_file() and part.stat().st_size == size:
            print(f"hashing {rel} ({size} bytes)", flush=True)
            if sha256_file(part) == sha256:
                part.rename(dst)
                state[rel] = {"size": size, "sha256": sha256}
                save_state(state)
                print(f"downloaded {rel} -> {size} bytes", flush=True)
                return True
            print(f"[warn] {rel} sha256 mismatch on full part; restarting fresh", flush=True)
            part.unlink()
            failures += 1
            continue
        failures += 1
        if failures >= MAX_CONSECUTIVE_FAILURES:
            return False
        time.sleep(BACKOFF_SECONDS)
    return False


def main() -> int:
    global DEST, STATE_FILE, MAX_CONSECUTIVE_FAILURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--max-failures", type=int, default=MAX_CONSECUTIVE_FAILURES)
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()

    DEST = Path(args.dest)
    STATE_FILE = DEST.parent / ".flux_slim_state.json"
    MAX_CONSECUTIVE_FAILURES = args.max_failures
    DEST.mkdir(parents=True, exist_ok=True)
    state = load_state()

    if not args.skip_copy:
        snap = _snapshot_dir()
        copied = 0
        for rel in CACHED_FILES:
            src = snap / rel
            dst = DEST / rel
            if not src.is_file():
                print(f"[warn] not in cache snapshot: {rel}", flush=True)
                continue
            if dst.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            print(f"copied {rel}", flush=True)
        print(f"copied {copied} cached files", flush=True)

    failed = False
    for rel, (size, sha256) in MANIFEST.items():
        if not download_resumable(rel, size, sha256, state):
            failed = True
            print(
                f"[error] giving up on {rel} after "
                f"{MAX_CONSECUTIVE_FAILURES} consecutive failures",
                flush=True,
            )
    if failed:
        print("FLUX slim dir NOT ready:", DEST, flush=True)
        return 1
    print("FLUX slim dir ready at", DEST, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
