import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


FRAME_KEYS = ("exp", "jaw_pose", "neck_pose", "eye_pose")
DEFAULT_INPUTS = ("data/talkvid/flame_tracker",)
DEFAULT_OUTPUT = "talkshow/collection/talkvid/talkvid_tracker_exp.npy"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert flame-head-tracker TalkVid outputs into RuiHeadStudio "
            "training pose collections."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=list(DEFAULT_INPUTS),
        help=(
            "Tracking result directories or parent directories searched recursively "
            f"for clip folders. Default: {' '.join(DEFAULT_INPUTS)}"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output RuiHeadStudio .npy path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append converted sequences to an existing RuiHeadStudio collection.",
    )
    return parser.parse_args(argv)


def is_tracking_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.npz"))


def collect_tracking_dirs(inputs: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if is_tracking_dir(path):
            paths.append(path)
            continue
        if path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_dir()):
                if is_tracking_dir(child):
                    paths.append(child)
            continue
        raise FileNotFoundError(f"Unsupported tracking input path: {path}")
    if not paths:
        raise FileNotFoundError("No tracking directories with frame .npz files were found.")
    return sorted(dict.fromkeys(paths))


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _load_metadata(clip_dir: Path) -> Dict[str, str]:
    metadata_path = clip_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}

    source_video = str(raw.get("source_video", raw.get("source_path", clip_dir)))
    return {
        "video_name": str(raw.get("video_name", clip_dir.parent.name or clip_dir.name)),
        "clip_name": str(raw.get("clip_name", clip_dir.name)),
        "source_file": str(raw.get("source_file", clip_dir.name)),
        "source_path": source_video,
        "source": "talkvid",
    }


def _load_frame(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    missing = [key for key in FRAME_KEYS if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {missing}")

    frame = {}
    for key in FRAME_KEYS:
        arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
        frame[key] = arr
    if frame["exp"].shape[0] != 100:
        raise ValueError(f"{npz_path} exp shape {frame['exp'].shape} is not compatible with FLAME 100D expression.")
    if frame["jaw_pose"].shape[0] != 3 or frame["neck_pose"].shape[0] != 3:
        raise ValueError(f"{npz_path} jaw_pose/neck_pose must be 3D vectors.")
    if frame["eye_pose"].shape[0] != 6:
        raise ValueError(f"{npz_path} eye_pose shape {frame['eye_pose'].shape} is not compatible with FLAME eye pose.")
    return frame


def convert_tracking_dir(clip_dir: Path) -> Dict[str, np.ndarray]:
    clip_dir = Path(clip_dir)
    frame_paths = sorted(clip_dir.glob("*.npz"), key=frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No frame .npz files found in {clip_dir}")

    expression: List[np.ndarray] = []
    jaw_pose: List[np.ndarray] = []
    neck_pose: List[np.ndarray] = []
    leye_pose: List[np.ndarray] = []
    reye_pose: List[np.ndarray] = []

    for frame_path in frame_paths:
        frame = _load_frame(frame_path)
        expression.append(frame["exp"])
        jaw_pose.append(frame["jaw_pose"])
        neck_pose.append(frame["neck_pose"])
        leye_pose.append(frame["eye_pose"][:3])
        reye_pose.append(frame["eye_pose"][3:])

    metadata = _load_metadata(clip_dir)
    return {
        "expression": np.stack(expression, axis=0).astype(np.float32),
        "jaw_pose": np.stack(jaw_pose, axis=0).astype(np.float32),
        "neck_pose": np.stack(neck_pose, axis=0).astype(np.float32),
        "leye_pose": np.stack(leye_pose, axis=0).astype(np.float32),
        "reye_pose": np.stack(reye_pose, axis=0).astype(np.float32),
        **metadata,
    }


def load_existing_collection(output_path: Path) -> List[Dict[str, np.ndarray]]:
    if not output_path.exists():
        return []
    existing = np.load(output_path, allow_pickle=True)
    return list(existing.tolist())


def save_collection(output_path: Path, sequences: List[Dict[str, np.ndarray]], append: bool) -> None:
    if not sequences:
        raise RuntimeError("No valid tracking sequences were converted.")
    merged = load_existing_collection(output_path) + sequences if append else sequences
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(merged, dtype=object), allow_pickle=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output)
    if output_path.suffix != ".npy":
        raise ValueError("Output path must end with .npy")

    tracking_dirs = collect_tracking_dirs([Path(item) for item in args.inputs])
    sequences = [convert_tracking_dir(path) for path in tracking_dirs]
    save_collection(output_path, sequences, append=args.append)

    print(f"saved {len(sequences)} sequences to {output_path}")
    print("sequence keys: expression, jaw_pose, leye_pose, reye_pose, neck_pose")


if __name__ == "__main__":
    main()
