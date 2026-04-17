import argparse
import io
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


REQUIRED_KEYS = (
    "expression",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "body_pose_axis",
)


def torch_load_cpu(bytes_obj):
    return torch.load(io.BytesIO(bytes_obj), map_location="cpu")


class CPUCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return torch_load_cpu
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TalkSHOW/SHOW tracking .pkl files to RuiHeadStudio training "
            "pose collection format."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .pkl files or directories. Directories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output .npy path. Recommended naming follows the existing training file "
            "pattern, e.g. talkshow/collection/myset_exp.npy"
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append converted sequences to an existing RuiHeadStudio training .npy file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when a .pkl file is missing required fields.",
    )
    return parser.parse_args()


def collect_pkl_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix == ".pkl":
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.pkl")))
            continue
        raise FileNotFoundError(f"Unsupported input path: {path}")
    if not paths:
        raise FileNotFoundError("No .pkl files found from the provided inputs.")
    return sorted(dict.fromkeys(paths))


def ensure_numpy_array(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"{name} is scalar, expected a per-frame array.")
    return array.astype(np.float32, copy=False)


def convert_sequence(data: Dict, source_path: Path) -> Dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"{source_path} is missing required keys: {missing}")

    expression = ensure_numpy_array("expression", data["expression"])
    jaw_pose = ensure_numpy_array("jaw_pose", data["jaw_pose"])
    leye_pose = ensure_numpy_array("leye_pose", data["leye_pose"])
    reye_pose = ensure_numpy_array("reye_pose", data["reye_pose"])
    body_pose_axis = ensure_numpy_array("body_pose_axis", data["body_pose_axis"])

    if body_pose_axis.ndim != 2 or body_pose_axis.shape[1] != 63:
        raise ValueError(
            f"{source_path} has body_pose_axis shape {body_pose_axis.shape}, expected [T, 63]."
        )

    num_frames = expression.shape[0]
    expected_first_dim = {
        "jaw_pose": jaw_pose.shape[0],
        "leye_pose": leye_pose.shape[0],
        "reye_pose": reye_pose.shape[0],
        "body_pose_axis": body_pose_axis.shape[0],
    }
    for name, length in expected_first_dim.items():
        if length != num_frames:
            raise ValueError(
                f"{source_path} has inconsistent frame count: "
                f"expression={num_frames}, {name}={length}."
            )

    neck_pose = body_pose_axis.reshape(-1, 21, 3)[:, 12]

    video_name = source_path.parent.parent.name if source_path.parent.parent != source_path.parent else source_path.stem
    clip_name = source_path.parent.name

    return {
        "expression": expression,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "neck_pose": neck_pose.astype(np.float32, copy=False),
        "video_name": video_name,
        "clip_name": clip_name,
        "source_file": source_path.name,
        "source_path": str(source_path),
    }


def load_sequence(path: Path, strict: bool) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = CPUCompatibleUnpickler(f).load()
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not deserialize to dict, got {type(data).__name__}.")
    try:
        return convert_sequence(data, path)
    except Exception:
        if strict:
            raise
        print(f"[skip] failed to convert {path}")
        return {}


def load_existing_collection(output_path: Path) -> List[Dict[str, np.ndarray]]:
    if not output_path.exists():
        return []
    existing = np.load(output_path, allow_pickle=True)
    return list(existing.tolist())


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.suffix != ".npy":
        raise ValueError("Output path must end with .npy")

    source_paths = collect_pkl_paths(args.inputs)
    sequences: List[Dict[str, np.ndarray]] = []

    for path in source_paths:
        converted = load_sequence(path, strict=args.strict)
        if converted:
            sequences.append(converted)

    if not sequences:
        raise RuntimeError("No valid sequences were converted.")

    if args.append:
        merged = load_existing_collection(output_path) + sequences
    else:
        merged = sequences

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(merged, dtype=object), allow_pickle=True)

    print(f"saved {len(merged)} sequences to {output_path}")
    print("sequence keys: expression, jaw_pose, leye_pose, reye_pose, neck_pose")


if __name__ == "__main__":
    main()
