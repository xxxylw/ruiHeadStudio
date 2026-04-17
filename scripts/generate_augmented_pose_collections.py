import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


POSE_KEYS = ("expression", "jaw_pose", "leye_pose", "reye_pose", "neck_pose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate medium-extent synthetic RuiHeadStudio pose collections."
    )
    parser.add_argument(
        "--input",
        default="talkshow/collection/project_converted_exp.npy",
        help="Input RuiHeadStudio object-array collection .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        default="talkshow/collection/synthetic_aug",
        help="Directory for generated .npy collections.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of output collection files to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260417,
        help="Random seed for deterministic generation.",
    )
    return parser.parse_args()


def load_collection(path: Path) -> List[Dict]:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype != object:
        raise ValueError(f"{path} is not a RuiHeadStudio object-array collection.")
    return list(arr.tolist())


def _resample_frames(values: np.ndarray, new_length: int) -> np.ndarray:
    old_length = values.shape[0]
    if old_length == new_length:
        return values.astype(np.float32, copy=True)

    old_x = np.linspace(0.0, 1.0, old_length, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, new_length, dtype=np.float32)
    out = np.empty((new_length, values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(new_x, old_x, values[:, dim]).astype(np.float32)
    return out


def _smooth_noise(length: int, dims: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=(length, dims)).astype(np.float32)
    kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
    padded = np.pad(noise, ((2, 2), (0, 0)), mode="edge")
    smoothed = np.empty_like(noise)
    for i in range(length):
        smoothed[i] = (padded[i : i + 5] * kernel[:, None]).sum(axis=0)
    return smoothed


def _make_emphasis_envelope(length: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    envelope = np.ones(length, dtype=np.float32)
    center_count = int(rng.integers(1, 4))
    for _ in range(center_count):
        center = float(rng.uniform(-0.5, 0.5))
        width = float(rng.uniform(0.15, 0.45))
        strength = float(rng.uniform(0.1, 0.35))
        envelope += strength * np.exp(-((x - center) ** 2) / (2 * width * width)).astype(
            np.float32
        )
    return envelope


def _augment_channel(
    values: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float,
    base_scale: float,
    emphasize: float,
) -> np.ndarray:
    augmented = values.astype(np.float32, copy=True)
    length = augmented.shape[0]
    dims = augmented.shape[1]

    time_scale = float(rng.uniform(0.92, 1.12))
    new_length = max(8, int(round(length * time_scale)))
    augmented = _resample_frames(augmented, new_length)
    augmented = _resample_frames(augmented, length)

    mean = augmented.mean(axis=0, keepdims=True)
    augmented = (augmented - mean) * base_scale + mean
    augmented += _smooth_noise(length, dims, rng, noise_sigma)

    envelope = _make_emphasis_envelope(length, rng)[:, None]
    augmented = mean + (augmented - mean) * (1.0 + emphasize * (envelope - 1.0))
    return augmented.astype(np.float32)


def _augment_sequence(sequence: Dict, file_index: int, seq_index: int, rng: np.random.Generator) -> Dict:
    out: Dict = {}
    out["expression"] = _augment_channel(
        np.asarray(sequence["expression"], dtype=np.float32),
        rng,
        noise_sigma=0.015,
        base_scale=float(rng.uniform(1.08, 1.22)),
        emphasize=0.55,
    )
    out["jaw_pose"] = _augment_channel(
        np.asarray(sequence["jaw_pose"], dtype=np.float32),
        rng,
        noise_sigma=0.01,
        base_scale=float(rng.uniform(1.12, 1.28)),
        emphasize=0.65,
    )
    out["leye_pose"] = _augment_channel(
        np.asarray(sequence["leye_pose"], dtype=np.float32),
        rng,
        noise_sigma=0.004,
        base_scale=float(rng.uniform(1.02, 1.10)),
        emphasize=0.20,
    )
    out["reye_pose"] = _augment_channel(
        np.asarray(sequence["reye_pose"], dtype=np.float32),
        rng,
        noise_sigma=0.004,
        base_scale=float(rng.uniform(1.02, 1.10)),
        emphasize=0.20,
    )
    out["neck_pose"] = _augment_channel(
        np.asarray(sequence["neck_pose"], dtype=np.float32),
        rng,
        noise_sigma=0.008,
        base_scale=float(rng.uniform(1.10, 1.25)),
        emphasize=0.50,
    )

    out["video_name"] = f"synthetic_medium_{file_index + 1:02d}"
    out["clip_name"] = f"{sequence.get('clip_name', 'clip')}_aug_{seq_index + 1:02d}"
    out["source_file"] = str(sequence.get("source_file", "unknown.pkl"))
    out["source_path"] = str(sequence.get("source_path", "unknown"))
    out["augmentation_tag"] = "medium_mix"
    out["augmentation_file_index"] = file_index
    return out


def generate_augmented_collection(
    source_sequences: Sequence[Dict], file_index: int, rng_seed: int
) -> List[Dict]:
    rng = np.random.default_rng(rng_seed + file_index)
    return [
        _augment_sequence(sequence, file_index=file_index, seq_index=seq_index, rng=rng)
        for seq_index, sequence in enumerate(source_sequences)
    ]


def save_collections(
    source_sequences: Sequence[Dict], output_dir: Path, num_files: int, seed: int
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    for file_index in range(num_files):
        augmented = generate_augmented_collection(source_sequences, file_index, seed)
        output_path = output_dir / f"aug_pose_mix_{file_index + 1:02d}.npy"
        np.save(output_path, np.array(augmented, dtype=object), allow_pickle=True)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    args = parse_args()
    source_sequences = load_collection(Path(args.input))
    output_paths = save_collections(
        source_sequences,
        Path(args.output_dir),
        num_files=args.num_files,
        seed=args.seed,
    )
    for path in output_paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
