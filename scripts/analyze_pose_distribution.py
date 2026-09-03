import argparse
import csv
import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib.pyplot as plt
import numpy as np
try:
    import numba
except ImportError:  # pragma: no cover - handled at runtime
    numba = None


def _strip_cache_kwargs(kwargs):
    if "cache" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("cache", None)
    return kwargs


def _njit_without_cache(*args, **kwargs):
    kwargs = _strip_cache_kwargs(kwargs)
    return _original_numba_njit(*args, **kwargs)


def _jit_without_cache(*args, **kwargs):
    kwargs = _strip_cache_kwargs(kwargs)
    return _original_numba_jit(*args, **kwargs)


def _vectorize_without_cache(*args, **kwargs):
    kwargs = _strip_cache_kwargs(kwargs)
    return _original_numba_vectorize(*args, **kwargs)


def _guvectorize_without_cache(*args, **kwargs):
    kwargs = _strip_cache_kwargs(kwargs)
    return _original_numba_guvectorize(*args, **kwargs)


if numba is not None:
    _original_numba_njit = numba.njit
    _original_numba_jit = numba.jit
    _original_numba_vectorize = numba.vectorize
    _original_numba_guvectorize = numba.guvectorize

    numba.njit = _njit_without_cache
    numba.jit = _jit_without_cache
    numba.vectorize = _vectorize_without_cache
    numba.guvectorize = _guvectorize_without_cache

try:
    import umap
except ImportError:  # pragma: no cover - handled at runtime
    umap = None

try:
    import hdbscan
except ImportError:  # pragma: no cover - handled at runtime
    hdbscan = None


POSE_KEYS = ("expression", "jaw_pose", "leye_pose", "reye_pose", "neck_pose")


@dataclass(frozen=True)
class InputSpec:
    input_path: Path
    group_label: str
    input_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frame- and sequence-level pose features from RuiHeadStudio "
            "training collections, then visualize the distribution with UMAP and HDBSCAN."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more RuiHeadStudio pose collection .npy files or directories containing .npy files.",
    )
    parser.add_argument(
        "--group-label",
        nargs="*",
        default=None,
        help=(
            "Optional labels for the top-level inputs. The count must match the number of positional "
            "inputs. Directory inputs apply their group label to every .npy file inside."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save feature tables and plots. Defaults to a sibling folder next to the input file.",
    )
    parser.add_argument(
        "--embedding-level",
        choices=("frame", "sequence"),
        default="frame",
        help="Use frame-level or sequence-level features to build the UMAP/HDBSCAN plots.",
    )
    parser.add_argument(
        "--sampler-mode",
        choices=("all_frames", "train_like"),
        default="all_frames",
        help="Frame sampling mode. train_like matches training by sampling sequence first, then frame.",
    )
    parser.add_argument(
        "--samples-per-input",
        type=int,
        default=None,
        help="When sampler-mode=train_like, draw this many samples per group label. Defaults to max_points // num_groups.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Maximum number of points used for UMAP/HDBSCAN/plotting. Full tables are still exported.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for subsampling and UMAP.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="HDBSCAN min_cluster_size parameter.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples parameter. Defaults to min_cluster_size // 2.",
    )
    return parser.parse_args()


def require_packages() -> None:
    missing = []
    if umap is None:
        missing.append("umap-learn")
    if hdbscan is None:
        missing.append("hdbscan")
    if missing:
        raise ImportError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install them in your analysis environment before running this script."
        )


def load_collection(path: Path):
    collection = np.load(path, allow_pickle=True)
    if collection.dtype != object:
        raise ValueError(f"{path} is not an object-array RuiHeadStudio collection.")
    return list(collection.tolist())


def expand_input_specs(input_paths: Sequence[Path], group_labels: Optional[Sequence[str]] = None) -> List[InputSpec]:
    normalized_paths = [Path(path) for path in input_paths]
    if group_labels is not None and len(group_labels) != len(normalized_paths):
        raise ValueError(
            f"group_labels length ({len(group_labels)}) must match inputs length ({len(normalized_paths)})"
        )

    specs: List[InputSpec] = []
    for index, input_path in enumerate(normalized_paths):
        group_label = group_labels[index] if group_labels is not None else input_path.stem
        if input_path.is_dir():
            children = sorted(child for child in input_path.glob("*.npy") if child.is_file())
            if not children:
                raise FileNotFoundError(f"No .npy files found in directory: {input_path}")
            for child in children:
                specs.append(InputSpec(input_path=child, group_label=group_label, input_name=child.stem))
        elif input_path.is_file():
            specs.append(InputSpec(input_path=input_path, group_label=group_label, input_name=input_path.stem))
        else:
            raise FileNotFoundError(input_path)
    return specs


def infer_sequence_metadata(input_spec: InputSpec, sequence: Dict, sequence_index: int) -> Dict[str, str]:
    source_path = sequence.get("source_path", "")
    source_file = sequence.get("source_file", "")
    video_name = sequence.get("video_name")
    clip_name = sequence.get("clip_name")

    stem = input_spec.input_path.stem
    if "__" in stem:
        stem_video, stem_clip = stem.split("__", 1)
    else:
        stem_video = stem
        stem_clip = f"seq_{sequence_index:05d}"

    if not video_name:
        video_name = stem_video
    if not clip_name:
        clip_name = stem_clip if sequence_index == 0 else f"{stem_clip}__seq_{sequence_index:05d}"

    source_name = f"{video_name}__{clip_name}"
    return {
        "input_path": str(input_spec.input_path),
        "input_name": input_spec.input_name,
        "group_label": input_spec.group_label,
        "sequence_index": str(sequence_index),
        "sequence_key": f"{input_spec.input_name}::seq_{sequence_index:05d}",
        "source_name": source_name,
        "video_name": video_name,
        "clip_name": clip_name,
        "source_path": source_path,
        "source_file": source_file,
    }


def stack_frame_features(sequence: Dict) -> np.ndarray:
    parts = []
    frame_count = None
    for key in POSE_KEYS:
        value = np.asarray(sequence[key], dtype=np.float32)
        if value.ndim != 2:
            raise ValueError(f"{key} must have shape [T, D], got {value.shape}")
        if frame_count is None:
            frame_count = value.shape[0]
        elif value.shape[0] != frame_count:
            raise ValueError(f"Inconsistent frame counts inside sequence: {key}={value.shape[0]}, expected {frame_count}")
        parts.append(value)
    return np.concatenate(parts, axis=1)


def sequence_statistics(frame_features: np.ndarray) -> np.ndarray:
    velocity = np.diff(frame_features, axis=0)
    acceleration = np.diff(velocity, axis=0)

    velocity_mean_abs = np.zeros(frame_features.shape[1], dtype=np.float32)
    velocity_max_abs = np.zeros(frame_features.shape[1], dtype=np.float32)
    acceleration_mean_abs = np.zeros(frame_features.shape[1], dtype=np.float32)

    if velocity.size > 0:
        velocity_mean_abs = np.mean(np.abs(velocity), axis=0).astype(np.float32)
        velocity_max_abs = np.max(np.abs(velocity), axis=0).astype(np.float32)
    if acceleration.size > 0:
        acceleration_mean_abs = np.mean(np.abs(acceleration), axis=0).astype(np.float32)

    stats = [
        np.mean(frame_features, axis=0),
        np.std(frame_features, axis=0),
        np.min(frame_features, axis=0),
        np.max(frame_features, axis=0),
        velocity_mean_abs,
        velocity_max_abs,
        acceleration_mean_abs,
    ]
    return np.concatenate(stats, axis=0).astype(np.float32)


def build_feature_tables(
    input_specs: Sequence[InputSpec],
) -> Tuple[np.ndarray, List[Dict[str, str]], np.ndarray, List[Dict[str, str]]]:
    frame_feature_list: List[np.ndarray] = []
    frame_metadata: List[Dict[str, str]] = []
    sequence_feature_list: List[np.ndarray] = []
    sequence_metadata: List[Dict[str, str]] = []

    for input_spec in input_specs:
        sequences = load_collection(input_spec.input_path)
        for sequence_index, sequence in enumerate(sequences):
            for key in POSE_KEYS:
                if key not in sequence:
                    raise KeyError(f"Sequence {sequence_index} in {input_spec.input_path} is missing key '{key}'")

            sequence_info = infer_sequence_metadata(input_spec, sequence, sequence_index)
            frame_features = stack_frame_features(sequence)
            frame_feature_list.append(frame_features)
            sequence_feature_list.append(sequence_statistics(frame_features))

            sequence_metadata.append(
                {
                    **sequence_info,
                    "num_frames": str(frame_features.shape[0]),
                }
            )

            for frame_index in range(frame_features.shape[0]):
                frame_metadata.append(
                    {
                        **sequence_info,
                        "frame_index": str(frame_index),
                    }
                )

    frame_feature_matrix = np.concatenate(frame_feature_list, axis=0).astype(np.float32)
    sequence_feature_matrix = np.stack(sequence_feature_list, axis=0).astype(np.float32)
    return frame_feature_matrix, frame_metadata, sequence_feature_matrix, sequence_metadata


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - mean) / std


def sample_indices(num_points: int, max_points: int, random_seed: int) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(num_points, size=max_points, replace=False))
    return indices.astype(np.int64)


def sample_train_like_indices(
    frame_metadata: Sequence[Dict[str, str]],
    samples_per_input: int,
    random_seed: int,
) -> np.ndarray:
    if samples_per_input <= 0:
        raise ValueError("samples_per_input must be a positive integer")

    by_input: Dict[str, Dict[str, List[int]]] = {}
    for row_index, row in enumerate(frame_metadata):
        input_name = row["group_label"]
        sequence_key = row["sequence_key"]
        by_input.setdefault(input_name, {}).setdefault(sequence_key, []).append(row_index)

    if not by_input:
        raise ValueError("No frame metadata available for train_like sampling")

    rng = np.random.default_rng(random_seed)
    sampled_indices: List[int] = []
    for input_name in sorted(by_input):
        sequence_map = by_input[input_name]
        sequence_keys = sorted(sequence_map)
        for _ in range(samples_per_input):
            chosen_sequence = sequence_keys[int(rng.integers(len(sequence_keys)))]
            frame_indices = sequence_map[chosen_sequence]
            sampled_indices.append(frame_indices[int(rng.integers(len(frame_indices)))])
    return np.asarray(sampled_indices, dtype=np.int64)


def run_embedding(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.random_seed,
    )
    return reducer.fit_transform(features).astype(np.float32)


def run_hdbscan(embedding: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    min_samples = args.min_samples
    if min_samples is None:
        min_samples = max(5, args.min_cluster_size // 2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(embedding).astype(np.int32)


def make_palette(num_colors: int) -> List[Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(num_colors)]


def plot_by_category(
    embedding: np.ndarray,
    metadata: List[Dict[str, str]],
    category_key: str,
    title: str,
    path: Path,
) -> None:
    values = [row[category_key] for row in metadata]
    unique_values = list(dict.fromkeys(values))
    palette = make_palette(len(unique_values))
    color_map = {name: palette[i] for i, name in enumerate(unique_values)}

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    for name in unique_values:
        idx = np.array([i for i, value in enumerate(values) if value == name], dtype=np.int64)
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=8,
            alpha=0.7,
            label=name,
            color=color_map[name],
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_by_cluster(embedding: np.ndarray, labels: np.ndarray, path: Path) -> None:
    unique_labels = sorted(np.unique(labels).tolist())
    non_noise = [label for label in unique_labels if label != -1]
    palette = make_palette(len(non_noise))
    color_map = {-1: (0.6, 0.6, 0.6, 0.35)}
    for i, label in enumerate(non_noise):
        color_map[label] = palette[i]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        label_name = "noise" if label == -1 else f"cluster_{label}"
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=8,
            alpha=0.75 if label != -1 else 0.35,
            label=label_name,
            color=color_map[label],
            edgecolors="none",
        )
    ax.set_title("UMAP of Pose Distribution Colored by HDBSCAN")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    input_specs: Sequence[InputSpec],
    frame_features: np.ndarray,
    frame_metadata: List[Dict[str, str]],
    sequence_features: np.ndarray,
    sequence_metadata: List[Dict[str, str]],
    sampled_indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    source_names = [row["source_name"] for row in frame_metadata]
    unique_sources = sorted(set(source_names))
    group_counts: Dict[str, int] = {}
    input_counts: Dict[str, int] = {}
    for row in frame_metadata:
        group_counts[row["group_label"]] = group_counts.get(row["group_label"], 0) + 1
        input_counts[row["input_name"]] = input_counts.get(row["input_name"], 0) + 1

    sampled_group_counts: Dict[str, int] = {}
    for row_index in sampled_indices.tolist():
        row = frame_metadata[row_index] if args.embedding_level == "frame" else sequence_metadata[row_index]
        sampled_group_counts[row["group_label"]] = sampled_group_counts.get(row["group_label"], 0) + 1

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = {}
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        cluster_counts["noise" if label == -1 else f"cluster_{label}"] = count

    summary = {
        "inputs": [str(path) for path in args.inputs],
        "expanded_inputs": [
            {
                "input_path": str(spec.input_path),
                "input_name": spec.input_name,
                "group_label": spec.group_label,
            }
            for spec in input_specs
        ],
        "embedding_level": args.embedding_level,
        "sampler_mode": args.sampler_mode,
        "samples_per_input": args.samples_per_input,
        "num_sequences": int(sequence_features.shape[0]),
        "num_frames": int(frame_features.shape[0]),
        "frame_feature_dim": int(frame_features.shape[1]),
        "sequence_feature_dim": int(sequence_features.shape[1]),
        "num_sources": int(len(unique_sources)),
        "sources": unique_sources,
        "group_counts": group_counts,
        "input_counts": input_counts,
        "sampled_points": int(sampled_indices.shape[0]),
        "sampled_group_counts": sampled_group_counts,
        "cluster_counts": cluster_counts,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    require_packages()

    input_paths = [Path(path) for path in args.inputs]
    input_specs = expand_input_specs(input_paths, args.group_label)

    if args.output_dir is None:
        if len(input_paths) == 1:
            output_dir = input_paths[0].with_suffix("")
            output_dir = output_dir.parent / f"{output_dir.name}_distribution"
        else:
            common_parent = Path(os.path.commonpath([str(path.resolve().parent) for path in input_paths]))
            try:
                rel_parent = common_parent.resolve().relative_to(
                    Path("collection/ruiheadstudio/flame_collections").resolve()
                )
                output_dir = (
                    Path("collection/ruiheadstudio/analysis/pose_distribution")
                    / rel_parent
                    / "multi_input_pose_distribution"
                )
            except ValueError:
                output_dir = common_parent / "multi_input_pose_distribution"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_features, frame_metadata, sequence_features, sequence_metadata = build_feature_tables(input_specs)

    np.save(output_dir / "frame_features.npy", frame_features)
    write_csv(output_dir / "frame_metadata.csv", frame_metadata)
    np.save(output_dir / "sequence_features.npy", sequence_features)
    write_csv(output_dir / "sequence_metadata.csv", sequence_metadata)

    if args.embedding_level == "frame":
        base_features = frame_features
        base_metadata = frame_metadata
        if args.sampler_mode == "all_frames":
            sampled_indices = sample_indices(base_features.shape[0], args.max_points, args.random_seed)
        else:
            samples_per_input = args.samples_per_input
            if samples_per_input is None:
                group_count = max(1, len({spec.group_label for spec in input_specs}))
                samples_per_input = max(1, args.max_points // group_count)
            sampled_indices = sample_train_like_indices(frame_metadata, samples_per_input, args.random_seed)
    else:
        if args.sampler_mode != "all_frames":
            raise ValueError("sampler_mode=train_like is only supported for frame-level embeddings")
        base_features = sequence_features
        base_metadata = sequence_metadata
        sampled_indices = sample_indices(base_features.shape[0], args.max_points, args.random_seed)
    sampled_features = standardize_features(base_features[sampled_indices])
    sampled_metadata = [base_metadata[i] for i in sampled_indices.tolist()]

    embedding = run_embedding(sampled_features, args)
    labels = run_hdbscan(embedding, args)

    np.save(output_dir / f"{args.embedding_level}_umap_embedding.npy", embedding)
    np.save(output_dir / f"{args.embedding_level}_hdbscan_labels.npy", labels)
    np.save(output_dir / f"{args.embedding_level}_sample_indices.npy", sampled_indices)

    plot_by_category(
        embedding,
        sampled_metadata,
        "source_name",
        "UMAP of Pose Distribution by Source",
        output_dir / f"umap_{args.embedding_level}_by_source.png",
    )
    plot_by_category(
        embedding,
        sampled_metadata,
        "group_label",
        "UMAP of Pose Distribution by Group",
        output_dir / f"umap_{args.embedding_level}_by_group.png",
    )
    plot_by_cluster(embedding, labels, output_dir / f"umap_{args.embedding_level}_by_cluster.png")
    write_summary(
        output_dir / "summary.json",
        args,
        input_specs,
        frame_features,
        frame_metadata,
        sequence_features,
        sequence_metadata,
        sampled_indices,
        labels,
    )

    print(f"saved analysis outputs to {output_dir}")
    print(f"frame_features: {frame_features.shape}")
    print(f"sequence_features: {sequence_features.shape}")
    print(
        f"embedding_level: {args.embedding_level}, sampler_mode: {args.sampler_mode}, "
        f"sampled_points: {sampled_indices.shape[0]}"
    )


if __name__ == "__main__":
    main()
