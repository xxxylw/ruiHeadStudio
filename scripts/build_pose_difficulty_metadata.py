from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def _max_frame_delta(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(values, axis=0), axis=1)))


def compute_sequence_stats(sequence: Dict[str, np.ndarray]) -> Dict[str, float]:
    jaw_pose = np.asarray(sequence["jaw_pose"], dtype=np.float32)
    neck_pose = np.asarray(sequence["neck_pose"], dtype=np.float32)
    expression = np.asarray(
        sequence.get("expression", np.zeros((jaw_pose.shape[0], 100), dtype=np.float32)),
        dtype=np.float32,
    )
    head_pitch = np.abs(neck_pose[:, 0])
    head_yaw = np.abs(neck_pose[:, 1])
    head_roll = np.abs(neck_pose[:, 2])
    return {
        "jaw_open_max": float(np.max(np.abs(jaw_pose[:, 2]))),
        "neck_rot_max": float(np.max(np.linalg.norm(neck_pose, axis=1))),
        "head_pitch_max": float(np.max(head_pitch)),
        "head_yaw_max": float(np.max(head_yaw)),
        "head_roll_max": float(np.max(head_roll)),
        "expression_norm_max": float(np.max(np.linalg.norm(expression, axis=1))),
        "jaw_delta_max": _max_frame_delta(jaw_pose),
        "neck_delta_max": _max_frame_delta(neck_pose),
        "expression_delta_max": _max_frame_delta(expression),
    }


def classify_sequence_bucket(stats: Dict[str, float]) -> str:
    if (
        stats["head_yaw_max"] >= 0.45
        or stats["expression_delta_max"] >= 12.0
        or stats["neck_delta_max"] >= 0.35
    ):
        return "rare"
    if (
        stats["head_yaw_max"] >= 0.30
        or stats["head_pitch_max"] >= 0.25
        or stats["expression_norm_max"] >= 8.0
        or stats["jaw_delta_max"] >= 0.20
    ):
        return "hard"
    if (
        stats["head_yaw_max"] >= 0.15
        or stats["head_pitch_max"] >= 0.12
        or stats["expression_norm_max"] >= 3.0
        or stats["jaw_open_max"] >= 0.12
    ):
        return "medium"
    return "easy"


def build_metadata_entry(item: Dict[str, np.ndarray]) -> Dict[str, float]:
    stats = compute_sequence_stats(item)
    source_name = item.get("source_name") or item.get("source_file")
    if not source_name:
        source_name = f"{item.get('video_name', 'unknown_video')}__{item.get('clip_name', 'unknown_clip')}"
    return {
        "source_name": source_name,
        "video_name": item.get("video_name", ""),
        "clip_name": item.get("clip_name", ""),
        "bucket": classify_sequence_bucket(stats),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = np.load(args.input, allow_pickle=True)
    metadata = [build_metadata_entry(item) for item in records]
    Path(args.output).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
