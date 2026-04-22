from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def compute_sequence_stats(sequence: Dict[str, np.ndarray]) -> Dict[str, float]:
    jaw_pose = np.asarray(sequence["jaw_pose"], dtype=np.float32)
    neck_pose = np.asarray(sequence["neck_pose"], dtype=np.float32)
    return {
        "jaw_open_max": float(np.max(np.abs(jaw_pose[:, 2]))),
        "neck_rot_max": float(np.max(np.linalg.norm(neck_pose, axis=1))),
    }


def classify_sequence_bucket(stats: Dict[str, float]) -> str:
    if stats["jaw_open_max"] < 0.08 and stats["neck_rot_max"] < 0.12:
        return "easy"
    if stats["jaw_open_max"] < 0.18 and stats["neck_rot_max"] < 0.22:
        return "medium"
    if stats["jaw_open_max"] < 0.30 and stats["neck_rot_max"] < 0.35:
        return "hard"
    return "rare"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = np.load(args.input, allow_pickle=True)
    metadata = []
    for item in records:
        stats = compute_sequence_stats(item)
        metadata.append(
            {
                "source_name": item["source_name"],
                "video_name": item["video_name"],
                "clip_name": item["clip_name"],
                "bucket": classify_sequence_bucket(stats),
                **stats,
            }
        )
    Path(args.output).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
