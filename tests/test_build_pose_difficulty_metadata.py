import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pose_difficulty_metadata import (
    build_metadata_entry,
    classify_sequence_bucket,
    compute_sequence_stats,
)


class TestBuildPoseDifficultyMetadata(unittest.TestCase):
    def test_compute_sequence_stats_and_bucket(self):
        sequence = {
            "jaw_pose": np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.03]], dtype=np.float32),
            "neck_pose": np.array([[0.02, 0.0, 0.0], [0.03, 0.0, 0.0]], dtype=np.float32),
        }

        stats = compute_sequence_stats(sequence)

        self.assertIn("jaw_open_max", stats)
        self.assertIn("neck_rot_max", stats)
        bucket = classify_sequence_bucket(stats)
        self.assertEqual(bucket, "easy")

    def test_compute_sequence_stats_includes_head_pose_ranges(self):
        sequence = {
            "jaw_pose": np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.03]], dtype=np.float32),
            "neck_pose": np.array([[0.10, 0.20, 0.30], [0.15, 0.25, 0.35]], dtype=np.float32),
            "expression": np.zeros((2, 100), dtype=np.float32),
        }

        stats = compute_sequence_stats(sequence)

        self.assertIn("head_yaw_max", stats)
        self.assertIn("head_pitch_max", stats)
        self.assertIn("head_roll_max", stats)
        self.assertIn("expression_norm_max", stats)

    def test_build_metadata_entry_falls_back_when_source_name_missing(self):
        sequence = {
            "video_name": "video.mp4",
            "clip_name": "clip_01",
            "source_file": "video__clip_01",
            "jaw_pose": np.array([[0.0, 0.0, 0.02]], dtype=np.float32),
            "neck_pose": np.array([[0.02, 0.0, 0.0]], dtype=np.float32),
        }

        entry = build_metadata_entry(sequence)

        self.assertEqual(entry["source_name"], "video__clip_01")
        self.assertEqual(entry["bucket"], "easy")


if __name__ == "__main__":
    unittest.main()
