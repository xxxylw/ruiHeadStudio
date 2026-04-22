import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pose_difficulty_metadata import classify_sequence_bucket, compute_sequence_stats


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


if __name__ == "__main__":
    unittest.main()
