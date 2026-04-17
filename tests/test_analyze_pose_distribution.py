import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_pose_distribution import (
    build_feature_tables,
    expand_input_specs,
    sample_train_like_indices,
)


POSE_KEYS = ("expression", "jaw_pose", "leye_pose", "reye_pose", "neck_pose")


class AnalyzePoseDistributionTests(unittest.TestCase):
    def _make_sequence(self, frames: int, tag: str):
        t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        return {
            "expression": np.stack([t + 0.01 for _ in range(100)], axis=1).astype(np.float32),
            "jaw_pose": np.stack([t, t * 0.5, t * 0.25], axis=1).astype(np.float32),
            "leye_pose": np.stack([t * 0.2, t * 0.1, t * 0.05], axis=1).astype(np.float32),
            "reye_pose": np.stack([t * 0.15, t * 0.05, t * 0.08], axis=1).astype(np.float32),
            "neck_pose": np.stack([t * 0.3, t * 0.2, t * 0.1], axis=1).astype(np.float32),
            "video_name": f"video_{tag}",
            "clip_name": f"clip_{tag}",
            "source_file": f"{tag}.pkl",
            "source_path": f"/tmp/{tag}.pkl",
        }

    def _write_collection(self, path: Path, *sequences):
        arr = np.array(list(sequences), dtype=object)
        np.save(path, arr, allow_pickle=True)

    def test_expand_input_specs_supports_file_and_directory_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.npy"
            aug_dir = root / "aug_dir"
            aug_dir.mkdir()
            aug_a = aug_dir / "aug_a.npy"
            aug_b = aug_dir / "aug_b.npy"

            self._write_collection(base_path, self._make_sequence(4, "base"))
            self._write_collection(aug_a, self._make_sequence(5, "aug_a"))
            self._write_collection(aug_b, self._make_sequence(6, "aug_b"))

            specs = expand_input_specs(
                [base_path, aug_dir],
                group_labels=["base", "aug"],
            )

            self.assertEqual([spec.group_label for spec in specs], ["base", "aug", "aug"])
            self.assertEqual([spec.input_path.name for spec in specs], ["base.npy", "aug_a.npy", "aug_b.npy"])

    def test_train_like_sampling_draws_equal_counts_per_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.npy"
            aug_path = root / "aug.npy"

            self._write_collection(
                base_path,
                self._make_sequence(2, "base_short"),
                self._make_sequence(7, "base_long"),
            )
            self._write_collection(
                aug_path,
                self._make_sequence(3, "aug_short"),
                self._make_sequence(8, "aug_long"),
            )

            specs = expand_input_specs([base_path, aug_path], group_labels=["base", "aug"])
            _, frame_metadata, _, _ = build_feature_tables(specs)
            sampled = sample_train_like_indices(frame_metadata, samples_per_input=5, random_seed=3)
            sampled_rows = [frame_metadata[index] for index in sampled.tolist()]

            self.assertEqual(len(sampled_rows), 10)
            self.assertEqual(sum(row["group_label"] == "base" for row in sampled_rows), 5)
            self.assertEqual(sum(row["group_label"] == "aug" for row in sampled_rows), 5)
            self.assertTrue(all("input_name" in row for row in sampled_rows))


if __name__ == "__main__":
    unittest.main()
