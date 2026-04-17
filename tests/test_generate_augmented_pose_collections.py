import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_augmented_pose_collections import (
    POSE_KEYS,
    generate_augmented_collection,
    save_collections,
)


class SyntheticPoseCollectionTests(unittest.TestCase):
    def _make_sequence(self, frames: int = 24):
        t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        return {
            "expression": np.stack([t for _ in range(100)], axis=1).astype(np.float32),
            "jaw_pose": np.stack([t, t * 0.5, t * 0.25], axis=1).astype(np.float32),
            "leye_pose": np.stack([t * 0.2, t * 0.1, t * 0.05], axis=1).astype(np.float32),
            "reye_pose": np.stack([t * 0.15, t * 0.05, t * 0.08], axis=1).astype(np.float32),
            "neck_pose": np.stack([t * 0.3, t * 0.2, t * 0.1], axis=1).astype(np.float32),
            "video_name": "src_video",
            "clip_name": "src_clip",
            "source_file": "src.pkl",
            "source_path": "/tmp/src.pkl",
        }

    def test_generate_augmented_collection_preserves_contract(self):
        source = [self._make_sequence()]
        augmented = generate_augmented_collection(source, file_index=0, rng_seed=7)

        self.assertEqual(len(augmented), 1)
        item = augmented[0]
        for key in POSE_KEYS:
            self.assertEqual(item[key].dtype, np.float32)
            self.assertEqual(item[key].shape[0], source[0][key].shape[0])
        self.assertIn("synthetic_medium", item["video_name"])
        self.assertNotEqual(
            float(np.mean(item["expression"])),
            float(np.mean(source[0]["expression"])),
        )

    def test_save_collections_writes_expected_number_of_files(self):
        source = [self._make_sequence()]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = save_collections(
                source,
                Path(tmpdir),
                num_files=3,
                seed=11,
            )

            self.assertEqual(len(output_paths), 3)
            for path in output_paths:
                self.assertTrue(path.exists())
                arr = np.load(path, allow_pickle=True)
                self.assertEqual(arr.dtype, object)
                self.assertEqual(arr.shape, (1,))


if __name__ == "__main__":
    unittest.main()
