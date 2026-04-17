import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.convert_talkvid_to_ruiheadstudio import (
    collect_tracking_dirs,
    convert_tracking_dir,
    save_collection,
)


class ConvertTalkVidToRuiHeadStudioTests(unittest.TestCase):
    def _write_frame(self, path: Path, value: float):
        np.savez(
            path,
            exp=np.full((1, 100), value, dtype=np.float32),
            jaw_pose=np.full((1, 3), value + 1.0, dtype=np.float32),
            neck_pose=np.full((1, 3), value + 2.0, dtype=np.float32),
            eye_pose=np.full((1, 6), value + 3.0, dtype=np.float32),
        )

    def _write_clip_dir(self, root: Path, clip_name: str):
        clip_dir = root / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "video_name": "speaker_001",
            "clip_name": clip_name,
            "source_video": f"/data/talkvid/clips/{clip_name}.mp4",
            "group_label": "train",
        }
        (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self._write_frame(clip_dir / "000000.npz", 0.1)
        self._write_frame(clip_dir / "000001.npz", 0.2)
        return clip_dir

    def test_convert_tracking_dir_maps_tracker_outputs_to_ruiheadstudio_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_dir = self._write_clip_dir(Path(tmpdir), "clip_0001")

            seq = convert_tracking_dir(clip_dir)

            self.assertEqual(seq["expression"].shape, (2, 100))
            self.assertEqual(seq["jaw_pose"].shape, (2, 3))
            self.assertEqual(seq["neck_pose"].shape, (2, 3))
            self.assertEqual(seq["leye_pose"].shape, (2, 3))
            self.assertEqual(seq["reye_pose"].shape, (2, 3))
            self.assertEqual(seq["video_name"], "speaker_001")
            self.assertEqual(seq["clip_name"], "clip_0001")
            self.assertEqual(seq["source_file"], "clip_0001")
            self.assertTrue(seq["source_path"].endswith("clip_0001.mp4"))
            np.testing.assert_allclose(seq["leye_pose"][0], np.full((3,), 3.1, dtype=np.float32))
            np.testing.assert_allclose(seq["reye_pose"][1], np.full((3,), 3.2, dtype=np.float32))

    def test_collect_and_save_collection_from_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clip_a = self._write_clip_dir(root / "tracker_outputs", "clip_a")
            clip_b = self._write_clip_dir(root / "tracker_outputs", "clip_b")
            output_path = root / "talkshow" / "collection" / "talkvid_exp.npy"

            found = collect_tracking_dirs([clip_a.parent])
            self.assertEqual([path.name for path in found], ["clip_a", "clip_b"])

            sequences = [convert_tracking_dir(path) for path in found]
            save_collection(output_path, sequences, append=False)

            arr = np.load(output_path, allow_pickle=True)
            items = arr.tolist()
            self.assertEqual(arr.dtype, object)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["clip_name"], "clip_a")
            self.assertEqual(items[1]["clip_name"], "clip_b")


if __name__ == "__main__":
    unittest.main()
