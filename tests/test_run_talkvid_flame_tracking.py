import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_talkvid_flame_tracking import (
    load_clip_jobs,
    prepare_tracker_import,
    write_tracking_metadata,
)


class RunTalkVidFlameTrackingTests(unittest.TestCase):
    def test_load_clip_jobs_resolves_clip_and_output_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clips_root = root / "clips"
            output_root = root / "tracker"
            clip_id = "video_demo-scene1"
            clip_dir = clips_root / clip_id
            clip_dir.mkdir(parents=True)
            (clip_dir / f"{clip_id}.mp4").write_bytes(b"fake")
            (clip_dir / "metadata.json").write_text(
                json.dumps({"video_name": clip_id, "source_path": str(clip_dir / f"{clip_id}.mp4")}),
                encoding="utf-8",
            )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps([{"id": clip_id}]), encoding="utf-8")

            jobs = load_clip_jobs(manifest_path, clips_root, output_root)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].clip_id, clip_id)
            self.assertEqual(jobs[0].video_path, clip_dir / f"{clip_id}.mp4")
            self.assertEqual(jobs[0].output_dir, output_root / clip_id)

    def test_write_tracking_metadata_copies_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clips_root = root / "clips"
            output_root = root / "tracker"
            clip_id = "video_demo-scene1"
            clip_dir = clips_root / clip_id
            clip_dir.mkdir(parents=True)
            source_video = clip_dir / f"{clip_id}.mp4"
            source_video.write_bytes(b"fake")
            source_metadata = {
                "video_name": "speaker_demo",
                "clip_name": clip_id,
                "source_file": source_video.name,
                "source_path": str(source_video),
            }
            (clip_dir / "metadata.json").write_text(json.dumps(source_metadata), encoding="utf-8")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps([{"id": clip_id}]), encoding="utf-8")
            job = load_clip_jobs(manifest_path, clips_root, output_root)[0]

            write_tracking_metadata(job)

            written = json.loads((job.output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(written["video_name"], "speaker_demo")
            self.assertEqual(written["clip_name"], clip_id)
            self.assertEqual(written["source_file"], source_video.name)
            self.assertEqual(written["source_path"], str(source_video))

    def test_prepare_tracker_import_adds_tracker_root_to_sys_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker_root = Path(tmpdir) / "external" / "flame-head-tracker"
            tracker_root.mkdir(parents=True)

            import sys

            before = list(sys.path)
            try:
                prepare_tracker_import(tracker_root)
                self.assertEqual(Path(sys.path[0]), tracker_root)
            finally:
                sys.path[:] = before


if __name__ == "__main__":
    unittest.main()
