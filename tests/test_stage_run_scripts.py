import os
import unittest
from pathlib import Path


class TestStageRunScripts(unittest.TestCase):
    def test_stage_run_scripts_exist(self):
        self.assertTrue(os.path.exists("scripts/run_stage1_prior.sh"))
        self.assertTrue(os.path.exists("scripts/run_stage2_text.sh"))

    def test_stage_run_scripts_prepare_pose_metadata(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("build_pose_difficulty_metadata.py", stage1)
        self.assertIn("build_pose_difficulty_metadata.py", stage2)
        self.assertIn("data.pose_metadata_inputs", stage1)
        self.assertIn("data.pose_metadata_inputs", stage2)

    def test_stage2_script_supports_stage1_checkpoint_handoff(self):
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("STAGE1_CKPT", stage2)
        self.assertIn("resume=", stage2)


if __name__ == "__main__":
    unittest.main()
