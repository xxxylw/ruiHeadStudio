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
        self.assertIn("system.weights=", stage2)
        self.assertNotIn("resume=", stage2)

    def test_stage_run_scripts_pin_bitsandbytes_to_torch_cuda(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn('export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-118}"', stage1)
        self.assertIn('export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-118}"', stage2)
        self.assertIn('export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib"', stage1)
        self.assertIn('export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib"', stage2)

    def test_stage_run_scripts_use_dedicated_training_env_prefix(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn('TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"', stage1)
        self.assertIn('TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"', stage2)
        self.assertIn('"$TRAIN_ENV_PREFIX/bin/python"', stage1)
        self.assertIn('"$TRAIN_ENV_PREFIX/bin/python"', stage2)

    def test_stage_run_scripts_launch_python_in_clean_env(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("env -i", stage1)
        self.assertIn("env -i", stage2)

    def test_stage_run_scripts_do_not_inherit_outer_path(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn('export PATH="$TRAIN_ENV_PREFIX/bin:$CUDA_HOME/bin:/usr/bin:/bin"', stage1)
        self.assertIn('export PATH="$TRAIN_ENV_PREFIX/bin:$CUDA_HOME/bin:/usr/bin:/bin"', stage2)


if __name__ == "__main__":
    unittest.main()
