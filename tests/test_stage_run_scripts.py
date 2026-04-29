import os
import unittest
from pathlib import Path


class TestStageRunScripts(unittest.TestCase):
    def test_stage_run_scripts_exist(self):
        self.assertTrue(os.path.exists("scripts/run_stage1_prior.sh"))
        self.assertTrue(os.path.exists("scripts/run_stage2_text.sh"))
        self.assertTrue(os.path.exists("scripts/run_two_stage.sh"))

    def test_stage_run_scripts_prepare_pose_metadata(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("build_pose_difficulty_metadata.py", stage1)
        self.assertIn("build_pose_difficulty_metadata.py", stage2)
        self.assertIn("data.pose_metadata_inputs", stage1)
        self.assertIn("data.pose_metadata_inputs", stage2)

    def test_stage2_script_supports_stage1_checkpoint_handoff(self):
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")
        two_stage = Path("scripts/run_two_stage.sh").read_text(encoding="utf-8")

        self.assertIn("STAGE1_CKPT", stage2)
        self.assertIn("system.weights=", stage2)
        self.assertNotIn("resume=", stage2)
        self.assertIn('STAGE1_CKPT="${OUTPUT_ROOT}/headstudio-stage1-prior/ckpts/last.ckpt"', two_stage)
        self.assertIn('export STAGE1_CKPT', two_stage)
        self.assertIn('bash scripts/run_stage2_text.sh', two_stage)

    def test_two_stage_script_shares_tag_and_timestamp_across_stages(self):
        two_stage = Path("scripts/run_two_stage.sh").read_text(encoding="utf-8")

        self.assertIn('RUN_TAG="${RUN_TAG:-silver_haired_scientist_portrait}"', two_stage)
        self.assertIn('RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"', two_stage)
        self.assertIn('OUTPUT_ROOT="outputs/${RUN_TAG}${RUN_TS}"', two_stage)
        self.assertIn('tag="${RUN_TAG}"', two_stage)
        self.assertIn('timestamp="${RUN_TS}"', two_stage)
        self.assertNotIn('@${RUN_TS}', two_stage)

    def test_two_stage_default_stage2_prompt_avoids_unsupported_clothing(self):
        two_stage = Path("scripts/run_two_stage.sh").read_text(encoding="utf-8").lower()

        self.assertIn("head and neck only", two_stage)
        self.assertIn("no clothing", two_stage)
        self.assertIn("no collar", two_stage)
        self.assertNotIn("turtleneck", two_stage)
        self.assertNotIn("shallow depth of field", two_stage)

    def test_stage_run_scripts_pin_bitsandbytes_to_torch_cuda(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn('export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-118}"', stage1)
        self.assertIn('export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-118}"', stage2)
        self.assertIn(
            'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/lib:/usr/lib/wsl/lib"',
            stage1,
        )
        self.assertIn(
            'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/lib:/usr/lib/wsl/lib"',
            stage2,
        )

    def test_stage_run_scripts_use_dedicated_training_env_prefix(self):
        stage1 = Path("scripts/run_stage1_prior.sh").read_text(encoding="utf-8")
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn('TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"', stage1)
        self.assertIn('TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-/home/rui/miniconda3/envs/ruiheadstudio}"', stage2)
        self.assertIn('export CONDA_PREFIX="$TRAIN_ENV_PREFIX"', stage1)
        self.assertIn('export CONDA_PREFIX="$TRAIN_ENV_PREFIX"', stage2)
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

    def test_fidelity_eval_script_uses_fixed_animation_protocol(self):
        script = Path("scripts/render_fidelity_eval.sh").read_text(encoding="utf-8")

        self.assertIn("FIDELITY_CKPT", script)
        self.assertIn("configs/headstudio_stage2_text.yaml", script)
        self.assertIn("trainer.max_steps=1", script)
        self.assertIn("--test", script)

    def test_stage2_script_accepts_opacity_repair_overrides(self):
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("OPACITY_COVERAGE_ENABLED", stage2)
        self.assertIn("REAR_OPACITY_ENABLED", stage2)
        self.assertIn("PRUNE_REGION_GUARD_ENABLED", stage2)
        self.assertIn("system.opacity_coverage.enabled=${OPACITY_COVERAGE_ENABLED}", stage2)
        self.assertIn(
            "system.loss.lambda_opacity_coverage=${LAMBDA_OPACITY_COVERAGE}",
            stage2,
        )

    def test_two_stage_script_forwards_opacity_repair_overrides(self):
        two_stage = Path("scripts/run_two_stage.sh").read_text(encoding="utf-8")

        self.assertIn("OPACITY_COVERAGE_ENABLED", two_stage)
        self.assertIn("LAMBDA_OPACITY_COVERAGE", two_stage)
        self.assertIn("REAR_OPACITY_ENABLED", two_stage)
        self.assertIn("LAMBDA_REAR_OPACITY", two_stage)
        self.assertIn("PRUNE_REGION_GUARD_ENABLED", two_stage)


if __name__ == "__main__":
    unittest.main()
