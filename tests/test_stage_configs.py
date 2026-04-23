import unittest
import importlib.util
import sys
import types
from pathlib import Path

from omegaconf import OmegaConf


def load_config_module():
    threestudio_stub = types.ModuleType("threestudio")
    threestudio_stub.warn = lambda *args, **kwargs: None

    utils_stub = types.ModuleType("threestudio.utils")
    typing_stub = types.ModuleType("threestudio.utils.typing")
    exec(
        "from typing import Any, Dict, List, Optional, Tuple, Union\n"
        "from omegaconf import DictConfig\n",
        typing_stub.__dict__,
    )

    sys.modules["threestudio"] = threestudio_stub
    sys.modules["threestudio.utils"] = utils_stub
    sys.modules["threestudio.utils.typing"] = typing_stub

    config_path = Path(__file__).resolve().parents[1] / "threestudio" / "utils" / "config.py"
    spec = importlib.util.spec_from_file_location("test_stage_config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_config = load_config_module().load_config


class TestStageConfigs(unittest.TestCase):
    def test_stage1_and_stage2_configs_define_distinct_modes(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertEqual(stage1.data.difficulty_sampling_mode, "curriculum")
        self.assertEqual(stage2.data.difficulty_sampling_mode, "curriculum")
        self.assertFalse(stage1.system.guidance.use_nfsd)
        self.assertTrue(stage2.system.guidance.use_nfsd)

    def test_stage1_uses_neutral_prompt_and_lower_sds_weight(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertNotEqual(
            stage1.system.prompt_processor.prompt,
            stage2.system.prompt_processor.prompt,
        )
        self.assertIn("human head", stage1.system.prompt_processor.prompt.lower())
        self.assertLess(stage1.system.loss.lambda_sds, stage2.system.loss.lambda_sds)

    def test_stage_configs_use_prompt_first_trial_layout(self):
        stage1 = load_config(
            "configs/headstudio_stage1_prior.yaml",
            cli_args=[
                "exp_root_dir=/tmp/ruiheadstudio-layout-tests",
                "use_timestamp=False",
                "system.prompt_processor.prompt=Elon Musk",
            ],
        )
        stage2 = load_config(
            "configs/headstudio_stage2_text.yaml",
            cli_args=[
                "exp_root_dir=/tmp/ruiheadstudio-layout-tests",
                "use_timestamp=False",
                "system.prompt_processor.prompt=Elon Musk",
            ],
        )

        self.assertTrue(stage1.trial_dir.endswith("Elon_Musk/headstudio-stage1-prior"))
        self.assertTrue(stage2.trial_dir.endswith("Elon_Musk/headstudio-stage2-text"))


if __name__ == "__main__":
    unittest.main()
