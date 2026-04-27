import unittest
import importlib.util
import sys
import types
from pathlib import Path

from omegaconf import OmegaConf


def load_config_module():
    for resolver in (
        "calc_exp_lr_decay_rate",
        "add",
        "sub",
        "mul",
        "div",
        "idiv",
        "basename",
        "rmspace",
        "tuple2",
        "gt0",
        "cmaxgt0",
        "not",
        "cmaxgt0orcmaxgt0",
    ):
        if OmegaConf.has_resolver(resolver):
            OmegaConf.clear_resolver(resolver)

    stubbed_modules = [
        "threestudio",
        "threestudio.utils",
        "threestudio.utils.typing",
    ]
    previous_modules = {name: sys.modules.get(name) for name in stubbed_modules}

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
    try:
        spec.loader.exec_module(module)
    finally:
        for name in stubbed_modules:
            if previous_modules[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_modules[name]
    return module


load_config = load_config_module().load_config


class TestStageConfigs(unittest.TestCase):
    def test_stage1_and_stage2_configs_define_distinct_modes(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertEqual(stage1.data.difficulty_sampling_mode, "curriculum")
        self.assertEqual(stage2.data.difficulty_sampling_mode, "curriculum")
        self.assertFalse(stage1.system.guidance.use_nfsd)
        self.assertFalse(stage2.system.guidance.use_nfsd)

    def test_stage1_uses_neutral_prompt_and_lower_sds_weight(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertNotEqual(
            stage1.system.prompt_processor.prompt,
            stage2.system.prompt_processor.prompt,
        )
        self.assertIn("human head", stage1.system.prompt_processor.prompt.lower())
        self.assertLess(stage1.system.loss.lambda_sds, stage2.system.loss.lambda_sds)
        self.assertLessEqual(stage2.system.loss.lambda_sds, 0.6)

    def test_stage2_keeps_light_stability_regularization(self):
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertGreater(stage2.system.loss.lambda_anchor, 0.0)
        self.assertGreater(stage2.system.loss.lambda_temporal_xyz, 0.0)

    def test_stage2_default_prompt_stays_within_head_and_neck_geometry(self):
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")
        prompt = stage2.system.prompt_processor.prompt.lower()

        self.assertIn("head and neck only", prompt)
        self.assertIn("no clothing", prompt)
        self.assertIn("no collar", prompt)
        self.assertNotIn("turtleneck", prompt)
        self.assertNotIn("skin pores", prompt)
        self.assertNotIn("shallow depth of field", prompt)

    def test_stage_configs_bias_toward_more_opaque_head_geometry(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertEqual(stage1.system.loss.lambda_sparsity, 0.2)
        self.assertEqual(stage2.system.loss.lambda_sparsity, 0.1)
        self.assertEqual(stage1.system.loss.lambda_opaque, 0.05)
        self.assertEqual(stage2.system.loss.lambda_opaque, 0.1)
        self.assertGreaterEqual(stage1.system.densify_min_opacity, 0.05)
        self.assertGreaterEqual(stage2.system.densify_min_opacity, 0.05)
        self.assertGreaterEqual(stage1.system.prune_only_min_opacity, 0.05)
        self.assertGreaterEqual(stage2.system.prune_only_min_opacity, 0.05)

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
