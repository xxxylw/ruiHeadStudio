import os
import sys
import tempfile
import unittest
import importlib.util
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
    spec = importlib.util.spec_from_file_location("test_config_module", config_path)
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


ExperimentConfig = load_config_module().ExperimentConfig


class TestExperimentConfigPaths(unittest.TestCase):
    def test_prompt_tagged_runs_use_prompt_first_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                name="headstudio-stage1-prior",
                tag="Elon_Musk",
                exp_root_dir=tmpdir,
                use_timestamp=False,
            )

            self.assertEqual(cfg.exp_dir, os.path.join(tmpdir, "Elon_Musk"))
            self.assertEqual(
                cfg.trial_dir,
                os.path.join(tmpdir, "Elon_Musk", "headstudio-stage1-prior"),
            )

    def test_prompt_tagged_runs_append_timestamp_before_stage_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                name="headstudio-stage2-text",
                tag="Elon_Musk",
                exp_root_dir=tmpdir,
                use_timestamp=True,
                timestamp="@20260423-120000",
            )

            self.assertEqual(
                cfg.exp_dir,
                os.path.join(tmpdir, "Elon_Musk@20260423-120000"),
            )
            self.assertEqual(
                cfg.trial_dir,
                os.path.join(
                    tmpdir,
                    "Elon_Musk@20260423-120000",
                    "headstudio-stage2-text",
                ),
            )

    def test_untagged_runs_keep_stage_first_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                name="headstudio-stage1-prior",
                tag="",
                exp_root_dir=tmpdir,
                use_timestamp=True,
                timestamp="@20260423-120000",
            )

            self.assertEqual(
                cfg.exp_dir,
                os.path.join(tmpdir, "headstudio-stage1-prior"),
            )
            self.assertEqual(
                cfg.trial_dir,
                os.path.join(
                    tmpdir,
                    "headstudio-stage1-prior",
                    "@20260423-120000",
                ),
            )


if __name__ == "__main__":
    unittest.main()
