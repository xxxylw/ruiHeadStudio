import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def load_uncond_rand_exp_module():
    module_name = "test_uncond_rand_exp_module"
    module_path = Path(__file__).resolve().parents[1] / "threestudio" / "data" / "uncond_rand_exp.py"

    threestudio_stub = types.ModuleType("threestudio")
    threestudio_stub.warn = lambda *args, **kwargs: None
    threestudio_stub.register = lambda name: (lambda cls: cls)

    pl_stub = types.ModuleType("pytorch_lightning")
    pl_stub.LightningDataModule = type("LightningDataModule", (), {})

    config_stub = types.ModuleType("threestudio.utils.config")
    config_stub.parse_structured = lambda cls, cfg=None: cfg

    base_stub = types.ModuleType("threestudio.utils.base")
    base_stub.Updateable = type("Updateable", (), {})

    misc_stub = types.ModuleType("threestudio.utils.misc")
    misc_stub.get_device = lambda: "cpu"

    ops_stub = types.ModuleType("threestudio.utils.ops")
    ops_stub.get_mvp_matrix = lambda *args, **kwargs: None
    ops_stub.get_projection_matrix = lambda *args, **kwargs: None
    ops_stub.get_ray_directions = lambda *args, **kwargs: None
    ops_stub.get_rays = lambda *args, **kwargs: None

    typing_stub = types.ModuleType("threestudio.utils.typing")
    typing_stub.Any = Any
    typing_stub.DictConfig = dict
    typing_stub.Float = Any
    typing_stub.List = List
    typing_stub.Optional = Optional
    typing_stub.Tensor = Any
    typing_stub.Tuple = Tuple
    typing_stub.Union = Union
    typing_stub.Dict = Dict

    head_stub = types.ModuleType("threestudio.utils.head_v2")
    head_stub.FlamePointswRandomExp = type("FlamePointswRandomExp", (), {})

    sys.modules.setdefault("threestudio", threestudio_stub)
    sys.modules.setdefault("pytorch_lightning", pl_stub)
    sys.modules.setdefault("threestudio.utils", types.ModuleType("threestudio.utils"))
    sys.modules["threestudio.utils.base"] = base_stub
    sys.modules["threestudio.utils.config"] = config_stub
    sys.modules["threestudio.utils.misc"] = misc_stub
    sys.modules["threestudio.utils.ops"] = ops_stub
    sys.modules["threestudio.utils.typing"] = typing_stub
    sys.modules["threestudio.utils.head_v2"] = head_stub

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MultiInputLoaderConfigTests(unittest.TestCase):
    def test_normalize_train_pose_inputs_falls_back_to_legacy_path(self):
        module = load_uncond_rand_exp_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "legacy.npy"
            cfg = {
                "talkshow_train_path": str(legacy_path),
                "train_pose_inputs": [],
                "train_pose_group_labels": [],
                "train_pose_group_weights": {},
            }

            normalized = module.normalize_train_pose_inputs(cfg)

            self.assertEqual(normalized["paths"], [str(legacy_path)])
            self.assertEqual(normalized["group_labels"], ["talkshow"])

    def test_expand_pose_input_specs_expands_directory_children(self):
        module = load_uncond_rand_exp_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            group_dir = Path(tmpdir) / "talkvid"
            group_dir.mkdir()
            (group_dir / "a.npy").write_bytes(b"x")
            (group_dir / "b.npy").write_bytes(b"x")

            specs = module.expand_pose_input_specs([str(group_dir)], ["talkvid"])

            self.assertEqual([spec.group_label for spec in specs], ["talkvid", "talkvid"])
            self.assertEqual([Path(spec.input_path).name for spec in specs], ["a.npy", "b.npy"])

    def test_expand_pose_input_specs_keeps_single_file(self):
        module = load_uncond_rand_exp_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "talkshow.npy"
            path.write_bytes(b"x")

            specs = module.expand_pose_input_specs([str(path)], ["talkshow"])

            self.assertEqual(len(specs), 1)
            self.assertEqual(Path(specs[0].input_path).name, "talkshow.npy")
            self.assertEqual(specs[0].group_label, "talkshow")


if __name__ == "__main__":
    unittest.main()
