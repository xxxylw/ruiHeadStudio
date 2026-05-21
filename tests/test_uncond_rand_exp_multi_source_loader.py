from pathlib import Path
import importlib.util
import random
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "threestudio/data/uncond_rand_exp.py"


def load_module():
    for name in [
        "pytorch_lightning",
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.utils",
        "torch.utils.data",
        "threestudio",
        "threestudio.utils.base",
        "threestudio.utils.config",
        "threestudio.utils.misc",
        "threestudio.utils.ops",
        "threestudio.utils.typing",
        "threestudio.utils.head_v2",
    ]:
        sys.modules.pop(name, None)

    torch_mod = types.ModuleType("torch")
    torch_mod.as_tensor = lambda *args, **kwargs: None
    torch_mod.rand = lambda *args, **kwargs: None
    torch_mod.randn = lambda *args, **kwargs: None
    torch_mod.full_like = lambda *args, **kwargs: None
    torch_mod.linspace = lambda *args, **kwargs: None
    torch_mod.stack = lambda *args, **kwargs: None
    torch_mod.zeros_like = lambda *args, **kwargs: None
    torch_mod.int32 = "int32"
    torch_mod.float32 = "float32"
    torch_mod.utils = types.SimpleNamespace(data=types.ModuleType("torch.utils.data"))
    class DataLoader:
        pass

    class Dataset:
        pass

    class IterableDataset:
        pass

    torch_mod.utils.data.DataLoader = DataLoader
    torch_mod.utils.data.Dataset = Dataset
    torch_mod.utils.data.IterableDataset = IterableDataset
    torch_mod.utils.data.default_collate = lambda batch: batch

    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    functional_mod.normalize = lambda *args, **kwargs: None
    utils_mod = types.ModuleType("torch.utils")
    data_mod = torch_mod.utils.data

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod

    pl_mod = types.ModuleType("pytorch_lightning")
    pl_mod.LightningDataModule = object
    sys.modules["pytorch_lightning"] = pl_mod

    threestudio_mod = types.ModuleType("threestudio")
    threestudio_mod.warn = lambda *args, **kwargs: None
    threestudio_mod.register = lambda name: (lambda cls: cls)
    sys.modules["threestudio"] = threestudio_mod

    base_mod = types.ModuleType("threestudio.utils.base")
    class Updateable:
        pass

    base_mod.Updateable = Updateable
    config_mod = types.ModuleType("threestudio.utils.config")
    config_mod.parse_structured = lambda cls, cfg: cfg
    misc_mod = types.ModuleType("threestudio.utils.misc")
    misc_mod.get_device = lambda: "cuda"
    ops_mod = types.ModuleType("threestudio.utils.ops")
    for name in ["get_mvp_matrix", "get_projection_matrix", "get_ray_directions", "get_rays"]:
        setattr(ops_mod, name, lambda *args, **kwargs: None)
    typing_mod = types.ModuleType("threestudio.utils.typing")
    for name in ["Any", "Dict", "List", "Optional", "Tuple", "Union"]:
        setattr(typing_mod, name, getattr(__import__("typing"), name))
    typing_mod.DictConfig = dict
    typing_mod.Float = object
    typing_mod.Tensor = object
    head_mod = types.ModuleType("threestudio.utils.head_v2")
    head_mod.FlamePointswRandomExp = object

    sys.modules["threestudio.utils.base"] = base_mod
    sys.modules["threestudio.utils.config"] = config_mod
    sys.modules["threestudio.utils.misc"] = misc_mod
    sys.modules["threestudio.utils.ops"] = ops_mod
    sys.modules["threestudio.utils.typing"] = typing_mod
    sys.modules["threestudio.utils.head_v2"] = head_mod

    spec = importlib.util.spec_from_file_location("uncond_rand_exp", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_sequence(value: float, frames: int = 1):
    return {
        "expression": np.arange(value, value + frames, dtype=np.float32)[:, None].repeat(100, axis=1),
        "jaw_pose": np.arange(value, value + frames, dtype=np.float32)[:, None].repeat(3, axis=1),
        "leye_pose": np.arange(value, value + frames, dtype=np.float32)[:, None].repeat(3, axis=1),
        "reye_pose": np.arange(value, value + frames, dtype=np.float32)[:, None].repeat(3, axis=1),
        "neck_pose": np.arange(value, value + frames, dtype=np.float32)[:, None].repeat(3, axis=1),
    }


def test_multi_source_pose_collection_loads_paths_and_weights(tmp_path):
    module = load_module()
    talkshow_path = tmp_path / "talkshow.npy"
    talkvid_path = tmp_path / "talkvid.npy"
    np.save(talkshow_path, np.array([make_sequence(1.0)], dtype=object), allow_pickle=True)
    np.save(talkvid_path, np.array([make_sequence(2.0), make_sequence(3.0)], dtype=object), allow_pickle=True)

    sources = module.load_pose_sources(
        [
            {"path": str(talkshow_path), "weight": 1.0, "name": "talkshow"},
            {"path": str(talkvid_path), "weight": 3.0, "name": "talkvid"},
        ],
        fallback_path="unused.npy",
    )

    assert [source.name for source in sources] == ["talkshow", "talkvid"]
    assert [source.weight for source in sources] == [1.0, 3.0]
    assert [len(source.sequences) for source in sources] == [1, 2]


def test_weighted_pose_source_sampling_happens_before_sequence_sampling():
    module = load_module()
    sources = [
        module.PoseSource("talkshow", 1.0, [make_sequence(1.0)]),
        module.PoseSource("talkvid", 3.0, [make_sequence(2.0), make_sequence(3.0)]),
    ]
    rng = random.Random(7)

    samples = [module.sample_pose_sequence(sources, rng)[0] for _ in range(1000)]

    talkvid_count = samples.count("talkvid")
    assert 700 <= talkvid_count <= 800


def test_temporal_window_sampling_uses_per_source_sequential_cursors():
    module = load_module()
    sources = [
        module.PoseSource("talkshow", 1.0, [make_sequence(10.0, frames=3), make_sequence(20.0, frames=2)]),
        module.PoseSource("talkvid", 1.0, [make_sequence(100.0, frames=3)]),
    ]
    cursors = module.create_pose_source_cursors(sources)

    first = module.sample_pose_window_from_source(
        sources,
        cursors,
        source_index=0,
        window_length=2,
        window_stride=1,
    )
    second = module.sample_pose_window_from_source(
        sources,
        cursors,
        source_index=0,
        window_length=2,
        window_stride=1,
    )
    talkvid = module.sample_pose_window_from_source(
        sources,
        cursors,
        source_index=1,
        window_length=2,
        window_stride=1,
    )
    third = module.sample_pose_window_from_source(
        sources,
        cursors,
        source_index=0,
        window_length=2,
        window_stride=1,
    )

    assert first["source_name"] == "talkshow"
    assert first["sequence_index"] == 0
    assert first["frame_indices"] == [0, 1]
    assert first["sequence"]["expression"][:, 0].tolist() == [10.0, 11.0, 12.0]

    assert second["sequence_index"] == 0
    assert second["frame_indices"] == [1, 2]

    assert talkvid["source_name"] == "talkvid"
    assert talkvid["sequence_index"] == 0
    assert talkvid["frame_indices"] == [0, 1]

    assert third["sequence_index"] == 1
    assert third["frame_indices"] == [0, 1]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_multi_source_pose_collection_loads_paths_and_weights(Path(tmpdir))
    test_weighted_pose_source_sampling_happens_before_sequence_sampling()
    test_temporal_window_sampling_uses_per_source_sequential_cursors()
