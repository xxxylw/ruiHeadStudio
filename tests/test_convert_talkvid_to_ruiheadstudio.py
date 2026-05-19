import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/convert_talkvid_to_ruiheadstudio.py"


def load_module():
    spec = importlib.util.spec_from_file_location("convert_talkvid_to_ruiheadstudio", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_frame(path: Path, value: float) -> None:
    np.savez(
        path,
        exp=np.full((1, 100), value, dtype=np.float32),
        jaw_pose=np.full((1, 3), value + 1, dtype=np.float32),
        neck_pose=np.full((1, 3), value + 2, dtype=np.float32),
        eye_pose=np.array([[value + 3, value + 4, value + 5, value + 6, value + 7, value + 8]], dtype=np.float32),
    )


def test_talkvid_converter_defaults_to_local_tracker_and_collection_paths():
    module = load_module()

    assert module.DEFAULT_INPUTS == ("data/talkvid/flame_tracker",)
    assert module.DEFAULT_OUTPUT == "talkshow/collection/talkvid/talkvid_tracker_exp.npy"


def test_talkvid_tracking_dir_converts_numeric_frame_order_and_eye_pose(tmp_path):
    module = load_module()
    clip_dir = tmp_path / "videoABC-scene1"
    clip_dir.mkdir()
    write_frame(clip_dir / "10.npz", 10.0)
    write_frame(clip_dir / "2.npz", 2.0)
    write_frame(clip_dir / "1.npz", 1.0)

    sequence = module.convert_tracking_dir(clip_dir)

    assert sequence["expression"].shape == (3, 100)
    assert sequence["jaw_pose"].shape == (3, 3)
    assert sequence["neck_pose"].shape == (3, 3)
    assert sequence["leye_pose"].shape == (3, 3)
    assert sequence["reye_pose"].shape == (3, 3)
    assert sequence["video_name"] == tmp_path.name
    assert sequence["clip_name"] == "videoABC-scene1"
    assert sequence["source"] == "talkvid"
    assert sequence["expression"][:, 0].tolist() == [1.0, 2.0, 10.0]
    assert sequence["leye_pose"][0].tolist() == [4.0, 5.0, 6.0]
    assert sequence["reye_pose"][0].tolist() == [7.0, 8.0, 9.0]


if __name__ == "__main__":
    test_talkvid_converter_defaults_to_local_tracker_and_collection_paths()
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_talkvid_tracking_dir_converts_numeric_frame_order_and_eye_pose(Path(tmpdir))
