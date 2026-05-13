from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_headstudio_defaults_disable_eye_and_neck_pose():
    config = (ROOT / "configs/headstudio.yaml").read_text()
    source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()

    assert "use_eye_pose: False" in config
    assert "use_neck_pose: False" in config
    assert "use_eye_pose: bool = False" in source
    assert "use_neck_pose: bool = False" in source


def test_headstudio_guards_eye_and_neck_pose_assignment():
    source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()

    assert "if self.cfg.use_eye_pose:" in source
    assert "self.gaussian._leye_pose = leye_pose.detach()" in source
    assert "self.gaussian._reye_pose = reye_pose.detach()" in source
    assert "if self.cfg.use_neck_pose and neck_pose is not None:" in source


if __name__ == "__main__":
    test_headstudio_defaults_disable_eye_and_neck_pose()
    test_headstudio_guards_eye_and_neck_pose_assignment()
