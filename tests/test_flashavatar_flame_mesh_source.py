from pathlib import Path
import importlib.util

import numpy as np


def load_flame_mesh_module():
    module_path = Path(__file__).resolve().parents[1] / "gaussiansplatting" / "scene" / "flame_mesh.py"
    spec = importlib.util.spec_from_file_location("test_flame_mesh_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flashavatar_flame_mesh_asset_is_vendored():
    flame_mesh = load_flame_mesh_module()

    assert flame_mesh.FLASHAVATAR_FLAME_MESH_PATH == Path("third_party/FlashAvatar-code/flame/FlameMesh.obj")
    assert flame_mesh.FLASHAVATAR_FLAME_MESH_PATH.exists()


def test_flashavatar_flame_faces_close_mouth_without_changing_vertex_count():
    flame_mesh = load_flame_mesh_module()
    faces = flame_mesh.load_flashavatar_flame_faces()

    assert faces.dtype == np.int32
    assert faces.shape == (10006, 3)
    assert faces.min() >= 0
    assert faces.max() < 5023


def test_gaussian_flame_model_uses_flashavatar_faces_for_surface_binding():
    source = Path("gaussiansplatting/scene/gaussian_flame_model.py").read_text()

    assert "from gaussiansplatting.scene.flame_mesh import load_flashavatar_flame_faces" in source
    assert "self.flame_faces = load_flashavatar_flame_faces()" in source
    assert "self.flame_faces" in source
    assert "faces = torch.tensor(self.flame_faces" in source


def test_head_condition_depth_uses_flashavatar_faces_without_reindexing_landmarks():
    source = Path("threestudio/utils/head_v2.py").read_text()

    assert "from gaussiansplatting.scene.flame_mesh import load_flashavatar_flame_faces" in source
    assert "self.flame_faces = load_flashavatar_flame_faces()" in source
    assert "get_cond_depth(vertices, self.flame_faces_tensor" in source
    assert "vertices2landmarks(" in source
    assert "self.official_flame_faces_tensor" in source
