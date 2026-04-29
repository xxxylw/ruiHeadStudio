import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def install_gaussian_flame_stubs():
    smplx_stub = types.ModuleType("smplx")
    smplx_stub.FLAME = object
    sys.modules["smplx"] = smplx_stub

    plyfile_stub = types.ModuleType("plyfile")
    plyfile_stub.PlyData = object
    plyfile_stub.PlyElement = object
    sys.modules["plyfile"] = plyfile_stub

    trimesh_stub = types.ModuleType("trimesh")
    trimesh_stub.Trimesh = object
    trimesh_stub.sample = types.SimpleNamespace(sample_surface=lambda mesh, n: None)
    sys.modules["trimesh"] = trimesh_stub

    simple_knn_stub = types.ModuleType("simple_knn")
    simple_knn_c_stub = types.ModuleType("simple_knn._C")
    simple_knn_c_stub.distCUDA2 = lambda values: values
    sys.modules["simple_knn"] = simple_knn_stub
    sys.modules["simple_knn._C"] = simple_knn_c_stub

    gaussian_model_stub = types.ModuleType("gaussiansplatting.scene.gaussian_model")

    class GaussianModel:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def get_opacity(self):
            return self.opacity_activation(self._opacity)

    gaussian_model_stub.GaussianModel = GaussianModel
    sys.modules["gaussiansplatting.scene.gaussian_model"] = gaussian_model_stub

    sh_utils_stub = types.ModuleType("gaussiansplatting.utils.sh_utils")
    sh_utils_stub.RGB2SH = lambda values: values
    sys.modules["gaussiansplatting.utils.sh_utils"] = sh_utils_stub

    system_utils_stub = types.ModuleType("gaussiansplatting.utils.system_utils")
    system_utils_stub.mkdir_p = lambda path: None
    sys.modules["gaussiansplatting.utils.system_utils"] = system_utils_stub

    flame_anchor_stub = types.ModuleType("gaussiansplatting.utils.flame_anchor")
    flame_anchor_stub.apply_face_transform = lambda xyz, t, r, s: xyz
    sys.modules["gaussiansplatting.utils.flame_anchor"] = flame_anchor_stub

    general_utils_stub = types.ModuleType("gaussiansplatting.utils.general_utils")
    general_utils_stub.strip_symmetric = lambda values: values
    general_utils_stub.build_scaling_rotation_only = lambda scaling, rotation: (
        torch.eye(3),
        torch.eye(3),
    )
    general_utils_stub.inverse_sigmoid = lambda values: values
    general_utils_stub.get_expon_lr_func = lambda **kwargs: None
    general_utils_stub.build_rotation = lambda values: torch.eye(3).repeat(
        values.shape[0], 1, 1
    )
    sys.modules["gaussiansplatting.utils.general_utils"] = general_utils_stub

    opacity_module_path = (
        REPO_ROOT / "threestudio" / "utils" / "opacity_diagnostics.py"
    )
    opacity_spec = importlib.util.spec_from_file_location(
        "threestudio.utils.opacity_diagnostics", opacity_module_path
    )
    opacity_module = importlib.util.module_from_spec(opacity_spec)
    assert opacity_spec.loader is not None
    sys.modules["threestudio"] = types.ModuleType("threestudio")
    sys.modules["threestudio.utils"] = types.ModuleType("threestudio.utils")
    sys.modules[opacity_spec.name] = opacity_module
    opacity_spec.loader.exec_module(opacity_module)


def load_gaussian_flame_model():
    install_gaussian_flame_stubs()
    module_path = REPO_ROOT / "gaussiansplatting" / "scene" / "gaussian_flame_model.py"
    spec = importlib.util.spec_from_file_location("gaussian_flame_model_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GaussianFlameModel


GaussianFlameModel = load_gaussian_flame_model()


class TestRegionPruneGuard(unittest.TestCase):
    def test_gaussian_region_labels_follow_bound_face_normals(self):
        model = object.__new__(GaussianFlameModel)
        rotations = torch.zeros((3, 3, 3), dtype=torch.float32)
        rotations[:, 0, 0] = 1.0
        rotations[:, 1, 1] = 1.0
        rotations[0, :, 2] = torch.tensor([1.0, 0.0, 0.0])
        rotations[1, :, 2] = torch.tensor([-1.0, 0.0, 0.0])
        rotations[2, :, 2] = torch.tensor([0.0, 1.0, 0.0])
        model.get_face_transform_components = lambda: (
            torch.zeros((3, 3)),
            rotations,
            torch.ones(3),
        )

        labels = model.get_gaussian_region_labels(
            front_threshold=0.35, rear_threshold=-0.35
        )

        self.assertEqual(labels, ["front", "rear", "side"])

    def test_region_opacity_prune_mask_can_lower_rear_threshold(self):
        model = object.__new__(GaussianFlameModel)
        model._opacity = torch.tensor([[0.03], [0.03], [0.01]], dtype=torch.float32)
        model.opacity_activation = lambda value: value
        model.get_gaussian_region_labels = lambda: ["front", "rear", "rear"]
        base_mask = torch.tensor([True, True, True], dtype=torch.bool)

        prune_mask = model._region_opacity_prune_mask(
            base_mask,
            {"rear": 0.02},
        )

        self.assertEqual(prune_mask.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
