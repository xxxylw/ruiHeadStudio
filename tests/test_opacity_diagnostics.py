import unittest
import importlib.util
import sys
from pathlib import Path

import torch


def load_opacity_diagnostics_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "threestudio"
        / "utils"
        / "opacity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("opacity_diagnostics_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


opacity_diagnostics = load_opacity_diagnostics_module()
classify_by_face_normals = opacity_diagnostics.classify_by_face_normals
summarize_gaussian_regions = opacity_diagnostics.summarize_gaussian_regions
summarize_tensor = opacity_diagnostics.summarize_tensor


class TestOpacityDiagnostics(unittest.TestCase):
    def test_summarize_tensor_returns_json_safe_stats(self):
        stats = summarize_tensor(torch.tensor([0.1, 0.2, 0.9], dtype=torch.float32))

        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 0.4, places=6)
        self.assertIn("p10", stats)
        self.assertIn("p50", stats)
        self.assertIn("p90", stats)

    def test_summarize_tensor_handles_empty_tensor(self):
        stats = summarize_tensor(torch.empty(0))

        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["p50"])

    def test_classify_by_face_normals_uses_x_axis_regions(self):
        face_normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        labels = classify_by_face_normals(
            face_normals, front_threshold=0.35, rear_threshold=-0.35
        )

        self.assertEqual(labels, ["front", "rear", "side", "side"])

    def test_summarize_gaussian_regions_groups_opacity_and_scaling(self):
        opacity = torch.tensor([[0.9], [0.2], [0.5]], dtype=torch.float32)
        scaling = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = ["front", "rear", "rear"]

        summary = summarize_gaussian_regions(opacity, scaling, labels)

        self.assertEqual(summary["total"]["count"], 3)
        self.assertEqual(summary["regions"]["front"]["count"], 1)
        self.assertEqual(summary["regions"]["rear"]["count"], 2)
        self.assertEqual(summary["regions"]["side"]["count"], 0)
        self.assertAlmostEqual(
            summary["regions"]["rear"]["opacity"]["mean"], 0.35, places=6
        )


if __name__ == "__main__":
    unittest.main()
