import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestOpacityAlphaPipeline(unittest.TestCase):
    def test_head_system_uses_rasterizer_alpha_as_opacity(self):
        source = (REPO_ROOT / "threestudio" / "systems" / "Head3DGSLKs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('render_pkg["opacity"] = alphas', source)
        self.assertNotIn('render_pkg["opacity"] = depths /', source)

    def test_gaussian_flame_densify_and_prune_uses_min_opacity(self):
        source = (
            REPO_ROOT / "gaussiansplatting" / "scene" / "gaussian_flame_model.py"
        ).read_text(encoding="utf-8")
        method_start = source.index("    def densify_and_prune")
        method_end = source.index("    def prune_only", method_start)
        method_source = source[method_start:method_end]

        self.assertIn("self.get_opacity < min_opacity", method_source)
        self.assertIn("torch.logical_or(prune_mask", method_source)


if __name__ == "__main__":
    unittest.main()
