import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussiansplatting.utils.flame_anchor import apply_face_transform


class TestGaussianFlameAnchorLoss(unittest.TestCase):
    def test_face_local_points_round_trip_to_face_transforms(self):
        local_points = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)
        T = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        R = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        S = torch.tensor([4.0], dtype=torch.float32)

        world_points = apply_face_transform(local_points, T, R, S)

        self.assertTrue(
            torch.allclose(
                world_points,
                torch.tensor([[1.2, 2.0, 3.0]], dtype=torch.float32),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
