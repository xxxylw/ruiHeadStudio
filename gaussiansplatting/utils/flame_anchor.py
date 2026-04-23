import torch


def apply_face_transform(local_points: torch.Tensor, T: torch.Tensor, R: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    scale = (S + 1e-10).sqrt().unsqueeze(-1)
    rotated = torch.bmm(R, local_points.unsqueeze(-1)).squeeze(-1)
    return scale * rotated + T
