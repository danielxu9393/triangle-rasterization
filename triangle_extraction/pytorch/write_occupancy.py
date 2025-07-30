import torch
from jaxtyping import Int32
from torch import Tensor

from ..misc import pack_occupancy


def write_occupancy(
    shell_indices: Int32[Tensor, " triangle"],
    voxel_indices: Int32[Tensor, " triangle"],
    occupancy_shape: tuple[int, int, int],
    min_shell: int,
    max_shell: int,
) -> Int32[Tensor, "i j k_packed"]:
    device = shell_indices.device
    i, j, k = occupancy_shape
    occupancy = torch.zeros((i, j, k), dtype=torch.bool, device=device)
    mask = (shell_indices >= min_shell) & (shell_indices < max_shell)
    voxel_indices = voxel_indices[mask]
    occupancy.view(-1)[voxel_indices] = True
    return pack_occupancy(occupancy)
