from typing import Protocol

from jaxtyping import Int32
from torch import Tensor


class WriteOccupancyFn(Protocol):
    def __call__(
        self,
        shell_indices: Int32[Tensor, " triangle"],
        voxel_indices: Int32[Tensor, " triangle"],
        occupancy_shape: tuple[int, int, int],
        min_shell: int,
        max_shell: int,
    ) -> Int32[Tensor, "i j k_packed"]:
        pass
