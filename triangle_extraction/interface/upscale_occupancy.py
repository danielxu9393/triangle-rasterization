from typing import Protocol

from jaxtyping import Int32
from torch import Tensor


class UpscaleOccupancyFn(Protocol):
    def __call__(
        self,
        occupancy: Int32[Tensor, "i j k_packed"],
        target_shape: tuple[int, int, int],
    ) -> Int32[Tensor, "i_target j_target k_packed_target"]:
        pass
