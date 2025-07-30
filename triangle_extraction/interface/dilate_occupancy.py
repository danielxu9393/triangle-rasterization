from typing import Protocol

from jaxtyping import Int32
from torch import Tensor


class DilateOccupancyFn(Protocol):
    def __call__(
        self,
        occupancy: Int32[Tensor, "i j k_packed"],
        dilation: int,
    ) -> Int32[Tensor, "i j k_packed"]:
        pass
