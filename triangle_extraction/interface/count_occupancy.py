from typing import Protocol

from jaxtyping import Int32
from torch import Tensor


class CountOccupancyFn(Protocol):
    def __call__(
        self,
        occupancy: Int32[Tensor, "i j k_packed"],
    ) -> Int32[Tensor, " (i+1)*(j+1)*(k_packed+1)+1"]:
        pass
