from typing import NamedTuple, Protocol

from jaxtyping import Int32
from torch import Tensor


class VertexOccupancyFnResult(NamedTuple):
    vertex_occupancy: Int32[Tensor, "i+1 j+1 k_packed+1"]

    # The first n - 1 entries should be row-major to match the occupancy above.
    vertex_counts: Int32[Tensor, " (i+1)*(j+1)*(k_packed+1)+1"]


class ComputeVertexOccupancyFn(Protocol):
    def __call__(
        self,
        occupancy: Int32[Tensor, "i j k_packed"],
    ) -> VertexOccupancyFnResult:
        pass
