from typing import NamedTuple, Protocol

from jaxtyping import Float, Int32, UInt8
from torch import Tensor


class CountTriangleVerticesResult(NamedTuple):
    vertex_counts: Int32[Tensor, " voxel_and_subvoxel+1"]
    vertex_counts_by_level_set: UInt8[Tensor, "level_set voxel_and_subvoxel"]


class CountTriangleVerticesFn(Protocol):
    def __call__(
        self,
        vertices: Float[Tensor, " sample xyz=3"],
        signed_distances: Float[Tensor, " sample"],
        lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        level_sets: Float[Tensor, " level_set"],
    ) -> CountTriangleVerticesResult:
        pass
