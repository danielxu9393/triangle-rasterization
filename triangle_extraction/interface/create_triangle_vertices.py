from typing import NamedTuple, Protocol

from jaxtyping import Float32, Int32, UInt8
from torch import Tensor


class CreateTriangleVerticesResult(NamedTuple):
    vertices: Float32[Tensor, "triangle_vertex xyz=3"]
    signed_distances: Float32[Tensor, " triangle_vertex"]
    spherical_harmonics: Float32[Tensor, "sh triangle_vertex rgb=3"]
    vertex_types: UInt8[Tensor, " triangle_vertex"]


class CreateTriangleVerticesFn(Protocol):
    def __call__(
        self,
        grid_vertices: Float32[Tensor, "sample xyz=3"],
        grid_signed_distances: Float32[Tensor, " sample"],
        grid_spherical_harmonics: Float32[Tensor, "sh sample rgb=3"],
        lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        vertex_counts: Int32[Tensor, " voxel_and_subvoxel+1"],
        level_sets: Float32[Tensor, " level_set"],
    ) -> CreateTriangleVerticesResult:
        pass
