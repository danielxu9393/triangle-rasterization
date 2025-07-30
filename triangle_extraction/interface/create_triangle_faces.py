from typing import NamedTuple, Protocol

from jaxtyping import Float, Int, UInt8
from torch import Tensor


class CreateTriangleFacesResult(NamedTuple):
    faces: Int[Tensor, "triangle_face corner=3"]
    shell_indices: Int[Tensor, " triangle_face"]
    voxel_indices: Int[Tensor, " triangle_face"] | None


class CreateTriangleFacesFn(Protocol):
    def __call__(
        self,
        signed_distances: Float[Tensor, " sample"],
        neighbors: Int[Tensor, "neighbor=7 voxel"],
        indices: Int[Tensor, " voxel"],
        vertex_counts: Int[Tensor, " voxel_and_subvoxel+1"],
        vertex_counts_by_level_set: UInt8[Tensor, "level_set voxel_and_subvoxel"],
        triangle_vertex_types: UInt8[Tensor, " triangle_vertex"],
        face_counts: Int[Tensor, " voxel+1"],
        voxel_cell_codes: UInt8[Tensor, "level_set voxel"],
        level_sets: Float[Tensor, " level_set"],
    ) -> CreateTriangleFacesResult:
        pass
