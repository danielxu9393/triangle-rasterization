from typing import NamedTuple, Protocol

from jaxtyping import Float, Int, Int32
from torch import Tensor


class Voxels(NamedTuple):
    vertices: Float[Tensor, "vertex xyz=3"]
    neighbors: Int[Tensor, "neighbor=7 voxel"]
    lower_corners: Int[Tensor, "corner=4 voxel_and_subvoxel"]
    upper_corners: Int[Tensor, "corner=4 voxel"]
    indices: Int[Tensor, " voxel"]
    grid_shape: tuple[int, int, int]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_voxels(self) -> int:
        return self.upper_corners.shape[1]

    @property
    def num_subvoxels(self) -> int:
        return self.lower_corners.shape[1] - self.upper_corners.shape[1]


class CreateVoxelsFn(Protocol):
    def __call__(
        self,
        occupancy: Int32[Tensor, "i j k_packed"],
        voxel_offsets: Int32[Tensor, "i+1 j+1 k_packed+1"],
        num_voxels: int,
        vertex_occupancy: Int32[Tensor, "i+1 j+1 k_packed+1"],
        vertex_offsets: Int32[Tensor, "i+1 j+1 k_packed+1"],
        num_vertices: int,
    ) -> Voxels:
        pass
