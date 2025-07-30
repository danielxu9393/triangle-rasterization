from typing import Protocol

from jaxtyping import Float32, Int32
from torch import Tensor

from triangle_rasterization.types import TileGrid


class CompositeFn(Protocol):
    def __call__(
        self,
        vertices: Float32[Tensor, "vertex xy=2"],
        colors: Float32[Tensor, "vertex rgb=3"],
        signed_distances: Float32[Tensor, " vertex"],
        faces: Int32[Tensor, "triangle corner=3"],
        shell_indices: Int32[Tensor, " triangle"],
        sharpness: Float32[Tensor, ""],
        num_shells: int,
        sorted_triangle_indices: Int32[Tensor, " key"],
        tile_boundaries: Int32[Tensor, "tile boundary=2"],
        grid: TileGrid,
        voxel_indices: Int32[Tensor, " triangle"] | None = None,
        occupancy: Int32[Tensor, "i j k_packed"] | None = None,
    ) -> Float32[Tensor, "4 height width"]:
        pass
