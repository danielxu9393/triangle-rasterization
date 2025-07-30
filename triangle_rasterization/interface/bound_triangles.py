from dataclasses import dataclass
from typing import Protocol

from jaxtyping import Float, Int
from torch import Tensor

from triangle_rasterization.types import TileGrid


@dataclass
class TriangleBounds:
    # Minimum tile indices (inclusive) that each triangle overlaps.
    tile_minima: Int[Tensor, "triangle xy=2"]

    # Maximum tile indices (exclusive) that each triangle overlaps.
    tile_maxima: Int[Tensor, "triangle xy=2"]

    # The number of tiles each triangle overlaps.
    num_tiles_overlapped: Int[Tensor, " triangle"]


class BoundTrianglesFn(Protocol):
    def __call__(
        self,
        vertices: Float[Tensor, "vertex xy=2"],
        faces: Int[Tensor, "triangle corner=3"],
        grid: TileGrid,
    ) -> TriangleBounds:
        pass
