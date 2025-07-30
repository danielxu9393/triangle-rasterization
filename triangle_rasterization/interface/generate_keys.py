from dataclasses import dataclass
from typing import Protocol

from jaxtyping import Float, Int, Int64
from torch import Tensor

from triangle_rasterization.types import TileGrid


@dataclass
class PairedKeys:
    keys: Int64[Tensor, " key"]
    triangle_indices: Int[Tensor, " key"]
    num_keys: int


class GenerateKeysFn(Protocol):
    def __call__(
        self,
        depths: Float[Tensor, " vertex"],
        faces: Int[Tensor, "triangle corner=3"],
        tile_minima: Int[Tensor, "triangle xy=2"],
        tile_maxima: Int[Tensor, "triangle xy=2"],
        num_tiles_overlapped: Int[Tensor, " triangle"],
        grid: TileGrid,
    ) -> PairedKeys:
        pass
