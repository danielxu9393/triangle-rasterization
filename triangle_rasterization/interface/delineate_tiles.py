from typing import Protocol

from jaxtyping import Int
from torch import Tensor

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid


class DelineateTilesFn(Protocol):
    def __call__(
        self,
        sorted_paired_keys: PairedKeys,
        grid: TileGrid,
    ) -> Int[Tensor, "tile boundary=2"]:
        pass
