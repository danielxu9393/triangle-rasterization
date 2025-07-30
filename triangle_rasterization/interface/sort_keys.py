from typing import Protocol

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid


class SortKeysFn(Protocol):
    def __call__(
        self,
        paired_keys: PairedKeys,
        grid: TileGrid,
    ) -> PairedKeys:
        pass
