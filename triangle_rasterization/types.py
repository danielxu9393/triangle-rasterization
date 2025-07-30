from dataclasses import dataclass, replace

from triangle_rasterization.misc import ceildiv
from math import prod


@dataclass(frozen=True)
class TileGrid:
    # One tile's shape in pixels as (height, width).
    tile_shape: tuple[int, int]

    # The entire image's shape in pixels as (height, width).
    image_shape: tuple[int, int]

    # The shape of the grid in terms of (rows, columns).
    grid_shape: tuple[int, int]

    # The area of the grid that's rendered (in tiles). The minimum (inclusive)
    # and maximum (exclusive) are specified as (row, column).
    active_minimum: tuple[int, int]
    active_maximum: tuple[int, int]

    @staticmethod
    def create(
        image_shape: tuple[int, int],
        tile_shape: tuple[int, int],
    ) -> "TileGrid":
        grid_shape = tuple(
            ceildiv(image_length, tile_length)
            for image_length, tile_length in zip(image_shape, tile_shape)
        )
        return TileGrid(tile_shape, image_shape, grid_shape, (0, 0), grid_shape)

    @property
    def num_tiles(self) -> int:
        return prod(self.grid_shape)

    @property
    def active_image_shape(self) -> tuple[int, int]:
        result = []
        for axis in range(2):
            minimum = self.active_minimum[axis] * self.tile_shape[axis]
            maximum = self.active_maximum[axis] * self.tile_shape[axis]
            maximum = min(maximum, self.image_shape[axis])
            result.append(maximum - minimum)
        return tuple(result)

    @property
    def active_grid_shape(self) -> tuple[int, int]:
        return tuple(
            maximum - minimum
            for maximum, minimum in zip(self.active_maximum, self.active_minimum)
        )

    def clamp_to_grid(self, index: tuple[int, int]) -> tuple[int, int]:
        return tuple(
            max(0, min(grid_length, length))
            for length, grid_length in zip(index, self.grid_shape)
        )

    def with_active_area(
        self,
        start: tuple[int, int],
        extent: tuple[int, int],
    ) -> "TileGrid":
        return replace(
            self,
            active_minimum=self.clamp_to_grid(start),
            active_maximum=self.clamp_to_grid(
                tuple(s + e for s, e in zip(start, extent))
            ),
        )
