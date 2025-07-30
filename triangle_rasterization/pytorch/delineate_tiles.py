import torch
from jaxtyping import Int
from torch import Tensor

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid


def delineate_tiles(
    sorted_paired_keys: PairedKeys,
    grid: TileGrid,
) -> Int[Tensor, "tile boundary=2"]:
    device = sorted_paired_keys.keys.device

    # Extract the tile indices (leftmost 32 bits) from the keys.
    tile_indices = (sorted_paired_keys.keys & (-1 << 32)) >> 32

    # Extract the depths (rightmost 32 bits) from the keys.
    depths = sorted_paired_keys.keys & 0xFFFFFFFF
    depths = depths.view(torch.int64).type(torch.int32).view(torch.float32)

    boundaries = torch.zeros(
        (grid.num_tiles, 2),
        dtype=torch.int32,
        device=device,
    )

    for i in range(sorted_paired_keys.num_keys):
        at_tile_start = (i == 0) or (tile_indices[i] != tile_indices[i - 1])
        at_tile_end = (i == sorted_paired_keys.num_keys - 1) or (
            tile_indices[i] != tile_indices[i + 1]
        )
        tile_index = tile_indices[i]

        # Handle empty tiles.
        if at_tile_end and (depths[i] <= 0):
            boundaries[tile_index, 0] = i + 1

        # Handle full tiles (start).
        if at_tile_start and (depths[i] > 0):
            boundaries[tile_index, 0] = i

        # Handle partially full tiles (start).
        if (
            i > 0
            and (tile_indices[i] == tile_indices[i - 1])
            and (depths[i] > 0)
            and (depths[i - 1] <= 0)
        ):
            boundaries[tile_index, 0] = i

        # Handle ends for all tiles.
        if at_tile_end:
            boundaries[tile_index, 1] = i + 1

    return boundaries
