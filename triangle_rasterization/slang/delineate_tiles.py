# File: python/triangle_rasterization/compilation/delineate_tiles.py

from pathlib import Path

import slangtorch
import torch
from torch.profiler import record_function

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid

from ..compilation import wrap_compilation

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "delineate_tiles.slang"),
        verbose=True,
    )
)

BLOCK_SIZE = 256


@record_function("delineate_tiles")
def delineate_tiles(
    sorted_paired_keys: PairedKeys,
    grid: TileGrid,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Given keys sorted by (tile << 32) | depthBits, compute for each tile the
    [start, end) range of indices whose depth > epsilon.

    Args:
        sorted_paired_keys: PairedKeys with .keys (int64) sorted by tile<<32 | depthBits
        grid: TileGrid (has .num_tiles)
        epsilon: depth threshold to consider “behind” vs “in front”

    Returns:
        IntTensor of shape (grid.num_tiles, 2) with start/end indices per tile.
    """
    device = sorted_paired_keys.keys.device

    # Zero-init (relies on kernel writing only the discovered boundaries)
    tile_boundaries = torch.zeros(
        (grid.num_tiles, 2), dtype=torch.int64, device=device
    )

    num_keys = int(sorted_paired_keys.num_keys)
    if num_keys > 0:
        # pass num_keys via a single-element int64 tensor (Slang reads [0])
        # num_keys_container = torch.tensor([num_keys], dtype=torch.int64, device=device)

        slang().delineate_tiles(
            sorted_keys=sorted_paired_keys.keys,
            # num_keys_container=num_keys_container,
            # num_keys=num_keys,
            out_tile_boundaries=tile_boundaries,
            epsilon=float(epsilon),
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=((num_keys + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
        )

        # match original behavior that synchronized after launch
        torch.cuda.synchronize(device)

    return tile_boundaries
