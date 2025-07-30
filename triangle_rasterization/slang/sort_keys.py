from pathlib import Path

import torch
from torch.profiler import record_function
from torch.utils import cpp_extension

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid

from ..compilation import wrap_compilation

cuda = wrap_compilation(
    lambda: cpp_extension.load(
        name="sort_keys",
        sources=[Path(__file__).parent / "sort_keys.cu"],
    )
)


@record_function("sort_keys")
def sort_keys(paired_keys: PairedKeys, grid: TileGrid) -> PairedKeys:
    num_keys = paired_keys.num_keys
    assert num_keys <= paired_keys.keys.shape[0]
    highest_tile_id_msb = grid.num_tiles.bit_length()
    keys, triangle_indices = cuda().sort_keys(
        paired_keys.keys,
        paired_keys.triangle_indices,
        num_keys,
        highest_tile_id_msb,
    )

    # The sorting may happen in a different stream, so this may be necessary.
    torch.cuda.synchronize(paired_keys.keys.device)

    return PairedKeys(keys, triangle_indices, num_keys)
