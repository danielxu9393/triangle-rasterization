# File: python/triangle_rasterization/compilation/generate_keys.py

from pathlib import Path

import torch
from torch.profiler import record_function
from torch.utils import cpp_extension

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.misc import TILE_KEYS, ceildiv
from triangle_rasterization.types import TileGrid

from ..compilation import wrap_compilation

# Compile & load the CUDA extension
cuda = wrap_compilation(
    lambda: cpp_extension.load(
        name="generate_keys",
        sources=[Path(__file__).parent / "generate_keys.cu"],
        verbose=True,
    )
)

BLOCK_SIZE = 256


@record_function("generate_keys")
def generate_keys(
    depths: torch.Tensor,
    faces: torch.Tensor,
    tile_minima: torch.Tensor,
    tile_maxima: torch.Tensor,
    num_tiles_overlapped: torch.Tensor,
    grid: TileGrid,
) -> PairedKeys:
    """
    For each triangle, emits one (tileIndex<<32 | depthBits) key per
    tile the triangle overlaps, along with the triangle index.

    Args:
        depths:         FloatTensor[Nv] giving per-vertex depth.
        faces:          IntTensor[Nf,3] listing vertex indices of each triangle.
        tile_minima:    IntTensor[Nf,2] giving min-x,min-y tile coords.
        tile_maxima:    IntTensor[Nf,2] giving max-x,max-y tile coords.
        num_tiles_overlapped: IntTensor[Nf] number of tiles each triangle spans.
        grid:           TileGrid with .grid_shape=(ny,nx), .active_minimum, etc.

    Returns:
        PairedKeys with
          - keys:          Int64Tensor[num_keys] packed (tileIndex<<32 | depthBits)
          - triangle_indices: Int32Tensor[num_keys] triangle index for each key.
    """
    device = depths.device

    # cumulative offsets into the flat key buffer
    offsets = num_tiles_overlapped.cumsum(dim=0, dtype=torch.int64)

    # total keys to emit
    num_keys = int(offsets[-1].item()) if offsets.numel() > 0 else 0
    num_faces = faces.shape[0]

    # allocate output buffers
    keys = torch.zeros(num_keys, dtype=torch.int64, device=device)
    triangle_indices = torch.zeros(num_keys, dtype=torch.int32, device=device)

    # launch CUDA kernel (if there are any faces)
    if num_faces > 0 and num_keys > 0:
        blocks = ceildiv(num_faces, BLOCK_SIZE)
        cuda().generate_keys(
            depths.contiguous(),
            faces.contiguous(),
            tile_minima.contiguous(),
            tile_maxima.contiguous(),
            offsets.contiguous(),
            grid.grid_shape[1],
            keys,
            triangle_indices,
            num_faces,
        )
        # ensure completion
        torch.cuda.synchronize(device)

    return PairedKeys(keys, triangle_indices, num_keys)
