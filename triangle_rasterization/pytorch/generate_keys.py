import torch
from jaxtyping import Float, Int
from torch import Tensor

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid

from triangle_rasterization.misc import TILE_KEYS


def generate_keys(
    depths: Float[Tensor, " vertex"],
    faces: Int[Tensor, "triangle corner=3"],
    tile_minima: Int[Tensor, "triangle xy=2"],
    tile_maxima: Int[Tensor, "triangle xy=2"],
    num_tiles_overlapped: Int[Tensor, " triangle"],
    grid: TileGrid,
) -> PairedKeys:
    device = depths.device
    depths = depths[faces]

    # Compute per-triangle offsets in the keys.
    cumsum = num_tiles_overlapped.cumsum(dim=0)
    num_keys = cumsum.max().item()
    offsets = torch.cat((torch.zeros_like(cumsum[:1]), cumsum[:-1]), dim=0)

    # Initialize the paired keys and triangle indices that will be filled in later.
    padded_num_keys = (
        ((num_keys + TILE_KEYS - 1) // TILE_KEYS) * TILE_KEYS if num_keys > 0 else 0
    )
    keys = torch.zeros(padded_num_keys, dtype=torch.int64, device=device)
    triangle_indices = torch.zeros(padded_num_keys, dtype=torch.int32, device=device)

    # Iterate over the number of tiles in the largest box.
    box = tile_maxima - tile_minima
    # box_width, box_height = box.unbind(dim=-1)
    box_height, box_width = box.unbind(dim=-1)
    for index_in_box in range(num_tiles_overlapped.max().item()):
        box_row = index_in_box // box_width
        box_col = index_in_box % box_width

        # Compute each tile's index within the image.
        # x_min, y_min = tile_minima.unbind(dim=-1)
        y_min, x_min = tile_minima.unbind(dim=-1)
        row = y_min + box_row
        col = x_min + box_col
        index_in_image = row * grid.grid_shape[1] + col

        # Compute each tile's index within the keys.
        index_in_keys = offsets + index_in_box

        # Skip invalid tiles (i.e., handle boxes that are smaller than the largest box).
        # x_max, y_max = tile_maxima.unbind(dim=-1)
        y_max, x_max = tile_maxima.unbind(dim=-1)
        valid = (row < y_max) & (col < x_max) & (box_width > 0) & (box_height > 0)

        index_in_keys = index_in_keys[valid]
        index_in_triangles = torch.arange(len(valid), device=device, dtype=torch.int32)[
            valid
        ]
        index_in_image = index_in_image[valid]
        depth = depths[valid].mean(dim=-1)
        depth[(depths[valid] <= 0).any(dim=-1)] = 0

        # Construct the key. The left 32 bits represent the tile index, while the right
        # 32 bits are a float32 depth.
        tile_part = index_in_image.type(torch.int64) << 32
        depth_part = depth.view(torch.uint32).type(torch.int64)
        keys[index_in_keys] = (tile_part + depth_part).type(torch.int64)

        # Save the key's corresponding triangle index.
        triangle_indices[index_in_keys] = index_in_triangles

    return PairedKeys(keys, triangle_indices, num_keys)
