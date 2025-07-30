import torch

from triangle_rasterization.interface.generate_keys import PairedKeys
from triangle_rasterization.types import TileGrid


def sort_keys(paired_keys: PairedKeys, grid: TileGrid) -> PairedKeys:
    # keys, sorted_indices = torch.sort(paired_keys.keys)
    # triangle_indices = paired_keys.triangle_indices[sorted_indices]
    # return PairedKeys(keys, triangle_indices)
    num_keys = paired_keys.num_keys
    if num_keys > 0:
        # extract valid portion
        valid_keys = paired_keys.keys[:num_keys]
        valid_tris = paired_keys.triangle_indices[:num_keys]
        # sort valid entries
        sorted_keys, sorted_idx = torch.sort(valid_keys)
        sorted_tris = valid_tris[sorted_idx]
        # prepare padded outputs
        total_len = paired_keys.keys.shape[0]
        keys = torch.zeros_like(paired_keys.keys)
        triangle_indices = torch.zeros_like(paired_keys.triangle_indices)
        # fill sorted valid range
        keys[:num_keys] = sorted_keys
        triangle_indices[:num_keys] = sorted_tris
    else:
        # no valid keys, preserve original buffers
        keys = paired_keys.keys.clone()
        triangle_indices = paired_keys.triangle_indices.clone()
    # return with same num_keys scalar
    return PairedKeys(keys, triangle_indices, paired_keys.num_keys)
