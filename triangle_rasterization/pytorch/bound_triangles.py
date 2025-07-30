import torch
from einops import reduce
from jaxtyping import Float, Int
from torch import Tensor

from triangle_rasterization.interface.bound_triangles import TriangleBounds
from triangle_rasterization.types import TileGrid


def bound_triangles(
    vertices: Float[Tensor, "vertex xy=2"],
    depths: Float[Tensor, " vertex"],
    faces: Int[Tensor, "triangle corner=3"],
    grid: TileGrid,
    near_plane: float = 0.2,
    backface_culling: int = 1,
) -> TriangleBounds:
    device = vertices.device
    vertices = vertices[faces].flip(-1)  # xy -> yx

    # Determine tile membership.
    tile_hw = torch.tensor(
        (grid.tile_shape),
        dtype=torch.int32,
        device=device,
    )

    tile_minima = reduce(vertices, "t c xy -> t xy", "min")
    tile_minima = tile_minima.type(torch.int32) // tile_hw

    tile_maxima = reduce(vertices, "t c xy -> t xy", "max")
    tile_maxima = (tile_maxima / tile_hw).ceil().type(torch.int32)

    upper_bound = torch.tensor(
        grid.grid_shape,
        dtype=torch.int32,
        device=device,
    )
    tile_minima = tile_minima.clip(min=0).clip(max=upper_bound)
    tile_maxima = tile_maxima.clip(min=0).clip(max=upper_bound)

    # zero out tile bounds for triangles with very small depth
    min_depths = depths[faces].min(dim=1).values  # shape (T,)
    mask = min_depths < near_plane
    tile_minima[mask] = 0
    tile_maxima[mask] = 0

    return TriangleBounds(
        tile_minima,
        tile_maxima,
        (tile_maxima - tile_minima).prod(dim=-1),
    )
