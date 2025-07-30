from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.bound_triangles import TriangleBounds
from ..misc import ceildiv
from ..types import TileGrid

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "bound_triangles.slang"),
        verbose=True,
    )
)


BLOCK_SIZE = 256


@record_function("bound_triangles")
def bound_triangles(
    vertices: Float[Tensor, "vertex xy=2"],
    depths: Float[Tensor, " vertex"],
    faces: Int[Tensor, "triangle corner=3"],
    grid: TileGrid,
    near_plane: float = 0.2,
    backface_culling: int = 1,
) -> TriangleBounds:
    assert backface_culling in [-1, 0, 1]  # Render behind, both, or front faces.

    device = vertices.device
    t, _ = faces.shape
    kwargs = {"device": device, "dtype": torch.int32}
    out_tile_minima = torch.empty((t, 2), **kwargs)
    out_tile_maxima = torch.empty((t, 2), **kwargs)
    out_num_tiles_touched = torch.empty((t,), **kwargs)

    slang().bound_triangles(
        vertices=vertices,
        depths=depths,
        faces=faces,
        gridTileHeight=grid.tile_shape[0],
        gridTileWidth=grid.tile_shape[1],
        gridRowMinimum=grid.active_minimum[0],
        gridColMinimum=grid.active_minimum[1],
        gridRowMaximum=grid.active_maximum[0],
        gridColMaximum=grid.active_maximum[1],
        outTileMinima=out_tile_minima,
        outTileMaxima=out_tile_maxima,
        outNumTilesTouched=out_num_tiles_touched,
        backfaceCulling=backface_culling,
        nearPlane=near_plane,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(t, BLOCK_SIZE), 1, 1),
    )

    return TriangleBounds(out_tile_minima, out_tile_maxima, out_num_tiles_touched)
