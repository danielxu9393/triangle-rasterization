from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32, Int32, Int64
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.composite import CompositeFn
from ..misc import TILE_KEYS
from ..types import TileGrid

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "composite.slang"),
        # defines={"TILE_HEIGHT": 16, "TILE_WIDTH": 16, "TILE_KEYS": TILE_KEYS},
        defines={"TILE_HEIGHT": 16, "TILE_WIDTH": 16},
        verbose=True,
    )
)


class Composite(torch.autograd.Function):
    @record_function("composite_forward")
    @staticmethod
    def forward(
        ctx,
        vertices: Float32[Tensor, "vertex xy=2"],
        colors: Float32[Tensor, "vertex rgb=3"],
        signed_distances: Float32[Tensor, " vertex"],
        faces: Int32[Tensor, "triangle corner=3"],
        shell_indices: Int32[Tensor, " triangle"],
        sharpness: Float32[Tensor, ""],
        num_shells: int,
        sorted_triangle_indices: Int32[Tensor, " key"],
        tile_boundaries: Int64[Tensor, "tile boundary=2"],
        grid: TileGrid,
        voxel_indices: Int32[Tensor, " triangle"] | None,
        occupancy: Int32[Tensor, "*shape"] | None,
    ):
        device = vertices.device

        # Handle marking occupied voxels during rendering.
        assert (voxel_indices is None) == (occupancy is None)
        mark_occupancy = occupancy is not None
        if voxel_indices is None:
            voxel_indices = torch.empty((), dtype=torch.int32, device=device)
            occupancy = voxel_indices

        # Composite the grid's active area.
        h, w = grid.active_image_shape
        out_image = torch.empty((4, h, w), dtype=torch.float32, device=device)

        slang().composite_forward(
            vertices=vertices,
            colors=colors,
            signedDistances=signed_distances,
            faces=faces,
            shellIndices=shell_indices,
            sharpness=sharpness,
            numShells=num_shells,
            sortedTriangleIndices=sorted_triangle_indices,
            tileBoundaries=tile_boundaries,
            gridNumCols=grid.grid_shape[1],
            gridRowMinimum=grid.active_minimum[0],
            gridColMinimum=grid.active_minimum[1],
            gridTileHeight=grid.tile_shape[0],
            gridTileWidth=grid.tile_shape[1],
            imageHeight=h,
            imageWidth=w,
            outAccumulators=out_image,
            voxelIndices=voxel_indices,
            occupancy=occupancy.view(-1),
            markOccupancy=mark_occupancy,
        ).launchRaw(
            blockSize=(grid.tile_shape[1], grid.tile_shape[0], 1),
            gridSize=(grid.active_grid_shape[1], grid.active_grid_shape[0], 1),
        )

        ctx.save_for_backward(
            vertices,
            colors,
            signed_distances,
            faces,
            shell_indices,
            sharpness,
            sorted_triangle_indices,
            tile_boundaries,
            out_image,
        )
        ctx.num_shells = num_shells
        ctx.grid = grid

        # This is returned because it's useful for logging, but dLoss isn't actually
        # used in the backward pass.
        return out_image

    @record_function("composite_backward")
    @staticmethod
    def backward(
        ctx,
        out_image_grad,
    ):
        (
            vertices,
            colors,
            signed_distances,
            faces,
            shell_indices,
            sharpness,
            sorted_triangle_indices,
            tile_boundaries,
            out_image,
        ) = ctx.saved_tensors
        num_shells = ctx.num_shells
        grid = ctx.grid

        # Note: The kernel for the forward pass also computes the backward pass.
        colors_grad = torch.zeros_like(colors)
        signed_distances_grad = torch.zeros_like(signed_distances)
        sharpness_grad = torch.zeros_like(sharpness)

        h, w = grid.active_image_shape
        slang().composite_backward(
            vertices=vertices,
            colors=(colors, colors_grad),
            signedDistances=(signed_distances, signed_distances_grad),
            faces=faces,
            shellIndices=shell_indices,
            sharpness=(sharpness, sharpness_grad),
            numShells=num_shells,
            sortedTriangleIndices=sorted_triangle_indices,
            tileBoundaries=tile_boundaries,
            gridNumCols=grid.grid_shape[1],
            gridRowMinimum=grid.active_minimum[0],
            gridColMinimum=grid.active_minimum[1],
            gridTileHeight=grid.tile_shape[0],
            gridTileWidth=grid.tile_shape[1],
            imageHeight=h,
            imageWidth=w,
            outAccumulators=(out_image, out_image_grad),
        ).launchRaw(
            blockSize=(grid.tile_shape[1], grid.tile_shape[0], 1),
            gridSize=(grid.active_grid_shape[1], grid.active_grid_shape[0], 1),
        )

        del vertices
        del colors
        del signed_distances
        del faces
        del shell_indices
        del sharpness
        del sorted_triangle_indices
        del tile_boundaries
        del out_image
        del ctx

        return (
            None,
            colors_grad,
            signed_distances_grad,
            None,
            None,
            sharpness_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


composite: CompositeFn = Composite.apply
