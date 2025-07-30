from pathlib import Path

import slangtorch
import torch
from jaxtyping import Int32
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.create_voxels import Voxels
from ..misc import ceildiv

# Since occupancies are packed along the last dimension, each block effectively operates
# on a (32, 32, 32) chunk of occupancies with a (32, 32, 1) block size.
BLOCK_SIZE = (32, 32, 1)


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "create_voxels.slang"),
        verbose=True,
    )
)


@record_function("create_voxels")
def create_voxels(
    occupancy: Int32[Tensor, "i j k_packed"],
    voxel_offsets: Int32[Tensor, "i+1 j+1 k_packed+1"],
    num_voxels: int,
    vertex_occupancy: Int32[Tensor, "i+1 j+1 k_packed+1"],
    vertex_offsets: Int32[Tensor, "i+1 j+1 k_packed+1"],
    num_vertices: int,
) -> Voxels:
    device = occupancy.device

    # There are three quantities we care about:
    # - Voxels (used to create triangle vertices and triangle faces)
    # - Subvoxels (only used to create triangle vertices)
    # - Vertices (used to sample the field)
    # The number of vertices is equal to the number of voxels plus the number of
    # subvoxels. Hence, the number of lower corners is the number of vertices.
    vertices = torch.empty((num_vertices, 3), dtype=torch.float32, device=device)
    neighbors = torch.empty((7, num_voxels), dtype=torch.int32, device=device)
    lower_corners = torch.empty((4, num_vertices), dtype=torch.int32, device=device)
    upper_corners = torch.empty((4, num_voxels), dtype=torch.int32, device=device)
    indices = torch.empty((num_voxels,), dtype=torch.int32, device=device)

    if num_voxels > 0:
        slang().create_voxels(
            voxelOccupancy=occupancy,
            voxelOffsets=voxel_offsets,
            vertexOccupancy=vertex_occupancy,
            vertexOffsets=vertex_offsets,
            vertices=vertices,
            neighbors=neighbors,
            lowerCorners=lower_corners,
            upperCorners=upper_corners,
            indices=indices,
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=tuple(
                ceildiv(dim, block)
                for dim, block in zip(vertex_occupancy.shape, BLOCK_SIZE)
            ),
        )

    i, j, k_packed = occupancy.shape
    return Voxels(
        vertices,
        neighbors,
        lower_corners,
        upper_corners,
        indices,
        (i, j, k_packed * 32),
    )
