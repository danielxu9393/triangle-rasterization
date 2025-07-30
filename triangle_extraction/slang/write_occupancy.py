from pathlib import Path

import slangtorch
import torch
from jaxtyping import Int32
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..misc import ceildiv

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "write_occupancy.slang"),
        verbose=True,
    )
)


@record_function("write_occupancy")
def write_occupancy(
    shell_indices: Int32[Tensor, " triangle"],
    voxel_indices: Int32[Tensor, " triangle"],
    packed_occupancy_shape: tuple[int, ...],
    min_shell: int,
    max_shell: int,
) -> Int32[Tensor, "*rest packed"]:
    (num_triangles,) = shell_indices.shape
    occupancy = torch.zeros(
        packed_occupancy_shape,
        dtype=torch.int32,
        device=shell_indices.device,
    )

    slang().write_occupancy(
        occupancy=occupancy.view(-1),
        shellIndices=shell_indices,
        voxelIndices=voxel_indices,
        minShell=min_shell,
        maxShell=max_shell,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_triangles, BLOCK_SIZE), 1, 1),
    )

    return occupancy
