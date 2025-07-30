from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float, Int32
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.count_triangle_vertices import CountTriangleVerticesResult
from ..misc import ceildiv

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "count_triangle_vertices.slang"),
        verbose=True,
    )
)


@record_function("count_triangle_vertices")
def count_triangle_vertices(
    signed_distances: Float[Tensor, " sample"],
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    level_sets: Float[Tensor, " level_set"],
) -> CountTriangleVerticesResult:
    device = signed_distances.device
    _, num_voxels = lower_corners.shape
    vertex_counts = torch.empty((num_voxels + 1,), dtype=torch.int32, device=device)
    vertex_counts[-1] = 0

    vertex_counts_by_level_set = torch.empty(
        (level_sets.shape[0], num_voxels),
        dtype=torch.uint8,
        device=device,
    )

    slang().countTriangleVertices(
        signedDistances=signed_distances,
        lowerCorners=lower_corners,
        levelSets=level_sets,
        vertexCounts=vertex_counts,
        vertexCountsByLevelSet=vertex_counts_by_level_set,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
    )

    return CountTriangleVerticesResult(vertex_counts, vertex_counts_by_level_set)
