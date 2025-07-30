from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float, Int32
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.count_triangle_faces import CountTriangleFacesResult
from ..misc import ceildiv

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "count_triangle_faces.slang"),
        verbose=True,
    )
)


@record_function("count_triangle_faces")
def count_triangle_faces(
    signed_distances: Float[Tensor, " sample"],
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    upper_corners: Int32[Tensor, "corner=4 voxel"],
    level_sets: Float[Tensor, " level_set"],
) -> CountTriangleFacesResult:
    device = lower_corners.device
    _, num_voxels = upper_corners.shape
    face_counts = torch.empty((num_voxels + 1,), dtype=torch.int32, device=device)
    face_counts[-1] = 0

    num_level_set = level_sets.shape[0]
    voxel_cell_codes = torch.empty(
        (num_level_set, num_voxels),
        dtype=torch.uint8,
        device=device,
    )

    slang().countTriangleFaces(
        signedDistances=signed_distances,
        lowerCorners=lower_corners,
        upperCorners=upper_corners,
        levelSets=level_sets,
        faceCounts=face_counts,
        voxelCellCodes=voxel_cell_codes,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
    )

    return CountTriangleFacesResult(face_counts, voxel_cell_codes)
