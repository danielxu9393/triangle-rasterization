from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.create_triangle_faces import CreateTriangleFacesResult
from ..misc import ceildiv

BLOCK_SIZE = 256


slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "create_triangle_faces.slang"),
        verbose=True,
    )
)


@record_function("create_triangle_faces")
def create_triangle_faces(
    signed_distances: Float[Tensor, " sample"],
    neighbors: Int[Tensor, "neighbor=7 voxel"],
    indices: Int[Tensor, " voxel"],
    vertex_counts: Int[Tensor, " voxel_and_subvoxel_plus_one"],
    vertex_counts_by_level_set: UInt8[
        Tensor, "level_set voxel_and_subvoxel_plus_one-1"
    ],
    triangle_vertex_types: UInt8[Tensor, " triangle_vertex"],
    face_counts: Int[Tensor, " voxel+1"],
    voxel_cell_codes: UInt8[Tensor, "level_set voxel"],
    level_sets: Float[Tensor, " level_set"],
) -> CreateTriangleFacesResult:
    device = signed_distances.device
    _, num_voxels = neighbors.shape
    num_faces = face_counts[-1].item() // 3

    triangle_faces = torch.empty((num_faces, 3), dtype=torch.int32, device=device)
    shell_indices = torch.empty((num_faces,), dtype=torch.int32, device=device)
    voxel_indices = torch.empty((num_faces,), dtype=torch.int32, device=device)

    slang().createTriangleFaces(
        signedDistances=signed_distances,
        neighbors=neighbors,
        indices=indices,
        vertexOffsets=vertex_counts,
        vertexCountsByLevelSet=vertex_counts_by_level_set,
        triangleVertexTypes=triangle_vertex_types,
        faceCounts=face_counts,
        voxelCellCodes=voxel_cell_codes,
        levelSets=level_sets,
        triangleFaces=triangle_faces,
        shellIndices=shell_indices,
        voxelIndices=voxel_indices,
    ).launchRaw(
        blockSize=(BLOCK_SIZE, 1, 1),
        gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
    )

    return CreateTriangleFacesResult(triangle_faces, shell_indices, voxel_indices)
