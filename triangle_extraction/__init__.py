import logging
from typing import NamedTuple

import torch
from jaxtyping import Float32, Int32
from torch import Tensor
from torch.profiler import record_function

from .interface.create_voxels import Voxels
from .misc import pack_occupancy as pack_occupancy
from .misc import unpack_occupancy as unpack_occupancy
from .pytorch.index_vertices import index_vertices as index_vertices
from .registry import (
    COMPUTE_EXCLUSIVE_CUMSUM,
    COMPUTE_VERTEX_OCCUPANCY,
    COUNT_OCCUPANCY,
    COUNT_TRIANGLE_FACES,
    COUNT_TRIANGLE_VERTICES,
    CREATE_TRIANGLE_FACES,
    CREATE_TRIANGLE_VERTICES,
    CREATE_VOXELS,
    Backend,
)

# These imports are forwarded for convenience.
from .slang.compute_sdf_regularizers import (
    compute_sdf_regularizers as compute_sdf_regularizers,
)
from .slang.dilate_occupancy import dilate_occupancy as dilate_occupancy
from .slang.get_occupancy_mesh import get_occupancy_mesh as get_occupancy_mesh
from .slang.upscale_occupancy import upscale_occupancy as upscale_occupancy
from .slang.upscale_parameters import UpscaledParameters as UpscaledParameters
from .slang.upscale_parameters import upscale_parameters as upscale_parameters
from .slang.write_occupancy import write_occupancy as write_occupancy


@record_function("extract_voxels")
def extract_voxels(
    occupancy: Int32[Tensor, "i j k_packed"],
    backend: Backend,
) -> Voxels:
    i, j, k_packed = occupancy.shape
    logging.debug(f"Extracting voxels for grid with shape ({i}, {j}, {k_packed * 32}).")

    # Compute vertex occupancy. A voxel is considered vertex-occupied if it or any
    # of its 7 lower neighbors (one index lower in each direction) are occupied.
    # Also compute the number of vertices owned by each voxel. Each vertex-occupied
    # voxel owns one vertex.
    vertex_occupancy, vertex_offsets = COMPUTE_VERTEX_OCCUPANCY[backend](occupancy)

    # Compute vertex offsets.
    COMPUTE_EXCLUSIVE_CUMSUM[backend](vertex_offsets)
    num_vertices = vertex_offsets[-1].item()
    vertex_offsets = vertex_offsets[:-1].view(vertex_occupancy.shape)

    # Compute voxel offsets.
    voxel_offsets = COUNT_OCCUPANCY[backend](occupancy)
    COMPUTE_EXCLUSIVE_CUMSUM[backend](voxel_offsets)
    num_voxels = voxel_offsets[-1].item()
    voxel_offsets = voxel_offsets[:-1].view(vertex_occupancy.shape)

    # Actually create the samples.
    logging.debug(f"Sampled {num_vertices} vertices and {num_voxels} voxels.")
    return CREATE_VOXELS[backend](
        occupancy,
        voxel_offsets,
        num_voxels,
        vertex_occupancy,
        vertex_offsets,
        num_vertices,
    )


class Triangles(NamedTuple):
    vertices: Float32[Tensor, "vertex xyz=3"]
    signed_distances: Float32[Tensor, " vertex"]
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"]
    faces: Int32[Tensor, "face corner=3"]
    shell_indices: Int32[Tensor, " face"]
    voxel_indices: Int32[Tensor, " face"]


@record_function("tessellate_voxels")
def tessellate_voxels(
    signed_distances: Float32[Tensor, " sample"],
    vertices: Float32[Tensor, "sample xyz=3"],
    spherical_harmonics: Float32[Tensor, "sh sample rgb=3"],
    neighbors: Int32[Tensor, "neighbor=7 voxel"],
    lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    upper_corners: Int32[Tensor, "corner=4 voxel"],
    indices: Int32[Tensor, " voxel"],
    level_sets: Float32[Tensor, " level_set"],
    backend: Backend,
) -> Triangles:
    logging.debug(f"Tessellating {neighbors.shape[1]} voxels.")

    # Non-differentiable operations:
    # Count the number of vertices created by each voxel.
    with torch.no_grad():
        vertex_counts, vertex_counts_by_level_set = COUNT_TRIANGLE_VERTICES[backend](
            signed_distances,
            lower_corners,
            level_sets,
        )
        COMPUTE_EXCLUSIVE_CUMSUM[backend](vertex_counts)
        num_vertices = vertex_counts[-1].item()

    # Differentiable operation:
    # Use the custom differentiable function to compute triangle vertices.
    # triangle_vertices, triangle_vertex_types = CreateTriangleVertices.apply(
    (
        triangle_vertices,
        triangle_signed_distances,
        triangle_spherical_harmonics,
        triangle_vertex_types,
    ) = CREATE_TRIANGLE_VERTICES[backend](
        vertices,
        signed_distances,
        spherical_harmonics,
        lower_corners,
        vertex_counts,
        level_sets,
    )

    # Continue with non-differentiable operations:
    # Count the number of faces created by each voxel.
    with torch.no_grad():
        face_counts, voxel_cell_codes = COUNT_TRIANGLE_FACES[backend](
            signed_distances,
            lower_corners,
            upper_corners,
            level_sets,
        )

        # Convert the face counts to face offsets in place.
        COMPUTE_EXCLUSIVE_CUMSUM[backend](face_counts)
        num_faces = face_counts[-1].item() // 3

        faces, shell_indices, voxel_indices = CREATE_TRIANGLE_FACES[backend](
            signed_distances,
            neighbors,
            indices,
            vertex_counts,
            vertex_counts_by_level_set,
            triangle_vertex_types,
            face_counts,
            voxel_cell_codes,
            level_sets,
        )

    logging.debug(f"Created {num_vertices} vertices and {num_faces} faces.")
    return Triangles(
        triangle_vertices,
        triangle_signed_distances,
        triangle_spherical_harmonics,
        faces,
        shell_indices,
        voxel_indices,
    )
