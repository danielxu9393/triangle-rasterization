from typing import Literal

from .interface.compute_exclusive_cumsum import ComputeExclusiveCumsumFn
from .interface.compute_vertex_occupancy import ComputeVertexOccupancyFn
from .interface.count_occupancy import CountOccupancyFn
from .interface.count_triangle_faces import CountTriangleFacesFn
from .interface.count_triangle_vertices import CountTriangleVerticesFn
from .interface.create_triangle_faces import CreateTriangleFacesFn
from .interface.create_triangle_vertices import (
    CreateTriangleVerticesFn,
)
from .interface.create_voxels import CreateVoxelsFn
from .pytorch.compute_exclusive_cumsum import (
    compute_exclusive_cumsum as compute_exclusive_cumsum_torch,
)
from .slang.compute_exclusive_cumsum import (
    compute_exclusive_cumsum as compute_exclusive_cumsum_slang,
)
from .slang.compute_vertex_occupancy import (
    compute_vertex_occupancy as compute_vertex_occupancy_slang,
)
from .slang.count_occupancy import count_occupancy as count_occupancy_slang
from .slang.count_triangle_faces import (
    count_triangle_faces as count_triangle_faces_slang,
)
from .slang.count_triangle_vertices import (
    count_triangle_vertices as count_triangle_vertices_slang,
)
from .slang.create_triangle_faces import (
    create_triangle_faces as create_triangle_faces_slang,
)
from .slang.create_triangle_vertices import (
    create_triangle_vertices as create_triangle_vertices_slang,
)
from .slang.create_voxels import create_voxels as create_voxels_slang

Backend = Literal["torch", "slang"]

COMPUTE_VERTEX_OCCUPANCY: dict[Backend, ComputeVertexOccupancyFn] = {
    "slang": compute_vertex_occupancy_slang,
}

COMPUTE_EXCLUSIVE_CUMSUM: dict[Backend, ComputeExclusiveCumsumFn] = {
    "slang": compute_exclusive_cumsum_slang,
    "torch": compute_exclusive_cumsum_torch,
}

COUNT_OCCUPANCY: dict[Backend, CountOccupancyFn] = {
    "slang": count_occupancy_slang,
}

CREATE_VOXELS: dict[Backend, CreateVoxelsFn] = {
    "slang": create_voxels_slang,
}

COUNT_TRIANGLE_VERTICES: dict[Backend, CountTriangleVerticesFn] = {
    "slang": count_triangle_vertices_slang,
}

CREATE_TRIANGLE_VERTICES: dict[Backend, CreateTriangleVerticesFn] = {
    "slang": create_triangle_vertices_slang,
}

COUNT_TRIANGLE_FACES: dict[Backend, CountTriangleFacesFn] = {
    "slang": count_triangle_faces_slang,
}

CREATE_TRIANGLE_FACES: dict[Backend, CreateTriangleFacesFn] = {
    "slang": create_triangle_faces_slang,
}
