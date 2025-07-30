import pytest
import torch

from triangle_rasterization.pytorch.bound_triangles import (
    bound_triangles as bound_triangles_torch,
)
from triangle_rasterization.pytorch.project_vertices import project_vertices
from triangle_rasterization.slang.bound_triangles import (
    bound_triangles as bound_triangles_slang,
)
from triangle_rasterization.types import TileGrid
from visualization.triangle_rasterization.cameras import CAMERAS, Camera
from visualization.triangle_rasterization.scenes import SCENES, Scene


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("scene_fn", tuple(SCENES.values()))
@pytest.mark.parametrize("camera_fn", tuple(CAMERAS.values()))
def test_operation(device, scene_fn, camera_fn):
    camera: Camera = camera_fn(device)
    scene: Scene = scene_fn(device)
    grid = TileGrid.create(camera.image_shape, (16, 16))
    dummy_active_sh = torch.zeros((1, scene.vertices.shape[0], 3), device=device)
    projection = project_vertices(
        scene.vertices, dummy_active_sh, camera.extrinsics, camera.intrinsics
    )

    triangle_bounds_torch = bound_triangles_torch(
        projection.positions, projection.depths, scene.faces, grid
    )
    triangle_bounds_slang = bound_triangles_slang(
        projection.positions, projection.depths, scene.faces, grid, backface_culling=0
    )

    assert (
        triangle_bounds_torch.tile_minima == triangle_bounds_slang.tile_minima
    ).all()
    assert (
        triangle_bounds_torch.tile_maxima == triangle_bounds_slang.tile_maxima
    ).all()
    assert (
        triangle_bounds_torch.num_tiles_overlapped
        == triangle_bounds_slang.num_tiles_overlapped
    ).all()
