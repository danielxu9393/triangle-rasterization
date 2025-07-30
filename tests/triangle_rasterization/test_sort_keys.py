import pytest
import torch

from triangle_rasterization import TileGrid
from triangle_rasterization.pytorch.bound_triangles import bound_triangles
from triangle_rasterization.pytorch.generate_keys import generate_keys
from triangle_rasterization.pytorch.project_vertices import project_vertices
from triangle_rasterization.pytorch.sort_keys import sort_keys as sort_keys_torch
from triangle_rasterization.slang.sort_keys import sort_keys as sort_keys_slang
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
    triangle_bounds = bound_triangles(
        projection.positions, projection.depths, scene.faces, grid
    )
    paired_keys = generate_keys(
        projection.depths,
        scene.faces,
        triangle_bounds.tile_minima,
        triangle_bounds.tile_maxima,
        triangle_bounds.num_tiles_overlapped,
        grid,
    )

    paired_keys_torch = sort_keys_torch(paired_keys, grid)
    paired_keys_slang = sort_keys_slang(paired_keys, grid)

    # Technically, we may want to worry about stable vs. non-stable sorts (since two
    # triangles in the same tile could have the same depth), but until a test fails
    # because of this, we'll ignore it.
    assert (paired_keys_torch.keys == paired_keys_slang.keys).all()
    assert (
        paired_keys_torch.triangle_indices == paired_keys_slang.triangle_indices
    ).all()
