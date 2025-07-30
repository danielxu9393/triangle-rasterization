import pytest
import torch

from triangle_rasterization import TileGrid
from triangle_rasterization.pytorch.bound_triangles import bound_triangles
from triangle_rasterization.pytorch.generate_keys import (
    generate_keys as generate_keys_torch,
)
from triangle_rasterization.pytorch.project_vertices import project_vertices
from triangle_rasterization.slang.generate_keys import (
    generate_keys as generate_keys_slang,
)
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

    keys_torch = generate_keys_torch(
        projection.depths,
        scene.faces,
        triangle_bounds.tile_minima,
        triangle_bounds.tile_maxima,
        triangle_bounds.num_tiles_overlapped,
        grid,
    )
    keys_slang = generate_keys_slang(
        projection.depths,
        scene.faces,
        triangle_bounds.tile_minima,
        triangle_bounds.tile_maxima,
        triangle_bounds.num_tiles_overlapped,
        grid,
    )

    # The ordering of the keys doesn't matter, so we sort them before comparing them.
    keys_torch = torch.sort(keys_torch.keys)[0]
    keys_slang = torch.sort(keys_slang.keys)[0]

    tile_indices_torch = keys_torch >> 32
    tile_indices_slang = keys_slang >> 32
    assert (tile_indices_torch == tile_indices_slang).all()

    depths_torch = (keys_torch & ((1 << 32) - 1)).type(torch.int32).view(torch.float32)
    depths_slang = (keys_slang & ((1 << 32) - 1)).type(torch.int32).view(torch.float32)

    assert torch.allclose(depths_torch, depths_slang)
