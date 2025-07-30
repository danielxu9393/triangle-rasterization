from itertools import permutations

import pytest
import torch

from triangle_rasterization import TileGrid
from triangle_rasterization.interface.delineate_tiles import PairedKeys
from triangle_rasterization.pytorch.bound_triangles import bound_triangles
from triangle_rasterization.pytorch.delineate_tiles import (
    delineate_tiles as delineate_tiles_torch,
)
from triangle_rasterization.pytorch.generate_keys import generate_keys
from triangle_rasterization.pytorch.project_vertices import project_vertices
from triangle_rasterization.pytorch.sort_keys import sort_keys

from triangle_rasterization.slang.delineate_tiles import (
    delineate_tiles as delineate_tiles_slang,
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
    paired_keys = generate_keys(
        projection.depths,
        scene.faces,
        triangle_bounds.tile_minima,
        triangle_bounds.tile_maxima,
        triangle_bounds.num_tiles_overlapped,
        grid,
    )
    paired_keys = sort_keys(paired_keys, grid)

    tile_boundaries_slang = delineate_tiles_slang(paired_keys, grid)
    tile_boundaries_torch = delineate_tiles_torch(paired_keys, grid)

    assert (tile_boundaries_slang == tile_boundaries_torch).all()


@pytest.mark.parametrize("backend", (delineate_tiles_torch, delineate_tiles_slang))
def test_isolated(device, backend):
    grid = TileGrid((16, 16), (48, 16), (3, 1), (0, 0), (3, 1))

    def make_key(tile: int, depth: float):
        depth = torch.tensor(depth, dtype=torch.float32, device=device)
        tile = torch.tensor(tile, dtype=torch.int64, device=device)
        return (tile << 32) + depth.view(torch.uint32).type(torch.int64)

    bundle = (
        ((1.5, 2.5, 3.5), (0, 3)),
        ((-1.5, 2.5, 3.5), (1, 3)),
        ((-3.5, -2.5, -1.5), (3, 3)),
    )
    for permutation in permutations(bundle):
        # Assemble the correct keys and boundaries.
        keys = []
        starts = []
        ends = []
        offset = 0
        for tile_index, (tile_depths, (start, end)) in enumerate(permutation):
            for depth in tile_depths:
                keys.append(make_key(tile_index, depth))
            starts.append(offset + start)
            ends.append(offset + end)
            offset += len(tile_depths)
        keys = torch.stack(keys)
        starts = torch.tensor(starts, device=device)
        ends = torch.tensor(ends, device=device)
        paired_keys = PairedKeys(
            keys, torch.zeros_like(keys).type(torch.int32), num_keys=keys.shape[0]
        )
        boundaries_gt = torch.stack((starts, ends), dim=-1)

        boundaries = backend(paired_keys, grid)

        assert (boundaries == boundaries_gt).all()
