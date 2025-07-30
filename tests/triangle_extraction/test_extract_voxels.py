from math import prod
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from jaxtyping import Bool, Int32, install_import_hook
from torch import Tensor

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction import Backend, extract_voxels
    from triangle_extraction.misc import pack_occupancy

VALID_BACKENDS = ("slang",)


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_test(
    shape: tuple[int, int, int],
    device: torch.device,
    backend: Backend,
    set_occupancy: Callable[[Bool[Tensor, "i j k"]], None],
) -> None:
    # Set up the packed occupancy grid.
    occupancy = torch.zeros(shape, dtype=torch.bool, device=device)
    set_occupancy(occupancy)
    correct_indices = torch.arange(prod(shape), device=device, dtype=torch.int32)
    correct_indices = correct_indices.reshape(shape)[occupancy]
    packed_occupancy = pack_occupancy(occupancy)

    # Extract voxels.
    vertices, neighbors, lower_corners, upper_corners, indices, _ = extract_voxels(
        packed_occupancy, backend
    )

    # Make sure the indices (in the full cube) associated with the voxels are correct.
    assert (indices == correct_indices).all()

    # Assembly the full voxels' corner positions.
    _, num_voxels = neighbors.shape
    corners = [
        lower_corners[0, :num_voxels],  # (0, 0, 0) origin
        lower_corners[1, :num_voxels],  # (0, 0, 1) +x
        lower_corners[2, :num_voxels],  # (0, 1, 0) +y
        upper_corners[3],  # (0, 1, 1) +xy
        lower_corners[3, :num_voxels],  # (1, 0, 0) +z
        upper_corners[2],  # (1, 0, 1) +xz
        upper_corners[1],  # (1, 1, 0) +yz
        upper_corners[0],  # (1, 1, 1) +xyz
    ]
    corners = torch.stack(corners, dim=-1)
    corners = rearrange(corners, "v (i j k) -> v i j k", i=2, j=2, k=2)

    # Make sure the voxels' corner spacings are correct.
    xyz = vertices[corners]
    delta_x = xyz.diff(dim=-2)[:, :, :, 0, 0]
    delta_y = xyz.diff(dim=-3)[:, :, 0, :, 1]
    delta_z = xyz.diff(dim=-4)[:, 0, :, :, 2]
    i, j, k = shape
    assert torch.allclose(delta_x, torch.tensor(1 / k, device=device))
    assert torch.allclose(delta_y, torch.tensor(1 / j, device=device))
    assert torch.allclose(delta_z, torch.tensor(1 / i, device=device))

    # Make sure the voxels' centroids are correct.
    centroids = reduce(xyz, "v i j k xyz -> v xyz", "mean")
    x = (torch.arange(k, device=device, dtype=torch.int32) + 0.5) / k
    y = (torch.arange(j, device=device, dtype=torch.int32) + 0.5) / j
    z = (torch.arange(i, device=device, dtype=torch.int32) + 0.5) / i
    expected_centroids = torch.meshgrid((z, y, x), indexing="ij")
    expected_centroids = torch.stack(expected_centroids[::-1], dim=-1)
    assert torch.allclose(centroids, expected_centroids[occupancy])

    # Make sure the voxels' neighbors are correct. This also checks the subvoxels'
    # lowest corner. The neighbor order is (+x, +y, +z, +xy, +yz, +xz, +xyz)
    assert (lower_corners[:, neighbors[0]][0] == corners[:, 0, 0, 1]).all()
    assert (lower_corners[:, neighbors[1]][0] == corners[:, 0, 1, 0]).all()
    assert (lower_corners[:, neighbors[2]][0] == corners[:, 1, 0, 0]).all()
    assert (lower_corners[:, neighbors[3]][0] == corners[:, 0, 1, 1]).all()
    assert (lower_corners[:, neighbors[4]][0] == corners[:, 1, 1, 0]).all()
    assert (lower_corners[:, neighbors[5]][0] == corners[:, 1, 0, 1]).all()
    assert (lower_corners[:, neighbors[6]][0] == corners[:, 1, 1, 1]).all()

    # Make sure the subvoxels' corners are correct. The subvoxel's origin corners are
    # assumed to be correct if the above assertions pass.

    def vertex_index_to_location(
        indices: Int32[Tensor, "*batch"],
    ) -> Int32[Tensor, "*batch ijk=3"]:
        total = torch.tensor((k, j, i), device=device, dtype=torch.float32)
        return (vertices[indices] * total).round().int().flip(-1)

    # Subvoxel Check 1: Edge corners are not defined.
    subvoxel_corners = lower_corners[:, num_voxels:]
    subvoxel_ijk = vertex_index_to_location(lower_corners[0, num_voxels:])
    assert (subvoxel_corners[0, :] >= 0).all()
    assert (subvoxel_corners[1, :][subvoxel_ijk[:, 2] == k] < 0).all()
    assert (subvoxel_corners[2, :][subvoxel_ijk[:, 1] == j] < 0).all()
    assert (subvoxel_corners[3, :][subvoxel_ijk[:, 0] == i] < 0).all()

    # Subvoxel Check 2: If corners are defined, they are correct.
    x_valid = subvoxel_corners[1, :] >= 0
    subvoxel_corners_x = subvoxel_corners[1, x_valid]
    subvoxel_corners_x = vertex_index_to_location(subvoxel_corners_x)
    di = torch.tensor((0, 0, 1), dtype=torch.int32, device=device)
    assert (subvoxel_ijk[x_valid] + di == subvoxel_corners_x).all()

    y_valid = subvoxel_corners[2, :] >= 0
    subvoxel_corners_y = subvoxel_corners[2, y_valid]
    subvoxel_corners_y = vertex_index_to_location(subvoxel_corners_y)
    dj = torch.tensor((0, 1, 0), dtype=torch.int32, device=device)
    assert (subvoxel_ijk[y_valid] + dj == subvoxel_corners_y).all()

    z_valid = subvoxel_corners[3, :] >= 0
    subvoxel_corners_z = subvoxel_corners[3, z_valid]
    subvoxel_corners_z = vertex_index_to_location(subvoxel_corners_z)
    dk = torch.tensor((1, 0, 0), dtype=torch.int32, device=device)
    assert (subvoxel_ijk[z_valid] + dk == subvoxel_corners_z).all()

    # Subvoxel Check 3: If corners are not defined, they indeed shouldn't be.
    neighbor_occupancy = F.pad(occupancy, (1, 2, 1, 2, 1, 2), "constant", 0)
    neighbor_occupancy = (
        neighbor_occupancy[:-1, :-1, :-1]
        | neighbor_occupancy[:-1, :-1, 1:]
        | neighbor_occupancy[:-1, 1:, :-1]
        | neighbor_occupancy[:-1, 1:, 1:]
        | neighbor_occupancy[1:, :-1, :-1]
        | neighbor_occupancy[1:, :-1, 1:]
        | neighbor_occupancy[1:, 1:, :-1]
        | neighbor_occupancy[1:, 1:, 1:]
    )

    ii, jj, kk = subvoxel_ijk[~x_valid].unbind(dim=-1)
    assert (~neighbor_occupancy[ii, jj, kk + 1]).all()

    ii, jj, kk = subvoxel_ijk[~y_valid].unbind(dim=-1)
    assert (~neighbor_occupancy[ii, jj + 1, kk]).all()

    ii, jj, kk = subvoxel_ijk[~z_valid].unbind(dim=-1)
    assert (~neighbor_occupancy[ii + 1, jj, kk]).all()


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_middle(device, backend):
    def set_input(occupancy):
        occupancy[10, 15, 20] = True

    run_test((32, 64, 96), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_000(device, backend):
    def set_input(occupancy):
        occupancy[0, 0, 0] = True

    run_test((64, 64, 32), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_k_boundary_left(device, backend):
    def set_input(occupancy):
        occupancy[10, 15, 31] = True

    run_test((32, 64, 96), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_k_boundary_right(device, backend):
    def set_input(occupancy):
        occupancy[10, 15, 32] = True

    run_test((32, 64, 96), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_high_k(device, backend):
    def set_input(occupancy):
        occupancy[31, 32, 63] = True

    run_test((64, 64, 64), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_high_ij(device, backend):
    def set_input(occupancy):
        occupancy[63, 20, 20] = True
        occupancy[20, 63, 20] = True

    run_test((64, 64, 64), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_low_ij(device, backend):
    def set_input(occupancy):
        occupancy[0, 20, 20] = True
        occupancy[20, 0, 20] = True

    run_test((64, 64, 64), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_low_k(device, backend):
    def set_input(occupancy):
        occupancy[20, 20, 0] = True

    run_test((64, 64, 64), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_adjacent(device, backend):
    def set_input(occupancy):
        occupancy[20, 20, 20] = True
        occupancy[20, 20, 21] = True

        occupancy[25, 25, 25] = True
        occupancy[25, 26, 25] = True

        occupancy[30, 30, 30] = True
        occupancy[31, 30, 30] = True

    run_test((64, 64, 64), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
def test_random_10_percent(device, backend, seed):
    def set_input(occupancy):
        generator = torch.Generator(device)
        generator.manual_seed(seed)
        occupancy[:] = (
            torch.rand(occupancy.shape, device=device, generator=generator) < 0.1
        )

    run_test((32, 64, 96), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
def test_random_50_percent(device, backend, seed):
    def set_input(occupancy):
        generator = torch.Generator(device)
        generator.manual_seed(seed)
        occupancy[:] = (
            torch.rand(occupancy.shape, device=device, generator=generator) < 0.5
        )

    run_test((96, 32, 128), device, backend, set_input)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_full(device, backend):
    def set_input(occupancy):
        occupancy[:] = True

    run_test((64, 64, 64), device, backend, set_input)
