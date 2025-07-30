import pytest
import torch
from diso import DiffMC
from jaxtyping import Bool, Float, install_import_hook
from torch import Tensor
from einops import rearrange

from tests.triangle_extraction.misc import (
    Index3DGrid,
    align_meshes,
    enforce_level_set_margin,
    filter_mesh,
    prune_unused_vertices,
    sample_grid,
    sort_mesh_vertices,
)

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction import Backend, extract_voxels, tessellate_voxels
    from triangle_extraction.misc import pack_occupancy

VALID_BACKENDS = ("slang",)


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_test(
    shape: tuple[int, int, int],
    device: torch.device,
    backend: Backend,
    sdf_grid: Float[Tensor, "i j k"],
    occupancy: Bool[Tensor, "i j k"],
    level_sets: Float[Tensor, " level_set"],
    assert_shapes_only: bool = False,
) -> None:
    """
    Small limitation on these tests: When we have a value in sdf_grid that is exactly 0,
    then we get a degenerate face (all 3 vertices are the same). My filtering code can't
    handle this (because it's underdetermined which voxel that face belongs to). So
    instead, I have another function which combines vertices with same coordinate, and
    filters out degenerate faces. This is not quite ideal, because this also has to
    apply to our code.

    Actually: Based on the voxel cell code, you could probably handle this case, but it
    is quite annoying to implement.

    But basically these test cases don't handle the degenerate case where you have a
    triangle with 0 area.

    Another limitation: DiffMC code seems to miss a vertex at the intersection of SDF
    and the grid edge (i.e. coordinate 32.0, 0.0, 25.4).
    """

    packed_occupancy = pack_occupancy(occupancy)

    vertices, neighbors, lower_corners, upper_corners, indices, _ = extract_voxels(
        packed_occupancy,
        backend,
    )

    ### Simulate sampling from the SDF field to match vertices.
    vertices_int = vertices * torch.tensor(shape, device=device)
    vertices_int = vertices_int.round().long()

    features_grid = torch.randn(
        (shape[2] + 1, shape[1] + 1, shape[0] + 1, 48),
        device=device,
        dtype=torch.float32,
    )
    features_grid.requires_grad = True

    ### Remember sdf_grid is zyx indexing!
    signed_distances = Index3DGrid.apply(sdf_grid, vertices_int)
    features = Index3DGrid.apply(features_grid, vertices_int)

    triangles = tessellate_voxels(
        signed_distances,
        vertices,
        rearrange(features, "s (sh rgb) -> sh s rgb", rgb=3).contiguous(),
        neighbors,
        lower_corners,
        upper_corners,
        indices,
        level_sets,
        backend,
    )

    triangle_features_interpolate = sample_grid(
        features_grid,
        triangles.vertices,
    )
    triangle_sh_interpolate = rearrange(
        triangle_features_interpolate, "v (sh rgb) -> sh v rgb", rgb=3
    )

    assert torch.allclose(
        triangles.spherical_harmonics,
        triangle_sh_interpolate,
        atol=1e-2,
    )

    triangle_vertices = triangles.vertices * torch.tensor(shape, device=device)

    triangle_vertices, triangle_faces, tri_perm_1 = sort_mesh_vertices(
        triangle_vertices, triangles.faces
    )
    triangle_vertices, triangle_faces, tri_unique_indices = prune_unused_vertices(
        triangle_vertices, triangle_faces
    )
    triangle_vertices, triangle_faces, tri_perm_2 = sort_mesh_vertices(
        triangle_vertices, triangle_faces
    )

    ### diff_mc takes in sdf_grid as xyz indexing!!
    ### So just be aware of this discrepancy!!
    ### In our model code, we were flipping the vertices output from diff_mc
    ### But instead, we should be flipping sdf_grid when input to diff_mc right?
    diff_mc = DiffMC()
    sdf_grid_xyz = sdf_grid.permute(2, 1, 0)

    expected_vertices = []
    expected_faces = []
    vertex_counter = 0

    for i in range(level_sets.shape[0]):
        expected_vertices_i, expected_faces_i = diff_mc(
            sdf_grid_xyz - level_sets[i],
            normalize=True,
        )

        expected_faces_i += vertex_counter
        vertex_counter += expected_vertices_i.shape[0]

        expected_vertices.append(expected_vertices_i)
        expected_faces.append(expected_faces_i)

    expected_vertices = torch.cat(expected_vertices, dim=0)
    expected_faces = torch.cat(expected_faces, dim=0)

    expected_vertices = expected_vertices * torch.tensor(shape, device=device)

    expected_vertices, expected_faces, _ = sort_mesh_vertices(
        expected_vertices, expected_faces
    )
    expected_filtered_vertices, expected_filtered_faces, _ = filter_mesh(
        expected_vertices, expected_faces, occupancy
    )
    expected_filtered_vertices, expected_filtered_faces, _ = sort_mesh_vertices(
        expected_filtered_vertices, expected_filtered_faces
    )
    (
        triangle_vertices,
        triangle_faces,
        _,
        expected_filtered_vertices,
        expected_filtered_faces,
        _,
    ) = align_meshes(
        triangle_vertices,
        triangle_faces,
        expected_filtered_vertices,
        expected_filtered_faces,
    )

    if assert_shapes_only:
        tol = 1e-4
        assert triangle_vertices.shape == expected_filtered_vertices.shape
        assert triangle_faces.shape == expected_filtered_faces.shape

        diff_vertices = torch.abs(triangle_vertices - expected_filtered_vertices)
        close_mask_vertices = diff_vertices <= tol
        fraction_close_vertices = close_mask_vertices.float().mean().item()
        assert fraction_close_vertices > 0.99

        diff_faces = torch.abs(triangle_faces - expected_filtered_faces)
        close_mask_faces = diff_faces <= tol
        fraction_close_faces = close_mask_faces.float().mean().item()
        assert fraction_close_faces > 0.99

    else:
        assert torch.allclose(triangle_vertices, expected_filtered_vertices, atol=1e-4)
        assert torch.allclose(triangle_faces, expected_filtered_faces, atol=1e-4)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_dense_sphere(device, backend):
    shape = (32, 64, 96)
    level_sets = torch.tensor(
        [-1.2, -1.0, -0.5, 0.0, 0.33, 0.7, 0.9],
        dtype=torch.float32,
        device=device,
    )

    ### occupancy is zyx indexing!!
    occupancy = torch.ones(tuple(reversed(shape)), dtype=torch.bool, device=device)

    ### Keep sdf_grid initialized with xyz indexing
    xyz = torch.meshgrid(
        (
            torch.arange(shape[0] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[1] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[2] + 1, device=device, dtype=torch.float32),
        ),
        indexing="ij",
    )
    xyz = torch.stack(xyz, dim=-1)
    origin = torch.tensor([10, 11, 12], dtype=torch.float32, device=device)
    sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 4.5

    ### Flip sdf_grid to zyx indexing
    sdf_grid = sdf_grid.permute(2, 1, 0)
    run_test(shape, device, backend, sdf_grid, occupancy, level_sets)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_dense_sphere_cutoff_edge(device, backend):
    shape = (32, 64, 96)
    level_sets = torch.tensor(
        [-1.2, -1.0, -0.4, 0.0, 0.33, 0.7, 0.9, 5.4],
        # [22],
        dtype=torch.float32,
        device=device,
    )

    ### occupancy is zyx indexing!!
    occupancy = torch.ones(tuple(reversed(shape)), dtype=torch.bool, device=device)

    ### Keep sdf_grid initialized with xyz indexing
    xyz = torch.meshgrid(
        (
            torch.arange(shape[0] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[1] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[2] + 1, device=device, dtype=torch.float32),
        ),
        indexing="ij",
    )
    xyz = torch.stack(xyz, dim=-1)
    origin = torch.tensor([5, 6, 7], dtype=torch.float32, device=device)
    sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 11

    ### Flip sdf_grid to zyx indexing
    sdf_grid = sdf_grid.permute(2, 1, 0)

    run_test(shape, device, backend, sdf_grid, occupancy, level_sets)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_sphere_cutoff_occupancy(device, backend):
    shape = (32, 64, 96)
    level_sets = torch.tensor(
        [-1.21, -1.0, -0.4, 0.0, 0.33, 0.7, 0.9, 5.4, 6.3],
        # [-1.2],
        dtype=torch.float32,
        device=device,
    )

    ### occupancy is zyx indexing!!
    occupancy = torch.ones(tuple(reversed(shape)), dtype=torch.bool, device=device)

    occupancy[7:, 6:, 5:] = False

    ### Keep sdf_grid initialized with xyz indexing
    xyz = torch.meshgrid(
        (
            torch.arange(shape[0] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[1] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[2] + 1, device=device, dtype=torch.float32),
        ),
        indexing="ij",
    )
    xyz = torch.stack(xyz, dim=-1)
    origin = torch.tensor([5, 6, 7], dtype=torch.float32, device=device)
    sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 14.2

    ### Flip sdf_grid to zyx indexing
    sdf_grid = sdf_grid.permute(2, 1, 0)

    run_test(shape, device, backend, sdf_grid, occupancy, level_sets)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_multiple_occupancy_cutoff(device, backend):
    shape = (32, 64, 96)
    level_sets = torch.tensor(
        [-1.2, -1.0, -0.5, 0.0, 0.33, 0.7, 0.9],
        # [0.0],
        dtype=torch.float32,
        device=device,
    )

    ### occupancy is zyx indexing!!
    occupancy = torch.ones(tuple(reversed(shape)), dtype=torch.bool, device=device)
    occupancy[10:, 11:, 12:] = False
    occupancy[:10, 11:, :12] = False
    occupancy[10:, :11, :12] = False
    occupancy[:10, :11, 12:] = False

    ### Keep sdf_grid initialized with xyz indexing
    xyz = torch.meshgrid(
        (
            torch.arange(shape[0] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[1] + 1, device=device, dtype=torch.float32),
            torch.arange(shape[2] + 1, device=device, dtype=torch.float32),
        ),
        indexing="ij",
    )
    xyz = torch.stack(xyz, dim=-1)
    origin = torch.tensor([10, 11, 12], dtype=torch.float32, device=device)
    sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 4.5

    ### Flip sdf_grid to zyx indexing
    sdf_grid = sdf_grid.permute(2, 1, 0)
    run_test(shape, device, backend, sdf_grid, occupancy, level_sets)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_rand_sdf(device, backend):
    for _ in range(20):
        shape = (32, 32, 32)
        level_sets = torch.tensor(
            [-2.5, 0.0, 1.33],
            dtype=torch.float32,
            device=device,
        )

        ### occupancy is zyx indexing!!
        occupancy = torch.ones(tuple(reversed(shape)), dtype=torch.bool, device=device)

        shape_plus_1 = (shape[0] + 1, shape[1] + 1, shape[2] + 1)
        sdf_grid = torch.rand(shape_plus_1, device=device) * 6.0 - 3.0
        sdf_grid = enforce_level_set_margin(sdf_grid, level_sets)

        ### Set boundaries to avoid bounadry issues
        sdf_grid[0, :, :] = 3
        sdf_grid[-1, :, :] = 3
        sdf_grid[:, 0, :] = 3
        sdf_grid[:, -1, :] = 3
        sdf_grid[:, :, 0] = 3
        sdf_grid[:, :, -1] = 3

        ### Flip sdf_grid to zyx indexing
        sdf_grid = sdf_grid.permute(2, 1, 0)
        run_test(
            shape,
            device,
            backend,
            sdf_grid,
            occupancy,
            level_sets,
        )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_rand_occupancy(device, backend):
    for i in range(100):
        shape = (32, 64, 96)
        level_sets = torch.tensor(
            [-1.2, -1.0, -0.45, 0.0, 0.33, 0.7, 0.9],
            # [0.0],
            dtype=torch.float32,
            device=device,
        )

        ### occupancy is zyx indexing!!
        occupancy = torch.rand(tuple(reversed(shape)), device=device) < 0.5

        ### Keep sdf_grid initialized with xyz indexing
        xyz = torch.meshgrid(
            (
                torch.arange(shape[0] + 1, device=device, dtype=torch.float32),
                torch.arange(shape[1] + 1, device=device, dtype=torch.float32),
                torch.arange(shape[2] + 1, device=device, dtype=torch.float32),
            ),
            indexing="ij",
        )
        xyz = torch.stack(xyz, dim=-1)
        origin = torch.tensor([10, 11, 12], dtype=torch.float32, device=device)
        sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 5.5

        ### Flip sdf_grid to zyx indexing
        sdf_grid = sdf_grid.permute(2, 1, 0)
        run_test(shape, device, backend, sdf_grid, occupancy, level_sets)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_rand_occupancy_sdf(device, backend):
    for i in range(100):
        shape = (32, 32, 32)
        level_sets = torch.tensor(
            [-1.2, -1.0, -0.5, 0.0, 0.33, 0.7, 0.9],
            dtype=torch.float32,
            device=device,
        )

        ### occupancy is zyx indexing!!
        occupancy = torch.rand(tuple(reversed(shape)), device=device) < 0.5

        shape_plus_1 = (shape[0] + 1, shape[1] + 1, shape[2] + 1)
        sdf_grid = torch.rand(shape_plus_1, device=device) * 6.0 - 3.0
        sdf_grid = enforce_level_set_margin(sdf_grid, level_sets)

        ### Set boundaries to avoid bounadry issues
        sdf_grid[0, :, :] = 3
        sdf_grid[-1, :, :] = 3
        sdf_grid[:, 0, :] = 3
        sdf_grid[:, -1, :] = 3
        sdf_grid[:, :, 0] = 3
        sdf_grid[:, :, -1] = 3

        ### Flip sdf_grid to zyx indexing
        sdf_grid = sdf_grid.permute(2, 1, 0)
        run_test(
            shape,
            device,
            backend,
            sdf_grid,
            occupancy,
            level_sets,
        )


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_flip_sdf(device, backend):
    for i in range(1):
        shape = (32, 32, 32)
        level_sets = torch.tensor(
            [0.0],
            dtype=torch.float32,
            device=device,
        )

        ### occupancy is zyx indexing!!
        occupancy = torch.ones(tuple(reversed(shape)), device=device)

        ### Keep sdf_grid initialized with xyz indexing
        xyz = torch.meshgrid(
            (
                torch.arange(shape[0] + 1, device=device, dtype=torch.float32),
                torch.arange(shape[1] + 1, device=device, dtype=torch.float32),
                torch.arange(shape[2] + 1, device=device, dtype=torch.float32),
            ),
            indexing="ij",
        )
        xyz = torch.stack(xyz, dim=-1)
        origin = torch.tensor([10, 11, 12], dtype=torch.float32, device=device)
        sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 4.5

        ### Flip sdf_grid to zyx indexing
        sdf_grid = sdf_grid.permute(2, 1, 0)

        packed_occupancy = pack_occupancy(occupancy)

        vertices, neighbors, lower_corners, upper_corners, _, _ = extract_voxels(
            packed_occupancy,
            backend,
        )

        ### Simulate sampling from the SDF field to match vertices.
        vertices_int = vertices * torch.tensor(shape, device=device)
        vertices_int = vertices_int.round().long()

        features_grid = torch.randn(
            (shape[2] + 1, shape[1] + 1, shape[0] + 1, 16),
            device=device,
            dtype=torch.float32,
        )
        features_grid.requires_grad = True

        ### Remember sdf_grid is zyx indexing!
        signed_distances = Index3DGrid.apply(sdf_grid, vertices_int)
        features = Index3DGrid.apply(features_grid, vertices_int)

        triangles = tessellate_voxels(
            signed_distances,
            vertices,
            features,
            neighbors,
            lower_corners,
            upper_corners,
            level_sets,
            backend,
        )

        triangle_vertices = triangles.vertices * torch.tensor(shape, device=device)

        triangle_vertices, triangle_faces, tri_perm_1 = sort_mesh_vertices(
            triangle_vertices, triangles.faces
        )
        triangle_vertices, triangle_faces, tri_unique_indices = prune_unused_vertices(
            triangle_vertices, triangle_faces
        )
        triangle_vertices, triangle_faces, tri_perm_2 = sort_mesh_vertices(
            triangle_vertices, triangle_faces
        )

        sdf_grid_neg = -sdf_grid.clone().detach()
        vertices_neg = vertices.clone().detach()
        features_grid_neg = features_grid.clone().detach()
        signed_distances_neg = Index3DGrid.apply(sdf_grid_neg, vertices_int)
        features_neg = Index3DGrid.apply(features_grid_neg, vertices_int)

        triangles_neg = tessellate_voxels(
            signed_distances_neg,
            vertices_neg,
            features_neg,
            neighbors,
            lower_corners,
            upper_corners,
            level_sets.clone().detach(),
            backend,
        )

        triangle_vertices_neg = triangles_neg.vertices * torch.tensor(
            shape, device=device
        )
        triangle_vertices_neg, triangle_faces_neg, tri_perm_1_neg = sort_mesh_vertices(
            triangle_vertices_neg, triangles_neg.faces
        )
        (
            triangle_vertices_neg,
            triangle_faces_neg,
            tri_unique_indices_neg,
        ) = prune_unused_vertices(triangle_vertices_neg, triangle_faces_neg)
        triangle_vertices_neg, triangle_faces_neg, tri_perm_2_neg = sort_mesh_vertices(
            triangle_vertices_neg, triangle_faces_neg
        )

        ### Align Mesh
        (
            triangle_vertices,
            triangle_faces,
            _,
            triangle_vertices_neg,
            triangle_faces_neg,
            _,
        ) = align_meshes(
            triangle_vertices,
            triangle_faces,
            triangle_vertices_neg,
            triangle_faces_neg,
        )

        assert torch.allclose(triangle_vertices, triangle_vertices_neg, atol=1e-4)
        assert torch.allclose(triangle_faces, triangle_faces_neg, atol=1e-4)
