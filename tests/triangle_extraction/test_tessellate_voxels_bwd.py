import pytest
import torch
from diso import DiffMC
from einops import rearrange
from jaxtyping import Bool, Float, install_import_hook
from torch import Tensor

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
) -> None:
    """
    Small limitation on these tests:
    When we have a value in sdf_grid that is exactly 0, then we get a degenerate face
    (all 3 vertices are the same). My filtering code can't handle this (because it's
    underdetermined which voxel that face belongs to). So instead, I have another
    function which combines vertices with same coordinate, and filters out degenerate
    faces. This is not quite ideal, because this also has to apply to our code.

    Actually: Based on the voxel cell code, you could probably handle this case, but it
    is quite annoying to implement.

    But basically these test cases don't handle the degenerate case where you have a
    triangle with 0 area.

    Another limitation: DiffMC code seems to miss a vertex at the intersection of SDF
    and the grid edge (i.e. coordinate 32.0, 0.0, 25.4).
    """

    torch.autograd.set_detect_anomaly(True)

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
    # deform_grid = torch.randn(
    deform_grid = torch.zeros(
        (shape[2] + 1, shape[1] + 1, shape[0] + 1, 3),
        device=device,
    )
    deform_grid.requires_grad = True

    ### Remember sdf_grid is zyx indexing!
    signed_distances = Index3DGrid.apply(sdf_grid, vertices_int)
    features = Index3DGrid.apply(features_grid, vertices_int)
    vertices_deformed = Index3DGrid.apply(deform_grid, vertices_int)

    vertices.requires_grad = True
    vertices = vertices + vertices_deformed
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

    triangle_vertices = triangles.vertices * torch.tensor(
        shape, dtype=torch.float32, device=device
    )

    (
        triangle_vertices_sorted,
        triangle_faces_sorted,
        triangles_perm,
    ) = sort_mesh_vertices(triangle_vertices, triangles.faces, round=None)
    (
        triangle_vertices_sorted,
        triangle_faces_sorted,
        triangles_unique_indices,
    ) = prune_unused_vertices(triangle_vertices_sorted, triangle_faces_sorted)
    (
        triangle_vertices_sorted,
        triangle_faces_sorted,
        triangles_perm_2,
    ) = sort_mesh_vertices(triangle_vertices_sorted, triangle_faces_sorted)
    triangle_features_sorted = rearrange(
        triangles.spherical_harmonics, "sh v rgb -> v (sh rgb)"
    )[triangles_perm][triangles_unique_indices][triangles_perm_2]

    # vertices_grad = torch.randn_like(triangle_vertices_sorted)
    vertices_grad = torch.ones_like(triangle_vertices_sorted)
    triangle_vertices_grad = torch.zeros_like(triangle_vertices)
    triangle_vertices_grad[
        triangles_perm[triangles_unique_indices[triangles_perm_2]]
    ] = vertices_grad

    features_grad = torch.randn_like(triangle_features_sorted)
    # features_grad = torch.ones_like(triangle_features_sorted)
    triangle_features_grad = torch.zeros_like(triangles.spherical_harmonics)

    triangle_features_grad = rearrange(triangle_features_grad, "sh v rgb -> v (sh rgb)")
    triangle_features_grad[
        triangles_perm[triangles_unique_indices[triangles_perm_2]]
    ] = features_grad
    triangle_features_grad = rearrange(
        triangle_features_grad, "v (sh rgb) -> sh v rgb", rgb=3
    )

    (
        triangle_sdf_grad,
        triangle_features_grad_final,
        triangle_deform_grad,
    ) = torch.autograd.grad(
        # inputs=(sdf_grid, features_grid, vertices),
        inputs=(sdf_grid, features_grid, deform_grid),
        outputs=(triangle_vertices, triangles.spherical_harmonics),
        grad_outputs=(triangle_vertices_grad, triangle_features_grad),
        retain_graph=True,
    )
    # For some reason, this sets triangle_features_grad to all 0s, idk if this is sus

    ### diff_mc takes in sdf_grid as xyz indexing!!
    ### So just be aware of this discrepancy!!
    ### In our model code, we were flipping the vertices output from diff_mc
    ### But instead, we should be flipping sdf_grid when input to diff_mc right?
    diff_mc = DiffMC()
    sdf_grid_xyz = sdf_grid.clone().detach().permute(2, 1, 0)
    sdf_grid_xyz.requires_grad = True
    deform_grid_xyz = deform_grid.clone().detach().permute(2, 1, 0, 3)
    deform_grid_xyz = deform_grid_xyz / torch.tensor(shape, device=device)
    deform_grid_xyz.requires_grad = True

    expected_vertices = []
    expected_faces = []
    vertex_counter = 0

    for i in range(level_sets.shape[0]):
        expected_vertices_i, expected_faces_i = diff_mc(
            sdf_grid_xyz.detach() - level_sets[i],
            deform_grid_xyz,
            normalize=True,
        )

        expected_faces_i += vertex_counter
        vertex_counter += expected_vertices_i.shape[0]

        expected_vertices.append(expected_vertices_i)
        expected_faces.append(expected_faces_i)

    expected_vertices = torch.cat(expected_vertices, dim=0)
    expected_faces = torch.cat(expected_faces, dim=0)

    expected_vertices = expected_vertices * torch.tensor(shape, device=device)

    expected_vertices_sorted, expected_faces_sorted, expected_perm = sort_mesh_vertices(
        expected_vertices,
        expected_faces,
    )
    expected_vertices_sorted, expected_faces_sorted, unique_indices = filter_mesh(
        expected_vertices_sorted, expected_faces_sorted, occupancy
    )
    (
        expected_vertices_sorted,
        expected_faces_sorted,
        expected_perm_2,
    ) = sort_mesh_vertices(expected_vertices_sorted, expected_faces_sorted)
    (
        triangle_vertices_sorted,
        triangle_faces_sorted,
        _,
        expected_vertices_sorted,
        expected_faces_sorted,
        expected_perm_3,
    ) = align_meshes(
        triangle_vertices_sorted,
        triangle_faces_sorted,
        expected_vertices_sorted,
        expected_faces_sorted,
    )

    ### Sample features at the DIFFMC original vertices
    features_grid_expected = features_grid.clone().detach()
    features_grid_expected.requires_grad = True

    expected_vertices_unscaled = expected_vertices / torch.tensor(shape, device=device)
    # expected_vertices_unscaled = expected_vertices_unscaled.detach()
    ### Should I detach expected_vertices_unscaled? Doesn't seem to make a difference
    expected_features = sample_grid(
        features_grid_expected,
        expected_vertices_unscaled,
    )
    expected_features_sorted = expected_features[expected_perm][unique_indices][
        expected_perm_2
    ][expected_perm_3]

    ### Vertices need to be the same, we are not testing occupancy/faces here
    assert torch.allclose(triangle_vertices_sorted, expected_vertices_sorted, atol=1e-4)
    assert torch.allclose(triangle_faces_sorted, expected_faces_sorted, atol=1e-4)
    assert torch.allclose(triangle_features_sorted, expected_features_sorted, atol=1e-2)

    expected_vertices_grad = torch.zeros_like(expected_vertices)
    expected_vertices_grad[
        expected_perm[unique_indices[expected_perm_2[expected_perm_3]]]
    ] = vertices_grad

    expected_features_grad = torch.zeros_like(expected_features)
    expected_features_grad[
        expected_perm[unique_indices[expected_perm_2[expected_perm_3]]]
    ] = features_grad

    # expected_features_grad_final, expected_sdf_grad_xyz, expected_deform_grad_xyz = (
    expected_features_grad_final, expected_deform_grad_xyz = torch.autograd.grad(
        # inputs=(features_grid_expected, sdf_grid_xyz, deform_grid_xyz),
        inputs=(features_grid_expected, deform_grid_xyz),
        outputs=(expected_features, expected_vertices),
        grad_outputs=(expected_features_grad, expected_vertices_grad),
        retain_graph=True,
    )

    # expected_sdf_grad = expected_sdf_grad_xyz.permute(2, 1, 0)
    # expected_deform_grad = expected_deform_grad_xyz.permute(2, 1, 0, 3)

    # assert torch.allclose(triangle_sdf_grad, expected_sdf_grad, atol=1e-4)
    assert torch.allclose(
        triangle_features_grad_final, expected_features_grad_final, atol=1e-1
    )
    # assert torch.allclose(triangle_deform_grad, expected_deform_grad, atol=1e-4)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_dense_sphere(device, backend):
    for _ in range(1):
        shape = (32, 32, 32)
        level_sets = torch.tensor(
            # [-1.0, 0.0, 1.2],
            [0.0],
            dtype=torch.float32,
            device=device,
        )

        ### occupancy is zyx indexing!!
        # occupancy = torch.ones(
        #     tuple(reversed(shape)),
        #     dtype=torch.bool,
        #     device=device,
        # )
        occupancy = torch.zeros(tuple(reversed(shape)), dtype=torch.bool, device=device)
        occupancy[10:13, 9:12, 8:11] = True

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
        sdf_grid = torch.linalg.norm(xyz - origin, dim=-1) - 0.5
        sdf_grid.requires_grad = True

        ### Flip sdf_grid to zyx indexing
        sdf_grid = sdf_grid.permute(2, 1, 0)
        run_test(shape, device, backend, sdf_grid, occupancy, level_sets)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_rand_occupancy(device, backend):
    for _ in range(100):
        shape = (32, 64, 96)
        level_sets = torch.tensor(
            [-1.2, -1.0, -0.45, 0.0, 0.33, 0.7, 0.9],
            # Having vertex at all integer coordds creates problems
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
        sdf_grid.requires_grad = True

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
        sdf_grid.requires_grad = True

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
def test_rand_sdf(device, backend):
    for i in range(1):
        shape = (32, 32, 64)
        level_sets = torch.tensor(
            [-0.5, -0.3, 0.0, 0.7, 0.9],
            # [0.0],
            dtype=torch.float32,
            device=device,
        )

        ### occupancy is zyx indexing!!
        occupancy = torch.ones(tuple(reversed(shape)), dtype=torch.bool, device=device)

        ### Keep sdf_grid initialized with xyz indexing
        shape_plus_1 = (shape[0] + 1, shape[1] + 1, shape[2] + 1)
        sdf_grid = torch.rand(shape_plus_1, device=device) * 6.0 - 3.0
        sdf_grid = enforce_level_set_margin(sdf_grid, level_sets, tol=0.3)

        ### Set boundaries to avoid bounadry issues
        sdf_grid[0, :, :] = 3
        sdf_grid[-1, :, :] = 3
        sdf_grid[:, 0, :] = 3
        sdf_grid[:, -1, :] = 3
        sdf_grid[:, :, 0] = 3
        sdf_grid[:, :, -1] = 3
        sdf_grid.requires_grad = True

        ### Flip sdf_grid to zyx indexing
        sdf_grid = sdf_grid.permute(2, 1, 0)
        run_test(shape, device, backend, sdf_grid, occupancy, level_sets)
