import pytest
import torch
from einops import rearrange, repeat
from jaxtyping import Float32, Int32
from torch import Tensor

from triangle_rasterization import TileGrid
from triangle_rasterization.interface.composite import CompositeFn
from triangle_rasterization.pytorch.composite import composite as composite_torch
from triangle_rasterization.slang.composite import (
    TILE_KEYS,
    composite as composite_slang,
)


def fit(
    composite: CompositeFn,
    vertices: Float32[Tensor, "vertex xy=2"],
    colors: Float32[Tensor, "vertex rgb=3"],
    signed_distances: Float32[Tensor, " vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_indices: Int32[Tensor, " triangle"],
    sharpness: Float32[Tensor, ""],
    num_shells: int,
    sorted_triangle_indices: Int32[Tensor, " key"],
    tile_boundaries: Int32[Tensor, "tile boundary=2"],
    target: Float32[Tensor, "rgb=3 height width"],
    background: Float32[Tensor, "rgb=3"],
    grid: TileGrid,
) -> Float32[Tensor, ""]:
    image = composite(
        vertices,
        colors,
        signed_distances,
        faces,
        shell_indices,
        sharpness,
        num_shells,
        sorted_triangle_indices,
        tile_boundaries,
        grid,
        None,
        None,
    )
    image = image[:3] + (1 - image[3:4]) * background[:, None, None]
    return ((image - target) ** 2).sum(dim=0).mean()


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_case_overlapping_triangles(
    signed_distances: list[float],
    shell_indices: list[int],
    sharpness: float,
):
    t = len(signed_distances)

    def case_overlapping_triangles(device: torch.device):
        float_kwargs = dict(dtype=torch.float32, device=device)
        int_kwargs = dict(dtype=torch.int32, device=device)

        generator = torch.Generator(device)
        generator.manual_seed(0)

        # To allow for per-triangle colors, each triangle has separate vertices.
        vertices = [[0.0, 0.0], [1.1, 0.0], [0.0, 1.1]]
        vertices = torch.tensor(vertices, **float_kwargs)
        vertices = repeat(vertices, "c xy -> (t c) xy", t=t)
        colors = torch.rand((t * 3, 3), generator=generator, **float_kwargs)
        _signed_distances = torch.tensor(signed_distances, **float_kwargs)
        _signed_distances = repeat(_signed_distances, "t -> (t c)", c=3)
        faces = torch.arange(t * 3, **int_kwargs)
        faces = rearrange(faces, "(t c) -> t c", t=t, c=3)
        _shell_indices = torch.tensor(shell_indices, **int_kwargs)
        _sharpness = torch.tensor(sharpness, **float_kwargs)
        num_shells = len(set(shell_indices))
        sorted_triangle_indices = torch.arange(t, **int_kwargs)
        # pad sorted_triangle_indices to multiple of TILE_KEYS
        sorted_triangle_indices = torch.cat(
            [
                sorted_triangle_indices,
                torch.full((TILE_KEYS - (t % TILE_KEYS),), -1, **int_kwargs).to(device),
            ]
        )

        tile_boundaries = torch.tensor([[0, t]], dtype=torch.int64, device=device)
        grid = TileGrid((16, 16), (1, 1), (7, 7), (0, 0), (1, 1))

        return (
            vertices.contiguous(),
            colors.contiguous(),
            _signed_distances.contiguous(),
            faces.contiguous(),
            _shell_indices.contiguous(),
            _sharpness.contiguous(),
            num_shells,
            sorted_triangle_indices.contiguous(),
            tile_boundaries.contiguous(),
            grid,
        )

    return case_overlapping_triangles


TEST_CASES = {
    "single_triangle": make_case_overlapping_triangles([0.0], [0], 1.0),
    "two_triangles": make_case_overlapping_triangles([1.0, 0.0], [0, 1], 1.0),
    "three_triangles": make_case_overlapping_triangles(
        [1.0, 0.0, -1.0], [0, 1, 2], 1.0
    ),
    "three_triangles_same_shell": make_case_overlapping_triangles(
        [1.0, 0.0, -1.0], [0, 0, 0], 1.0
    ),
    "skipping_triangles": make_case_overlapping_triangles(
        [1.0, 1.0, 0.0, 0.0, -1.0, -1.0], [0, 0, 1, 1, 2, 2], 1.0
    ),
    "block_boundary": make_case_overlapping_triangles(
        [*([2.0] * 126), 1.0, 0.0, -1.0, -2.0], [*([0] * 126), 1, 2, 3, 4], 1.0
    ),
    "in_and_out": make_case_overlapping_triangles(
        [1.0, 0.0, -1.0, -1.0, 0.0, 1.0], [0, 1, 2, 2, 1, 0], 1.0
    ),
    "in_and_out_and_in": make_case_overlapping_triangles(
        [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0],
        [0, 1, 2, 2, 1, 0, 0, 1, 2],
        1.0,
    ),
}


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_forward_pass(test_case, device):
    image_slang = composite_slang(*TEST_CASES[test_case](device), None, None)
    image_torch = composite_torch(*TEST_CASES[test_case](device), None, None)
    assert torch.allclose(image_slang, image_torch)


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_backward_pass(test_case, device):
    (
        vertices_torch,
        colors_torch,
        signed_distances_torch,
        faces_torch,
        shell_indices_torch,
        sharpness_torch,
        num_shells_torch,
        sorted_triangle_indices_torch,
        tile_boundaries_torch,
        grid_torch,
    ) = TEST_CASES[test_case](device)
    colors_torch.requires_grad_(True)
    signed_distances_torch.requires_grad_(True)
    sharpness_torch.requires_grad_(True)

    (
        vertices_slang,
        colors_slang,
        signed_distances_slang,
        faces_slang,
        shell_indices_slang,
        sharpness_slang,
        num_shells_slang,
        sorted_triangle_indices_slang,
        tile_boundaries_slang,
        grid_slang,
    ) = TEST_CASES[test_case](device)
    colors_slang.requires_grad_(True)
    signed_distances_slang.requires_grad_(True)
    sharpness_slang.requires_grad_(True)

    g = torch.Generator(device)
    g.manual_seed(0)
    background = torch.rand(3, dtype=torch.float32, device=device, generator=g)
    h, w = grid_torch.image_shape
    target = torch.rand(3, h, w, dtype=torch.float32, device=device, generator=g)

    loss_torch = fit(
        composite_torch,
        vertices_torch,
        colors_torch,
        signed_distances_torch,
        faces_torch,
        shell_indices_torch,
        sharpness_torch,
        num_shells_torch,
        sorted_triangle_indices_torch,
        tile_boundaries_torch,
        target,
        background,
        grid_torch,
    )
    torch.autograd.backward(
        loss_torch,
        torch.ones_like(loss_torch),
        inputs=(colors_torch, signed_distances_torch, sharpness_torch),
    )

    loss_slang = fit(
        composite_slang,
        vertices_slang,
        colors_slang,
        signed_distances_slang,
        faces_slang,
        shell_indices_slang,
        sharpness_slang,
        num_shells_slang,
        sorted_triangle_indices_slang,
        tile_boundaries_slang,
        target,
        background,
        grid_slang,
    )
    torch.autograd.backward(
        loss_slang,
        torch.ones_like(loss_slang),
        inputs=(colors_slang, signed_distances_slang, sharpness_slang),
    )

    assert torch.allclose(loss_torch, loss_slang)
    assert torch.allclose(colors_torch.grad, colors_slang.grad, atol=1e-5)
    assert torch.allclose(
        signed_distances_torch.grad, signed_distances_slang.grad, atol=1e-5
    )
    assert torch.allclose(sharpness_torch.grad, sharpness_slang.grad, atol=1e-5)
