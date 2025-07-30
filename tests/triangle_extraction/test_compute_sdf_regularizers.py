from typing import Callable

import pytest
import torch
from jaxtyping import Bool, install_import_hook
from torch import Tensor

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction import extract_voxels
    from triangle_extraction.misc import pack_occupancy
    from triangle_extraction.pytorch.compute_sdf_regularizers import (
        compute_sdf_regularizers as compute_sdf_regularizers_torch,
    )
    from triangle_extraction.slang.compute_sdf_regularizers import (
        compute_sdf_regularizers as compute_sdf_regularizers_slang,
    )


def run_test(
    shape: tuple[int, int, int],
    device: torch.device,
    set_occupancy: Callable[[Bool[Tensor, "i j k"]], None],
    seed: int,
) -> None:
    occupancy = torch.zeros(shape, dtype=torch.bool, device=device)
    set_occupancy(occupancy)
    occupancy = pack_occupancy(occupancy)
    vertices, neighbors, lower_corners, upper_corners, _, _ = extract_voxels(
        occupancy, "slang"
    )

    generator = torch.Generator(device)
    generator.manual_seed(seed)
    _, num_voxels = neighbors.shape
    num_vertices, _ = vertices.shape
    sdf = torch.randn((num_vertices,), device=device, generator=generator)
    output_gradients = torch.randn((num_voxels,), device=device, generator=generator)

    for loss_index in range(2):
        sdf_torch = sdf.clone().requires_grad_(True)
        loss_torch = compute_sdf_regularizers_torch(
            sdf_torch,
            neighbors,
            lower_corners,
            upper_corners,
            shape,
        )[loss_index]
        (grad_torch,) = torch.autograd.grad(loss_torch, sdf_torch, output_gradients)

        sdf_slang = sdf.clone().requires_grad_(True)
        loss_slang = compute_sdf_regularizers_slang(
            sdf_slang,
            neighbors,
            lower_corners,
            upper_corners,
            shape,
        )[loss_index]
        (grad_slang,) = torch.autograd.grad(loss_slang, sdf_slang, output_gradients)

        # Allow a really small number of gradients to be different because of numerical
        # weirdness. The number should be smaller than any single axis length because
        # edge cases exist for edges and faces of the voxel grid. Technically, corners
        # (especially the +xyz one) are also an edge case, but that one probably doesn't
        # matter enough to be concerned about.
        assert torch.allclose(loss_torch, loss_slang, rtol=1e-4, atol=1e-4)
        assert (
            ~torch.isclose(grad_torch, grad_slang, rtol=1e-4, atol=1e-4)
        ).sum() < min(shape)


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_dense(seed):
    def set_occupancy(occupancy: Bool[Tensor, "i j k"]) -> None:
        occupancy[:] = True

    run_test((32, 32, 32), torch.device("cuda"), set_occupancy, seed)


@pytest.mark.parametrize("seed", (1, 2))
@pytest.mark.parametrize("p_occupied", (0.1, 0.25, 0.5, 0.75, 0.9))
def test_sparse(seed, p_occupied):
    device = torch.device("cuda")
    generator = torch.Generator(device)
    generator.manual_seed(seed)

    def set_occupancy(occupancy: Bool[Tensor, "i j k"]) -> None:
        occupancy[:] = torch.bernoulli(
            torch.full_like(occupancy, p_occupied, dtype=torch.float32)
        ).bool()

    run_test((32, 32, 32), device, set_occupancy, seed)
