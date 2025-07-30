from typing import Callable

import pytest
import torch
from jaxtyping import Bool, install_import_hook
from torch import Tensor

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction.misc import pack_occupancy
    from triangle_extraction.pytorch.upscale_occupancy import (
        upscale_occupancy as upscale_occupancy_torch,
    )
    from triangle_extraction.slang.upscale_occupancy import (
        upscale_occupancy as upscale_occupancy_slang,
    )


def run_test(
    shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    set_occupancy: Callable[[Bool[Tensor, "i j k"]], None],
) -> None:
    # Set up the packed occupancy grid.
    occupancy = torch.zeros(shape, dtype=torch.bool, device=torch.device("cuda"))
    set_occupancy(occupancy)
    occupancy = pack_occupancy(occupancy)

    upscaled_torch = upscale_occupancy_torch(occupancy, target_shape)
    upscaled_slang = upscale_occupancy_slang(occupancy, target_shape)
    assert (upscaled_torch == upscaled_slang).all()


@pytest.mark.parametrize("i", (0, 31, 63))
@pytest.mark.parametrize("j", (0, 32, 63))
@pytest.mark.parametrize("k", (0, 1, 15, 31, 32, 63))
def test_single_voxel_2x(i, j, k):
    def set_input(occupancy):
        occupancy[i, j, k] = True

    run_test((64, 64, 64), (128, 128, 128), set_input)


@pytest.mark.parametrize("i", (0, 1, 31, 63))
@pytest.mark.parametrize("j", (0, 1, 32, 63))
@pytest.mark.parametrize("k", (0, 1, 15, 31, 32, 63))
def test_single_voxel_50_percent(i, j, k):
    def set_input(occupancy):
        occupancy[i, j, k] = True

    run_test((64, 64, 64), (96, 96, 96), set_input)


@pytest.mark.parametrize("i", (0, 1, 31))
@pytest.mark.parametrize("j", (0, 1, 32, 63))
@pytest.mark.parametrize("k", (0, 1, 15, 31, 32, 63, 64, 95))
def test_single_voxel_odd_sizes(i, j, k):
    def set_input(occupancy):
        occupancy[i, j, k] = True

    run_test((32, 64, 96), (160, 128, 96), set_input)


@pytest.mark.parametrize(
    "old_shape",
    (
        (32, 32, 32),
        (64, 32, 32),
        (32, 64, 32),
        (32, 32, 64),
        (32, 64, 64),
        (64, 32, 64),
        (64, 64, 32),
        (64, 64, 64),
    ),
)
@pytest.mark.parametrize(
    "new_shape",
    (
        (64, 64, 64),
        (64, 96, 128),
        (128, 64, 96),
        (128, 128, 128),
    ),
)
@pytest.mark.parametrize("probability", (0.01, 0.05, 0.1, 0.5, 0.8))
@pytest.mark.parametrize("seed", (0, 123))
def test_random(old_shape, new_shape, probability, seed):
    device = torch.device("cuda")
    generator = torch.Generator(device)
    generator.manual_seed(seed)

    def set_input(occupancy):
        p = torch.full(occupancy.shape, probability, dtype=torch.float32, device=device)
        occupancy[:] = torch.bernoulli(p).bool()

    run_test(old_shape, new_shape, set_input)
