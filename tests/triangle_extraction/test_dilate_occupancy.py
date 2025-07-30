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
    from triangle_extraction.pytorch.dilate_occupancy import (
        dilate_occupancy as dilate_occupancy_torch,
    )
    from triangle_extraction.slang.dilate_occupancy import (
        dilate_occupancy as dilate_occupancy_slang,
    )


def run_test(
    shape: tuple[int, int, int],
    dilation: int,
    set_occupancy: Callable[[Bool[Tensor, "i j k"]], None],
) -> None:
    # Set up the packed occupancy grid.
    occupancy = torch.zeros(shape, dtype=torch.bool, device=torch.device("cuda"))
    set_occupancy(occupancy)
    occupancy = pack_occupancy(occupancy)

    dilated_torch = dilate_occupancy_torch(occupancy, dilation)
    dilated_slang = dilate_occupancy_slang(occupancy, dilation)
    assert (dilated_torch == dilated_slang).all()


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_middle(dilation):
    def set_input(occupancy):
        occupancy[10, 15, 20] = True

    run_test((32, 64, 96), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_000(dilation):
    def set_input(occupancy):
        occupancy[0, 0, 0] = True

    run_test((64, 64, 32), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_k_boundary(dilation):
    def set_input(occupancy):
        occupancy[10, 15, 31] = True

    run_test((32, 64, 96), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_high_k(dilation):
    def set_input(occupancy):
        occupancy[31, 32, 63] = True

    run_test((64, 64, 64), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_high_ij(dilation):
    def set_input(occupancy):
        occupancy[63, 20, 20] = True
        occupancy[20, 63, 20] = True

    run_test((64, 64, 64), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_low_ij(dilation):
    def set_input(occupancy):
        occupancy[0, 20, 20] = True
        occupancy[20, 0, 20] = True

    run_test((64, 64, 64), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
def test_low_k(dilation):
    def set_input(occupancy):
        occupancy[20, 20, 0] = True

    run_test((64, 64, 64), dilation, set_input)


@pytest.mark.parametrize("dilation", (1, 2, 3, 6, 11))
@pytest.mark.parametrize("probability", (0.01, 0.05, 0.1, 0.5, 0.8))
@pytest.mark.parametrize("seed", (0, 123))
def test_random(dilation, probability, seed):
    device = torch.device("cuda")
    generator = torch.Generator(device)
    generator.manual_seed(seed)

    def set_input(occupancy):
        p = torch.full(occupancy.shape, probability, dtype=torch.float32, device=device)
        occupancy[:] = torch.bernoulli(p).bool()

    run_test((128, 128, 128), dilation, set_input)
