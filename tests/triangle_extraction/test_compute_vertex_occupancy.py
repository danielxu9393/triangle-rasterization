from typing import Callable

import pytest
import torch
from jaxtyping import Bool, install_import_hook
from torch import Tensor

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction.misc import pack_occupancy, unpack_occupancy
    from triangle_extraction.registry import COMPUTE_VERTEX_OCCUPANCY, Backend

VALID_BACKENDS = ("slang",)


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_test(
    shape: tuple[int, int, int],
    device: torch.device,
    backend: Backend,
    set_occupancy: Callable[[Bool[Tensor, "i j k"]], None],
    set_expected_vertex_occupancy: Callable[[Bool[Tensor, "i j k"]], None],
) -> None:
    # Set up the packed occupancy grid.
    occupancy = torch.zeros(shape, dtype=torch.bool, device=device)
    set_occupancy(occupancy)
    occupancy = pack_occupancy(occupancy)

    # Compute vertex occupancy.
    i, j, k_packed = occupancy.shape
    vertex_occupancy, vertex_counts = COMPUTE_VERTEX_OCCUPANCY[backend](occupancy)

    # Set up the expected vertex occupancy grid.
    expected_vertex_occupancy = torch.zeros(
        (i + 1, j + 1, (k_packed + 1) * 32),
        dtype=torch.int32,
        device=device,
    )
    set_expected_vertex_occupancy(expected_vertex_occupancy)

    # Assert that the vertex occupancies match.
    assert (unpack_occupancy(vertex_occupancy) == expected_vertex_occupancy).all()
    assert vertex_counts.sum() == expected_vertex_occupancy.sum()


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_middle(device, backend):
    def set_input(occupancy):
        occupancy[10, 15, 20] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[10:12, 15:17, 20:22] = True

    run_test((32, 64, 96), device, backend, set_input, set_output)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_000(device, backend):
    def set_input(occupancy):
        occupancy[0, 0, 0] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[:2, :2, :2] = True

    run_test((64, 64, 32), device, backend, set_input, set_output)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_k_boundary(device, backend):
    def set_input(occupancy):
        occupancy[10, 15, 31] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[10:12, 15:17, 31:33] = True

    run_test((32, 64, 96), device, backend, set_input, set_output)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_high_k(device, backend):
    def set_input(occupancy):
        occupancy[31, 32, 63] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[31:33, 32:34, 63:65] = True

    run_test((64, 64, 64), device, backend, set_input, set_output)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_high_ij(device, backend):
    def set_input(occupancy):
        occupancy[63, 20, 20] = True
        occupancy[20, 63, 20] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[63:65, 20:22, 20:22] = True
        vertex_occupancy[20:22, 63:65, 20:22] = True

    run_test((64, 64, 64), device, backend, set_input, set_output)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_low_ij(device, backend):
    def set_input(occupancy):
        occupancy[0, 20, 20] = True
        occupancy[20, 0, 20] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[0:2, 20:22, 20:22] = True
        vertex_occupancy[20:22, 0:2, 20:22] = True

    run_test((64, 64, 64), device, backend, set_input, set_output)


@pytest.mark.parametrize("backend", VALID_BACKENDS)
def test_low_k(device, backend):
    def set_input(occupancy):
        occupancy[20, 20, 0] = True

    def set_output(vertex_occupancy):
        vertex_occupancy[20:22, 20:22, 0:2] = True

    run_test((64, 64, 64), device, backend, set_input, set_output)
