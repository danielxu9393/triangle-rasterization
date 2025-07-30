from typing import Callable

import pytest
import torch
from jaxtyping import Float, Int, Int64, install_import_hook
from torch import Tensor

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction.pytorch.index_vertices import index_vertices, index_keys


@pytest.mark.parametrize(
    "points,result",
    [
        # 1) all distinct: expect [0,1]
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int),
        ),
        # 2) exact duplicates: both map to 0
        (
            torch.tensor([[5.5, 5.5, 5.5], [5.5, 5.5, 5.5]], dtype=torch.float32),
            torch.tensor([0, 0], dtype=torch.int),
        ),
        # 3) interleaved duplicates
        (
            torch.tensor(
                [
                    [1.1, 1.1, 1.1],
                    [2.2, 2.2, 2.2],
                    [1.1, 1.1, 1.1],
                    [3.3, 3.3, 3.3],
                    [2.2, 2.2, 2.2],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([0, 1, 0, 3, 1], dtype=torch.int),
        ),
        # 4) block duplicates then new
        (
            torch.tensor(
                [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], dtype=torch.float32
            ),
            torch.tensor([0, 0, 2], dtype=torch.int),
        ),
        # 5) non-adjacent repeats
        (
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [0.0, 1.0, 2.0],
                    [6.0, 7.0, 8.0],
                    [3.0, 4.0, 5.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([0, 1, 0, 3, 1], dtype=torch.int),
        ),
    ],
)
def test_index_points(
    points: Float[Tensor, "vertex 3"],
    result: Int[Tensor, " vertex"],
) -> None:
    idx = index_vertices(points)  # forward pass, O(V) gather lookup
    assert torch.equal(
        idx, result
    ), f"Expected {result.tolist()}, but got {idx.tolist()}"


def test_index_points_david() -> None:
    voxels = torch.load("/data/scene-rep/u/charatan/bad_vertices.torch")
    index = index_vertices(
        voxels.vertices, tol=1e-3
    )  # forward pass, O(V) gather lookup

    target = voxels.vertices[voxels.lower_corners]
    pred = voxels.vertices[index[voxels.lower_corners]]

    assert torch.equal(pred, target)


@pytest.mark.parametrize(
    "keys,result",
    [
        # all distinct
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor([0, 1, 2, 3], dtype=torch.int),
        ),
        # all same
        (
            torch.tensor([5, 5, 5], dtype=torch.int64),
            torch.tensor([0, 0, 0], dtype=torch.int),
        ),
        # repeats interleaved
        (
            torch.tensor([1, 2, 1, 3, 2], dtype=torch.int64),
            torch.tensor([0, 1, 0, 3, 1], dtype=torch.int),
        ),
        # block of duplicates then new
        (
            torch.tensor([10, 10, 10, 8, 8, 9], dtype=torch.int64),
            torch.tensor([0, 0, 0, 3, 3, 5], dtype=torch.int),
        ),
        # not in order originally
        (
            torch.tensor([10, 8, 7, 4, 6, 10, 9, 7, 8], dtype=torch.int64),
            torch.tensor([0, 1, 2, 3, 4, 0, 6, 2, 1], dtype=torch.int),
        ),
    ],
)
def test_index_keys(keys: Int64[Tensor, " vertex"], result: Int[Tensor, " vertex"]):
    index = index_keys(keys)
    assert torch.all(index == result), f"Expected {result} but got {index}"
