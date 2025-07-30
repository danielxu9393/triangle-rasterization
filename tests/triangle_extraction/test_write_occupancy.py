from math import prod

import pytest
import torch
from jaxtyping import Int32, install_import_hook
from torch import Tensor

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction.pytorch.write_occupancy import (
        write_occupancy as write_occupancy_torch,
    )
    from triangle_extraction.slang.write_occupancy import (
        write_occupancy as write_occupancy_slang,
    )


def run_test(
    shell_indices: Int32[Tensor, " triangle"],
    voxel_indices: Int32[Tensor, " triangle"],
    occupancy_shape: tuple[int, int, int],
    min_shell: int,
    max_shell: int,
) -> None:
    result_torch = write_occupancy_torch(
        shell_indices,
        voxel_indices,
        occupancy_shape,
        min_shell,
        max_shell,
    )
    result_slang = write_occupancy_slang(
        shell_indices,
        voxel_indices,
        occupancy_shape,
        min_shell,
        max_shell,
    )
    assert (result_torch == result_slang).all()


@pytest.mark.parametrize("seed", (0, 123))
@pytest.mark.parametrize("min_shell", (0, 2, 3))
@pytest.mark.parametrize("max_shell", (3, 5))
def test_random(seed, min_shell, max_shell):
    device = torch.device("cuda")
    generator = torch.Generator(device)
    generator.manual_seed(seed)

    shape = (64, 64, 64)
    num_voxels = prod(shape)
    num_triangles = num_voxels // 2
    shell_indices = torch.randint(
        0,
        5,
        (num_triangles,),
        device=device,
        generator=generator,
        dtype=torch.int32,
    )
    voxel_indices = torch.randint(
        0,
        num_voxels,
        (num_triangles,),
        device=device,
        generator=generator,
        dtype=torch.int32,
    )
    result_torch = write_occupancy_torch(
        shell_indices,
        voxel_indices,
        shape,
        min_shell,
        max_shell,
    )
    result_slang = write_occupancy_slang(
        shell_indices,
        voxel_indices,
        shape,
        min_shell,
        max_shell,
    )
    assert (result_torch == result_slang).all()
