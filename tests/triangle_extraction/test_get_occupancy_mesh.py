from itertools import product

import torch
from einops import reduce
from jaxtyping import install_import_hook

with install_import_hook(
    ("triangle_extraction", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction.misc import pack_occupancy
    from triangle_extraction.slang.get_occupancy_mesh import get_occupancy_mesh


def test_single_voxel_occupancy():
    device = torch.device("cuda")
    shape = (64, 64, 64)
    positions = [0, 1, 16, 30, 31, 32, 33, 48, 62, 63]
    colors = torch.full((3, 2, 3), 0.5, device=device, dtype=torch.float32)

    for i, j, k in product(positions, positions, positions):
        occupancy = torch.zeros(shape, dtype=torch.bool, device=device)
        occupancy[i, j, k] = True
        occupancy = pack_occupancy(occupancy)

        vertices, _ = get_occupancy_mesh(occupancy, colors)
        assert vertices.shape[0] == 12

        mean = reduce(vertices, "t c xyz -> xyz", "mean")
        expected = (
            (k + 0.5) / 64,
            (j + 0.5) / 64,
            (i + 0.5) / 64,
        )
        expected = torch.tensor(expected, dtype=torch.float32, device=device)
        assert torch.allclose(mean, expected)


def test_pair_voxel_occupancy():
    device = torch.device("cuda")
    shape = (64, 64, 64)
    positions = [0, 1, 16, 30, 31, 32, 33, 48, 61, 62, 63]
    colors = torch.full((3, 2, 3), 0.5, device=device, dtype=torch.float32)

    for axis in range(3):
        for i, j, k in product(positions, positions, positions):
            occupancy = torch.zeros(shape, dtype=torch.bool, device=device)
            occupancy[i, j, k] = True
            axes = [i, j, k]
            axes[axis] += 1
            if axes[axis] == 64:
                continue
            occupancy[*axes] = True
            occupancy = pack_occupancy(occupancy)

            vertices, _ = get_occupancy_mesh(occupancy, colors)
            assert vertices.shape[0] == 20
