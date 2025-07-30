import numpy as np
import torch
from einops import rearrange
from jaxtyping import install_import_hook
from trimesh import Trimesh

with install_import_hook(
    ("triangle_rasterization", "visualization"),
    "beartype.beartype",
):
    from triangle_extraction.misc import pack_occupancy
    from triangle_extraction.slang.get_occupancy_mesh import get_occupancy_mesh


if __name__ == "__main__":
    shape = (64, 64, 64)
    device = torch.device("cuda")
    generator = torch.Generator(device)
    generator.manual_seed(0)

    # Create a random occupancy grid.
    probability = torch.full(shape, 0.1, dtype=torch.float32, device=device)
    occupancy = torch.bernoulli(probability, generator=generator).bool()
    occupancy_packed = pack_occupancy(occupancy)

    vertices, colors = get_occupancy_mesh(
        occupancy_packed,
        torch.full((3, 2, 3), 0.5, device=device, dtype=torch.float32),
    )

    Trimesh(
        rearrange(vertices.detach().cpu().numpy(), "t c xyz -> (t c) xyz"),
        rearrange(np.arange(vertices.shape[0] * 3), "(t c) -> t c", c=3),
    ).export("test.obj")
