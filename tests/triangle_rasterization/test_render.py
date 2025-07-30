import pytest
import torch

from triangle_rasterization import render
from visualization.triangle_rasterization.cameras import viewing_origin
from visualization.triangle_rasterization.scenes import sphere


@pytest.mark.parametrize("backend", ["torch", "slang"])
def test_sphere_behind_camera(backend):
    device = torch.device("cuda")

    triangles = sphere(device)
    triangles.vertices[..., -1] -= 20
    extrinsics, intrinsics, image_shape = viewing_origin(device)

    image = render(
        *triangles,
        torch.zeros(triangles.faces.shape[0], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.float32, device=device),
        1,
        extrinsics[None],
        intrinsics[None],
        image_shape,
        backend,
    )
    assert (image == 0).all()
