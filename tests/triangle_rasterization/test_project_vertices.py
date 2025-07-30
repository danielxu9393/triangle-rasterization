import pytest
import torch

from tests.misc import dataclass_allclose, grad_allclose
from triangle_rasterization.pytorch.project_vertices import (
    project_vertices as project_vertices_torch,
)
from triangle_rasterization.slang.project_vertices import (
    project_vertices as project_vertices_slang,
)
from visualization.triangle_rasterization.cameras import CAMERAS, Camera
from visualization.triangle_rasterization.scenes import SCENES, Scene


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("scene_fn", tuple(SCENES.values()))
@pytest.mark.parametrize("camera_fn", tuple(CAMERAS.values()))
def test_forward_pass(device, scene_fn, camera_fn):
    camera: Camera = camera_fn(device)
    scene: Scene = scene_fn(device)
    projection_torch = project_vertices_torch(
        scene.vertices,
        scene.sh_coeffs,
        camera.extrinsics,
        camera.intrinsics,
        scene.active_sh,
    )
    projection_slang = project_vertices_slang(
        scene.vertices,
        scene.sh_coeffs,
        camera.extrinsics,
        camera.intrinsics,
        scene.active_sh,
    )
    assert dataclass_allclose(projection_torch, projection_slang, rtol=1e-2)


@pytest.mark.parametrize("scene_fn", tuple(SCENES.values()))
@pytest.mark.parametrize("camera_fn", tuple(CAMERAS.values()))
def test_backward_pass(device, scene_fn, camera_fn):
    camera: Camera = camera_fn(device)
    scene_torch: Scene = scene_fn(device)
    scene_slang: Scene = scene_fn(device)

    scene_torch.vertices.requires_grad_()
    scene_slang.vertices.requires_grad_()
    scene_torch.sh_coeffs.requires_grad_()
    scene_slang.sh_coeffs.requires_grad_()

    projection_torch = project_vertices_torch(
        scene_torch.vertices,
        scene_torch.sh_coeffs,
        camera.extrinsics,
        camera.intrinsics,
        scene_torch.active_sh,
    )
    projection_slang = project_vertices_slang(
        scene_slang.vertices,
        scene_slang.sh_coeffs,
        camera.extrinsics,
        camera.intrinsics,
        scene_slang.active_sh,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    positions_grad = torch.rand(
        projection_torch.positions.shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    depths_grad = torch.rand(
        projection_torch.depths.shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    colors_grad = torch.rand(
        projection_torch.colors.shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    torch.autograd.backward(
        (projection_torch.positions, projection_torch.depths, projection_torch.colors),
        (positions_grad, depths_grad, colors_grad),
        inputs=scene_torch.vertices,
    )

    torch.autograd.backward(
        (projection_slang.positions, projection_slang.depths, projection_slang.colors),
        (positions_grad, depths_grad, colors_grad),
        inputs=scene_slang.vertices,
    )

    assert grad_allclose(
        scene_slang.vertices.grad,
        scene_torch.vertices.grad,
        atol=4e-5,
    )
    assert grad_allclose(
        scene_slang.sh_coeffs.grad,
        scene_torch.sh_coeffs.grad,
        atol=4e-5,
    )
