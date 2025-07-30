from typing import NamedTuple, Protocol

import torch
import trimesh
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor


class Scene(NamedTuple):
    vertices: Float[Tensor, "vertex xyz=3"]
    sh_coeffs: Float[Tensor, "sh vertex rgb=3"]
    active_sh: int
    alphas: Float[Tensor, "vertex"]
    faces: Int[Tensor, "triangle corner=3"]


class SceneFn(Protocol):
    def __call__(self, device: torch.device) -> Scene:
        pass


def convert_colors_to_sh(colors: torch.Tensor) -> torch.Tensor:
    """
    Convert per-vertex colors (V, 3) into spherical harmonic coefficients (V, 16, 3)
    where only the l=0 coefficient (index 0) contains the color and all others are zero.
    """
    V = colors.shape[0]
    sh_coeffs = torch.zeros((16, V, 3), dtype=colors.dtype, device=colors.device)
    C0 = 0.28209479177387814
    colors = (colors - 0.5) / C0
    sh_coeffs[0, :, :] = colors
    return sh_coeffs.half()


def single_triangle(device: torch.device) -> Scene:
    positions = [
        [-0.5, 0.5, 0],  # bottom left
        [0.5, 0.5, 0],  # bottom right
        [0, -0.5, 0],  # top
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    alphas = [1, 1, 0.5]
    faces = torch.arange(3, dtype=torch.int32, device=device)[None]

    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
    sh_coeffs = convert_colors_to_sh(colors_tensor)
    active_sh = 0

    return Scene(
        torch.tensor(positions, dtype=torch.float32, device=device),
        sh_coeffs,
        active_sh,
        torch.tensor(alphas, dtype=torch.float32, device=device),
        faces,
    )


def two_overlapping_triangles(device: torch.device) -> Scene:
    corners = [
        # front triangle
        [-0.75, 0.25, -0.75],  # bottom left
        [0.25, 0.25, -0.75],  # bottom right
        [-0.25, -0.75, -0.75],  # top
        # back triangle
        [-0.25, -0.25, 0.75],  # top left
        [0.75, -0.25, 0.75],  # top right
        [0.25, 0.75, 0.75],  # bottom
    ]
    colors = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
    ]
    alphas = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    faces = rearrange(
        torch.arange(6, dtype=torch.int32, device=device), "(t c) -> t c", c=3
    )

    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
    sh_coeffs = convert_colors_to_sh(colors_tensor)
    active_sh = 0

    return Scene(
        torch.tensor(corners, dtype=torch.float32, device=device),
        sh_coeffs,
        active_sh,
        torch.tensor(alphas, dtype=torch.float32, device=device),
        faces,
    )


def sphere(device: torch.device) -> Scene:
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)

    # Add a small offset to vertices to avoid edge-case issues in projection.
    vertices = (
        torch.tensor(sphere_mesh.vertices, dtype=torch.float32, device=device) + 0.007
    )
    faces = torch.tensor(sphere_mesh.faces, dtype=torch.int32, device=device)

    # Define distinct colors for each triangle.
    colors = torch.eye(3, dtype=torch.float32, device=device)
    v, _ = vertices.shape
    colors = repeat(colors, "b rgb -> (a b) rgb", a=v // 3).contiguous()

    sh_coeffs = convert_colors_to_sh(colors)
    active_sh = 0

    # Use a uniform alpha for all vertices.
    alphas = torch.full((v,), 0.5, dtype=torch.float32, device=device)

    return Scene(
        vertices,
        sh_coeffs,
        active_sh,
        alphas,
        faces,
    )


def sphere_specular(device: torch.device) -> Scene:
    sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)

    # Add a small offset to vertices to avoid edge-case issues in projection.
    vertices = (
        torch.tensor(sphere_mesh.vertices, dtype=torch.float32, device=device) + 0.007
    )
    faces = torch.tensor(sphere_mesh.faces, dtype=torch.int32, device=device)

    # Define distinct colors for each triangle.
    colors = torch.eye(3, dtype=torch.float32, device=device)
    v, _ = vertices.shape
    colors = repeat(colors, "b rgb -> (a b) rgb", a=v // 3).contiguous()

    sh_coeffs = convert_colors_to_sh(colors)
    generator = torch.Generator(device).manual_seed(42)
    sh_coeffs = torch.randn(
        (16, v, 3), generator=generator, dtype=torch.float32, device=device
    )
    active_sh = 4

    # Use a uniform alpha for all vertices.
    alphas = torch.full((v,), 0.5, dtype=torch.float32, device=device)

    return Scene(
        vertices,
        sh_coeffs,
        active_sh,
        alphas,
        faces,
    )


def giant_sphere(device: torch.device) -> Scene:
    vertices, sh_coeffs, active_sh, alphas, faces = sphere(device)
    vertices = vertices * 100
    return Scene(vertices, sh_coeffs, active_sh, alphas, faces)


def sphere_outside_frustum(device: torch.device) -> Scene:
    vertices, sh_coeffs, active_sh, alphas, faces = sphere(device)
    vertices = vertices + 1000
    return Scene(vertices, sh_coeffs, active_sh, alphas, faces)


SCENES: dict[str, SceneFn] = {
    "single_triangle": single_triangle,
    "two_overlapping_triangles": two_overlapping_triangles,
    "sphere": sphere,
    "sphere_specular": sphere_specular,
    "sphere_outside_frustum": sphere_outside_frustum,
    "giant_sphere": giant_sphere,
}
