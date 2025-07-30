import torch
from einops import einsum, rearrange
from jaxtyping import Float, Float32
from torch import Tensor
from torch.nn import functional as F

from triangle_rasterization.interface.project_vertices import ProjectedVertices

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def project_vertices(
    vertices: Float[Tensor, "vertex xyz=3"],
    sh_coeffs: Float32[Tensor, "sh vertex rgb=3"],
    extrinsics: Float[Tensor, "4 4"],
    intrinsics: Float[Tensor, "3 3"],
    active_sh: int = 0,
) -> ProjectedVertices:
    # Convert the SH coefficients to RGB colors.
    # We need to convert extrinsics to c2w for camera position
    sh_coeffs = rearrange(sh_coeffs, "sh v rgb -> v sh rgb")
    extrinsics_inv = torch.inverse(extrinsics)
    cam_pos = extrinsics_inv[:3, 3]  # (3,)

    # Compute the directions from the camera to the vertices.
    dirs = vertices - cam_pos.unsqueeze(0)  # (V, 3)
    dirs = F.normalize(dirs, dim=-1)  # normalize each vector

    # Start with the l=0 term.
    rgb = SH_C0 * sh_coeffs[:, 0, :]  # (V, 3)
    if active_sh > 0:
        # l=1 terms
        rgb = rgb + (
            -SH_C1 * dirs[:, 1:2] * sh_coeffs[:, 1, :]
            + SH_C1 * dirs[:, 2:3] * sh_coeffs[:, 2, :]
            - SH_C1 * dirs[:, 0:1] * sh_coeffs[:, 3, :]
        )
        if active_sh > 1:
            # Precompute direction squared terms.
            xx = dirs[:, 0:1] ** 2
            yy = dirs[:, 1:2] ** 2
            zz = dirs[:, 2:3] ** 2
            xy = dirs[:, 0:1] * dirs[:, 1:2]
            yz = dirs[:, 1:2] * dirs[:, 2:3]
            xz = dirs[:, 0:1] * dirs[:, 2:3]
            # l=2 terms
            rgb = rgb + (
                SH_C2[0] * xy * sh_coeffs[:, 4, :]
                + SH_C2[1] * yz * sh_coeffs[:, 5, :]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
                + SH_C2[3] * xz * sh_coeffs[:, 7, :]
                + SH_C2[4] * (xx - yy) * sh_coeffs[:, 8, :]
            )
            if active_sh > 2:
                # l=3 terms
                rgb = rgb + (
                    SH_C3[0] * dirs[:, 1:2] * (3.0 * xx - yy) * sh_coeffs[:, 9, :]
                    + SH_C3[1] * xy * dirs[:, 2:3] * sh_coeffs[:, 10, :]
                    + SH_C3[2]
                    * dirs[:, 1:2]
                    * (4.0 * zz - xx - yy)
                    * sh_coeffs[:, 11, :]
                    + SH_C3[3]
                    * dirs[:, 2:3]
                    * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    * sh_coeffs[:, 12, :]
                    + SH_C3[4]
                    * dirs[:, 0:1]
                    * (4.0 * zz - xx - yy)
                    * sh_coeffs[:, 13, :]
                    + SH_C3[5] * dirs[:, 2:3] * (xx - yy) * sh_coeffs[:, 14, :]
                    + SH_C3[6] * dirs[:, 0:1] * (xx - 3.0 * yy) * sh_coeffs[:, 15, :]
                )
    # Add constant offset and clamp negative values.
    rgb = rgb + 0.5
    rgb = torch.clamp(rgb, min=0.0).float()

    # Convert the vertices to homogeneous coordinates.
    vertices = torch.cat((vertices, torch.ones_like(vertices[..., :1])), dim=-1)

    # Transform the vertices into camera space.
    vertices = einsum(extrinsics, vertices, "i j, v j -> v i")

    # Save the vertices' depths for later.
    depths = vertices[..., 2]

    # Project the vertices.
    vertices = vertices[..., :3] / vertices[..., 2:3]
    vertices = einsum(intrinsics, vertices, "i j, v j -> v i")[..., :2]

    return ProjectedVertices(vertices, depths, rgb)
