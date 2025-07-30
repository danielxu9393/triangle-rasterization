from pathlib import Path

import torch
import typer
from jaxtyping import install_import_hook

with install_import_hook(
    ("triangle_rasterization", "visualization"),
    "beartype.beartype",
):
    from triangle_rasterization import render
    from visualization.image_io import save_image
    from visualization.triangle_rasterization.cameras import CAMERAS
    from visualization.triangle_rasterization.scenes import SCENES


def main(
    camera_name: str = "viewing_origin",
    scene_name: str = "single_triangle",
    tile_width: int = 16,
    tile_height: int = 16,
    backend: str = "slang",
    workspace: Path = Path("scratch/visualization"),
) -> None:
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extrinsics, intrinsics, image_shape = CAMERAS[camera_name](device)
    # vertices, colors, alphas, faces = SCENES[scene_name](device)
    vertices, sh_coeffs, active_sh, alphas, faces = SCENES[scene_name](device)
    vertices.requires_grad_()
    sh_coeffs.requires_grad_()
    num_faces, _ = faces.shape
    image = render(
        vertices,
        # colors,
        sh_coeffs,
        active_sh,
        alphas,
        faces,
        torch.zeros((num_faces,), dtype=torch.int32, device=device),
        1,
        extrinsics[None],
        intrinsics[None],
        image_shape,
        backend,
        (tile_height, tile_width),
        msaa=2,
    )
    save_image(image[0, :3], workspace / "render.png")
    save_image(image[0, 3], workspace / "alpha.png")

    # Smoke test for gradients
    image_grad = torch.rand(image.shape, device=device)
    sh_coeffs_grad = torch.autograd.grad(
        image, sh_coeffs, image_grad, retain_graph=True
    )[0]


typer.run(main)
