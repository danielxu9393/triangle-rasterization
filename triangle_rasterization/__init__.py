import torch
from einops import reduce
from jaxtyping import Float32, Int32
from torch import Tensor
from torch.profiler import record_function

from triangle_rasterization.registry import (
    BOUND_TRIANGLES,
    COMPOSITE,
    DELINEATE_TILES,
    GENERATE_KEYS,
    PROJECT_VERTICES,
    SORT_KEYS,
    Backend,
)
from triangle_rasterization.types import TileGrid


@record_function("render")
def render(
    vertices: Float32[Tensor, "vertex xyz=3"],
    spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
    active_sh: int,
    signed_distances: Float32[Tensor, " vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_indices: Int32[Tensor, " triangle"],
    sharpness: Float32[Tensor, ""],
    num_shells: int,
    extrinsics: Float32[Tensor, "camera 4 4"],
    intrinsics: Float32[Tensor, "camera 3 3"],
    image_shape: tuple[int, int],
    backend: Backend,
    tile_shape: tuple[int, int] = (16, 16),
    grid_batch_shape: tuple[int, int] = (1024, 1024),
    near_plane: float = 0.2,
    ssaa: int = 1,
    backface_culling: int = 1,
    voxel_indices: Int32[Tensor, " triangle"] | None = None,
    occupancy: Int32[Tensor, "*shape"] | None = None,
) -> Float32[Tensor, "camera 4 height width"]:
    """Render the specified triangles from the specified camera poses.

    Extrinsics are OpenCV-style world-to-camera matrices (+Z look vector, -Y up vector,
    +X right vector). Intrinsics are unnormalized.
    """
    # Guard against empty scenes.
    b, _, _ = extrinsics.shape
    if vertices.numel() == 0 or faces.numel() == 0:
        return torch.zeros(
            (b, 4, *image_shape),
            device=vertices.device,
            dtype=torch.float32,
        )

    # Modify the image shape and intrinsics if super-sampling anti-aliasing is desired.
    if ssaa > 1:
        h, w = image_shape
        image_shape = (h * ssaa, w * ssaa)
        multiplier = torch.tensor((ssaa, ssaa, 1), device=intrinsics.device)
        intrinsics = intrinsics * multiplier[:, None]

    tile_height, tile_width = tile_shape
    grid = TileGrid.create(image_shape, (tile_height, tile_width))

    # TODO: We may want to make the kernels natively support batching.
    images = []
    for one_extrinsics, one_intrinsics in zip(extrinsics, intrinsics):
        # Project the vertices to image space.
        projection = PROJECT_VERTICES[backend](
            vertices,
            spherical_harmonics,
            one_extrinsics,
            one_intrinsics,
            active_sh,
        )

        # Slang has a limit of 2**30 elements per tensor. If there are a lot of
        # triangles, the number of rendering keys can exceed this limit. As a
        # workaround, we render the image in tiles.
        image = []
        for row in range(0, grid.grid_shape[0], grid_batch_shape[0]):
            row_image = []
            for col in range(0, grid.grid_shape[1], grid_batch_shape[1]):
                grid = grid.with_active_area((row, col), grid_batch_shape)

                # Compute triangle tile membership.
                triangle_bounds = BOUND_TRIANGLES[backend](
                    projection.positions,
                    projection.depths,
                    faces,
                    grid,
                    near_plane,
                    backface_culling,
                )

                # Generate keys (one per triangle-tile overlap).
                paired_keys = GENERATE_KEYS[backend](
                    projection.depths,
                    faces,
                    triangle_bounds.tile_minima,
                    triangle_bounds.tile_maxima,
                    triangle_bounds.num_tiles_overlapped,
                    grid,
                )

                # Sort the keys. After sorting, keys from the same tile will be
                # contiguous and ordered by ascending depth.
                paired_keys = SORT_KEYS[backend](paired_keys, grid)

                # Delineate tile boundaries.
                tile_boundaries = DELINEATE_TILES[backend](paired_keys, grid)

                # Alpha-composite triangles within each tile.
                active_image = COMPOSITE[backend](
                    projection.positions,
                    projection.colors,
                    signed_distances,
                    faces,
                    shell_indices,
                    sharpness,
                    num_shells,
                    paired_keys.triangle_indices,
                    tile_boundaries,
                    grid,
                    voxel_indices,
                    occupancy,
                )
                row_image.append(active_image)
                del active_image

            # Concatenate the row.
            row_image = torch.cat(row_image, dim=-1)
            image.append(row_image)
            del row_image

        # Concatenate the rows.
        image = torch.cat(image, dim=-2)

        # Scale the image if MSAA is desired.
        if ssaa > 1:
            image = reduce(image, "c (h mh) (w mw) -> c h w", "mean", mh=ssaa, mw=ssaa)

        images.append(image)

    return torch.stack(images)


@record_function("render_on_background")
def render_on_background(
    vertices: Float32[Tensor, "vertex xyz=3"],
    sh_coeffs: Float32[Tensor, "sh vertex rgb=3"],
    active_sh: int,
    signed_distances: Float32[Tensor, " vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_indices: Int32[Tensor, " triangle"],
    sharpness: Float32[Tensor, ""],
    num_shells: int,
    extrinsics: Float32[Tensor, "camera 4 4"],
    intrinsics: Float32[Tensor, "camera 3 3"],
    image_shape: tuple[int, int],
    background: Float32[Tensor, "rgb=3"],
    backend: Backend,
    tile_shape: tuple[int, int] = (16, 16),
    grid_batch_shape: tuple[int, int] = (1024, 1024),
    near_plane: float = 0.2,
    ssaa: int = 1,
    backface_culling: int = 1,
) -> Float32[Tensor, "camera rgb=3 height width"]:
    # Render RGBA images.
    image = render(
        vertices,
        sh_coeffs,
        active_sh,
        signed_distances,
        faces,
        shell_indices,
        sharpness,
        num_shells,
        extrinsics,
        intrinsics,
        image_shape,
        backend,
        tile_shape=tile_shape,
        grid_batch_shape=grid_batch_shape,
        near_plane=near_plane,
        ssaa=ssaa,
        backface_culling=backface_culling,
    )

    # Composite the images onto the background.
    image = image[:, :3] + (1 - image[:, 3:4]) * background[:, None, None]

    # Ensure that the images are within a valid range.
    return image.clip(min=0, max=1)
