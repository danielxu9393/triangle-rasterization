import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float32, Int32, Shaped
from torch import Tensor

from ..types import TileGrid


def compute_compositing_weights(
    alpha: Float32[Tensor, "*batch sample"],
) -> Float32[Tensor, "*batch sample"]:
    # Compute occlusion for each sample. The 1e-10 is from the original NeRF.
    shifted_alpha = torch.cat(
        (torch.ones_like(alpha[..., :1]), 1 - alpha[..., :-1] + 1e-10),
        dim=-1,
    )
    occlusion = torch.cumprod(shifted_alpha, dim=-1)

    # Combine alphas with occlusion effects to get the final weights.
    return alpha * occlusion


def compute_barycentric_coordinates(
    points: Float32[Tensor, "*#batch xy=2"],
    triangles: Float32[Tensor, "*#batch corner=3 xy=2"],
) -> Float32[Tensor, "*batch uvw=3"]:
    a, b, c = triangles.unbind(dim=-2)

    v0_x, v0_y = (b - a).unbind(dim=-1)
    v1_x, v1_y = (c - a).unbind(dim=-1)
    v2_x, v2_y = (points - a).unbind(dim=-1)
    coefficient = 1 / (v0_x * v1_y - v1_x * v0_y)
    v = (v2_x * v1_y - v1_x * v2_y) * coefficient
    w = (v0_x * v2_y - v2_x * v0_y) * coefficient
    u = 1 - v - w
    return torch.stack((u, v, w), dim=-1)


def mask_first_value(
    x: Shaped[Tensor, "*batch entry"],
    mask: Bool[Tensor, "*batch entry"],
) -> Bool[Tensor, "*batch entry"]:
    # This is probably much slower than it could be, but it's only used for testing...
    *batch, _ = x.shape
    x = rearrange(x, "... entry -> (...) entry")
    mask = rearrange(mask, "... entry -> (...) entry")
    result = torch.zeros_like(x, dtype=torch.bool)
    for i, entries in enumerate(x):
        seen = set()
        for j, entry in enumerate(entries):
            entry_item = entry.item()
            if entry_item in seen or not mask[i, j]:
                continue
            seen.add(entry_item)
            result[i, j] = True
    return result.reshape((*batch, -1))


def composite(
    vertices: Float32[Tensor, "vertex xy=2"],
    colors: Float32[Tensor, "vertex rgb=3"],
    signed_distances: Float32[Tensor, " vertex"],
    faces: Int32[Tensor, "triangle corner=3"],
    shell_indices: Int32[Tensor, " triangle"],
    sharpness: Float32[Tensor, ""],
    num_shells: int,
    sorted_triangle_indices: Int32[Tensor, " key"],
    tile_boundaries: Int32[Tensor, "tile boundary=2"],
    grid: TileGrid,
    voxel_indices: Int32[Tensor, " triangle"] | None = None,
    occupancy: Int32[Tensor, "i j k_packed"] | None = None,
) -> Float32[Tensor, "4 height width"]:
    device = sorted_triangle_indices.device
    vertices = vertices[faces]
    colors = colors[faces]
    signed_distances = signed_distances[faces]
    if voxel_indices is not None or occupancy is not None:
        raise NotImplementedError("Occupancy marking is not implemented for PyTorch.")

    image = torch.zeros((4, *grid.image_shape), dtype=torch.float32, device=device)
    _, h, w = image.shape

    for tile_index, (start, end) in enumerate(tile_boundaries):
        # Skip empty tiles.
        if start == end:
            continue

        # Retrieve triangles for the key range.
        indices = sorted_triangle_indices[start:end]
        tile_vertices = vertices[indices]
        tile_colors = colors[indices]
        tile_signed_distances = signed_distances[indices]
        tile_shell_indices = shell_indices[indices]

        # Compute XY coordinates within the tile.
        grid_row = tile_index // grid.grid_shape[1]
        grid_col = tile_index % grid.grid_shape[1]
        grd_x = grid_col * grid.tile_shape[1]
        grd_y = grid_row * grid.tile_shape[0]
        x = torch.arange(grid.tile_shape[1], device=device) + 0.5 + grd_x
        y = torch.arange(grid.tile_shape[0], device=device) + 0.5 + grd_y
        xy = torch.stack(torch.meshgrid((x, y), indexing="xy"), dim=-1)

        # Compute barycentric coordinates.
        uvw = compute_barycentric_coordinates(xy[:, :, None], tile_vertices)
        hit = ((uvw >= 0) & (uvw <= 1)).all(dim=-1)
        tile_signed_distances = einsum(
            uvw,
            tile_signed_distances,
            "h w t uvw, t uvw -> h w t",
        )
        tile_colors = einsum(uvw, tile_colors, "h w t uvw, t uvw c -> h w t c")

        # Just do this one pixel at a time since it's simpler and this will only be used
        # for testing...
        th, tw, _ = hit.shape
        for row in range(th):
            for col in range(tw):
                # Skip out-of-bounds pixels.
                if grd_x + col >= w or grd_y + row >= h:
                    continue

                px_hit = hit[row, col]

                # Ensure that the shell indices for the pixel are strictly increasing.
                px_shell_index = -1
                for i in range(px_hit.shape[0]):
                    if px_hit[i] and tile_shell_indices[i] > px_shell_index:
                        px_shell_index = tile_shell_indices[i].item()
                    else:
                        # The autograd engine complains if you just set px_hit[i] to
                        # False, so we do this instead.
                        px_hit_update = torch.ones_like(px_hit)
                        px_hit_update[i] = False
                        px_hit = px_hit & px_hit_update

                # Filter out all triangles that aren't being composited.
                px_sdf = tile_signed_distances[row, col][px_hit]
                px_color = tile_colors[row, col][px_hit]

                # Compute alphas using the NeuS formula. Note, however, that our
                # convention is that negative signed distances are inside the object.
                px_cdf = (px_sdf * sharpness).sigmoid()
                px_cdf = torch.cat((torch.ones_like(px_sdf[:1]), px_cdf))
                px_cdf_inner = px_cdf[1:]
                px_cdf_outer = px_cdf[:-1]
                px_alpha = torch.clip(
                    (px_cdf_outer - px_cdf_inner + 1e-5) / (px_cdf_outer + 1e-5), 0, 1
                )
                px_weights = compute_compositing_weights(px_alpha)
                px_color = einsum(px_weights, px_color, "t, t rgb -> rgb")

                image[:3, grd_y + row, grd_x + col] = px_color
                image[3, grd_y + row, grd_x + col] = px_weights.sum()

    return image
