import torch
from jaxtyping import Float, Int32
from torch import Tensor

from ..interface.compute_sdf_regularizers import SDFRegularizerLosses


def compute_eikonal_loss(
    grid_signed_distances: Float[Tensor, " sample"],
    voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
    grid_size: tuple[int, int, int],
) -> tuple[Float[Tensor, " voxel"], Float[Tensor, " voxel"]]:
    device = grid_signed_distances.device

    # Ignore subvoxel corners.
    _, num_voxels = voxel_upper_corners.shape
    voxel_lower_corners = voxel_lower_corners[:, :num_voxels]

    # Get each voxel's eight corners.
    d000 = grid_signed_distances[voxel_lower_corners[0]]  # origin
    d001 = grid_signed_distances[voxel_lower_corners[1]]  # +x
    d010 = grid_signed_distances[voxel_lower_corners[2]]  # +y
    d011 = grid_signed_distances[voxel_upper_corners[3]]  # +xy
    d100 = grid_signed_distances[voxel_lower_corners[3]]  # +z
    d101 = grid_signed_distances[voxel_upper_corners[2]]  # +xz
    d110 = grid_signed_distances[voxel_upper_corners[1]]  # +yz
    d111 = grid_signed_distances[voxel_upper_corners[0]]  # +xyz

    def sample_trilinearly(
        location: Float[Tensor, "sample xyz=3"],
    ) -> Float[Tensor, " sample"]:
        x, y, z = location.unbind(dim=-1)
        return (
            d000 * (1 - x) * (1 - y) * (1 - z)
            + d001 * x * (1 - y) * (1 - z)
            + d010 * (1 - x) * y * (1 - z)
            + d011 * x * y * (1 - z)
            + d100 * (1 - x) * (1 - y) * z
            + d101 * x * (1 - y) * z
            + d110 * (1 - x) * y * z
            + d111 * x * y * z
        )

    with torch.enable_grad():
        sample_points = torch.full(
            (num_voxels, 3),
            0.5,
            device=device,
            requires_grad=True,
        )
        sdfs = sample_trilinearly(sample_points)
        (gradient,) = torch.autograd.grad(
            sdfs,
            sample_points,
            torch.ones_like(sdfs),
            # These flags allow backpropagation through the gradient.
            create_graph=True,
            retain_graph=True,
        )

    # Scale the gradient to accoutn for the voxel size.
    i, j, k = grid_size
    factor = torch.tensor((k, j, i), device=device, dtype=torch.float32)
    gradient = gradient * factor

    # Compute the difference between the gradient's norm and 1 (possibly one-sided) and
    # return an L2 loss on that.
    delta = gradient.norm(dim=-1) - 1
    eikonal_loss_pos = torch.where(delta > 0, delta * delta, torch.zeros_like(delta))
    eikonal_loss_neg = torch.where(delta < 0, delta * delta, torch.zeros_like(delta))

    return eikonal_loss_pos, eikonal_loss_neg


def compute_curvature_loss(
    grid_signed_distances: Float[Tensor, " sample"],
    voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
    voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
) -> Float[Tensor, " voxel"]:
    # Get the corners that definitely exist.
    d011 = grid_signed_distances[voxel_upper_corners[3]]  # +xy
    d101 = grid_signed_distances[voxel_upper_corners[2]]  # +xz
    d110 = grid_signed_distances[voxel_upper_corners[1]]  # +yz
    d111 = grid_signed_distances[voxel_upper_corners[0]]  # +xyz

    total = d011 + d101 + d110
    num_valid = torch.full_like(d011, 3, dtype=torch.int32)

    # Get the corners that maybe exist.
    neighbor = voxel_neighbors[6]
    for axis in (1, 2, 3):
        corner_indices = voxel_lower_corners[axis, neighbor]
        mask = corner_indices >= 0
        num_valid += mask
        total = total + mask * grid_signed_distances[corner_indices]

    total = total - num_valid * d111

    # Return an L2 loss on the Laplacian filter's output.
    return total * total


def compute_sdf_regularizers(
    grid_signed_distances: Float[Tensor, " sample"],
    voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
    voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
    grid_size: tuple[int, int, int],
) -> SDFRegularizerLosses:
    eikonal_loss_pos, eikonal_loss_neg = compute_eikonal_loss(
        grid_signed_distances,
        voxel_lower_corners,
        voxel_upper_corners,
        grid_size,
    )
    curvature_loss = compute_curvature_loss(
        grid_signed_distances,
        voxel_neighbors,
        voxel_lower_corners,
        voxel_upper_corners,
    )
    return SDFRegularizerLosses(eikonal_loss_pos, eikonal_loss_neg, curvature_loss)
