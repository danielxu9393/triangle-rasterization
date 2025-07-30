from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float, Int32
from torch import Tensor
from torch.profiler import record_function

from ..compilation import wrap_compilation
from ..interface.compute_sdf_regularizers import SDFRegularizerLosses
from ..misc import ceildiv

BLOCK_SIZE = 256

slang = wrap_compilation(
    lambda: slangtorch.loadModule(
        str(Path(__file__).parent / "compute_sdf_regularizers.slang"),
        verbose=True,
    )
)


class ComputeSDFRegularizers(torch.autograd.Function):
    @record_function("compute_sdf_regularizers_forward")
    @staticmethod
    def forward(
        ctx,
        grid_signed_distances: Float[Tensor, " sample"],
        voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
        voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
        grid_size: tuple[int, int, int],
    ) -> tuple[
        Float[Tensor, " voxel"], Float[Tensor, " voxel"], Float[Tensor, " voxel"]
    ]:
        device = grid_signed_distances.device

        # Allocate space for the output.
        _, num_voxels = voxel_upper_corners.shape
        eikonal_loss_pos = torch.empty(
            (num_voxels,), dtype=torch.float32, device=device
        )
        eikonal_loss_neg = torch.empty(
            (num_voxels,), dtype=torch.float32, device=device
        )
        curvature_loss = torch.empty((num_voxels,), dtype=torch.float32, device=device)

        # Call the Slang kernel.
        i, j, k = grid_size
        slang().computeSDFRegularizers(
            gridSignedDistances=grid_signed_distances,
            voxelNeighbors=voxel_neighbors,
            voxelLowerCorners=voxel_lower_corners,
            voxelUpperCorners=voxel_upper_corners,
            voxelX=k,
            voxelY=j,
            voxelZ=i,
            eikonalLossPos=eikonal_loss_pos,
            eikonalLossNeg=eikonal_loss_neg,
            curvatureLoss=curvature_loss,
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
        )

        # Save tensors needed for the backward pass.
        ctx.save_for_backward(
            grid_signed_distances,
            voxel_neighbors,
            voxel_lower_corners,
            voxel_upper_corners,
            eikonal_loss_pos,
            eikonal_loss_neg,
            curvature_loss,
        )
        ctx.grid_size = grid_size

        return eikonal_loss_pos, eikonal_loss_neg, curvature_loss

    @staticmethod
    def backward(
        ctx, grad_eikonal_loss_pos, grad_eikonal_loss_neg, grad_curvature_loss
    ):
        # Retrieve information from the context.
        (
            grid_signed_distances,
            voxel_neighbors,
            voxel_lower_corners,
            voxel_upper_corners,
            eikonal_loss_pos,
            eikonal_loss_neg,
            curvature_loss,
        ) = ctx.saved_tensors
        i, j, k = ctx.grid_size
        _, num_voxels = voxel_upper_corners.shape

        # Allocate space for the inputs' gradients.
        grad_grid_signed_distances = torch.zeros_like(grid_signed_distances)
        grad_eikonal_loss_pos = grad_eikonal_loss_pos.contiguous()
        grad_eikonal_loss_neg = grad_eikonal_loss_neg.contiguous()
        grad_curvature_loss = grad_curvature_loss.contiguous()

        # Call the Slang kernel.
        slang().computeSDFRegularizers.bwd(
            gridSignedDistances=(grid_signed_distances, grad_grid_signed_distances),
            voxelNeighbors=voxel_neighbors,
            voxelLowerCorners=voxel_lower_corners,
            voxelUpperCorners=voxel_upper_corners,
            voxelX=k,
            voxelY=j,
            voxelZ=i,
            eikonalLossPos=(eikonal_loss_pos, grad_eikonal_loss_pos),
            eikonalLossNeg=(eikonal_loss_neg, grad_eikonal_loss_neg),
            curvatureLoss=(curvature_loss, grad_curvature_loss),
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(num_voxels, BLOCK_SIZE), 1, 1),
        )

        # This seems to be necessary to avoid memory leaks.
        del ctx
        del grid_signed_distances
        del voxel_neighbors
        del voxel_lower_corners
        del voxel_upper_corners
        del eikonal_loss_pos
        del eikonal_loss_neg
        del curvature_loss

        return grad_grid_signed_distances, None, None, None, None, None


def compute_sdf_regularizers(
    grid_signed_distances: Float[Tensor, " sample"],
    voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
    voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
    voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
    grid_size: tuple[int, int, int],
) -> SDFRegularizerLosses:
    return SDFRegularizerLosses(
        *ComputeSDFRegularizers.apply(
            grid_signed_distances,
            voxel_neighbors,
            voxel_lower_corners,
            voxel_upper_corners,
            grid_size,
        )
    )
