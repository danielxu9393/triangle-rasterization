from typing import NamedTuple, Protocol

from jaxtyping import Float, Int32
from torch import Tensor


class SDFRegularizerLosses(NamedTuple):
    eikonal_loss_pos: Float[Tensor, " voxel"]
    eikonal_loss_neg: Float[Tensor, " voxel"]
    curvature_loss: Float[Tensor, " voxel"]


class ComputeSDFRegularizersFn(Protocol):
    def __call__(
        self,
        grid_signed_distances: Float[Tensor, " sample"],
        voxel_neighbors: Int32[Tensor, "neighbor=7 voxel"],
        voxel_lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        voxel_upper_corners: Int32[Tensor, "corner=4 voxel"],
        grid_size: tuple[int, int, int],
    ) -> SDFRegularizerLosses:
        pass
