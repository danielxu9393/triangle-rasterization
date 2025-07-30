from typing import NamedTuple, Protocol

from jaxtyping import Float, Int32, UInt8
from torch import Tensor


class CountTriangleFacesResult(NamedTuple):
    face_counts: Int32[Tensor, " voxel+1"]
    cell_codes: UInt8[Tensor, "level_set voxel"]


class CountTriangleFacesFn(Protocol):
    def __call__(
        self,
        lower_corners: Int32[Tensor, "corner=4 voxel_and_subvoxel"],
        upper_corners: Int32[Tensor, "corner=4 voxel"],
        level_sets: Float[Tensor, " level_set"],
    ) -> CountTriangleFacesResult:
        pass
