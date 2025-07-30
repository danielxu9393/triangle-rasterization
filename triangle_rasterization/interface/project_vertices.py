from dataclasses import dataclass
from typing import Protocol

from jaxtyping import Float, Float32
from torch import Tensor


@dataclass
class ProjectedVertices:
    # Projected 2D vertex locations in pixel space (unnormalized).
    positions: Float[Tensor, "vertex xy=2"]

    # Camera-space depths for each vertex.
    depths: Float32[Tensor, " vertex"]

    # Projected colors
    colors: Float[Tensor, "vertex rgb=3"]


class ProjectVerticesFn(Protocol):
    def __call__(
        self,
        vertices: Float[Tensor, "vertex xyz=3"],
        spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
        extrinsics: Float[Tensor, "4 4"],
        intrinsics: Float[Tensor, "3 3"],
        active_sh: int = 0,
    ) -> ProjectedVertices:
        pass
