from typing import Protocol

from jaxtyping import Int32
from torch import Tensor


class ComputeExclusiveCumsumFn(Protocol):
    def __call__(self, x: Int32[Tensor, " entry"]) -> None:
        """This should be an in-place operation."""
        pass
