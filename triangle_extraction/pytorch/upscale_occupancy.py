import torch.nn.functional as F
from jaxtyping import Int32
from torch import Tensor

from ..misc import pack_occupancy, unpack_occupancy


def upscale_occupancy(
    occupancy: Int32[Tensor, "i j k_packed"],
    target_shape: tuple[int, int, int],
) -> Int32[Tensor, "i_target j_target k_packed_target"]:
    occupancy = unpack_occupancy(occupancy)
    occupancy = F.interpolate(
        occupancy[None, None, :, :, :].float(),
        target_shape,
        mode="area",
    )[0, 0]
    return pack_occupancy(occupancy > 0)
