import torch
from jaxtyping import Int32
from torch import Tensor

from ..misc import pack_occupancy, unpack_occupancy


def dilate_occupancy(
    occupancy: Int32[Tensor, "i j k_packed"],
    dilation: int,
) -> Int32[Tensor, "i j k_packed"]:
    occupancy = unpack_occupancy(occupancy)
    kernel_size = 2 * dilation + 1
    occupancy = torch.nn.functional.max_pool3d(
        occupancy[None].half(),
        kernel_size=(kernel_size, kernel_size, kernel_size),
        stride=1,
        padding=kernel_size // 2,
    )[0].bool()
    return pack_occupancy(occupancy)
