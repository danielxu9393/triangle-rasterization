import torch
from jaxtyping import Int32
from torch import Tensor


def compute_exclusive_cumsum(x: Int32[Tensor, " entry"]) -> None:
    padded = torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0)
    x[:] = padded.cumsum(0)
