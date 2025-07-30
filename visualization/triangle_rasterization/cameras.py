from typing import NamedTuple, Protocol

import torch
from jaxtyping import Float
from torch import Tensor


class Camera(NamedTuple):
    extrinsics: Float[Tensor, "4 4"]
    intrinsics: Float[Tensor, "3 3"]
    image_shape: tuple[int, int]


class CameraFn(Protocol):
    def __call__(self, device: torch.device) -> Camera:
        pass


def viewing_origin(device: torch.device) -> Camera:
    intrinsics = [
        [512, 0, 768 // 2],
        [0, 512, 512 // 2],
        [0, 0, 1],
    ]
    extrinsics = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -5],
        [0, 0, 0, 1],
    ]

    return Camera(
        torch.tensor(extrinsics, dtype=torch.float32, device=device).inverse(),
        torch.tensor(intrinsics, dtype=torch.float32, device=device),
        (512, 768),
    )


def almost_at_origin_looking_back(device: torch.device) -> Camera:
    intrinsics = [
        [256, 0, 768 // 2],
        [0, 256, 512 // 2],
        [0, 0, 1],
    ]
    extrinsics = [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, -0.2],
        [0, 0, 0, 1],
    ]

    return Camera(
        torch.tensor(extrinsics, dtype=torch.float32, device=device).inverse(),
        torch.tensor(intrinsics, dtype=torch.float32, device=device),
        (512, 768),
    )


CAMERAS: dict[str, CameraFn] = {
    "viewing_origin": viewing_origin,
    "almost_at_origin_looking_back": almost_at_origin_looking_back,
}
