# ruff: noqa: E731

from pathlib import Path

import slangtorch
import torch
from jaxtyping import Float32
from torch import Tensor
from torch.profiler import record_function

from triangle_rasterization.interface.project_vertices import (
    ProjectedVertices,
    ProjectVerticesFn,
)
from triangle_rasterization.misc import ceildiv

from ..compilation import wrap_compilation

slang = wrap_compilation(
    lambda num_spherical_harmonics: slangtorch.loadModule(
        str(Path(__file__).parent / "project_vertices.slang"),
        defines={"NUM_SPHERICAL_HARMONICS": num_spherical_harmonics},
        verbose=True,
    )
)


BLOCK_SIZE = 256


class ProjectVertices(torch.autograd.Function):
    @record_function("project_vertices_forward")
    @staticmethod
    def forward(
        ctx,
        vertices: Float32[Tensor, "vertex xyz=3"],
        spherical_harmonics: Float32[Tensor, "sh vertex rgb=3"],
        extrinsics: Float32[Tensor, "4 4"],
        intrinsics: Float32[Tensor, "3 3"],
        active_sh: int = 0,
    ):
        device = vertices.device
        v, _ = vertices.shape
        kwargs = {"device": device, "dtype": torch.float32}

        extrinsics_inv = torch.inverse(extrinsics)
        cam_pos = extrinsics_inv[:3, 3]  # (3,)

        out_vertices = torch.empty((v, 2), **kwargs)
        out_depths = torch.empty((v,), **kwargs)
        out_colors = torch.empty((v, 3), **kwargs)

        dummy_grad = torch.empty(
            (), dtype=torch.float32, device=spherical_harmonics.device
        )

        slang(16).project_vertices(
            vertices=vertices,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            camPos=cam_pos,
            activeSH=active_sh,
            outVertices=out_vertices,
            outDepths=out_depths,
            outColors=out_colors,
            sphericalHarmonics=(spherical_harmonics, dummy_grad),
        ).launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(v, BLOCK_SIZE), 1, 1),
        )

        ctx.save_for_backward(
            vertices,
            spherical_harmonics,
            extrinsics,
            intrinsics,
            cam_pos,
            out_vertices,
            out_depths,
            out_colors,
        )
        ctx.active_sh = active_sh

        # The PyTorch autograd system doesn't work if you return dataclasses, so we
        # return a tuple and wrap this function instead.
        return (out_vertices, out_depths, out_colors)

    @record_function("project_vertices_backward")
    @staticmethod
    def backward(
        ctx,
        out_vertices_grad: Float32[Tensor, "vertex xy=2"],
        out_depths_grad: Float32[Tensor, " vertex"],
        out_colors_grad: Float32[Tensor, "vertex rgb=3"],
    ):
        # Slang doesn't support non-contiguous input gradients.
        out_vertices_grad = out_vertices_grad.contiguous()
        out_depths_grad = out_depths_grad.contiguous()
        out_colors_grad = out_colors_grad.contiguous()

        # Retrieve the saved tensors and non-tensor metadata.
        (
            vertices,
            spherical_harmonics,
            extrinsics,
            intrinsics,
            cam_pos,
            out_vertices,
            out_depths,
            out_colors,
        ) = ctx.saved_tensors
        active_sh = ctx.active_sh

        v, _ = vertices.shape
        harmonics_grad = torch.zeros_like(spherical_harmonics)

        # Note: Don't confuse the gradients for the input vertices (3D, defined here)
        # with the gradients for the output vertices (2D, input to the backward pass).
        vertices_grad = torch.zeros_like(vertices)
        kernel_with_args = slang(16).project_vertices.bwd(
            vertices=(vertices, vertices_grad),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            camPos=cam_pos,
            activeSH=active_sh,
            outVertices=(out_vertices, out_vertices_grad),
            outDepths=(out_depths, out_depths_grad),
            outColors=(out_colors, out_colors_grad),
            sphericalHarmonics=(spherical_harmonics, harmonics_grad),
        )
        kernel_with_args.launchRaw(
            blockSize=(BLOCK_SIZE, 1, 1),
            gridSize=(ceildiv(v, BLOCK_SIZE), 1, 1),
        )

        # This seems to be necessary to avoid memory leaks.
        del ctx
        del vertices
        del spherical_harmonics
        del extrinsics
        del intrinsics
        del cam_pos
        del out_vertices
        del out_depths
        del out_colors

        return vertices_grad, harmonics_grad, None, None, None


# Wrap the above function so that it returns a dataclass.
project_vertices: ProjectVerticesFn = lambda *args, **kwargs: ProjectedVertices(
    *ProjectVertices.apply(*args, **kwargs)
)
