import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def compute_compositing_weights(
    alpha: Float[Tensor, "*batch sample"],
) -> Float[Tensor, "*batch sample"]:
    # Compute occlusion for each sample. The 1e-10 is from the original NeRF.
    shifted_alpha = torch.cat(
        (torch.ones_like(alpha[..., :1]), 1 - alpha[..., :-1]),
        dim=-1,
    )
    occlusion = torch.cumprod(shifted_alpha, dim=-1)

    # Combine alphas with occlusion effects to get the final weights.
    return alpha * occlusion


def composite(
    alphas: Float[Tensor, " layer"],
    colors: Float[Tensor, "layer channel"],
) -> Float[Tensor, " channel"]:
    weights = compute_compositing_weights(alphas)
    return einsum(weights, colors, "l, l c -> c")


@torch.no_grad()
def custom_backward(
    alphas: Float[Tensor, " layer"],
    colors: Float[Tensor, "layer channel"],
    rgb_ray: Float[Tensor, " channel"],
    output_gradient: Float[Tensor, " channel"],
) -> tuple[
    Float[Tensor, " layer"],  # alpha gradients
    Float[Tensor, "layer channel"],  # color gradients
]:
    alpha_gradients = torch.zeros_like(alphas)
    color_gradients = torch.zeros_like(colors)

    rgb_ray2 = 0
    t = 1
    for i, (alpha, rgb) in enumerate(zip(alphas, colors)):
        weight = alpha * t
        rgb_ray2 += weight * rgb

        suffix = rgb_ray - rgb_ray2
        color_gradients[i] = weight * output_gradient
        alpha_gradients[i] = einsum(
            output_gradient, t * rgb - suffix / (1 - alpha), "c, c ->"
        )

        t *= 1 - alpha

    return alpha_gradients, color_gradients


if __name__ == "__main__":
    NUM_LAYERS = 5
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    alphas = torch.rand(NUM_LAYERS, device=device, generator=generator).requires_grad_(
        True
    )
    colors = torch.rand(
        (NUM_LAYERS, 3), device=device, generator=generator
    ).requires_grad_(True)
    output_gradient = torch.rand((3,), device=device, generator=generator)

    # Compute the ground-truth gradients.
    output_color = composite(alphas, colors)
    alpha_gradients, color_gradients = torch.autograd.grad(
        output_color,
        (alphas, colors),
        output_gradient,
    )

    # Compute the forward gradients.
    alpha_gradients_custom, color_gradients_custom = custom_backward(
        alphas,
        colors,
        output_color,
        output_gradient,
    )

    a = 1
