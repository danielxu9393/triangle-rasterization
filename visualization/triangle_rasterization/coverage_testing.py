import torch
from einops import reduce
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import trange


def compute_barycentric_coordinates(
    points: Float[Tensor, "*#batch xy=2"],
    triangles: Float[Tensor, "*#batch corner=3 xy=2"],
) -> Float[Tensor, "*batch uvw=3"]:
    a, b, c = triangles.unbind(dim=-2)

    v0_x, v0_y = (b - a).unbind(dim=-1)
    v1_x, v1_y = (c - a).unbind(dim=-1)
    v2_x, v2_y = (points - a).unbind(dim=-1)
    coefficient = 1 / (v0_x * v1_y - v1_x * v0_y)
    v = (v2_x * v1_y - v1_x * v2_y) * coefficient
    w = (v0_x * v2_y - v2_x * v0_y) * coefficient
    u = 1 - v - w
    return torch.stack((u, v, w), dim=-1)


def compute_edge_distances(
    points: Float[Tensor, "*#batch xy=2"],
    triangles: Float[Tensor, "*#batch corner=3 xy=2"],
) -> Float[Tensor, "*batch uvw=3"]:
    # Extract vertices
    v0 = triangles[..., 0, :]
    v1 = triangles[..., 1, :]
    v2 = triangles[..., 2, :]

    # Compute triangle orientation (positive for CCW, negative for CW)
    orientation = (v1[..., 0] - v0[..., 0]) * (v2[..., 1] - v0[..., 1]) - (
        v2[..., 0] - v0[..., 0]
    ) * (v1[..., 1] - v0[..., 1])
    sign_flip = torch.sign(orientation).unsqueeze(-1)

    # Edge v0->v1
    dx01 = v1[..., 0] - v0[..., 0]
    dy01 = v1[..., 1] - v0[..., 1]
    num0 = dx01 * (v0[..., 1] - points[..., 1]) - (v0[..., 0] - points[..., 0]) * dy01
    dist0 = num0 / torch.sqrt(dx01 * dx01 + dy01 * dy01)

    # Edge v1->v2
    dx12 = v2[..., 0] - v1[..., 0]
    dy12 = v2[..., 1] - v1[..., 1]
    num1 = dx12 * (v1[..., 1] - points[..., 1]) - (v1[..., 0] - points[..., 0]) * dy12
    dist1 = num1 / torch.sqrt(dx12 * dx12 + dy12 * dy12)

    # Edge v2->v0
    dx20 = v0[..., 0] - v2[..., 0]
    dy20 = v0[..., 1] - v2[..., 1]
    num2 = dx20 * (v2[..., 1] - points[..., 1]) - (v2[..., 0] - points[..., 0]) * dy20
    dist2 = num2 / torch.sqrt(dx20 * dx20 + dy20 * dy20)

    distances = torch.stack([dist0, dist1, dist2], dim=-1)
    return distances * sign_flip


def compute_coverage(
    points: Float[Tensor, "*#batch xy=2"],
    triangles: Float[Tensor, "*#batch corner=3 xy=2"],
) -> Float[Tensor, "*batch"]:
    distances = compute_edge_distances(points, triangles)
    return (0.5 - distances).clip(min=0, max=1).prod(dim=-1)


def compute_binary_coverage(
    points: Float[Tensor, "*#batch xy=2"],
    triangles: Float[Tensor, "*#batch corner=3 xy=2"],
) -> Bool[Tensor, "*batch"]:
    return (compute_edge_distances(points, triangles) < 0).all(dim=-1)


def get_image_xy(
    image_shape: tuple[int, int],
    device: torch.device,
) -> Float[Tensor, "height width xy=2"]:
    h, w = image_shape
    x = torch.arange(w, device=device)
    y = torch.arange(h, device=device)
    return torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1) + 0.5


RESOLUTION = 128
CHUNK_SIZE = 512
NUM_CHUNKS = 256

if __name__ == "__main__":
    device = torch.device("cuda")

    xy = get_image_xy((RESOLUTION, RESOLUTION), device) / RESOLUTION
    one_sample = torch.full((2,), 0.5, device=device)

    true_coverages = []
    estimated_coverages = []

    label = "Up To 3x3 Triangle"
    for _ in trange(NUM_CHUNKS, desc="Simulating"):
        a = torch.empty((CHUNK_SIZE, 2), device=device).uniform_(-1, 2)
        b = torch.empty((CHUNK_SIZE, 2), device=device).uniform_(-1, 2)
        c = torch.empty((CHUNK_SIZE, 2), device=device).uniform_(-1, 2)
        triangles = torch.stack((a, b, c), dim=1)

        # Approximate the true coverage by sampling on a 128x128 grid.
        true_coverage = compute_binary_coverage(
            xy[:, :, None],
            triangles,
        )
        true_coverage = reduce(true_coverage.float(), "h w b -> b", "mean")

        # Compute estimated coverage.
        estimated_coverage = compute_coverage(one_sample, triangles)

        true_coverages.append(true_coverage)
        estimated_coverages.append(estimated_coverage)

    true_coverages = torch.cat(true_coverages)
    estimated_coverages = torch.cat(estimated_coverages)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.grid(color="#eeeeee", zorder=0)

    plt.scatter(
        true_coverages.cpu().numpy(),
        estimated_coverages.cpu().numpy(),
        alpha=0.01,
        s=2,
    )
    plt.xlabel("True Coverage")
    plt.ylabel("Estimated Coverage")

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.plot([0, 1], [0, 1], "k-", alpha=0.75, zorder=1_000_000_000)

    plt.title(f"Estimated vs. True Coverage: {label}")

    plt.savefig("test.png")

    a = 1
