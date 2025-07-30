import itertools

import torch
from jaxtyping import Float, Float32
from torch import Tensor


def lexicographic_sort(tensor, round=None):
    """
    Sorts a tensor lexicographically in (x, y, z) order.
    """
    if round is not None:
        round_value = 10**round
        tensor_round = torch.round(tensor * round_value) / round_value
    else:
        tensor_round = tensor

    perm = torch.arange(tensor.size(0), device=tensor.device)
    for i in [2, 1, 0]:  # Sorting by (z, y, x) in reverse order
        # _, order = torch.sort(tensor[:, i], stable=True)
        _, order = torch.sort(tensor_round[:, i], stable=True)
        tensor = tensor[order]
        tensor_round = tensor_round[order]
        perm = perm[order]
    return tensor, perm


def sort_mesh_vertices(vertices, faces, round=None):
    """
    Sorts the vertices of a triangle mesh in lexicographic order (x, y, z),
    permutes the faces so that each triplet is in increasing order,
    and updates the face indices accordingly.

    Parameters:
        vertices: A (V, 3) tensor of vertex positions on CUDA.
        faces: A (F, 3) tensor of face indices on CUDA.

    Returns:
        A new dictionary with sorted vertices and updated faces.
    """
    # Sort vertices lexicographically
    vertices, perm = lexicographic_sort(vertices, round=round)

    # Compute inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=vertices.device)

    # Update face indices
    faces = inv_perm[faces]

    # Ensure each face is in increasing order
    faces, _ = torch.sort(faces, dim=1)

    # Sort faces lexicographically
    faces, _ = lexicographic_sort(faces)

    return vertices, faces, perm


def align_meshes(mesh1_vertices, mesh1_faces, mesh2_vertices, mesh2_faces, tol=0.01):
    """
    Aligns the vertices of two meshes so that they are in the same order.
    Assumes that mesh1_vertices and mesh2_vertices are already sorted (e.g. using
    lexicographic_sort). First, it vectorizes the comparison: for each index i, if
    |mesh1[i] - mesh2[i]| <= tol in all coordinates, that vertex is directly matched.
    Then, for the remaining unmatched indices, it searches only among the unmatched
    vertices in mesh2 to find a match within tolerance.

    Parameters:
        mesh1_vertices: (V, 3) tensor of vertices for mesh 1.
        mesh1_faces:    (F, 3) tensor of face indices for mesh 1.
        mesh2_vertices: (V, 3) tensor of vertices for mesh 2.
        mesh2_faces:    (F, 3) tensor of face indices for mesh 2.
        tol:            Tolerance used to consider two vertices as equal.

    Returns:
        mesh1_vertices, mesh1_faces, perm1, aligned_v2, aligned_faces2, perm2_aligned

        - mesh1_vertices, mesh1_faces: Mesh 1's vertices/faces (assumed sorted).
        - perm1: Identity permutation for mesh1 (since it’s already sorted).
        - aligned_v2, aligned_faces2: Mesh 2's vertices/faces re-ordered to exactly
          match mesh1.
        - perm2_aligned: A permutation tensor such that aligned_v2 =
          mesh2_vertices[perm2_aligned].
    """
    num_vertices = mesh1_vertices.size(0)

    # First pass: Vectorized comparison.
    # Compare each vertex pair (mesh1[i] and mesh2[i]) across all coordinates.
    diff = torch.abs(mesh1_vertices - mesh2_vertices)  # (V, 3)
    direct_matches = torch.all(diff <= tol, dim=1)  # (V,) boolean mask

    # Build new_order for mesh2: for indices with direct match, new_order[i] = i.
    new_order = [-1] * num_vertices
    used = [False] * num_vertices  # Marks which indices in mesh2 are already matched.

    indices = torch.arange(num_vertices, device=mesh1_vertices.device)
    direct_indices = indices[direct_matches].tolist()
    for i in direct_indices:
        new_order[i] = i
        used[i] = True

    # Second pass: Only unmatched indices need to be compared.
    unmatched = indices[~direct_matches].tolist()
    # We'll only search among unmatched indices from mesh2.
    unmatched_candidates = unmatched.copy()

    for i in unmatched:
        v1 = mesh1_vertices[i]
        # Create a tensor of candidate indices from unmatched_candidates.
        candidate_idxs = torch.tensor(
            unmatched_candidates, device=mesh1_vertices.device, dtype=torch.long
        )
        v2_candidates = mesh2_vertices[candidate_idxs]  # (num_candidates, 3)
        # Compute differences between v1 and all candidate vertices.
        diff_candidates = torch.abs(v1 - v2_candidates)  # (num_candidates, 3)
        within_tol = torch.all(diff_candidates <= tol, dim=1)  # (num_candidates,)
        matching_candidates = candidate_idxs[within_tol]
        if matching_candidates.numel() == 0:
            raise ValueError(
                f"Couldn't match vertex {i} from mesh1 in mesh2 within tolerance {tol}."
            )
        # Choose the first matching candidate.
        chosen = int(matching_candidates[0].item())
        new_order[i] = chosen
        unmatched_candidates.remove(chosen)

    new_order_tensor = torch.tensor(
        new_order, device=mesh2_vertices.device, dtype=torch.long
    )

    # Reorder mesh2's vertices using the found permutation.
    aligned_v2 = mesh2_vertices[new_order_tensor]

    # Update mesh2's face indices: compute inverse permutation.
    inv_new_order = torch.empty_like(new_order_tensor)
    inv_new_order[new_order_tensor] = torch.arange(
        num_vertices, device=mesh2_vertices.device
    )
    aligned_faces2 = inv_new_order[mesh2_faces]

    # Optionally, sort each face's indices in increasing order,
    # then lexicographically sort the faces (using your lexicographic_sort).
    aligned_faces2, _ = torch.sort(aligned_faces2, dim=1)
    aligned_faces2, _ = lexicographic_sort(aligned_faces2)

    # For mesh1, the permutation is simply the identity.
    perm1 = torch.arange(num_vertices, device=mesh1_vertices.device)

    return (
        mesh1_vertices,
        mesh1_faces,
        perm1,
        aligned_v2,
        aligned_faces2,
        new_order_tensor,
    )


def filter_mesh_faces(vertices, faces, occupancy, tol=1e-6):
    """
    vertices: (V, 3) tensor of vertex coordinates.
    faces: (F, 3) tensor of vertex indices.
    occupancy: (D, H, W) tensor where occupancy[z, y, x] indicates that
               the voxel [x:x+1] x [y:y+1] x [z:z+1] is occupied.

    For each face (assumed to have vertices on voxel boundaries), we:
      1. Round each vertex coordinate to the nearest integer.
      2. For each coordinate of each vertex, define its candidate voxel indices as
         {r-1, r}.
      3. For each face and each coordinate, take the intersection of the candidate sets.
         In practice, if all three rounded values are equal then the candidate set is
         {r-1, r}. If the rounded values differ by 1 (i.e. two vertices yield r and one
         yields r+1), then the only candidate common to all is r.
      4. Form the Cartesian product (up to 8 candidate voxels) and check occupancy.

    If at least one candidate voxel for the face is occupied, the face is kept.
    Otherwise, the face is deleted.

    Returns:
      new_faces: faces (with the same vertex indices) that pass the test.
    """
    # Number of faces.
    F = faces.shape[0]

    # Gather face vertex coordinates. Shape: (F, 3, 3)
    face_vertices = vertices[faces]

    # Round the vertex coordinates (each face’s vertex should be near an integer).
    # This gives an integer tensor of shape (F, 3, 3)
    face_r = torch.round(face_vertices).long()

    # Split into coordinates: note that occupancy is indexed as [z, y, x]
    face_rx = face_r[..., 0]  # (F, 3)
    face_ry = face_r[..., 1]  # (F, 3)
    face_rz = face_r[..., 2]  # (F, 3)

    face_f = torch.floor(face_vertices).long()  # (F, 3, 3)
    face_fx = face_f[..., 0]  # (F, 3)
    face_fy = face_f[..., 1]  # (F, 3)
    face_fz = face_f[..., 2]  # (F, 3)

    face_i = torch.abs(face_vertices - face_r.float()) < tol  # (F, 3, 3)
    face_ix = face_i[..., 0]  # (F, 3)
    face_iy = face_i[..., 1]  # (F, 3)
    face_iz = face_i[..., 2]  # (F, 3)

    # If close to integer, then use rx-1, else just floor
    vertex_candidate_x1 = torch.where(face_ix, face_rx - 1, face_fx)  # (F, 3)
    vertex_candidate_x2 = torch.where(face_ix, face_rx, face_fx)  # (F, 3)
    vertex_candidate_y1 = torch.where(face_iy, face_ry - 1, face_fy)  # (F, 3)
    vertex_candidate_y2 = torch.where(face_iy, face_ry, face_fy)  # (F, 3)
    vertex_candidate_z1 = torch.where(face_iz, face_rz - 1, face_fz)  # (F, 3)
    vertex_candidate_z2 = torch.where(face_iz, face_rz, face_fz)  # (F, 3)

    candidate_x1 = torch.max(vertex_candidate_x1, dim=1)[0]  # (F,)
    candidate_x2 = torch.min(vertex_candidate_x2, dim=1)[0]  # (F,)
    candidate_y1 = torch.max(vertex_candidate_y1, dim=1)[0]
    candidate_y2 = torch.min(vertex_candidate_y2, dim=1)[0]
    candidate_z1 = torch.max(vertex_candidate_z1, dim=1)[0]
    candidate_z2 = torch.min(vertex_candidate_z2, dim=1)[0]

    # There should only be 1 candidate voxel for each face.
    if not torch.allclose(candidate_x1, candidate_x2):
        raise ValueError("Candidate x-coordinates are not equal.")
    assert torch.allclose(candidate_x1, candidate_x2)
    assert torch.allclose(candidate_y1, candidate_y2)
    assert torch.allclose(candidate_z1, candidate_z2)

    # The unique candidate voxel for each face.
    candidate_x = candidate_x1
    candidate_y = candidate_y1
    candidate_z = candidate_z1

    # Get occupancy dimensions. Remember: occupancy is indexed as [z, y, x]
    D, H, W = occupancy.shape

    # Check candidate indices for bounds.
    valid = (
        (candidate_x >= 0)
        & (candidate_x < W)
        & (candidate_y >= 0)
        & (candidate_y < H)
        & (candidate_z >= 0)
        & (candidate_z < D)
    )  # shape: (F,)

    # Compute linear indices for occupancy lookup (linear index = z * (H*W) + y * W + x)
    lin_idx = candidate_z * (H * W) + candidate_y * W + candidate_x  # shape: (F,)

    occupancy_flat = occupancy.reshape(-1)
    occ_vals = torch.zeros(F, dtype=torch.bool, device=occupancy.device)

    ### This type of indexing works if the inner most things have same shape both sides
    occ_vals[valid] = occupancy_flat[lin_idx[valid]]

    # Keep the face if its candidate voxel is occupied.
    face_valid = occ_vals
    new_faces = faces[face_valid]

    return new_faces


def prune_unused_vertices(vertices, faces):
    """
    Removes vertices that are not referenced in the faces tensor and reindexes faces
    accordingly.

    Parameters:
      vertices: (V, 3) tensor of vertex coordinates.
      faces: (F, 3) tensor of vertex indices.

    Returns:
      new_vertices: (V', 3) tensor containing only the vertices referenced by faces.
      new_faces: (F, 3) tensor with reindexed vertex indices.
    """
    # Get the unique vertex indices used in faces and the inverse mapping.
    # unique_indices: sorted unique vertex indices referenced by faces.
    # inverse_indices: a 1D tensor such that unique_indices[inverse_indices]
    # reconstructs faces.flatten().
    unique_indices, inverse_indices = torch.unique(
        faces, sorted=True, return_inverse=True
    )

    # Create the new vertex tensor.
    new_vertices = vertices[unique_indices]

    # Reshape the inverse mapping to have the same shape as faces.
    new_faces = inverse_indices.view(faces.shape)

    return new_vertices, new_faces, unique_indices


def filter_mesh(vertices, faces, occupancy, tol=1e-6):
    faces = filter_mesh_faces(vertices, faces, occupancy, tol)

    vertices, faces, unique_indices = prune_unused_vertices(vertices, faces)

    return vertices, faces, unique_indices


def combine_duplicate_vertices(vertices, faces, tol=1e-6):
    """
    vertices: (V, 3) tensor of vertex coordinates (already sorted).
    faces: (F, 3) tensor of vertex indices.
    tol: tolerance for considering two vertices as duplicates.

    This function merges adjacent vertices that are within tol of each other.
    It then builds a new vertex list (keeping the first vertex in each group)
    and updates face indices accordingly.

    Returns:
      new_vertices: (V_new, 3) tensor of unique vertices.
      new_faces: (F, 3) tensor of re-indexed face indices.
    """
    V = vertices.shape[0]
    if V == 0:
        return vertices, faces

    # Compute differences between consecutive vertices.
    # If all coordinate differences are less than tol, then the vertices are duplicates.
    diffs = torch.abs(vertices[1:] - vertices[:-1])
    # duplicates is True for vertex i+1 if it is a duplicate of vertex i.
    duplicates = torch.all(diffs < tol, dim=1)  # shape: (V-1,)

    # The first vertex is always unique.
    unique_mask = torch.cat([torch.tensor([True], device=vertices.device), ~duplicates])
    new_vertices = vertices[unique_mask]

    # Build a mapping from old vertex indices to new vertex indices.
    # For duplicate vertices, they get the same new index as their first occurrence.
    mapping = torch.cumsum(unique_mask.to(torch.int64), dim=0) - 1
    new_faces = mapping[faces]

    # Remove degenerate faces that have any repeated vertex.
    valid_face = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 0] != new_faces[:, 2])
        & (new_faces[:, 1] != new_faces[:, 2])
    )
    new_faces = new_faces[valid_face]

    return new_vertices, new_faces


def enforce_level_set_margin(sdf_grid, level_sets, tol=1e-1):
    """
    Adjusts sdf_grid in place so that every value is at least tol away from each level
    set value.

    Parameters:
      sdf_grid (torch.Tensor): A tensor (of any shape) of SDF values.
      level_sets (torch.Tensor): A 1D tensor containing level set values (unsorted).
      tol (float): The minimum distance required (default: 1e-1).

    Returns:
      torch.Tensor: The adjusted sdf_grid.
    """
    # Iterate through each level set value.
    for level in level_sets:
        # Compute difference between each grid value and the current level set.
        diff = sdf_grid - level
        # Identify grid locations that are too close (within tol) to the level set.
        mask = diff.abs() < tol
        if mask.any():
            # If a grid value is below the level, adjust it to level - tol.
            # If it is above (or exactly equal), adjust it to level + tol.
            sdf_grid[mask] = torch.where(
                sdf_grid[mask] < level, level - tol, level + tol
            )
    return sdf_grid


class Index3DGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, vertices_int):
        """
        Forward: Extracts values from a multi-dimensional grid using integer indexing.
        We assume grid is using zyx indexing!!!

        Arguments:
        - grid: Tensor of shape (D, H, W, ...), where `...` are arbitrary trailing
          dimensions.
        - vertices_int: Tensor of shape (V, 3), containing integer indices for (D, H, W)
          dimensions.

        Returns:
        - Indexed values: Tensor of shape (V, ...) matching the trailing dimensions of
          grid.
        """
        ctx.save_for_backward(vertices_int, torch.tensor(grid.shape))

        # Extract values using advanced indexing
        indexed_values = grid[
            vertices_int[:, 2], vertices_int[:, 1], vertices_int[:, 0]
        ]  # Shape: (V, ...)

        return indexed_values

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors.
        vertices_int, grid_shape_tensor = ctx.saved_tensors
        grid_shape = tuple(grid_shape_tensor.tolist())

        # Create a zero tensor with the same shape as the input grid.
        grad_grid = torch.zeros(
            grid_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        grad_grid = grad_grid.contiguous()

        # Accumulate gradients: for each vertex index, add the corresponding
        # grad_output. Note that we use the same ordering (z, y, x) as in the forward.
        grad_grid.index_put_(
            (vertices_int[:, 2], vertices_int[:, 1], vertices_int[:, 0]),
            grad_output,
            accumulate=True,
        )

        # No gradient for vertices_int (which is an integer tensor)
        return grad_grid, None


def sample_grid(
    grid: Float32[Tensor, "i j k channel"],
    samples: Float[Tensor, "*batch xyz=3"],
) -> Float32[Tensor, "*batch channel"]:
    """Same as sample_grid_original, except that it's differentiable w.r.t. samples.

    (code from Claude)
    """
    # Expect grid to be zyx order
    # Also we expect the samples to be in [0, 1]

    # Get grid dimensions
    i, j, k, channels = grid.shape
    dims = [k, j, i]  # Note: xyz order

    # Scale samples by grid dimensions
    scaled_samples = samples * torch.tensor(
        [(d - 1) for d in dims], device=samples.device
    )

    # Get integer indices for the 8 surrounding points
    floor = torch.floor(scaled_samples).long()
    ceil = floor + 1

    # Clamp indices to valid range
    max_indices = [d - 1 for d in dims]  # -2 because we need room for ceil
    floor = torch.stack(
        [torch.clamp(floor[..., i], min=0, max=max_indices[i]) for i in range(3)],
        dim=-1,
    )
    ceil = torch.stack(
        [torch.clamp(ceil[..., i], min=0, max=max_indices[i]) for i in range(3)],
        dim=-1,
    )

    # Calculate interpolation weights
    alpha = scaled_samples - floor.float()

    # Initialize result tensor
    *batch_dims, _ = samples.shape
    result = torch.zeros((*batch_dims, channels), device=grid.device, dtype=grid.dtype)

    # Perform trilinear interpolation
    for x in range(2):
        for y in range(2):
            for z in range(2):
                # Compute indices
                x_idx = ceil[..., 0] if x else floor[..., 0]
                y_idx = ceil[..., 1] if y else floor[..., 1]
                z_idx = ceil[..., 2] if z else floor[..., 2]

                # Compute weights
                wx = alpha[..., 0] if x else (1 - alpha[..., 0])
                wy = alpha[..., 1] if y else (1 - alpha[..., 1])
                wz = alpha[..., 2] if z else (1 - alpha[..., 2])
                weight = wx.unsqueeze(-1) * wy.unsqueeze(-1) * wz.unsqueeze(-1)

                # Get values and accumulate
                values = grid[z_idx, y_idx, x_idx]
                result += weight * values

    return result


def gradient_magnitude(
    sdf_grid: Float[Tensor, "i j k"],
    minima: Float[Tensor, "xyz=3"],
    maxima: Float[Tensor, "xyz=3"],
) -> Tensor:
    grid_size = sdf_grid.shape[0]
    edge_length = (maxima - minima) / (grid_size - 1)

    def shift_slice(offset):
        return slice(offset, offset + grid_size - 1)

    gradient_magnitudes = []
    for dx, dy, dz in itertools.product([0, 1], repeat=3):
        s_vertex = (shift_slice(dx), shift_slice(dy), shift_slice(dz))
        s_x = (shift_slice(1 - dx), shift_slice(dy), shift_slice(dz))
        s_y = (shift_slice(dx), shift_slice(1 - dy), shift_slice(dz))
        s_z = (shift_slice(dx), shift_slice(dy), shift_slice(1 - dz))

        vertex = sdf_grid[s_vertex]

        # Differences in x, y, and z directions
        x_difference = sdf_grid[s_x] - vertex
        y_difference = sdf_grid[s_y] - vertex
        z_difference = sdf_grid[s_z] - vertex

        # Compute the gradient
        gradient = (
            torch.stack([x_difference, y_difference, z_difference], dim=-1)
            / edge_length
        )
        gradient_magnitudes.append(torch.norm(gradient, p=2, dim=-1))

    return torch.stack(gradient_magnitudes, dim=-1)  # shape: (i-1, j-1, k-1, 8)


def eikonal_loss(field) -> Tensor:
    gradient_magnitudes = gradient_magnitude(
        field,
        0,
        1,
    )
    loss = ((gradient_magnitudes - 1) ** 2).mean(dim=-1)

    return loss
