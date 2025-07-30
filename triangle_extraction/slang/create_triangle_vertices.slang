import "../../slang_common/large_arrays.slang";

struct Voxel : IDifferentiable {
  // The origin (always valid).
  float d0;
  float3 v0;
  float sh0[NUM_SPHERICAL_HARMONICS][3];

  // The axes (sometimes invalid).
  float d1[3];
  float3 v1[3];
  float sh1[3][NUM_SPHERICAL_HARMONICS][3];
}

[Differentiable]
Voxel loadVoxel(
    const int index,
    const DiffTensorView gridVertices,
    const DiffTensorView gridSignedDistances,
    const DiffTensorViewLarge gridSphericalHarmonics,
    const TensorView<int> voxelLowerCorners,
    const TensorView<int> voxelVertexCounts,
) {
  // Load the origin's signed distance (d0), vertex location (v0), and features (f0).
  let i0 = voxelLowerCorners[uint2(0, index)];
  let d0 = gridSignedDistances[i0];
  let v0 = float3(
      gridVertices[uint2(i0, 0)],
      gridVertices[uint2(i0, 1)],
      gridVertices[uint2(i0, 2)],
  );
  float sh0[NUM_SPHERICAL_HARMONICS][3];
  $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
    $for(j in Range(0, 3)) {
      // sh0[s][j] = gridSphericalHarmonics[s].load(uint2(i0, j));
      sh0[s][j] = gridSphericalHarmonics.load(uint3(s, i0, j));
    }
  }

  // Load the signed distance (d1), vertex location (v1), and features (f1) for the
  // +X, +Y, and +Z directions.
  float d1[3];
  float3 v1[3];
  float sh1[3][NUM_SPHERICAL_HARMONICS][3];
  for (int axis = 0; axis < 3; axis++) {
    if (voxelLowerCorners[uint2(axis + 1, index)] >= 0) {
      // Handle the case where the axis is valid.
      let i1 = voxelLowerCorners[uint2(axis + 1, index)];
      d1[axis] = gridSignedDistances[i1];
      v1[axis] = float3(
          gridVertices[uint2(i1, 0)],
          gridVertices[uint2(i1, 1)],
          gridVertices[uint2(i1, 2)],
      );
      $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
        $for(j in Range(0, 3)) {
          // sh1[axis][s][j] = gridSphericalHarmonics[s].load(uint2(i1, j));
          sh1[axis][s][j] = gridSphericalHarmonics.load(uint3(s, i1, j));
        }
      }
    } else {
      // Handle the case where the axis is invalid. Simply setting d1[axis] to d0
      // prevents a vertex from being placed.
      d1[axis] = d0;
    }
  }

  return { d0, v0, sh0, d1, v1, sh1 };
}

bool shouldComputeTriangleVertex(
    const Voxel voxel,
    const int axis,
    no_diff const float levelSet,
) {
  // Vertices always lie along voxel edges. A vertex is created wherever an edge's
  // endpoints' signed distances are on opposite sides of the level set.
  if (voxel.d0 < levelSet && voxel.d1[axis] < levelSet) {
    return false;
  }
  if (voxel.d0 >= levelSet && voxel.d1[axis] >= levelSet) {
    return false;
  }
  return true;
}

struct TriangleVertex : IDifferentiable {
  float3 v;
  float sh[NUM_SPHERICAL_HARMONICS][3];
  float d;
  uint8_t vertexType;
}

[Differentiable]
TriangleVertex computeTriangleVertex(
    const Voxel voxel,
    const int axis,
    no_diff const float levelSet,
) {
  // Compute the interpolation weight along the edge.
  // Detach removes gradients from voxel sdf -> mesh vertex
  let t = detach(
      (voxel.d0 == voxel.d1[axis])
          ? 0.5
          : clamp((levelSet - voxel.d0) / (voxel.d1[axis] - voxel.d0), 0, 1)
  );
  // let t = (voxel.d0 == voxel.d1[axis])
  //             ? 0.5
  //             : clamp((levelSet - voxel.d0) / (voxel.d1[axis] - voxel.d0), 0, 1);

  // Compute and the vertex location.
  // Gradients from voxel vertex -> mesh vertex.
  // The t here gives gradients from voxel sdf -> mesh vertex (but is detached)
  let v = voxel.v0 + t * (voxel.v1[axis] - voxel.v0);

  // Compute the feature values.
  float sh[NUM_SPHERICAL_HARMONICS][3];
  $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
    $for(j in Range(0, 3)) {
      sh[s][j] = voxel.sh0[s][j] + float(t) * (voxel.sh1[axis][s][j] - voxel.sh0[s][j]);
    }
  }

  // Compute the vertex's signed distance.
  // This gives gradients from voxel sdf -> output sdf.
  // The t here is another path from voxel sdf -> output sdf (but is detached)
  // but this path is a more complex function of the voxel sdf.
  let d = voxel.d0 + t * (voxel.d1[axis] - voxel.d0);

  return { v, sh, d, uint8_t(axis) };
}

[Differentiable]
void storeTriangleVertex(
    const int index,
    const TriangleVertex triangleVertex,
    const DiffTensorView triangleVertices,
    const DiffTensorViewLarge triangleSphericalHarmonics,
    const DiffTensorView triangleSignedDistances,
    const TensorView<uint8_t> triangleVertexTypes,
) {
  triangleVertices.storeOnce(uint2(index, 0), triangleVertex.v.x);
  triangleVertices.storeOnce(uint2(index, 1), triangleVertex.v.y);
  triangleVertices.storeOnce(uint2(index, 2), triangleVertex.v.z);
  $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
    $for(j in Range(0, 3)) {
      triangleSphericalHarmonics.storeOnce(uint3(s, index, j), triangleVertex.sh[s][j]);
    }
  }
  triangleSignedDistances.storeOnce(index, triangleVertex.d);
  triangleVertexTypes[index] = triangleVertex.vertexType;
}

[BackwardDerivative(createTriangleVerticesInnerBackward)]
void createTriangleVerticesInner(
    const int index,
    const DiffTensorView gridVertices,
    const DiffTensorView gridSignedDistances,
    const DiffTensorViewLarge gridSphericalHarmonics,
    const TensorView<int> voxelLowerCorners,
    const TensorView<int> voxelVertexCounts,
    const TensorView<float> levelSets,
    const DiffTensorView triangleVertices,
    const DiffTensorView triangleSignedDistances,
    const DiffTensorViewLarge triangleSphericalHarmonics,
    const TensorView<uint8_t> triangleVertexTypes,
) {
  // Load relevant grid information for the current voxel.
  let voxel = loadVoxel(
      index,
      gridVertices,
      gridSignedDistances,
      gridSphericalHarmonics,
      voxelLowerCorners,
      voxelVertexCounts
  );

  // Loop through level sets and axes. At most, there's one vertex per axis.
  var vertexIndex = voxelVertexCounts[index];
  for (int i = 0; i < levelSets.size(0); i++) {
    let levelSet = levelSets[i];
    for (int axis = 0; axis < 3; axis++) {
      // Skip past invalid vertices.
      if (!shouldComputeTriangleVertex(voxel, axis, levelSet)) {
        continue;
      }

      // Compute and store the triangle vertex.
      let triangleVertex = computeTriangleVertex(voxel, axis, levelSet);
      storeTriangleVertex(
          vertexIndex,
          triangleVertex,
          triangleVertices,
          triangleSphericalHarmonics,
          triangleSignedDistances,
          triangleVertexTypes,
      );

      // Update the vertex index outside the differentiable functions.
      vertexIndex++;
    }
  }
}

void createTriangleVerticesInnerBackward(
    const int index,
    const DiffTensorView gridVertices,
    const DiffTensorView gridSignedDistances,
    const DiffTensorViewLarge gridSphericalHarmonics,
    const TensorView<int> voxelLowerCorners,
    const TensorView<int> voxelVertexCounts,
    const TensorView<float> levelSets,
    const DiffTensorView triangleVertices,
    const DiffTensorView triangleSignedDistances,
    const DiffTensorViewLarge triangleSphericalHarmonics,
    const TensorView<uint8_t> triangleVertexTypes,
) {
  // Load relevant grid information for the current voxel.
  let voxel = loadVoxel(
      index,
      gridVertices,
      gridSignedDistances,
      gridSphericalHarmonics,
      voxelLowerCorners,
      voxelVertexCounts
  );

  // Initialize an empty differential.
  Voxel.Differential dVoxel = diffPair(voxel).d;

  // Loop through level sets and axes. At most, there's one vertex per axis.
  var vertexIndex = voxelVertexCounts[index];
  for (int i = 0; i < levelSets.size(0); i++) {
    let levelSet = levelSets[i];
    for (int axis = 0; axis < 3; axis++) {
      // Skip past invalid vertices.
      if (!shouldComputeTriangleVertex(voxel, axis, levelSet)) {
        continue;
      }

      // Backpropagate through computing and storing the triangle vertex.
      let triangleVertex = computeTriangleVertex(voxel, axis, levelSet);
      var dpTriangleVertex = diffPair(triangleVertex);
      bwd_diff(storeTriangleVertex)(
          vertexIndex,
          dpTriangleVertex,
          triangleVertices,
          triangleSphericalHarmonics,
          triangleSignedDistances,
          triangleVertexTypes,
      );

      var dpVoxel = diffPair(voxel);
      bwd_diff(computeTriangleVertex)(dpVoxel, axis, levelSet, dpTriangleVertex.d);

      // Accumulate the voxel differential. Defining dpVoxel outside the loops and
      // passing it to bwd_diff(computeTriangleVertex) would cause it to be overwritten,
      // which is why we need to do this.
      dVoxel.d0 += dpVoxel.d.d0;
      dVoxel.d1[axis] += dpVoxel.d.d1[axis];
      dVoxel.v0 += dpVoxel.d.v0;
      dVoxel.v1[axis] += dpVoxel.d.v1[axis];
      $for(s in Range(0, NUM_SPHERICAL_HARMONICS)) {
        $for(j in Range(0, 3)) {
          dVoxel.sh0[s][j] += dpVoxel.d.sh0[s][j];
          dVoxel.sh1[axis][s][j] += dpVoxel.d.sh1[axis][s][j];
        }
      }

      // Update the vertex index outside the differentiable functions.
      vertexIndex++;
    }
  }

  // Backpropagate through loading the voxel.
  bwd_diff(loadVoxel)(
      index,
      gridVertices,
      gridSignedDistances,
      gridSphericalHarmonics,
      voxelLowerCorners,
      voxelVertexCounts,
      dVoxel,
  );
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void createTriangleVertices(
    const DiffTensorView gridVertices,
    const DiffTensorView gridSignedDistances,
    const DiffTensorViewLarge gridSphericalHarmonics,
    const TensorView<int> voxelLowerCorners,
    const TensorView<int> voxelVertexCounts,
    const TensorView<float> levelSets,
    const DiffTensorView triangleVertices,
    const DiffTensorView triangleSignedDistances,
    const DiffTensorViewLarge triangleSphericalHarmonics,
    const TensorView<uint8_t> triangleVertexTypes,
) {
  // Skip out-of-bounds voxels.
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= voxelVertexCounts.size(0) - 1) {
    return;
  }

  // The inner function allows us to define a custom backward pass.
  createTriangleVerticesInner(
      index,
      gridVertices,
      gridSignedDistances,
      gridSphericalHarmonics,
      voxelLowerCorners,
      voxelVertexCounts,
      levelSets,
      triangleVertices,
      triangleSignedDistances,
      triangleSphericalHarmonics,
      triangleVertexTypes,
  );
}
