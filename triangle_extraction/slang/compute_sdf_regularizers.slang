static const int NUM_EIKONAL_SAMPLES = 1;

[Differentiable]
float stableNorm(float a, float b, float c) {
  // This will not compile on some nodes with a let here.
  const float scale = max(max(abs(a), abs(b)), abs(c));
  if (scale < 1e-10) {
    return 0;
  }
  let invScale = 1 / scale;
  let scaledA = invScale * a;
  let scaledB = invScale * b;
  let scaledC = invScale * c;
  let sum = scaledA * scaledA + scaledB * scaledB + scaledC * scaledC;
  return scale * sqrt(sum);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void computeSDFRegularizers(
    const DiffTensorView gridSignedDistances,
    const TensorView<int> voxelNeighbors,
    const TensorView<int> voxelLowerCorners,
    const TensorView<int> voxelUpperCorners,
    const int voxelX,
    const int voxelY,
    const int voxelZ,
    const DiffTensorView eikonalLossPos,
    const DiffTensorView eikonalLossNeg,
    const DiffTensorView curvatureLoss,
) {
  // Skip out-of-bounds voxels.
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  let numVoxels = voxelUpperCorners.size(1);
  if (index >= numVoxels) {
    return;
  }

  // Load the current voxel's signed distances.
  let d000 = gridSignedDistances[voxelLowerCorners[uint2(0, index)]]; // origin
  let d001 = gridSignedDistances[voxelLowerCorners[uint2(1, index)]]; // +x
  let d010 = gridSignedDistances[voxelLowerCorners[uint2(2, index)]]; // +y
  let d100 = gridSignedDistances[voxelLowerCorners[uint2(3, index)]]; // +z
  let d111 = gridSignedDistances[voxelUpperCorners[uint2(0, index)]]; // +xyz
  let d110 = gridSignedDistances[voxelUpperCorners[uint2(1, index)]]; // +yz
  let d101 = gridSignedDistances[voxelUpperCorners[uint2(2, index)]]; // +xz
  let d011 = gridSignedDistances[voxelUpperCorners[uint2(3, index)]]; // +xy

  // Compute the Eikonal loss.
  const float3 sampleLocations[NUM_EIKONAL_SAMPLES] = {
    float3(0.5, 0.5, 0.5),
  };
  float eikonalSumPos = 0;
  float eikonalSumNeg = 0;
  for (int i = 0; i < NUM_EIKONAL_SAMPLES; i++) {
    let tx = sampleLocations[i][0];
    let ty = sampleLocations[i][1];
    let tz = sampleLocations[i][2];
    let tx1 = 1 - tx;
    let ty1 = 1 - ty;
    let tz1 = 1 - tz;

    // Compute the magnitude of the trilinearly interpolated signed distance's gradient.
    let dx = ty1 * tz1 * (d001 - d000) + ty * tz1 * (d011 - d010) +
             ty1 * tz * (d101 - d100) + ty * tz * (d111 - d110);
    let dy = tx1 * tz1 * (d010 - d000) + tx * tz1 * (d011 - d001) +
             tx1 * tz * (d110 - d100) + tx * tz * (d111 - d101);
    let dz = tx1 * ty1 * (d100 - d000) + tx * ty1 * (d101 - d001) +
             tx1 * ty * (d110 - d010) + tx * ty * (d111 - d011);

    // Compute the Eikonal loss as the square of the difference between the above
    // gradient magnitude and 1. Note that we scale the gradient magnitude by the grid
    // size.
    let delta = stableNorm(voxelX * dx, voxelY * dy, voxelZ * dz) - 1;
    if (delta < 0) {
      eikonalSumNeg += delta * delta;
    }
    if (delta > 0) {
      eikonalSumPos += delta * delta;
    }
  }
  eikonalLossPos[index] = eikonalSumPos / NUM_EIKONAL_SAMPLES;
  eikonalLossNeg[index] = eikonalSumNeg / NUM_EIKONAL_SAMPLES;

  // Compute the curvature loss. The curvature loss looks like a 3D Laplacian filter:
  //
  // Top layer:
  // -- -- --
  // --  1 --
  // -- -- --
  //
  // Middle layer:
  // --  1 --
  //  1 -6  1
  // --  1 --
  //
  // Bottom layer:
  // -- -- --
  // --  1 --
  // -- -- --
  //
  // The value that corresponds to the -6 weight is the current voxel's d111 corner.
  // Three of the values that correspond to 1 weights are the current voxel's d110,
  // d101, and d011 corners. The other three values that correspond to 1 weights may be
  // absent, since they're associated with the subvoxel with offset (1, 1, 1) from the
  // current voxel. If the values are absent, we make the -6 weight smaller.

  // Initialize the curvature sum with the signed distances that are always present.
  float curvatureSum = d110 + d101 + d011;
  int numCounted = 3;

  // Add whatever signed distances are present in the neighboring subvoxel.
  let subvoxelIndex = voxelNeighbors[uint2(6, index)];
  for (int i = 1; i < 4; i++) {
    let subvoxelCornerIndex = voxelLowerCorners[uint2(i, subvoxelIndex)];
    if (subvoxelCornerIndex >= 0) {
      curvatureSum += gridSignedDistances[subvoxelCornerIndex];
      numCounted++;
    }
  }

  // Weight the middle signed distance by the number of present signed distances.
  curvatureSum -= numCounted * d111;

  // Turn the result into an L2 loss on curvature.
  curvatureLoss[index] = curvatureSum * curvatureSum;
}
