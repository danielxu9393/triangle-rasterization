[AutoPyBindCUDA]
[CUDAKernel]
void countTriangleVertices(
    const TensorView<float> signedDistances,
    const TensorView<int> lowerCorners,
    const TensorView<float> levelSets,
    TensorView<int> vertexCounts,
    TensorView<uint8_t> vertexCountsByLevelSet,
) {
  let index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= vertexCounts.size(0) - 1) {
    return;
  }

  // The order is (origin, +x, +y, +z).
  let d0 = signedDistances[lowerCorners[uint2(0, index)]];
  let dx = (lowerCorners[uint2(1, index)] >= 0)
               ? signedDistances[lowerCorners[uint2(1, index)]]
               : d0;
  let dy = (lowerCorners[uint2(2, index)] >= 0)
               ? signedDistances[lowerCorners[uint2(2, index)]]
               : d0;
  let dz = (lowerCorners[uint2(3, index)] >= 0)
               ? signedDistances[lowerCorners[uint2(3, index)]]
               : d0;

  int numVertices = 0;
  for (int levelSet = 0; levelSet < levelSets.size(0); levelSet++) {
    let iso = levelSets[levelSet];
    int numVerticesInLevelSet = 0;
    if ((d0 < iso && dx >= iso) || (dx < iso && d0 >= iso)) {
      numVerticesInLevelSet += 1;
    }
    if ((d0 < iso && dy >= iso) || (dy < iso && d0 >= iso)) {
      numVerticesInLevelSet += 1;
    }
    if ((d0 < iso && dz >= iso) || (dz < iso && d0 >= iso)) {
      numVerticesInLevelSet += 1;
    }

    vertexCountsByLevelSet[uint2(levelSet, index)] = uint8_t(numVerticesInLevelSet);
    numVertices += numVerticesInLevelSet;
  }

  vertexCounts[index] = numVertices;
}
