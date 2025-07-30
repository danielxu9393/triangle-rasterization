int logicalRightShift(const int x, const int shift) {
  // In Slang, int uses arithmetic right shift, and uint uses logical right shift. Thus,
  // to get a logical right shift for int, we have to cast to uint and back.
  return reinterpret<int>(reinterpret<uint>(x) >> shift);
}

[AutoPyBindCUDA]
[CUDAKernel]
void compute_vertex_occupancy(
    const TensorView<int> occupancy,
    TensorView<int> vertexOccupancy,
    TensorView<int> vertexCounts,
) {
  // Read the packed occupancy grid's dimensions.
  let iMax = occupancy.size(0);
  let jMax = occupancy.size(1);
  let kPackedMax = occupancy.size(2);

  // Guard against indices that are invalid in the vertex occupancy grid. Note that
  // the highest indices in the vertex occupancy grid are out of bounds in the
  // occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let kPacked = index.z;
  if (i > iMax || j > jMax || kPacked > kPackedMax) {
    return;
  }

  // A particular voxel is considered vertex-occupied if it or any of its lower
  // neighbors are occupied. Along the i and j axes, we simply compare packed vectors
  // to the corresponding packed vectors that are one index lower. Along the k axis,
  // we compare the packed vectors to versions of themselves that are shifted by one
  // bit. We fill each shifted packed vector's missing bit using the most significant
  // bit from the previous packed vector.
  int result = 0;
  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      // Handle the low edge case along the i and j axes.
      if ((i == 0 && di == 1) || (j == 0 && dj == 1)) {
        continue;
      }

      // Handle the high edge case along the i and j axes.
      if ((i == iMax && di == 0) || (j == jMax && dj == 0)) {
        continue;
      }

      // Fetch the packed vector at the current index.
      let current = (kPacked == kPackedMax) ? 0 : occupancy[i - di, j - dj, kPacked];

      // Fetch the packed vector at the previous index along the k axis.
      let previousPacked = (kPacked == 0) ? 0 : occupancy[i - di, j - dj, kPacked - 1];

      // Compare packed vectors along the k axis as described above.
      let previous = logicalRightShift(previousPacked, 31) | (current << 1);
      result |= (current | previous);
    }
  }

  // Set the vertex occupancy.
  vertexOccupancy[index] = result;

  // Set the number of vertices owned by the group of packed voxels.
  vertexCounts[index] = countbits(reinterpret<uint>(result));
}
