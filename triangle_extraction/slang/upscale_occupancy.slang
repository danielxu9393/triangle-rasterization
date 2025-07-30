bool sampleCorner(
    const TensorView<int> occupancy,
    const int i,
    const int j,
    const int k,
    const int iMax,
    const int jMax,
    const int kMax,
    const int iSide,
    const int jSide,
    const int kSide,
) {
  // The coordinates are integers but represent corners. We use integer arithmetic to
  // sample the lower-resolution occupancy grid to avoid rounding problems.
  let iMaxSampled = occupancy.size(0);
  let jMaxSampled = occupancy.size(1);
  let kMaxSampled = occupancy.size(2) << 5;
  let iSampled = ((i + iSide) * iMaxSampled - iSide) / iMax;
  let jSampled = ((j + jSide) * jMaxSampled - jSide) / jMax;
  let kSampled = ((k + kSide) * kMaxSampled - kSide) / kMax;
  let bit = reinterpret<int>(1u << (kSampled % 32));
  return (occupancy[iSampled, jSampled, kSampled >> 5] & bit) != 0;
}

bool sampleVoxel(
    const TensorView<int> occupancy,
    const int i,
    const int j,
    const int k,
    const int iMax,
    const int jMax,
    const int kMax,
) {
  bool value = false;
  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      for (int dk = 0; dk < 2; dk++) {
        value |= sampleCorner(occupancy, i, j, k, iMax, jMax, kMax, di, dj, dk);
      }
    }
  }
  return value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void upscale_occupancy(
    const TensorView<int> occupancy,
    const TensorView<int> newOccupancy,
) {
  // Read the packed new occupancy grid's dimensions.
  let iMax = newOccupancy.size(0);
  let jMax = newOccupancy.size(1);
  let kPackedMax = newOccupancy.size(2);

  // Guard against indices that are invalid in the new occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let kPacked = index.z;
  if (i >= iMax || j >= jMax || kPacked >= kPackedMax) {
    return;
  }
  let kMax = kPackedMax << 5;
  let kBase = kPacked << 5;

  // Work through the bits in this thread's entry in the new occupancy grid one by one.
  int resultPacked = 0;
  for (int bit = 0; bit < 32; bit++) {
    let resultBit = sampleVoxel(occupancy, i, j, kBase + bit, iMax, jMax, kMax);
    resultPacked |= (int(resultBit) << bit);
  }
  newOccupancy[i, j, kPacked] = resultPacked;
}
