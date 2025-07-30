[AutoPyBindCUDA]
[CUDAKernel]
void count_occupancy(const TensorView<int> occupancy, TensorView<int> voxelCounts) {
  // Read the packed occupancy grid's dimensions.
  let iMax = occupancy.size(0);
  let jMax = occupancy.size(1);
  let kPackedMax = occupancy.size(2);

  // Guard against indices that are invalid in the occupancy grid.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let kPacked = index.z;
  if (i >= iMax || j >= jMax || kPacked >= kPackedMax) {
    return;
  }

  // Count the number of voxels in the current packed vector.
  voxelCounts[index] = countbits(occupancy[index]);
}
