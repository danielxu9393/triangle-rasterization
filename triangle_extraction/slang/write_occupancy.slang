[AutoPyBindCUDA]
[CUDAKernel]
void write_occupancy(
    const TensorView<int> occupancy,
    const TensorView<int> shellIndices,
    const TensorView<int> voxelIndices,
    const int minShell,
    const int maxShell,
) {
  // Guard against indices that are invalid in the new occupancy grid.
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= shellIndices.size(0)) {
    return;
  }

  // Skip past triangles whose shells should not be included.
  let shellIndex = shellIndices[index];
  if (shellIndex < minShell || shellIndex >= maxShell) {
    return;
  }

  // Mark the triangle's location in the occupancy grid as occupied.
  let voxelIndex = voxelIndices[index];
  let arrayIndex = voxelIndex >> 5;
  let bit = reinterpret<int>(1u << (voxelIndex % 32));
  __atomic_or(*occupancy.data_ptr_at(arrayIndex), bit, MemoryOrder.Relaxed);
}
