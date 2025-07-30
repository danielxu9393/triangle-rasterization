import "./constants.slang";

[AutoPyBindCUDA]
[CUDAKernel]
void createTriangleFaces(
    const TensorView<float> signedDistances,
    const TensorView<int> neighbors,
    const TensorView<int> indices,
    const TensorView<int> vertexOffsets,
    const TensorView<uint8_t> vertexCountsByLevelSet,
    const TensorView<uint8_t> triangleVertexTypes,
    const TensorView<int> faceCounts,
    const TensorView<uint8_t> voxelCellCodes,
    const TensorView<float> levelSets,
    const TensorView<int> triangleFaces,
    const TensorView<int> shellIndices,
    const TensorView<int> voxelIndices,
) {
  let index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= faceCounts.size(0) - 1) {
    return;
  }

  // This value is 3 * num_tris, so it is total number of vertices to add to our buffer
  let num = faceCounts[index + 1] - faceCounts[index];
  int triangle_write_ctr = faceCounts[index];

  for (int levelSet = 0; levelSet < levelSets.size(0); levelSet++) {
    const int cubeCode = voxelCellCodes[uint2(levelSet, index)];
    let num_i = firstMarchingCubesId[cubeCode + 1] - firstMarchingCubesId[cubeCode];
    let firstIn = firstMarchingCubesId[cubeCode];
    for (int j = 0; j < num_i; j++) {
      // Edge id
      let eid = marchingCubesIds[firstIn + j];

      // Which neighbor owns this edge?
      let neighbor_index = mcEdgeLocations[eid][0];

      // Basically the direction of the edge
      let edgeNr = mcEdgeLocations[eid][1];

      int neighbor_voxel_index;
      if (neighbor_index == -1) {
        neighbor_voxel_index = index;
      } else {
        neighbor_voxel_index = neighbors[uint2(neighbor_index, index)];
      }

      // For that neighbor, we need to access the triangle vertices it owns
      int neighbor_num_vertices =
          vertexCountsByLevelSet[uint2(levelSet, neighbor_voxel_index)];
      int neighbor_vertex_starting_idx = vertexOffsets[neighbor_voxel_index];
      for (int k = 0; k < levelSet; k++) {
        // We need navigate to the right level set
        neighbor_vertex_starting_idx +=
            vertexCountsByLevelSet[uint2(k, neighbor_voxel_index)];
      }

      // Now we are going to check all triangles vertices (with the right level set)
      // owned by the neighbor, and find the unique one that is the same orientation!
      int target_vertex = -1;
      for (int k = 0; k < neighbor_num_vertices; k++) {
        let query_vertex_idx = neighbor_vertex_starting_idx + k;
        if (triangleVertexTypes[query_vertex_idx] == edgeNr) {
          target_vertex = query_vertex_idx;
          break;
        }
      }

      // Now write the vertex idx to triangle faces buffer
      // We need to convert from 3*num_tris to num_tris, 3
      let triangle_num = (triangle_write_ctr + j) / 3;
      let triangle_offset = 2 - (triangle_write_ctr + j) % 3;
      triangleFaces[uint2(triangle_num, triangle_offset)] = target_vertex;
      if (triangle_offset == 0) {
        shellIndices[triangle_num] = levelSet;
        voxelIndices[triangle_num] = indices[index];
      }
    }
    triangle_write_ctr += num_i;
  }
}
