[AutoPyBindCUDA]
[CUDAKernel]
void create_voxels(
    const TensorView<int> voxelOccupancy,
    const TensorView<int> voxelOffsets,
    const TensorView<int> vertexOccupancy,
    const TensorView<int> vertexOffsets,
    TensorView<float> vertices,
    TensorView<int> neighbors,
    TensorView<int> lowerCorners,
    TensorView<int> upperCorners,
    TensorView<int> indices,
) {
  // Read the packed occupancy grid's dimensions.
  let iMax = voxelOccupancy.size(0);
  let jMax = voxelOccupancy.size(1);
  let kPackedMax = voxelOccupancy.size(2);

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

  // Do some math that will be needed to set vertex positions later.
  let kUnpacked = 32 * kPacked;
  let invIMax = 1 / float(iMax);
  let invJMax = 1 / float(jMax);
  let invKMax = 1 / float(32 * kPackedMax);

  // Allocate vertices.
  var vertexIndex = vertexOffsets[index];
  var vertexOccupancyChunk = vertexOccupancy[index];
  var k = kUnpacked;
  for (int bit = 0; bit < 32; bit++) {
    // If the least significant bit is set, the voxel is vertex-occupied.
    if ((vertexOccupancyChunk & 1) == 1) {
      // Allocate the vertex.
      vertices[uint2(vertexIndex, 0)] = k * invKMax;
      vertices[uint2(vertexIndex, 1)] = j * invJMax;
      vertices[uint2(vertexIndex, 2)] = i * invIMax;
      vertexIndex++;
    }

    // Step to the next bit.
    vertexOccupancyChunk >>= 1;
    k++;
  }

  // Allocate voxels and subvoxels.
  let numVoxels = neighbors.size(1);
  int localVoxelIndices[2][2];
  int localVoxelOccupancies[2][2];
  int localVertexIndices[2][2];
  int localVertexOccupancies[2][2];
  int localSubvoxelIndices[2][2];
  int localSubvoxelOccupancies[2][2];
  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      let ii = i + di;
      let jj = j + dj;
      let localIndex = uint3(ii, jj, kPacked);

      // Handle out-of-bounds indices.
      if (ii > iMax || jj > jMax) {
        localVoxelIndices[di][dj] = -64;
        localVoxelOccupancies[di][dj] = 0;
        localVertexIndices[di][dj] = -64;
        localVertexOccupancies[di][dj] = 0;
        localSubvoxelIndices[di][dj] = -64;
        localSubvoxelOccupancies[di][dj] = 0;
        continue;
      }

      let validVoxel = (ii < iMax && jj < jMax && kPacked < kPackedMax);
      let localVoxelIndex = voxelOffsets[localIndex];
      let localVoxelOccupancy = validVoxel ? voxelOccupancy[localIndex] : 0;
      let localVertexIndex = vertexOffsets[localIndex];
      let localVertexOccupancy = vertexOccupancy[localIndex];

      localVoxelIndices[di][dj] = localVoxelIndex;
      localVoxelOccupancies[di][dj] = localVoxelOccupancy;
      localVertexIndices[di][dj] = localVertexIndex;
      localVertexOccupancies[di][dj] = localVertexOccupancy;
      localSubvoxelIndices[di][dj] = numVoxels + localVertexIndex - localVoxelIndex;
      localSubvoxelOccupancies[di][dj] = localVertexOccupancy & ~localVoxelOccupancy;
    }
  }

  let kMax = kPackedMax * 32;
  k = kUnpacked;
  for (int bit = 0; bit < 32; bit++) {
    let isVoxel = (localVoxelOccupancies[0][0] & 1) != 0;
    let isSubvoxel = (localSubvoxelOccupancies[0][0] & 1) != 0;
    let voxelIndex = localVoxelIndices[0][0];

    // Set the lower corners. The order is (origin, +x, +y, +z). Note that for
    // subvoxels, the indices can be negative (invalid).
    if (isVoxel || isSubvoxel) {
      let chosenIndex = isVoxel ? voxelIndex : localSubvoxelIndices[0][0];

      // The origin is easy, since it's always defined.
      lowerCorners[uint2(0, chosenIndex)] = localVertexIndices[0][0];

      // The other corners have to be handled carefully for subvoxels because they may
      // be invalid. Here, the vertex occupancy is used as a proxy for whether the
      // adjacent entry is either a voxel or a subvoxel. If so, the corner is valid.
      // Note that the last bit has to be special-cased for subvoxels. This happens
      // later, outside the loop.
      let xValid = isVoxel || ((localVertexOccupancies[0][0] & 2) != 0);
      let yValid = isVoxel || ((localVertexOccupancies[0][1] & 1) != 0);
      let zValid = isVoxel || ((localVertexOccupancies[1][0] & 1) != 0);
      lowerCorners[uint2(1, chosenIndex)] = xValid ? localVertexIndices[0][0] + 1 : -64;
      lowerCorners[uint2(2, chosenIndex)] = yValid ? localVertexIndices[0][1] : -64;
      lowerCorners[uint2(3, chosenIndex)] = zValid ? localVertexIndices[1][0] : -64;
    }

    // Set the upper corners. The order is (+xyz, +yz, +xz, +xy).
    if (isVoxel) {
      upperCorners[uint2(0, voxelIndex)] = localVertexIndices[1][1] + 1;
      upperCorners[uint2(1, voxelIndex)] = localVertexIndices[1][1];
      upperCorners[uint2(2, voxelIndex)] = localVertexIndices[1][0] + 1;
      upperCorners[uint2(3, voxelIndex)] = localVertexIndices[0][1] + 1;
      indices[voxelIndex] = kMax * (jMax * i + j) + k;
    }

    // Set the neighbors. The order is (+x, +y, +z, +xy, +yz, +xz). Note that the last
    // bit has to be special-cased because it depends on the next packed occupancy.
    if (isVoxel && bit != 31) {
      // Set neighbor 0 (+x).
      let xIsSubvoxel = (localSubvoxelOccupancies[0][0] & 2) != 0;
      neighbors[uint2(0, voxelIndex)] =
          xIsSubvoxel ? localSubvoxelIndices[0][0] : localVoxelIndices[0][0] + 1;

      // Set neighbor 1 (+y).
      let yIsSubvoxel = (localSubvoxelOccupancies[0][1] & 1) != 0;
      neighbors[uint2(1, voxelIndex)] =
          yIsSubvoxel ? localSubvoxelIndices[0][1] : localVoxelIndices[0][1];

      // Set neighbor 2 (+z).
      let zIsSubvoxel = (localSubvoxelOccupancies[1][0] & 1) != 0;
      neighbors[uint2(2, voxelIndex)] =
          zIsSubvoxel ? localSubvoxelIndices[1][0] : localVoxelIndices[1][0];

      // Set neighbor 3 (+xy).
      let xyIsSubvoxel = (localSubvoxelOccupancies[0][1] & 2) != 0;
      neighbors[uint2(3, voxelIndex)] =
          xyIsSubvoxel
              ? (yIsSubvoxel ? localSubvoxelIndices[0][1] + 1
                             : localSubvoxelIndices[0][1])
              : (yIsSubvoxel ? localVoxelIndices[0][1] : localVoxelIndices[0][1] + 1);

      // Set neighbor 4 (+yz).
      let yzIsSubvoxel = (localSubvoxelOccupancies[1][1] & 1) != 0;
      neighbors[uint2(4, voxelIndex)] =
          yzIsSubvoxel ? localSubvoxelIndices[1][1] : localVoxelIndices[1][1];

      // Set neighbor 5 (+xz).
      let xzIsSubvoxel = (localSubvoxelOccupancies[1][0] & 2) != 0;
      neighbors[uint2(5, voxelIndex)] =
          xzIsSubvoxel
              ? (zIsSubvoxel ? localSubvoxelIndices[1][0] + 1
                             : localSubvoxelIndices[1][0])
              : (zIsSubvoxel ? localVoxelIndices[1][0] : localVoxelIndices[1][0] + 1);

      // Set neighbor 6 (+xyz).
      let xyzIsSubvoxel = (localSubvoxelOccupancies[1][1] & 2) != 0;
      neighbors[uint2(6, voxelIndex)] =
          xyzIsSubvoxel
              ? (yzIsSubvoxel ? localSubvoxelIndices[1][1] + 1
                              : localSubvoxelIndices[1][1])
              : (yzIsSubvoxel ? localVoxelIndices[1][1] : localVoxelIndices[1][1] + 1);
    }

    // Step to the next bit.
    if (bit != 31) {
      for (int di = 0; di < 2; di++) {
        for (int dj = 0; dj < 2; dj++) {
          localVoxelIndices[di][dj] += (localVoxelOccupancies[di][dj] & 1);
          localVoxelOccupancies[di][dj] >>= 1;

          localVertexIndices[di][dj] += (localVertexOccupancies[di][dj] & 1);
          localVertexOccupancies[di][dj] >>= 1;

          localSubvoxelIndices[di][dj] += (localSubvoxelOccupancies[di][dj] & 1);
          localSubvoxelOccupancies[di][dj] >>= 1;
        }
      }
    }

    k++;
  }

  // Handle +x neighbors for subvoxels at the last bit.
  if ((localSubvoxelOccupancies[0][0] & 1) != 0) {
    let xValid = (kPacked < kPackedMax) &&
                 ((vertexOccupancy[uint3(i, j, kPacked + 1)] & 1) != 0);
    lowerCorners[uint2(1, localSubvoxelIndices[0][0])] =
        xValid ? localVertexIndices[0][0] + 1 : -64;
  }

  // Set the neighbors for the last bit in the case where it represents a voxel. This
  // requires a special case because it relies on the next packed occupancy vector.
  // The order is again (+x, +y, +z, +xy, +yz, +xz).
  if ((localVoxelOccupancies[0][0] & 1) == 0) {
    return;
  }

  let voxelIndex = localVoxelIndices[0][0];
  let canIncrementX = (kPacked < kPackedMax - 1);
  let canIncrementY = (j < jMax - 1);
  let canIncrementZ = (i < iMax - 1);

  // Set neighbor 0 (+x).
  let xIsSubvoxel =
      canIncrementX ? ((voxelOccupancy[uint3(i, j, kPacked + 1)] & 1) == 0) : true;
  neighbors[uint2(0, voxelIndex)] =
      xIsSubvoxel ? localSubvoxelIndices[0][0] : localVoxelIndices[0][0] + 1;

  // Set neighbor 1 (+y).
  let yIsSubvoxel = canIncrementY
                        ? ((voxelOccupancy[uint3(i, j + 1, kPacked)] & (1 << 31)) == 0)
                        : true;
  neighbors[uint2(1, voxelIndex)] =
      yIsSubvoxel ? localSubvoxelIndices[0][1] : localVoxelIndices[0][1];

  // Set neighbor 2 (+z).
  let zIsSubvoxel = canIncrementZ
                        ? ((voxelOccupancy[uint3(i + 1, j, kPacked)] & (1 << 31)) == 0)
                        : true;
  neighbors[uint2(2, voxelIndex)] =
      zIsSubvoxel ? localSubvoxelIndices[1][0] : localVoxelIndices[1][0];

  // Set neighbor 3 (+xy).
  let xyIsSubvoxel = (canIncrementX && canIncrementY)
                         ? ((voxelOccupancy[uint3(i, j + 1, kPacked + 1)] & 1) == 0)
                         : true;
  neighbors[uint2(3, voxelIndex)] =
      xyIsSubvoxel
          ? (yIsSubvoxel ? localSubvoxelIndices[0][1] + 1 : localSubvoxelIndices[0][1])
          : (yIsSubvoxel ? localVoxelIndices[0][1] : localVoxelIndices[0][1] + 1);

  // Set neighbor 4 (+yz).
  let yzIsSubvoxel =
      (canIncrementY && canIncrementZ)
          ? ((voxelOccupancy[uint3(i + 1, j + 1, kPacked)] & (1 << 31)) == 0)
          : true;
  neighbors[uint2(4, voxelIndex)] =
      yzIsSubvoxel ? localSubvoxelIndices[1][1] : localVoxelIndices[1][1];

  // Set neighbor 5 (+xz).
  let xzIsSubvoxel = (canIncrementX && canIncrementZ)
                         ? ((voxelOccupancy[uint3(i + 1, j, kPacked + 1)] & 1) == 0)
                         : true;
  neighbors[uint2(5, voxelIndex)] =
      xzIsSubvoxel
          ? (zIsSubvoxel ? localSubvoxelIndices[1][0] + 1 : localSubvoxelIndices[1][0])
          : (zIsSubvoxel ? localVoxelIndices[1][0] : localVoxelIndices[1][0] + 1);

  // Set neighbor 6 (+xyz).
  let xyzIsSubvoxel =
      (canIncrementX && canIncrementY && canIncrementZ)
          ? ((voxelOccupancy[uint3(i + 1, j + 1, kPacked + 1)] & 1) == 0)
          : true;
  neighbors[uint2(6, voxelIndex)] =
      xyzIsSubvoxel
          ? (yzIsSubvoxel ? localSubvoxelIndices[1][1] + 1 : localSubvoxelIndices[1][1])
          : (yzIsSubvoxel ? localVoxelIndices[1][1] : localVoxelIndices[1][1] + 1);
}
