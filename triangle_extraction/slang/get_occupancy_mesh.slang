[AutoPyBindCUDA]
[CUDAKernel]
void countTriangles(
    const TensorView<int> occupancy,
    const TensorView<int> triangleCounts,
) {
  // Read the packed occupancy grid's dimensions.
  let iMax = occupancy.size(0);
  let jMax = occupancy.size(1);
  let kPackedMax = occupancy.size(2);

  // Guard against invalid indices. Note that indices where i == iMax, j == jMax, or
  // k == kPackedMax are considered valid, since we special-case these below.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let kPacked = index.z;
  if (i > iMax || j > jMax || kPacked > kPackedMax) {
    return;
  }

  // Fetch occupancy from the current packed vector and its neighbors.
  let iLim = (i == iMax);
  let jLim = (j == jMax);
  let kLim = (kPacked == kPackedMax);
  var atHere = (iLim || jLim || kLim) ? 0 : occupancy[uint3(i, j, kPacked)];
  var atMinusI = (jLim || kLim || (i == 0)) ? 0 : occupancy[uint3(i - 1, j, kPacked)];
  var atMinusJ = (iLim || kLim || (j == 0)) ? 0 : occupancy[uint3(i, j - 1, kPacked)];
  var bitAtMinusK = (iLim || jLim || (kPacked == 0))
                        ? false
                        : ((occupancy[uint3(i, j, kPacked - 1)] & (1 << 31)) != 0);

  // Iterate through the packed bits and count triangles.
  int numFaces = 0;
  for (int bit = 0; bit < 32; bit++) {
    // Extract the current bit.
    let bitAtHere = (atHere & 1) != 0;
    let bitAtMinusI = (atMinusI & 1) != 0;
    let bitAtMinusJ = (atMinusJ & 1) != 0;

    // Count the faces created by the current voxel.
    numFaces += int(bitAtHere != bitAtMinusI);
    numFaces += int(bitAtHere != bitAtMinusJ);
    numFaces += int(bitAtHere != bitAtMinusK);

    // Walk to the next bit.
    bitAtMinusK = bitAtHere;
    atHere >>= 1;
    atMinusI >>= 1;
    atMinusJ >>= 1;
  }

  triangleCounts[i, j, kPacked] = 2 * numFaces;
}

void createFace(
    const int index,
    const int ijk[3],
    const int ijkMax[3],
    const int axis,
    const bool upperBit,
    const TensorView<float> triangleVertices,
    const TensorView<float> triangleColors,
    const TensorView<float> colors,
) {
  let axis0 = (axis + 1) % 3;
  let axis1 = (axis + 2) % 3;

  // Define the vertex locations.
  float3 v00;
  float3 v01;
  float3 v10;
  float3 v11;

  let plane = float(ijk[axis]) / ijkMax[axis];
  v00[axis] = plane;
  v01[axis] = plane;
  v10[axis] = plane;
  v11[axis] = plane;

  let v0x = float(ijk[axis0]) / ijkMax[axis0];
  let v1x = float(ijk[axis0] + 1) / ijkMax[axis0];
  let vx0 = float(ijk[axis1]) / ijkMax[axis1];
  let vx1 = float(ijk[axis1] + 1) / ijkMax[axis1];

  v00[axis0] = v0x;
  v00[axis1] = vx0;

  v01[axis0] = v0x;
  v01[axis1] = vx1;

  v10[axis0] = v1x;
  v10[axis1] = vx0;

  v11[axis0] = v1x;
  v11[axis1] = vx1;

  // Determine the face color.
  let side = int(upperBit);
  let faceColor = float3(
      colors[uint3(axis, side, 0)],
      colors[uint3(axis, side, 1)],
      colors[uint3(axis, side, 2)],
  );

  // Write out the resulting triangles.
  const float3 triangles[2][3] = {
    { v00, upperBit ? v01 : v10, upperBit ? v10 : v01 },
    { v11, upperBit ? v10 : v01, upperBit ? v01 : v10 },
  };
  $for(i in Range(0, 2)) {
    let offset = index + i;
    $for(j in Range(0, 3)) {
      // The triangles are indexed by axis (ijk) but need to be flipped to xyz.
      triangleVertices[uint3(offset, j, 0)] = triangles[i][j].z;
      triangleVertices[uint3(offset, j, 1)] = triangles[i][j].y;
      triangleVertices[uint3(offset, j, 2)] = triangles[i][j].x;
    }
    triangleColors[uint2(offset, 0)] = faceColor.r;
    triangleColors[uint2(offset, 1)] = faceColor.g;
    triangleColors[uint2(offset, 2)] = faceColor.b;
  }
}

[AutoPyBindCUDA]
[CUDAKernel]
void createTriangles(
    const TensorView<int> occupancy,
    const TensorView<int> triangleOffsets,
    const TensorView<float> triangleVertices,
    const TensorView<float> triangleColors,
    const TensorView<float> colors,
) {
  // Read the packed occupancy grid's dimensions.
  let iMax = occupancy.size(0);
  let jMax = occupancy.size(1);
  let kPackedMax = occupancy.size(2);

  // Guard against invalid indices. Note that indices where i == iMax, j == jMax, or
  // k == kPackedMax are considered valid, since we special-case these below.
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  let i = index.x;
  let j = index.y;
  let kPacked = index.z;
  if (i > iMax || j > jMax || kPacked > kPackedMax) {
    return;
  }

  // Fetch occupancy from the current packed vector and its neighbors.
  let iLim = (i == iMax);
  let jLim = (j == jMax);
  let kLim = (kPacked == kPackedMax);
  var atHere = (iLim || jLim || kLim) ? 0 : occupancy[uint3(i, j, kPacked)];
  var atMinusI = (jLim || kLim || (i == 0)) ? 0 : occupancy[uint3(i - 1, j, kPacked)];
  var atMinusJ = (iLim || kLim || (j == 0)) ? 0 : occupancy[uint3(i, j - 1, kPacked)];
  var bitAtMinusK = (iLim || jLim || (kPacked == 0))
                        ? false
                        : ((occupancy[uint3(i, j, kPacked - 1)] & (1 << 31)) != 0);

  // Iterate through the packed bits and count triangles.
  int triangleIndex = triangleOffsets[uint3(i, j, kPacked)];
  int k = kPacked * 32;
  const int ijkMax[3] = { iMax, jMax, kPackedMax * 32 };
  for (int bit = 0; bit < 32; bit++) {
    // Extract the current bit.
    let bitAtHere = (atHere & 1) != 0;
    let bitAtMinusI = (atMinusI & 1) != 0;
    let bitAtMinusJ = (atMinusJ & 1) != 0;
    const int ijk[3] = { i, j, k };

    // Create faces with Z-aligned normals.
    if (bitAtHere != bitAtMinusI) {
      createFace(
          triangleIndex,
          ijk,
          ijkMax,
          0,
          bitAtHere,
          triangleVertices,
          triangleColors,
          colors,
      );
      triangleIndex += 2;
    }

    // Create faces with Y-aligned normals.
    if (bitAtHere != bitAtMinusJ) {
      createFace(
          triangleIndex,
          ijk,
          ijkMax,
          1,
          bitAtHere,
          triangleVertices,
          triangleColors,
          colors,
      );
      triangleIndex += 2;
    }

    // Create faces with X-aligned normals.
    if (bitAtHere != bitAtMinusK) {
      createFace(
          triangleIndex,
          ijk,
          ijkMax,
          2,
          bitAtHere,
          triangleVertices,
          triangleColors,
          colors,
      );
      triangleIndex += 2;
    }

    // Walk to the next bit.
    bitAtMinusK = bitAtHere;
    atHere >>= 1;
    atMinusI >>= 1;
    atMinusJ >>= 1;
    k++;
  }
}
