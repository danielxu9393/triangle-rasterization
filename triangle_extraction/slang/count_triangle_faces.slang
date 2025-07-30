import "./constants.slang";

int getCubeCode(const float cubeSDF[8], const float iso) {
  int code = 0;
  for (int i = 0; i < 8; i++) {
    if (cubeSDF[i] >= iso) {
      code |= 1 << i;
    }
  }
  return code;
}

[AutoPyBindCUDA]
[CUDAKernel]
void countTriangleFaces(
    const TensorView<float> signedDistances,
    const TensorView<int> lowerCorners,
    const TensorView<int> upperCorners,
    const TensorView<float> levelSets,
    TensorView<int> faceCounts,
    TensorView<uint8_t> voxelCellCodes,
) {
  int index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= upperCorners.size(1)) {
    return;
  }

  // We arrange the corners to match cumc.cu's mcCorners.
  const float cubeSDF[8] = {
    signedDistances[lowerCorners[uint2(0, index)]], // origin
    signedDistances[lowerCorners[uint2(1, index)]], // x
    signedDistances[upperCorners[uint2(3, index)]], // xy
    signedDistances[lowerCorners[uint2(2, index)]], // y
    signedDistances[lowerCorners[uint2(3, index)]], // z
    signedDistances[upperCorners[uint2(2, index)]], // xy
    signedDistances[upperCorners[uint2(0, index)]], // xyz
    signedDistances[upperCorners[uint2(1, index)]], // yz
  };

  int numFaces = 0;
  for (int levelSet = 0; levelSet < levelSets.size(0); levelSet++) {
    let iso = levelSets[levelSet];

    let cubeCode = getCubeCode(cubeSDF, iso);
    voxelCellCodes[uint2(levelSet, index)] = uint8_t(cubeCode);

    numFaces += firstMarchingCubesId[cubeCode + 1] - firstMarchingCubesId[cubeCode];
  }

  faceCounts[index] = numFaces;
}
