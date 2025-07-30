int logicalRightShift(const int x, const int shift) {
  // In Slang, int uses arithmetic right shift, and uint uses logical right shift. Thus,
  // to get a logical right shift for int, we have to cast to uint and back.
  return reinterpret<int>(reinterpret<uint>(x) >> shift);
}

[AutoPyBindCUDA]
[CUDAKernel]
void dilate_occupancy(
    const TensorView<int> occupancy,
    const TensorView<int> newOccupancy,
    const int dilation,
) {
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

  int result = 0;

  // Handle the previous k index.
  if (kPacked > 0) {
    int atPrevious = 0;
    for (int di = -dilation; di <= dilation; di++) {
      for (int dj = -dilation; dj <= dilation; dj++) {
        let ii = i + di;
        let jj = j + dj;
        if (ii < 0 || ii >= iMax || jj < 0 || jj >= jMax) {
          continue;
        }
        atPrevious |= occupancy[ii, jj, kPacked - 1];
      }
    }
    atPrevious = logicalRightShift(atPrevious, 32 - dilation);
    for (int bit = 0; bit < dilation; bit++) {
      result |= atPrevious;
      atPrevious = logicalRightShift(atPrevious, 1);
    }
  }

  // Handle the current k index.
  int atCurrent = 0;
  for (int di = -dilation; di <= dilation; di++) {
    for (int dj = -dilation; dj <= dilation; dj++) {
      let ii = i + di;
      let jj = j + dj;
      if (ii < 0 || ii >= iMax || jj < 0 || jj >= jMax) {
        continue;
      }
      atCurrent |= occupancy[ii, jj, kPacked];
    }
  }
  result |= atCurrent;
  int atCurrentLeft = atCurrent;
  int atCurrentRight = atCurrent;
  for (int bit = 0; bit < dilation; bit++) {
    atCurrentRight = logicalRightShift(atCurrentRight, 1);
    atCurrentLeft <<= 1;
    result |= atCurrentLeft;
    result |= atCurrentRight;
  }

  // Handle the next k index.
  if (kPacked < kPackedMax - 1) {
    int atNext = 0;
    for (int di = -dilation; di <= dilation; di++) {
      for (int dj = -dilation; dj <= dilation; dj++) {
        let ii = i + di;
        let jj = j + dj;
        if (ii < 0 || ii >= iMax || jj < 0 || jj >= jMax) {
          continue;
        }
        atNext |= occupancy[ii, jj, kPacked + 1];
      }
    }
    atNext <<= (32 - dilation);
    for (int bit = 0; bit < dilation; bit++) {
      result |= atNext;
      atNext <<= 1;
    }
  }

  // Write the result to the new occupancy grid.
  newOccupancy[i, j, kPacked] = result;
}
