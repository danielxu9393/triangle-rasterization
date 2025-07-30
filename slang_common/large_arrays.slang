uint64_t compute_offset<T, int N>(TensorView<T> x, vector<uint64_t, N> index) {
  uint64_t offset = 0;
  uint64_t stride = sizeof(T);
  for (int i = N - 1; i >= 1; i--) {
    offset += index[i] * stride;
    stride *= (uint64_t)x.size(i);
  }
  offset += index[0] * stride;
  return offset;
}

T load<T, int N>(TensorView<T> x, vector<uint64_t, N> index) {
  uint64_t offset = compute_offset(x, index);
  return *((T *)(((uint8_t *)x.data_ptr()) + offset));
}

T store<T, int N>(TensorView<T> x, vector<uint64_t, N> index, T value) {
  uint64_t offset = compute_offset(x, index);
  *((T *)(((uint8_t *)x.data_ptr()) + offset)) = value;
  return value;
}

uint64_t calc64bitOffset<T, int N>(TensorView<T> x, vector<uint, N> idx) {
  uint64_t offset = 0;
  for (int i = 0; i < N; ++i) {
    offset += idx[i] * (uint64_t)x.stride(i);
  }
  return offset;
}

T load64bit<T, int N>(TensorView<T> x, vector<uint, N> idx) {
  uint64_t offset = calc64bitOffset(x, idx);
  return *((T *)(((uint8_t *)x.data_ptr()) + offset));
  // x.data_ptr() returns x's data pointer of type T*
  // We then cast to uint8 to get bytes since offset must be in singular bytes
  // *((T *) .. ) first casts to T*, then dereferences
}

T store64bit<T, int N>(TensorView<T> x, vector<uint, N> idx, T value) {
  uint64_t offset = calc64bitOffset(x, idx);
  *((T *)(((uint8_t *)x.data_ptr()) + offset)) = value;
  return value;
}

void InterlockedAdd64bit<T, int N>(
    TensorView<T> x,
    vector<uint, N> idx,
    T val,
    out T oldVal
) {
  // 1) Compute 64-bit byte offset into the buffer
  uint64_t offset = calc64bitOffset<T, N>(x, idx);
  let address = *((T *)(((uint8_t *)x.data_ptr()) + offset));

  InterlockedAdd64bitInner<T>(x, offset, val, oldVal);
}

[require(cuda)]
void InterlockedAdd64bitInner<T>(
    TensorView<T> x,
    uint64_t offset,
    T val,
    out T oldVal
) {

  // 2) Perform the atomic add in CUDA via inline PTX
  // $0 = x, $1 = offset, $2 = val, $3 = &oldVal
  // clang-format off
  __target_switch {
    case cuda:
      __intrinsic_asm "*($3) = atomicAdd(($T2*)(((uint8_t*)($0.data_ptr<$T2>())) + $1), $2)";
  }
  // clang-format on
}

[AutoPyBindCUDA]
[CUDAKernel]
void largeArrays(const TensorView<float> x, const TensorView<float> y) {
  let index = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
  if (index.x >= x.size(0) || index.y >= x.size(1)) {
    return;
  }

  store64bit(y, index.xy, load64bit(x, index.xy) + 1);
}

struct DiffTensorViewLarge {
  TensorView<float> primal;
  TensorView<float> diff;

  // Constructors
  __init(TensorView<float> primal, TensorView<float> diff) {
    this.primal = primal;
    this.diff = diff;
  }

  __init(TensorView<float> primal) {
    this.primal = primal;
  }

  [BackwardDerivative(loadBwd)]
  __generic<let N : int> float load(vector<uint, N> index) {
    return load64bit(primal, index);
  }

  void loadBwd<int N>(vector<uint, N> index, float.Differential dOut) {
    float oldVal;
    InterlockedAdd64bit(diff, index, dOut, oldVal);
  }

  [BackwardDerivative(storeOnceBwd)]
  __generic<let N : int> void storeOnce(vector<uint, N> index, float value) {
    store64bit(primal, index, value);
  }

  void storeOnceBwd<int N>(vector<uint, N> index, inout DifferentialPair<float> dpval) {
    dpval = diffPair(dpval.p, load64bit(diff, index));
  }
}
