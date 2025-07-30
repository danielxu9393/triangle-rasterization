// File: triangle_rasterization/slang/generate_keys.cu

#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel: one thread per triangle
__global__ void generate_keys_kernel(
    const float *depths,       // [Nv]
    const int32_t *faces,      // [Nf × 3]
    const int32_t *tileMinima, // [Nf × 2]
    const int32_t *tileMaxima, // [Nf × 2]
    const int64_t *offsets,    // [Nf]
    int gridWidth,
    int64_t *outKeys,       // [num_keys]
    int32_t *outTriIndices, // [num_keys]
    int64_t numFaces
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numFaces) {
    return;
  }

  // start position in the flat key buffer
  int64_t writePos = (idx == 0 ? 0 : offsets[idx - 1]);

  // unpack tile bounds
  int32_t yMin = tileMinima[idx * 2 + 0];
  int32_t xMin = tileMinima[idx * 2 + 1];
  int32_t yMax = tileMaxima[idx * 2 + 0];
  int32_t xMax = tileMaxima[idx * 2 + 1];

  // per-vertex depths
  float dA = depths[faces[idx * 3 + 0]];
  float dB = depths[faces[idx * 3 + 1]];
  float dC = depths[faces[idx * 3 + 2]];
  bool invalid = (dA <= 0.0f) || (dB <= 0.0f) || (dC <= 0.0f);
  float meanDepth = invalid ? 0.0f : (dA + dB + dC) / 3.0f;
  uint32_t depthBits = __float_as_uint(meanDepth);

  // emit one key per covered tile
  for (int y = yMin; y < yMax; ++y) {
    for (int x = xMin; x < xMax; ++x) {
      uint64_t tileIndex = uint64_t(y) * uint64_t(gridWidth) + uint64_t(x);
      uint64_t key = (tileIndex << 32) | uint64_t(depthBits);
      outKeys[writePos] = static_cast<int64_t>(key);
      outTriIndices[writePos] = idx;
      ++writePos;
    }
  }
}

// C++ launcher that wraps the CUDA kernel
void generate_keys(
    at::Tensor depths,
    at::Tensor faces,
    at::Tensor tileMinima,
    at::Tensor tileMaxima,
    at::Tensor offsets,
    int gridWidth,
    at::Tensor outKeys,
    at::Tensor outTriIndices,
    int64_t numFaces
) {
  // type and device checks
  TORCH_CHECK(depths.dtype() == torch::kFloat, "depths must be float32");
  TORCH_CHECK(faces.dtype() == torch::kInt32, "faces must be int32");
  TORCH_CHECK(tileMinima.dtype() == torch::kInt32, "tileMinima must be int32");
  TORCH_CHECK(tileMaxima.dtype() == torch::kInt32, "tileMaxima must be int32");
  TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(outKeys.dtype() == torch::kInt64, "outKeys must be int64");
  TORCH_CHECK(outTriIndices.dtype() == torch::kInt32, "outTriIndices must be int32");

  TORCH_CHECK(
      depths.is_cuda() && faces.is_cuda() && tileMinima.is_cuda() &&
          tileMaxima.is_cuda() && offsets.is_cuda() && outKeys.is_cuda() &&
          outTriIndices.is_cuda(),
      "All tensors must be CUDA."
  );

  auto d_contig = depths.contiguous();
  auto f_contig = faces.contiguous();
  auto tmin_cont = tileMinima.contiguous();
  auto tmax_cont = tileMaxima.contiguous();
  auto off_cont = offsets.contiguous();
  auto k_cont = outKeys.contiguous();
  auto i_cont = outTriIndices.contiguous();

  if (numFaces <= 0) {
    return;
  }

  const int threads = 256;
  int blocks = int((numFaces + threads - 1) / threads);

  generate_keys_kernel<<<blocks, threads>>>(
      d_contig.data_ptr<float>(),
      f_contig.data_ptr<int32_t>(),
      tmin_cont.data_ptr<int32_t>(),
      tmax_cont.data_ptr<int32_t>(),
      off_cont.data_ptr<int64_t>(),
      gridWidth,
      k_cont.data_ptr<int64_t>(),
      i_cont.data_ptr<int32_t>(),
      numFaces
  );

  // explicit launch error check
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "generate_keys_kernel launch failed: ",
      cudaGetErrorString(err)
  );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "generate_keys",
      &generate_keys,
      "Generate (tile<<32|depth) keys & triangle indices (CUDA)",
      py::arg("depths"),
      py::arg("faces"),
      py::arg("tileMinima"),
      py::arg("tileMaxima"),
      py::arg("offsets"),
      py::arg("gridWidth"),
      py::arg("outKeys"),
      py::arg("outTriangleIndices"),
      py::arg("numFaces")
  );
}