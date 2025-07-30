import "../../slang_common/large_arrays.slang";
static const float EPSILON = 1e-6;
static const int64_t TILE_KEYS = 1000;

struct Vertex : IDifferentiable {
  no_diff float2 position;
  float3 color;
  float signedDistance;
}

struct Triangle : IDifferentiable {
  int shell;
  Vertex a;
  Vertex b;
  Vertex c;
};

// Define shared memory used during the forward and backward passes.
static const uint NUM_SHARED_TRIANGLES = TILE_HEIGHT * TILE_WIDTH;
groupshared Triangle sharedTriangles[NUM_SHARED_TRIANGLES];
groupshared int sharedTriangleIndices[NUM_SHARED_TRIANGLES];
groupshared int sharedVoxelIndices[NUM_SHARED_TRIANGLES];

[Differentiable]
Vertex loadVertex(
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signedDistances,
    const int index,
) {
  let position = float2(vertices[uint2(index, 0)], vertices[uint2(index, 1)], );
  let color = float3(
      colors[uint2(index, 0)],
      colors[uint2(index, 1)],
      colors[uint2(index, 2)],
  );
  let signedDistance = signedDistances[index];
  return { position, color, signedDistance };
}

[Differentiable]
Triangle loadTriangle(
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signedDistances,
    const TensorView<int> faces,
    const TensorView<int> shellIndices,
    const int index,
) {
  return {
    shellIndices[index],
    loadVertex(vertices, colors, signedDistances, faces[uint2(index, 0)]),
    loadVertex(vertices, colors, signedDistances, faces[uint2(index, 1)]),
    loadVertex(vertices, colors, signedDistances, faces[uint2(index, 2)]),
  };
}

[Differentiable]
float3
computeBarycentricCoordinates(const Triangle triangle, no_diff const float2 point) {
  let p1 = triangle.a.position;
  let p2 = triangle.b.position;
  let p3 = triangle.c.position;

  // Compute dot products
  let d_12_12 = dot(p2 - p1, p2 - p1);
  let d_12_23 = dot(p2 - p1, p3 - p2);
  let d_12_13 = dot(p2 - p1, p3 - p1);
  let d_13_13 = dot(p3 - p1, p3 - p1);
  let d_13_23 = dot(p3 - p1, p3 - p2);
  let d_23_23 = dot(p3 - p2, p3 - p2);
  let d_1x_12 = dot(point - p1, p2 - p1);
  let d_1x_13 = dot(point - p1, p3 - p1);
  let d_2x_12 = dot(point - p2, p2 - p1);
  let d_2x_13 = dot(point - p2, p3 - p1);

  // Compute barycentric coordinates
  // The most stable estimate of the area of the triangle is whatever uses
  // the smallest line segment as the thing to be squared (I think).
  float denom;
  if (abs(d_12_13) <= abs(d_12_23) && abs(d_12_13) <= abs(d_13_23)) {
    denom = d_12_12 * d_13_13 - d_12_13 * d_12_13;
  } else if (abs(d_12_23) <= abs(d_12_13) && abs(d_12_23) <= abs(d_13_23)) {
    denom = d_12_12 * d_23_23 - d_12_23 * d_12_23;
  } else if (abs(d_13_23) <= abs(d_12_13) && abs(d_13_23) <= abs(d_12_23)) {
    denom = d_13_13 * d_23_23 - d_13_23 * d_13_23;
  }

  float v = d_13_13 * d_1x_12 - d_12_13 * d_1x_13;
  float w = d_12_12 * d_1x_13 - d_12_13 * d_1x_12;
  float u = d_12_23 * d_2x_13 - d_13_23 * d_2x_12;

  let eps = 1e-8;
  v /= denom + eps;
  w /= denom + eps;
  u /= denom + eps;

  return { u, v, w };
}

// This gets created when you evaluate a triangle at a point.
struct Sample : IDifferentiable {
  float3 rgb; // color
  float sdf;  // signed distance
  bool hit;   // whether the triangle was intersected
}

[Differentiable]
Sample sampleTriangle(
    const Triangle triangle,
    no_diff const float2 point,
    const int lastShell,
) {
  // First, check the triangle's shell index to see if it should be skipped.
  if (triangle.shell <= lastShell) {
    return { float3(0.f, 0.f, 0.f), 0.f, false };
  }

  // Compute barycentric coordinates for the triangle.
  let uvw = computeBarycentricCoordinates(triangle, point);

  // If the barycentric coordinates land outside the triangle or appear to be
  // degenerate, return full transparency.
  let uvwSum = uvw.x + uvw.y + uvw.z;
  if (isnan(uvw.x) || isnan(uvw.y) || isnan(uvw.z) || uvw.x < 0 || uvw.x > 1 ||
      uvw.y < 0 || uvw.y > 1 || uvw.z < 0 || uvw.z > 1 || uvwSum < 0.999 ||
      uvwSum > 1.001) {
    return { float3(0.f, 0.f, 0.f), 0.f, false };
  }

  // Interpolate color and signed distance from the vertices.
  let signedDistance = uvw.x * triangle.a.signedDistance +
                       uvw.y * triangle.b.signedDistance +
                       uvw.z * triangle.c.signedDistance;
  let rgb =
      uvw.x * triangle.a.color + uvw.y * triangle.b.color + uvw.z * triangle.c.color;
  return { rgb, signedDistance, true };
}

// This is used to accumulate visibility.
struct Accumulator : IDifferentiable {
  float3 rgb;
  float transmittance;
}

[Differentiable]
float sdfToCdf(const float sdf, const float sharpness) {
  return 1 / (1 + exp(-sdf * sharpness));
}

[Differentiable]
float computeAlpha(const Sample inner, const Sample outer, const float sharpness) {
  let innerCdf = sdfToCdf(inner.sdf, sharpness);
  let outerCdf = sdfToCdf(outer.sdf, sharpness);
  return clamp((outerCdf - innerCdf + 1e-5) / (outerCdf + 1e-5), 0, 1);
}

[Differentiable]
Accumulator accumulate(
    const Accumulator accumulator,
    const Sample inner,
    const Sample outer,
    const float sharpness,
) {
  let alpha = computeAlpha(inner, outer, sharpness);
  let color = accumulator.rgb + inner.rgb * accumulator.transmittance * alpha;
  let transmittance = accumulator.transmittance * (1 - alpha);
  return { color, transmittance };
}

[Differentiable]
void storeAccumulator(
    const Accumulator accumulator,
    const DiffTensorView outAccumulators,
    const uint2 pixelCoordinates,
) {
  // Save the RGB channels.
  outAccumulators.storeOnce(
      uint3(0, pixelCoordinates.y, pixelCoordinates.x),
      accumulator.rgb.r,
  );
  outAccumulators.storeOnce(
      uint3(1, pixelCoordinates.y, pixelCoordinates.x),
      accumulator.rgb.g,
  );
  outAccumulators.storeOnce(
      uint3(2, pixelCoordinates.y, pixelCoordinates.x),
      accumulator.rgb.b,
  );

  // Instead of saving transmittance directly, save alpha.
  outAccumulators.storeOnce(
      uint3(3, pixelCoordinates.y, pixelCoordinates.x),
      1 - accumulator.transmittance,
  );
}

[Differentiable]
Accumulator
loadAccumulator(const DiffTensorView accumulators, const uint2 pixelCoordinates) {
  return {
    float3(
        accumulators[uint3(0, pixelCoordinates.y, pixelCoordinates.x)],
        accumulators[uint3(1, pixelCoordinates.y, pixelCoordinates.x)],
        accumulators[uint3(2, pixelCoordinates.y, pixelCoordinates.x)]
    ),
    1 - accumulators[uint3(3, pixelCoordinates.y, pixelCoordinates.x)],
  };
}

[Differentiable]
float loadSharpness(const DiffTensorView sharpness) {
  return sharpness[0];
}

[AutoPyBindCUDA]
[CUDAKernel]
void composite_forward(
    const TensorView<int> sortedTriangleIndices,
    const TensorView<int64_t> tileBoundaries,
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signedDistances,
    const TensorView<int> faces,
    const TensorView<int> shellIndices,
    const DiffTensorView sharpness,
    const int numShells,
    const int gridNumCols,
    const int gridRowMinimum,
    const int gridColMinimum,
    const int gridTileHeight,
    const int gridTileWidth,
    const int imageHeight,
    const int imageWidth,
    const DiffTensorView outAccumulators,
    const TensorView<int> voxelIndices,
    const TensorView<int> occupancy,
    const bool markOccupancy,
) {
  // Define the current thread's corresponding integer pixel xy coordinates.
  let localCoordinates = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).xy;
  let globalCoordinates =
      localCoordinates +
      uint2(gridColMinimum * TILE_WIDTH, gridRowMinimum * TILE_HEIGHT);

  // Define the current thread's corresponding global tile index.
  const int tileIndex = (cudaBlockIdx().y + gridRowMinimum) * gridNumCols +
                        (cudaBlockIdx().x + gridColMinimum);

  // Define the current thread's index within the CUDA block.
  const int indexInBlock = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

  // Load the sharpness, which is needed to compute alpha values from signed distances.
  let loadedSharpness = loadSharpness(sharpness);

  // Check whether the current thread's pixel is within the image.
  let isInImage =
      (localCoordinates.x < imageWidth) && (localCoordinates.y < imageHeight);

  // Keep track of the pixel's RGB and transmittance, plus whether triangles should be
  // counted. Triangles are not counted if the current thread's pixel is not in the
  // image or if the transmittance becomes close to zero.
  Accumulator accumulator = { float3(0.f, 0.f, 0.f), 1.f };
  var isActive = isInImage;

  // This counter indicates which shell index was last rendered. A triangle is only
  // rendered if its shell index is greater than all previously rendered triangles'
  // shell indices. If no triangle has been rendered yet, the outer sample is a dummy
  // sample with CDF == 0.
  int lastShell = -1;
  Sample outerSample;

  // Triangles within the tile are processed in block-sized chunks. Calculate the number
  // of such chunks that are needed to process all triangles.
  let tileStart = tileBoundaries[uint2(tileIndex, 0)];
  let tileEnd = tileBoundaries[uint2(tileIndex, 1)];
  let blockSize = gridTileHeight * gridTileHeight;
  let numChunks = (tileEnd - tileStart + blockSize - 1) / blockSize;
  var remainingTriangles = tileEnd - tileStart;
  for (int chunkIndex = 0; chunkIndex < numChunks; chunkIndex++) {
    // Collectively fetch per-triangle data to shared memory.
    AllMemoryBarrierWithGroupSync();
    int64_t keyIndex = tileStart + chunkIndex * blockSize + indexInBlock;
    if (keyIndex < tileEnd) {
      // let loadedTriangleIndex = sortedTriangleIndices[keyIndex];
      let loadedTriangleIndex = load(sortedTriangleIndices, keyIndex);
      sharedTriangles[indexInBlock] = loadTriangle(
          vertices,
          colors,
          signedDistances,
          faces,
          shellIndices,
          loadedTriangleIndex
      );
      if (markOccupancy) {
        sharedVoxelIndices[indexInBlock] = voxelIndices[loadedTriangleIndex];
      }
    }
    AllMemoryBarrierWithGroupSync();

    if (isActive) {
      for (int i = 0; i < min(blockSize, remainingTriangles); i++) {
        let triangle = sharedTriangles[i];

        // Evaluate the current triangle and skip it if it wasn't hit.
        let innerSample = sampleTriangle(triangle, globalCoordinates + 0.5f, lastShell);
        if (!innerSample.hit) {
          continue;
        }

        // For the first sample, use a dummy outer sample with a CDF of zero.
        accumulator = accumulate(
            accumulator,
            innerSample,
            (lastShell == -1) ? Sample(float3(0, 0, 0), 1e10, true) : outerSample,
            loadedSharpness
        );

        // Update the occupancy grid.
        if (markOccupancy) {
          let voxelIndex = sharedVoxelIndices[i];
          let arrayIndex = voxelIndex >> 5;
          let bit = reinterpret<int>(1u << (voxelIndex % 32));
          __atomic_or(*occupancy.data_ptr_at(arrayIndex), bit, MemoryOrder.Relaxed);
        }

        // Update the bookkeeping values.
        outerSample = innerSample;
        lastShell = triangle.shell;
        if (lastShell == numShells - 1) {
          isActive = false;
          break;
        }
      }

      remainingTriangles -= blockSize;
    }
  }

  if (isInImage) {
    storeAccumulator(accumulator, outAccumulators, localCoordinates);
  }
}

[AutoPyBindCUDA]
[CUDAKernel]
void composite_backward(
    const TensorView<int> sortedTriangleIndices,
    const TensorView<int64_t> tileBoundaries,
    const TensorView<float> vertices,
    const DiffTensorView colors,
    const DiffTensorView signedDistances,
    const TensorView<int> faces,
    const TensorView<int> shellIndices,
    const DiffTensorView sharpness,
    const int numShells,
    const int gridNumCols,
    const int gridRowMinimum,
    const int gridColMinimum,
    const int gridTileHeight,
    const int gridTileWidth,
    const int imageHeight,
    const int imageWidth,
    const DiffTensorView outAccumulators,
) {
  // Define the current thread's corresponding integer pixel xy coordinates.
  let localCoordinates = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).xy;
  let globalCoordinates =
      localCoordinates +
      uint2(gridColMinimum * TILE_WIDTH, gridRowMinimum * TILE_HEIGHT);

  // Define the current thread's corresponding global tile index.
  const int tileIndex = (cudaBlockIdx().y + gridRowMinimum) * gridNumCols +
                        (cudaBlockIdx().x + gridColMinimum);

  // Define the current thread's index within the CUDA block.
  const int indexInBlock = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

  // Load the sharpness, which is needed to compute alpha values from signed distances.
  let loadedSharpness = loadSharpness(sharpness);

  // Check whether the current thread's pixel is within the image.
  let isInImage =
      (localCoordinates.x < imageWidth) && (localCoordinates.y < imageHeight);

  // Keep track of the pixel's RGB and transmittance, plus whether triangles should be
  // counted. Triangles are not counted if the current thread's pixel is not in the
  // image or if the transmittance becomes close to zero.
  Accumulator accumulator = { float3(0.f, 0.f, 0.f), 1.f };

  // Reset the bookkeeping values for the second loop.
  int lastShell = -1;
  Sample outerSample;
  var isActive = isInImage;

  // Since each alpha value depends on two samples' SDF values, we need to accumulate
  // each sample's differential across two triangles/loop iterations.
  var dSamplePartial = Sample.Differential(float3(0, 0, 0), 0);
  var dSamplePartialIndex = -1;

  // Load the accumulator and its gradients.
  Accumulator finalAccumulator;
  if (isInImage) {
    finalAccumulator = loadAccumulator(outAccumulators, localCoordinates);
  }
  var dpFinalAccumulator = diffPair(finalAccumulator);
  if (isInImage) {
    bwd_diff(storeAccumulator)(dpFinalAccumulator, outAccumulators, localCoordinates);
  }

  // Triangles within the tile are processed in block-sized chunks. Calculate the
  // number of such chunks that are needed to process all triangles.
  let tileStart = tileBoundaries[uint2(tileIndex, 0)];
  let tileEnd = tileBoundaries[uint2(tileIndex, 1)];
  let blockSize = gridTileHeight * gridTileHeight;
  let numChunks = (tileEnd - tileStart + blockSize - 1) / blockSize;
  float dSharpness = 0;
  Triangle lastTriangle;
  var remainingTriangles = tileEnd - tileStart;
  for (int chunkIndex = 0; chunkIndex < numChunks; chunkIndex++) {
    // Collectively fetch per-triangle data to shared memory.
    AllMemoryBarrierWithGroupSync();
    uint64_t keyIndex = tileStart + chunkIndex * blockSize + indexInBlock;
    if (keyIndex < tileEnd) {
      // let loadedTriangleIndex = sortedTriangleIndices[keyIndex];
      let loadedTriangleIndex = load(sortedTriangleIndices, keyIndex);
      sharedTriangles[indexInBlock] = loadTriangle(
          vertices,
          colors,
          signedDistances,
          faces,
          shellIndices,
          loadedTriangleIndex
      );
      sharedTriangleIndices[indexInBlock] = loadedTriangleIndex;
    }
    AllMemoryBarrierWithGroupSync();

    if (isActive) {
      for (int i = 0; i < min(blockSize, remainingTriangles); i++) {
        let triangle = sharedTriangles[i];

        // Evaluate the current triangle and skip it if it wasn't hit.
        let innerSample = sampleTriangle(triangle, globalCoordinates + 0.5f, lastShell);
        if (!innerSample.hit) {
          continue;
        }
        lastTriangle = triangle;

        // For the first sample, use a dummy outer sample with a CDF of zero.
        let dummy = Sample(float3(0, 0, 0), 1e10, true);
        let outerSampleOrDummy = (lastShell == -1) ? dummy : outerSample;
        let alpha = computeAlpha(innerSample, outerSampleOrDummy, loadedSharpness);

        // Use the classic suffix trick to compute gradients for this layer's color and
        // alpha values. Note that dColor applies to the inner sample.
        let weight = alpha * accumulator.transmittance;
        accumulator.rgb += weight * innerSample.rgb;
        let suffix = finalAccumulator.rgb - accumulator.rgb;
        let inverseAlpha = 1 / (1 - alpha + EPSILON);
        let dColor = weight * dpFinalAccumulator.d.rgb;
        let dAlphaFromColor =
            dot(dpFinalAccumulator.d.rgb,
                accumulator.transmittance * innerSample.rgb - suffix * inverseAlpha);
        let dAlphaFromTransmittance = -dpFinalAccumulator.d.transmittance *
                                      finalAccumulator.transmittance * inverseAlpha;
        let dAlpha = dAlphaFromColor + dAlphaFromTransmittance;

        // Backpropagate through computing the alpha value.
        var dpInnerSample = diffPair(innerSample);
        var dpOuterSample = diffPair(outerSampleOrDummy);
        var dpSharpness = diffPair(loadedSharpness);
        bwd_diff(computeAlpha)(dpInnerSample, dpOuterSample, dpSharpness, dAlpha);
        dSharpness += dpSharpness.d;

        // For all triangles except the first triangle, accumulate the resulting
        // gradients in dSamplePartial, then backpropagate through sampling and loading
        // the triangle. dSamplePartial already contains the gradients from when the
        // triangle corresponded to the inner sample; we add the gradients from when
        // it corresponds to the outer sample.
        if (lastShell != -1) {
          // Note that the outer sample does not contribute to the color.
          dSamplePartial.sdf += dpOuterSample.d.sdf;

          // Backpropagate through loading and evaluating the triangle.
          var dpTriangle = diffPair(triangle);
          bwd_diff(sampleTriangle)(
              dpTriangle,
              globalCoordinates + 0.5f,
              lastShell,
              dSamplePartial,
          );
          bwd_diff(loadTriangle)(
              vertices,
              colors,
              signedDistances,
              faces,
              shellIndices,
              dSamplePartialIndex,
              dpTriangle.d,
          );
        }

        // The next dSamplePartial becomes dSample from the inner sample. Note that
        // backpropagating through computeAlpha does not send gradients to the sample
        // colors; the sample color gradients come directly from compositing.
        dSamplePartial = Sample.Differential(dColor, dpInnerSample.d.sdf);
        dSamplePartialIndex = sharedTriangleIndices[i];

        // This has to be updated after computing the gradients.
        accumulator.transmittance *= 1 - alpha;

        // Update the bookkeeping values.
        outerSample = innerSample;
        lastShell = triangle.shell;
        if (lastShell == numShells - 1) {
          isActive = false;
          break;
        }
      }

      remainingTriangles -= blockSize;
    }
  }

  // Everything after this point is only relevant if the thread corresponds to a pixel
  // that's in the image.
  if (!isInImage) {
    return;
  }

  if (lastShell != -1) {
    // Decrement lastShell so that sampleTriangle doesn't exit early.
    lastShell--;

    // Backpropagate through loading and evaluating the last triangle.
    var dpTriangle = diffPair(lastTriangle);
    bwd_diff(sampleTriangle)(
        dpTriangle,
        globalCoordinates + 0.5f,
        lastShell,
        dSamplePartial,
    );
    bwd_diff(loadTriangle)(
        vertices,
        colors,
        signedDistances,
        faces,
        shellIndices,
        dSamplePartialIndex,
        dpTriangle.d,
    );
  }

  bwd_diff(loadSharpness)(sharpness, dSharpness);
}
