float2 loadVertex(const DiffTensorView vertices, const uint index) {
  return float2(vertices[uint2(index, 0)], vertices[uint2(index, 1)]);
}

float loadDepth(const DiffTensorView depths, const uint index) {
  return depths[uint2(index, 0)];
}

struct Triangle {
  float2 a;
  float2 b;
  float2 c;
  float da; // depth
  float db;
  float dc;
}

Triangle loadTriangle(
    const DiffTensorView vertices,
    const DiffTensorView depths,
    const TensorView<int> faces,
    const uint index,
) {
  return {
    loadVertex(vertices, faces[uint2(index, 0)]),
    loadVertex(vertices, faces[uint2(index, 1)]),
    loadVertex(vertices, faces[uint2(index, 2)]),
    loadDepth(depths, faces[uint2(index, 0)]),
    loadDepth(depths, faces[uint2(index, 1)]),
    loadDepth(depths, faces[uint2(index, 2)]),
  };
}

struct TileMembership {
  int2 y;
  int2 x;
}

int2 computeTileBounds(
    const float minimum,
    const float maximum,
    const int tileSize,
    const int minBound,
    const int maxBound,
) {
  // Compute the minimum (inclusive) and maximum (exclusive) tile bounds for the given
  // minimum and maximum float values.
  let tileMinimum = int(minimum) / tileSize;
  let tileMaximum = int(ceil(maximum / tileSize));

  // Ensure that the tile bounds are valid by clamping them.
  return int2(
      clamp(tileMinimum, minBound, maxBound),
      clamp(tileMaximum, minBound, maxBound),
  );
}

TileMembership computeTileMembership(
    const Triangle triangle,
    const int gridTileHeight,
    const int gridTileWidth,
    const int gridRowMinimum,
    const int gridColMinimum,
    const int gridRowMaximum,
    const int gridColMaximum,
    const int backfaceCulling,
    const float nearPlane,
) {
  // Do back-face culling by assigned back-facing triangles to no tiles.
  let a = triangle.a;
  let b = triangle.b;
  let c = triangle.c;
  let da = triangle.da;
  let db = triangle.db;
  let dc = triangle.dc;
  let check = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  let minDepth = min(min(da, db), dc);
  if (check < 0 && backfaceCulling == 1) {
    // The triangle is back-facing, so we assign it to no tiles.
    return TileMembership(int2(0, 0), int2(0, 0));
  } else if (check > 0 && backfaceCulling == -1) {
    // The triangle is front-facing, so we assign it to no tiles.
    return TileMembership(int2(0, 0), int2(0, 0));
  } else if (minDepth < nearPlane) {
    // The triangle is too close to the near plane, so don't render it.
    return TileMembership(int2(0, 0), int2(0, 0));
  }

  let xMin = min(min(triangle.a.xy.x, triangle.b.xy.x), triangle.c.xy.x);
  let xMax = max(max(triangle.a.xy.x, triangle.b.xy.x), triangle.c.xy.x);

  let yMin = min(min(triangle.a.xy.y, triangle.b.xy.y), triangle.c.xy.y);
  let yMax = max(max(triangle.a.xy.y, triangle.b.xy.y), triangle.c.xy.y);

  return {
    computeTileBounds(yMin, yMax, gridTileHeight, gridRowMinimum, gridRowMaximum),
    computeTileBounds(xMin, xMax, gridTileWidth, gridColMinimum, gridColMaximum),
  };
}

void storeTileMembership(
    const TileMembership tileMembership,
    TensorView<int32_t> outTileMinima,
    TensorView<int32_t> outTileMaxima,
    TensorView<int32_t> outNumTilesTouched,
    const uint32_t index,
) {
  outTileMinima[uint2(index, 0)] = tileMembership.y[0];
  outTileMinima[uint2(index, 1)] = tileMembership.x[0];

  outTileMaxima[uint2(index, 0)] = tileMembership.y[1];
  outTileMaxima[uint2(index, 1)] = tileMembership.x[1];

  outNumTilesTouched[index] = (tileMembership.y[1] - tileMembership.y[0]) *
                              (tileMembership.x[1] - tileMembership.x[0]);
}

[AutoPyBindCUDA]
[CUDAKernel]
void bound_triangles(
    const DiffTensorView vertices,
    const DiffTensorView depths,
    const TensorView<int> faces,
    const int gridTileHeight,
    const int gridTileWidth,
    const int gridRowMinimum,
    const int gridColMinimum,
    const int gridRowMaximum,
    const int gridColMaximum,
    TensorView<int32_t> outTileMinima,
    TensorView<int32_t> outTileMaxima,
    TensorView<int32_t> outNumTilesTouched,
    const int backfaceCulling,
    const float nearPlane,
) {
  // Skip past invalid triangles.
  let index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= faces.size(0)) {
    return;
  }

  // Compute tile overlaps for the projected triangle.
  let tileMembership = computeTileMembership(
      loadTriangle(vertices, depths, faces, index),
      gridTileHeight,
      gridTileWidth,
      gridRowMinimum,
      gridColMinimum,
      gridRowMaximum,
      gridColMaximum,
      backfaceCulling,
      nearPlane,
  );

  // Store the results.
  storeTileMembership(
      tileMembership,
      outTileMinima,
      outTileMaxima,
      outNumTilesTouched,
      index,
  );
}
