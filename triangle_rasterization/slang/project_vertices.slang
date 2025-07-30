import spherical_harmonics;
import "../../slang_common/large_arrays.slang";

struct Camera {
  float4x4 extrinsics;
  float3x3 intrinsics;
  float3 position;
}

Camera loadCamera(
    const TensorView<float> extrinsics,
    const TensorView<float> intrinsics,
    const float3 camPos
) {
  let extrinsicsMatrix = float4x4(
      extrinsics[uint2(0, 0)],
      extrinsics[uint2(0, 1)],
      extrinsics[uint2(0, 2)],
      extrinsics[uint2(0, 3)],
      extrinsics[uint2(1, 0)],
      extrinsics[uint2(1, 1)],
      extrinsics[uint2(1, 2)],
      extrinsics[uint2(1, 3)],
      extrinsics[uint2(2, 0)],
      extrinsics[uint2(2, 1)],
      extrinsics[uint2(2, 2)],
      extrinsics[uint2(2, 3)],
      extrinsics[uint2(3, 0)],
      extrinsics[uint2(3, 1)],
      extrinsics[uint2(3, 2)],
      extrinsics[uint2(3, 3)]
  );
  let intrinsicsMatrix = float3x3(
      intrinsics[uint2(0, 0)],
      intrinsics[uint2(0, 1)],
      intrinsics[uint2(0, 2)],
      intrinsics[uint2(1, 0)],
      intrinsics[uint2(1, 1)],
      intrinsics[uint2(1, 2)],
      intrinsics[uint2(2, 0)],
      intrinsics[uint2(2, 1)],
      intrinsics[uint2(2, 2)],
  );
  return { extrinsicsMatrix, intrinsicsMatrix, camPos };
}

struct Vertex2D : IDifferentiable {
  float2 xy;
  float depth;
  float3 rgb;
}

[Differentiable]
Vertex2D projectVertex(
    const float3 vertex,
    const SpherHarmCoeffs shCoeffs,
    const Camera camera,
    const int activeSH
) {
  // Transform the vertex into camera space.
  let cameraSpaceVertex = mul(camera.extrinsics, float4(vertex, 1));

  // Compute the vertex color from SH coeffs
  float3 rgb =
      compute_color_from_sh_coeffs(shCoeffs, vertex, camera.position, activeSH);

  // Apply the intrinsics and divide by depth.
  let depth = cameraSpaceVertex.z;
  let xy = mul(camera.intrinsics, cameraSpaceVertex.xyz).xy / depth;
  return { xy, depth, rgb };
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void project_vertices(
    const DiffTensorView vertices,
    const DiffTensorViewLarge sphericalHarmonics,
    const TensorView<float> extrinsics,
    const TensorView<float> intrinsics,
    const TensorView<float> camPos,
    const int activeSH,
    DiffTensorView outVertices,
    DiffTensorView outDepths,
    DiffTensorView outColors,
) {
  // Skip past invalid vertices.
  let index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
  if (index >= vertices.size(0)) {
    return;
  }

  let shCoeffs = read_spherical_harmonics_coeffs(index, sphericalHarmonics, activeSH);
  float3 camPos = float3(camPos[0], camPos[1], camPos[2]);

  // Load the vertex and the camera.
  let vertex = float3(
      vertices[uint2(index, 0)],
      vertices[uint2(index, 1)],
      vertices[uint2(index, 2)]
  );
  let camera = loadCamera(extrinsics, intrinsics, camPos);

  let projected = projectVertex(vertex, shCoeffs, camera, activeSH);

  // Save the vertex's projected position and depth.
  outVertices.storeOnce(uint2(index, 0), projected.xy.x);
  outVertices.storeOnce(uint2(index, 1), projected.xy.y);
  outDepths.storeOnce(index, projected.depth);
  outColors.storeOnce(uint2(index, 0), projected.rgb.x);
  outColors.storeOnce(uint2(index, 1), projected.rgb.y);
  outColors.storeOnce(uint2(index, 2), projected.rgb.z);
}
