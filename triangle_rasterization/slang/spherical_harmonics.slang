// Taken from George Kopanas
import "../../slang_common/large_arrays.slang";

static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
  1.0925484305920792f,
  -1.0925484305920792f,
  0.31539156525252005f,
  -1.0925484305920792f,
  0.5462742152960396f
};
static const float SH_C3[] = {
  -0.5900435899266435f,
  2.890611442640554f,
  -0.4570457994644658f,
  0.3731763325901154f,
  -0.4570457994644658f,
  1.445305721320277f,
  -0.5900435899266435f
};

struct SpherHarmCoeffs : IDifferentiable {
  // Band 0
  float3 coeff0;

  // Band 1
  float3 coeff1;
  float3 coeff2;
  float3 coeff3;

  // Band2
  float3 coeff4;
  float3 coeff5;
  float3 coeff6;
  float3 coeff7;
  float3 coeff8;

  // Band3
  float3 coeff9;
  float3 coeff10;
  float3 coeff11;
  float3 coeff12;
  float3 coeff13;
  float3 coeff14;
  float3 coeff15;
};

[Differentiable]
SpherHarmCoeffs read_spherical_harmonics_coeffs(
    const uint32_t g_idx,
    const DiffTensorViewLarge sphericalHarmonics,
    const uint32_t active_sh
) {
  SpherHarmCoeffs g_sh_coeffs;
  g_sh_coeffs.coeff0 = float3(
      sphericalHarmonics.load(uint3(0, g_idx, 0)),
      sphericalHarmonics.load(uint3(0, g_idx, 1)),
      sphericalHarmonics.load(uint3(0, g_idx, 2))
  );

  if (active_sh > 0) {
    g_sh_coeffs.coeff1 = float3(
        sphericalHarmonics.load(uint3(1, g_idx, 0)),
        sphericalHarmonics.load(uint3(1, g_idx, 1)),
        sphericalHarmonics.load(uint3(1, g_idx, 2))
    );
    g_sh_coeffs.coeff2 = float3(
        sphericalHarmonics.load(uint3(2, g_idx, 0)),
        sphericalHarmonics.load(uint3(2, g_idx, 1)),
        sphericalHarmonics.load(uint3(2, g_idx, 2))
    );
    g_sh_coeffs.coeff3 = float3(
        sphericalHarmonics.load(uint3(3, g_idx, 0)),
        sphericalHarmonics.load(uint3(3, g_idx, 1)),
        sphericalHarmonics.load(uint3(3, g_idx, 2))
    );

    if (active_sh > 1) {
      g_sh_coeffs.coeff4 = float3(
          sphericalHarmonics.load(uint3(4, g_idx, 0)),
          sphericalHarmonics.load(uint3(4, g_idx, 1)),
          sphericalHarmonics.load(uint3(4, g_idx, 2))
      );
      g_sh_coeffs.coeff5 = float3(
          sphericalHarmonics.load(uint3(5, g_idx, 0)),
          sphericalHarmonics.load(uint3(5, g_idx, 1)),
          sphericalHarmonics.load(uint3(5, g_idx, 2))
      );
      g_sh_coeffs.coeff6 = float3(
          sphericalHarmonics.load(uint3(6, g_idx, 0)),
          sphericalHarmonics.load(uint3(6, g_idx, 1)),
          sphericalHarmonics.load(uint3(6, g_idx, 2))
      );
      g_sh_coeffs.coeff7 = float3(
          sphericalHarmonics.load(uint3(7, g_idx, 0)),
          sphericalHarmonics.load(uint3(7, g_idx, 1)),
          sphericalHarmonics.load(uint3(7, g_idx, 2))
      );
      g_sh_coeffs.coeff8 = float3(
          sphericalHarmonics.load(uint3(8, g_idx, 0)),
          sphericalHarmonics.load(uint3(8, g_idx, 1)),
          sphericalHarmonics.load(uint3(8, g_idx, 2))
      );

      if (active_sh > 2) {
        g_sh_coeffs.coeff9 = float3(
            sphericalHarmonics.load(uint3(9, g_idx, 0)),
            sphericalHarmonics.load(uint3(9, g_idx, 1)),
            sphericalHarmonics.load(uint3(9, g_idx, 2))
        );
        g_sh_coeffs.coeff10 = float3(
            sphericalHarmonics.load(uint3(10, g_idx, 0)),
            sphericalHarmonics.load(uint3(10, g_idx, 1)),
            sphericalHarmonics.load(uint3(10, g_idx, 2))
        );
        g_sh_coeffs.coeff11 = float3(
            sphericalHarmonics.load(uint3(11, g_idx, 0)),
            sphericalHarmonics.load(uint3(11, g_idx, 1)),
            sphericalHarmonics.load(uint3(11, g_idx, 2))
        );
        g_sh_coeffs.coeff12 = float3(
            sphericalHarmonics.load(uint3(12, g_idx, 0)),
            sphericalHarmonics.load(uint3(12, g_idx, 1)),
            sphericalHarmonics.load(uint3(12, g_idx, 2))
        );
        g_sh_coeffs.coeff13 = float3(
            sphericalHarmonics.load(uint3(13, g_idx, 0)),
            sphericalHarmonics.load(uint3(13, g_idx, 1)),
            sphericalHarmonics.load(uint3(13, g_idx, 2))
        );
        g_sh_coeffs.coeff14 = float3(
            sphericalHarmonics.load(uint3(14, g_idx, 0)),
            sphericalHarmonics.load(uint3(14, g_idx, 1)),
            sphericalHarmonics.load(uint3(14, g_idx, 2))
        );
        g_sh_coeffs.coeff15 = float3(
            sphericalHarmonics.load(uint3(15, g_idx, 0)),
            sphericalHarmonics.load(uint3(15, g_idx, 1)),
            sphericalHarmonics.load(uint3(15, g_idx, 2))
        );
      }
    }
  }
  return g_sh_coeffs;
}

[Differentiable]
float3 compute_color_from_sh_coeffs(
    SpherHarmCoeffs sh,
    float3 vertex_pos,
    float3 cam_pos,
    uint32_t active_sh
) {
  float3 dir = vertex_pos - cam_pos;
  dir = normalize(dir);

  float3 rgb = SH_C0 * sh.coeff0;
  if (active_sh > 0) {
    rgb = rgb - SH_C1 * dir.y * sh.coeff1 + SH_C1 * dir.z * sh.coeff2 -
          SH_C1 * dir.x * sh.coeff3;
    if (active_sh > 1) {
      float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
      float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;
      rgb = rgb + SH_C2[0] * xy * sh.coeff4 + SH_C2[1] * yz * sh.coeff5 +
            SH_C2[2] * (2.0f * zz - xx - yy) * sh.coeff6 + SH_C2[3] * xz * sh.coeff7 +
            SH_C2[4] * (xx - yy) * sh.coeff8;

      if (active_sh > 2) {
        rgb = rgb + SH_C3[0] * dir.y * (3.0f * xx - yy) * sh.coeff9 +
              SH_C3[1] * xy * dir.z * sh.coeff10 +
              SH_C3[2] * dir.y * (4.0f * zz - xx - yy) * sh.coeff11 +
              SH_C3[3] * dir.z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh.coeff12 +
              SH_C3[4] * dir.x * (4.0f * zz - xx - yy) * sh.coeff13 +
              SH_C3[5] * dir.z * (xx - yy) * sh.coeff14 +
              SH_C3[6] * dir.x * (xx - 3.0f * yy) * sh.coeff15;
      }
    }
  }

  rgb += 0.5;

  return max(rgb, 0.0);
}
