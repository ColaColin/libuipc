// s08 static-cost probe: attribute the FP64 instruction count of
// make_spd_translation_free_4x3_blocked<1> between (a) the K16 block
// assembly (forward restriction + back-assemble) and (b) the 9x9 QL
// eigen-solve. Static counts; the QL loops are runtime loops so the solver
// side is undercounted statically by (iterations-1) x body.
#include <utils/make_spd.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
// variant under test selected by template tag
// 0 = full helper (assembly + solver)
// 1 = assembly only (solver skipped: forward + back on raw Hr)
// 2 = forward assembly only (Hr dumped into H to stay live)
template <int Variant>
__global__ void probe_kernel(const Float* in, Float* out, int n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Matrix12x12         H;
        Eigen::Matrix<Float, 9, 9> Hr;
#pragma unroll
        for(int i = 0; i < 12; ++i)
#pragma unroll
            for(int j = 0; j < 12; ++j)
                H(i, j) = in[I * 144 + i * 12 + j];

        if constexpr(Variant == 0)
        {
            make_spd_translation_free_4x3_blocked<1>(H);
        }
        else
        {
            constexpr Float r2      = 0.70710678118654752440;
            constexpr Float r6      = 0.40824829046386301637;
            constexpr Float r12     = 0.28867513459481288225;
            constexpr Float h[3][4] = {{r2, -r2, 0.0, 0.0},
                                       {r6, r6, -2.0 * r6, 0.0},
                                       {r12, r12, r12, -3.0 * r12}};
            constexpr int nz[3] = {2, 3, 4};
            for(int j = 0; j < 3; ++j)
                for(int k = 0; k < 3; ++k)
                {
                    Matrix3x3 B = Matrix3x3::Zero();
                    for(int a = 0; a < nz[j]; ++a)
                        for(int b = 0; b < nz[k]; ++b)
                            B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
                    Hr.template block<3, 3>(3 * j, 3 * k) = B;
                }
            if constexpr(Variant == 1)
            {
                for(int a = 0; a < 4; ++a)
                    for(int b = 0; b < 4; ++b)
                    {
                        Matrix3x3 B = Matrix3x3::Zero();
                        for(int j = 0; j < 3; ++j)
                        {
                            if(a >= nz[j])
                                continue;
                            for(int k = 0; k < 3; ++k)
                            {
                                if(b >= nz[k])
                                    continue;
                                B += (h[j][a] * h[k][b]) * Hr.template block<3, 3>(3 * j, 3 * k);
                            }
                        }
                        H.template block<3, 3>(3 * a, 3 * b) = B;
                    }
            }
            else
            {
                // dump Hr into H to keep it live
                for(int i = 0; i < 9; ++i)
                    for(int j = 0; j < 9; ++j)
                        H(i < 3 ? i : (i + 3 > 11 ? 11 : i + 3), j) = Hr(i, j);
            }
        }

#pragma unroll
        for(int i = 0; i < 12; ++i)
#pragma unroll
            for(int j = 0; j < 12; ++j)
                out[I * 144 + i * 12 + j] = H(i, j);
    }
template __global__ void probe_kernel<0>(const Float*, Float*, int);
template __global__ void probe_kernel<1>(const Float*, Float*, int);
template __global__ void probe_kernel<2>(const Float*, Float*, int);
}  // namespace uipc::backend::cuda
