#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
template <int N>
UIPC_GENERIC void make_spd(Matrix<Float, N, N>& H)
{
    Vector<Float, N>    eigen_values;
    Matrix<Float, N, N> eigen_vectors;
    cuda_tool::eigen::template evd<Float, N>(H, eigen_values, eigen_vectors);
    for(int i = 0; i < N; ++i)
    {
        auto& v = eigen_values(i);
        v       = v < 0.0 ? 0.0 : v;
    }
    H = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}

// perf/kernels (K7): PSD projection of a 12x12 four-vertex element Hessian
// that annihilates rigid translations (H t = 0 for t = (e_k, e_k, e_k, e_k),
// k = x, y, z), e.g. a discrete-shell hinge whose energy depends on the
// dihedral angle only. The non-trivial part of H lives in the 9-dim
// orthogonal complement of the translations, which is the same constant
// subspace for every element: with Q (12x9, orthonormal columns = Helmert
// basis of R^4 (x) I_3) we have H = Q (Q^T H Q) Q^T exactly, the eigenvalues
// of the 9x9 restriction are exactly the non-zero eigenvalues of H, and
// clamping them and mapping back gives the same projection as make_spd<12>
// (up to floating-point rounding) at a fraction of the eigen-solver cost.
// perf/kernels (K16): the same projection assembled from 3x3 blocks with
// the constant Helmert weights (no 12x9 basis matrix, no 12x9 temporaries;
// zero weights skipped exactly) to cut the register/local-memory traffic of
// the hinge kernel. Same math, different summation order (rounding-level).
inline UIPC_GENERIC void make_spd_translation_free_4x3_blocked(Matrix12x12& H)
{
    constexpr Float r2  = 0.70710678118654752440;
    constexpr Float r6  = 0.40824829046386301637;
    constexpr Float r12 = 0.28867513459481288225;
    constexpr Float h[3][4] = {{r2, -r2, 0.0, 0.0},
                               {r6, r6, -2.0 * r6, 0.0},
                               {r12, r12, r12, -3.0 * r12}};
    constexpr int   nz[3] = {2, 3, 4};  // non-zero weights of row j: a < nz[j]
    Eigen::Matrix<Float, 9, 9> Hr;
    for(int j = 0; j < 3; ++j)
        for(int k = 0; k < 3; ++k)
        {
            Matrix3x3 B = Matrix3x3::Zero();
            for(int a = 0; a < nz[j]; ++a)
                for(int b = 0; b < nz[k]; ++b)
                    B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
            Hr.template block<3, 3>(3 * j, 3 * k) = B;
        }
    make_spd<9>(Hr);
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

inline UIPC_GENERIC void make_spd_translation_free_4x3(Matrix12x12& H)
{
    // Helmert rows: orthonormal, each orthogonal to (1,1,1,1)
    constexpr Float r2 = 0.70710678118654752440;   // 1/sqrt(2)
    constexpr Float r6 = 0.40824829046386301637;   // 1/sqrt(6)
    constexpr Float r12 = 0.28867513459481288225;  // 1/sqrt(12)
    constexpr Float h[3][4] = {{r2, -r2, 0.0, 0.0},
                               {r6, r6, -2.0 * r6, 0.0},
                               {r12, r12, r12, -3.0 * r12}};
    Eigen::Matrix<Float, 12, 9> Q = Eigen::Matrix<Float, 12, 9>::Zero();
    for(int j = 0; j < 3; ++j)
        for(int a = 0; a < 4; ++a)
            for(int k = 0; k < 3; ++k)
                Q(3 * a + k, 3 * j + k) = h[j][a];
    Eigen::Matrix<Float, 9, 9> Hr = Q.transpose() * H * Q;
    make_spd<9>(Hr);
    H = Q * Hr * Q.transpose();
}
}  // namespace uipc::backend::cuda