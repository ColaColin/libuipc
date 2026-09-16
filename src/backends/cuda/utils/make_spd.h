#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
// perf/round5 (w0, s19): `Solver` picks the eigen-solve behind the PSD
// projection. 0 = Eigen's SelfAdjointEigenSolver (the path every round up to
// 4 used, kept as the rollback and as the A/B reference), 1 = the fixed-size
// Householder + implicit-QL of `cuda_tool::eigen::evd_tridiag_ql`, which runs
// the same algorithm without Eigen's dynamic-size block expressions and its
// out-of-line selfadjoint_matrix_vector_product. N <= 3 always takes Eigen's
// closed-form `computeDirect`, which is cheaper than either. It is a template
// parameter, not a runtime flag, so each instantiation carries only one code
// path's stack frame (the s14 lesson). The default is 0 so that s19 changed
// exactly the two call sites it measured (the discrete-shell hinge and the ABD
// ortho potential). The contact branches were wired on to the same axis later
// -- s25 for part 1 (PT+EE, `SpdTql`) and s31 for part 2 (PE+PP, `Spd2`) --
// so only the cold call sites still take Solver = 0. The env switch that
// drives the two s19 sites is named UIPC_MAKE_SPD_JACOBI for historical
// reasons: it selects `evd_tridiag_ql`, NOT a Jacobi sweep. Round 5 measured
// cyclic Jacobi here and rejected it (R1); `evd_jacobi` in cuda_tool/eigen has
// no callers at all.
template <int N, int Solver = 0>
UIPC_GENERIC void make_spd(Matrix<Float, N, N>& H)
{
    Vector<Float, N>    eigen_values;
    Matrix<Float, N, N> eigen_vectors;
    if constexpr(Solver == 0 || N <= 3)
    {
        cuda_tool::eigen::template evd<Float, N>(H, eigen_values, eigen_vectors);
        for(int i = 0; i < N; ++i)
        {
            auto& v = eigen_values(i);
            v       = v < 0.0 ? 0.0 : v;
        }
        H = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
    }
    else
    {
        cuda_tool::eigen::template evd_tridiag_ql<Float, N>(H, eigen_values, eigen_vectors);
        for(int i = 0; i < N; ++i)
        {
            auto& v = eigen_values(i);
            v       = v < 0.0 ? 0.0 : v;
        }
        // V diag(w) V^T is symmetric by construction: only the upper triangle
        // is summed and mirrored, which halves the reconstruction and drops the
        // N x N temporary of the Eigen product.
#pragma unroll
        for(int i = 0; i < N; ++i)
#pragma unroll
            for(int j = i; j < N; ++j)
            {
                Float sum = 0.0;
#pragma unroll
                for(int k = 0; k < N; ++k)
                    sum += eigen_vectors(i, k) * eigen_values(k) * eigen_vectors(j, k);
                H(i, j) = sum;
                H(j, i) = sum;
            }
    }
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
//
// perf/round7 (s08): SymAsm picks how much of the block assembly's redundant
// triangle work is skipped. The full-triangle assembly computes Hr's upper
// triangle (blocks j < k and the upper entries of the diagonal blocks) even
// though *neither* eigen-solver behind make_spd<9, N> reads it -- Eigen's
// SelfAdjointEigenSolver::compute and the fixed-size tred2 both reference the
// lower triangle only (proved on device: 2e5 NaN-poisoned random matrices per
// N in {6, 9} per solver, 0 output words differ; the lower-triangle poison
// control mismatches 100%) -- and it back-assembles all 16 blocks of the
// symmetric result H although the lower 6 are the transposed upper blocks.
//   SymAsm = 0  the full-triangle assembly (the pre-s08 code, the rollback;
//               kept verbatim so the instantiation is byte-identical)
//   SymAsm = 1  forward cut only (the numerics-isolation arm: the kept
//               entries run the same expressions in the same order as 0, the
//               dead entries are zero -> bit-identical outputs)
//   SymAsm = 2  forward cut + mirrored back-assembly: the upper 10 blocks of
//               the returned H are computed with 0's exact expressions/order
//               (bit-identical), the lower 6 are their exact transposes
//               (reassociation-level difference only). Default.
// The static FP64-arithmetic count of the assembly drops 1458 -> 876 FMAs
// (-40%). Callers that do not pass SymAsm get the fast default. The env
// switch that drives the wired call sites is UIPC_MAKE_SPD_BLOCKED_HALF=0
// (one shared helper-level name after the UIPC_MAKE_SPD_JACOBI precedent).
template <int Solver = 0, int SymAsm = 2>
inline UIPC_GENERIC void make_spd_translation_free_4x3_blocked(Matrix12x12& H)
{
    constexpr Float r2      = 0.70710678118654752440;
    constexpr Float r6      = 0.40824829046386301637;
    constexpr Float r12     = 0.28867513459481288225;
    constexpr Float h[3][4] = {{r2, -r2, 0.0, 0.0},
                               {r6, r6, -2.0 * r6, 0.0},
                               {r12, r12, r12, -3.0 * r12}};
    constexpr int   nz[3] = {2, 3, 4};  // non-zero weights of row j: a < nz[j]
    Eigen::Matrix<Float, 9, 9> Hr;
    if constexpr(SymAsm == 0)
    {
        for(int j = 0; j < 3; ++j)
            for(int k = 0; k < 3; ++k)
            {
                Matrix3x3 B = Matrix3x3::Zero();
                for(int a = 0; a < nz[j]; ++a)
                    for(int b = 0; b < nz[k]; ++b)
                        B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
                Hr.template block<3, 3>(3 * j, 3 * k) = B;
            }
    }
    else
    {
        // dead-triangle cut: only Hr's lower triangle is assembled. The dead
        // entries are zeroed so no uninitialized value is ever stored.
        Hr.setZero();
#pragma unroll
        for(int j = 0; j < 3; ++j)
#pragma unroll
            for(int k = 0; k <= j; ++k)
            {
                if(j == k)
                {
                    // diagonal block: only its lower triangle is read
#pragma unroll
                    for(int p = 0; p < 3; ++p)
#pragma unroll
                        for(int q = 0; q <= p; ++q)
                        {
                            Float s = 0.0;
#pragma unroll
                            for(int a = 0; a < nz[j]; ++a)
#pragma unroll
                                for(int b = 0; b < nz[j]; ++b)
                                    s += (h[j][a] * h[j][b]) * H(3 * a + p, 3 * b + q);
                            Hr(3 * j + p, 3 * j + q) = s;
                        }
                    continue;
                }
                Matrix3x3 B = Matrix3x3::Zero();
                for(int a = 0; a < nz[j]; ++a)
                    for(int b = 0; b < nz[k]; ++b)
                        B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
                Hr.template block<3, 3>(3 * j, 3 * k) = B;
            }
    }
    make_spd<9, Solver>(Hr);
    if constexpr(SymAsm == 2)
    {
        // mirrored back-assembly: the projected H is symmetric (Q P Q^T with
        // P symmetric), so the lower 3x3 blocks are the transposed upper
        // blocks. The upper blocks run 0's exact expression order.
#pragma unroll
        for(int a = 0; a < 4; ++a)
#pragma unroll
            for(int b = a; b < 4; ++b)
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
                if(a != b)
                    H.template block<3, 3>(3 * b, 3 * a) = B.transpose();
            }
    }
    else
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
}

// perf/round7 (s07): the same projection for a THREE-vertex element Hessian
// (9x9) that annihilates rigid translations, e.g. a triangle membrane
// (NeoHookeanShell2D) whose energy depends on the vertices only through edge
// differences. The non-trivial part of H lives in the 6-dim orthogonal
// complement of the translations (Helmert basis of R^3 (x) I_3), so the
// eigenproblem shrinks 9x9 -> 6x6 and the eigenvalues of the restriction are
// exactly the non-zero eigenvalues of H. Same math up to rounding as
// make_spd<9>, assembled from 3x3 blocks with the constant Helmert weights
// (the K16 construction, one vertex fewer).
//
// perf/round7 (s08) NOTE: the s08 dead-triangle cut + mirrored back-assembly
// (see make_spd_translation_free_4x3_blocked, SymAsm) was tried here too and
// REJECTED on measurement: the 6x6 solvers likewise ignore the upper
// triangle (the NaN-poison proof covers N = 6), but three formulations of
// the cut (entry-wise, block-shaped with a zero store, block-shaped with a
// mirrored store) all measured ~4 % SLOWER than this full assembly in
// isolation (68.5-68.8 vs 71.2-72.3 ns/matrix, interleaved) -- at this size
// the deleted ~126 FMAs are worth less than the schedule the compiler had.
// Do not retry without a microbenchmark.
template <int Solver = 0>
inline UIPC_GENERIC void make_spd_translation_free_3x3_blocked(Matrix9x9& H)
{
    constexpr Float r2      = 0.70710678118654752440;
    constexpr Float r6      = 0.40824829046386301637;
    constexpr Float h[2][3] = {{r2, -r2, 0.0}, {r6, r6, -2.0 * r6}};
    constexpr int   nz[2]   = {2, 3};  // non-zero weights of row j: a < nz[j]
    Eigen::Matrix<Float, 6, 6> Hr;
    for(int j = 0; j < 2; ++j)
        for(int k = 0; k < 2; ++k)
        {
            Matrix3x3 B = Matrix3x3::Zero();
            for(int a = 0; a < nz[j]; ++a)
                for(int b = 0; b < nz[k]; ++b)
                    B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
            Hr.template block<3, 3>(3 * j, 3 * k) = B;
        }
    make_spd<6, Solver>(Hr);
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
        {
            Matrix3x3 B = Matrix3x3::Zero();
            for(int j = 0; j < 2; ++j)
            {
                if(a >= nz[j])
                    continue;
                for(int k = 0; k < 2; ++k)
                {
                    if(b >= nz[k])
                        continue;
                    B += (h[j][a] * h[k][b]) * Hr.template block<3, 3>(3 * j, 3 * k);
                }
            }
            H.template block<3, 3>(3 * a, 3 * b) = B;
        }
}

template <int Solver = 0>
inline UIPC_GENERIC void make_spd_translation_free_4x3(Matrix12x12& H)
{
    // Helmert rows: orthonormal, each orthogonal to (1,1,1,1)
    constexpr Float             r2      = 0.70710678118654752440;  // 1/sqrt(2)
    constexpr Float             r6      = 0.40824829046386301637;  // 1/sqrt(6)
    constexpr Float             r12     = 0.28867513459481288225;  // 1/sqrt(12)
    constexpr Float             h[3][4] = {{r2, -r2, 0.0, 0.0},
                                           {r6, r6, -2.0 * r6, 0.0},
                                           {r12, r12, r12, -3.0 * r12}};
    Eigen::Matrix<Float, 12, 9> Q       = Eigen::Matrix<Float, 12, 9>::Zero();
    for(int j = 0; j < 3; ++j)
        for(int a = 0; a < 4; ++a)
            for(int k = 0; k < 3; ++k)
                Q(3 * a + k, 3 * j + k) = h[j][a];
    Eigen::Matrix<Float, 9, 9> Hr = Q.transpose() * H * Q;
    make_spd<9, Solver>(Hr);
    H = Q * Hr * Q.transpose();
}

// perf/round7 (s07): the dense-basis (K7-style) counterpart of the blocked
// 3-vertex projection above, for A/B of the block assembly.
template <int Solver = 0>
inline UIPC_GENERIC void make_spd_translation_free_3x3(Matrix9x9& H)
{
    // Helmert rows: orthonormal, each orthogonal to (1,1,1)
    constexpr Float            r2      = 0.70710678118654752440;  // 1/sqrt(2)
    constexpr Float            r6      = 0.40824829046386301637;  // 1/sqrt(6)
    constexpr Float            h[2][3] = {{r2, -r2, 0.0}, {r6, r6, -2.0 * r6}};
    Eigen::Matrix<Float, 9, 6> Q       = Eigen::Matrix<Float, 9, 6>::Zero();
    for(int j = 0; j < 2; ++j)
        for(int a = 0; a < 3; ++a)
            for(int k = 0; k < 3; ++k)
                Q(3 * a + k, 3 * j + k) = h[j][a];
    Eigen::Matrix<Float, 6, 6> Hr = Q.transpose() * H * Q;
    make_spd<6, Solver>(Hr);
    H = Q * Hr * Q.transpose();
}
}  // namespace uipc::backend::cuda