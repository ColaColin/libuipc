#pragma once
#include <type_define.h>
#include <Eigen/Eigenvalues>
namespace uipc::backend::cuda_tool
{
namespace eigen
{
    template <typename T, int N>
    UIPC_GENERIC void evd(const Eigen::Matrix<T, N, N>& M,
                          Eigen::Vector<T, N>&          eigen_values,
                          Eigen::Matrix<T, N, N>&       eigen_vectors)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N>> eigen_solver;
        // NOTE:
        //  On CUDA, if N <= 3, compute() is not supported.
        //  So, we use computeDirect() instead.
        if constexpr(N <= 3)
            eigen_solver.computeDirect(M);
        else
            eigen_solver.compute(M);
        eigen_values  = eigen_solver.eigenvalues();
        eigen_vectors = eigen_solver.eigenvectors();
    }


    /**
     * @brief Symmetric eigen-decomposition of a small fixed-size matrix by
     * Householder tridiagonalisation + implicit-shift QL (the same algorithm
     * Eigen's SelfAdjointEigenSolver::compute() runs), written as fixed-size
     * loops so that nvcc can unroll it and keep the working set in registers.
     *
     * perf/round5 (w0): Eigen's version goes through its dynamic-size block
     * expressions and an out-of-line selfadjoint_matrix_vector_product<double,
     * long> (142 registers), which costs a 3 488 B stack frame for N = 9
     * against 1 440 B here. Eigenvalues are *not* sorted (make_spd does not
     * need them to be). Rounding-level equal to `evd`, not bit-identical.
     *
     * M = V * diag(eigen_values) * V^T, V = eigen_vectors.
     */
    template <typename T, int N>
    UIPC_GENERIC void evd_tridiag_ql(const Eigen::Matrix<T, N, N>& M,
                                     Eigen::Vector<T, N>&          eigen_values,
                                     Eigen::Matrix<T, N, N>&       eigen_vectors)
    {
        static_assert(N >= 2, "evd_tridiag_ql needs N >= 2");
        T V[N][N];
        T d[N];
        T e[N];
#pragma unroll
        for(int i = 0; i < N; ++i)
#pragma unroll
            for(int j = 0; j < N; ++j)
                V[i][j] = M(i, j);

        // --- Householder reduction to tridiagonal form (EISPACK tred2)
#pragma unroll
        for(int j = 0; j < N; ++j)
            d[j] = V[N - 1][j];
        for(int i = N - 1; i > 0; --i)
        {
            T scale = 0.0;
            T h     = 0.0;
            for(int k = 0; k < i; ++k)
                scale += abs(d[k]);
            if(scale == 0.0)
            {
                e[i] = d[i - 1];
                for(int j = 0; j < i; ++j)
                {
                    d[j]    = V[i - 1][j];
                    V[i][j] = 0.0;
                    V[j][i] = 0.0;
                }
            }
            else
            {
                for(int k = 0; k < i; ++k)
                {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                T f = d[i - 1];
                T g = sqrt(h);
                if(f > 0)
                    g = -g;
                e[i]     = scale * g;
                h        = h - f * g;
                d[i - 1] = f - g;
                for(int j = 0; j < i; ++j)
                    e[j] = 0.0;
                for(int j = 0; j < i; ++j)
                {
                    f       = d[j];
                    V[j][i] = f;
                    g       = e[j] + V[j][j] * f;
                    for(int k = j + 1; k <= i - 1; ++k)
                    {
                        g += V[k][j] * d[k];
                        e[k] += V[k][j] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for(int j = 0; j < i; ++j)
                {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                T hh = f / (h + h);
                for(int j = 0; j < i; ++j)
                    e[j] -= hh * d[j];
                for(int j = 0; j < i; ++j)
                {
                    f = d[j];
                    g = e[j];
                    for(int k = j; k <= i - 1; ++k)
                        V[k][j] -= (f * e[k] + g * d[k]);
                    d[j]    = V[i - 1][j];
                    V[i][j] = 0.0;
                }
            }
            d[i] = h;
        }
        for(int i = 0; i < N - 1; ++i)
        {
            V[N - 1][i] = V[i][i];
            V[i][i]     = 1.0;
            T h         = d[i + 1];
            if(h != 0.0)
            {
                for(int k = 0; k <= i; ++k)
                    d[k] = V[k][i + 1] / h;
                for(int j = 0; j <= i; ++j)
                {
                    T g = 0.0;
                    for(int k = 0; k <= i; ++k)
                        g += V[k][i + 1] * V[k][j];
                    for(int k = 0; k <= i; ++k)
                        V[k][j] -= g * d[k];
                }
            }
            for(int k = 0; k <= i; ++k)
                V[k][i + 1] = 0.0;
        }
#pragma unroll
        for(int j = 0; j < N; ++j)
        {
            d[j]        = V[N - 1][j];
            V[N - 1][j] = 0.0;
        }
        V[N - 1][N - 1] = 1.0;
        e[0]            = 0.0;

        // --- implicit-shift QL on the tridiagonal (EISPACK tql2)
        for(int i = 1; i < N; ++i)
            e[i - 1] = e[i];
        e[N - 1] = 0.0;
        T f      = 0.0;
        T tst1   = 0.0;
        // 2^-52: the QL deflation threshold, relative to the running row norm
        constexpr T eps = 2.220446049250313e-16;
        for(int l = 0; l < N; ++l)
        {
            T tl = abs(d[l]) + abs(e[l]);
            if(tl > tst1)
                tst1 = tl;
            int m = l;
            while(m < N)
            {
                if(abs(e[m]) <= eps * tst1)
                    break;
                ++m;
            }
            if(m > l)
            {
                int iter = 0;
                do
                {
                    ++iter;
                    T g = d[l];
                    T p = (d[l + 1] - g) / (2.0 * e[l]);
                    T r = hypot(p, T(1.0));
                    if(p < 0)
                        r = -r;
                    d[l]     = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    T dl1    = d[l + 1];
                    T h      = g - d[l];
                    for(int i = l + 2; i < N; ++i)
                        d[i] -= h;
                    f += h;
                    p     = d[m];
                    T c   = 1.0;
                    T c2  = c;
                    T c3  = c;
                    T el1 = e[l + 1];
                    T s   = 0.0;
                    T s2  = 0.0;
                    for(int i = m - 1; i >= l; --i)
                    {
                        c3       = c2;
                        c2       = c;
                        s2       = s;
                        g        = c * e[i];
                        h        = c * p;
                        r        = hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s        = e[i] / r;
                        c        = p / r;
                        p        = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
#pragma unroll
                        for(int k = 0; k < N; ++k)
                        {
                            h           = V[k][i + 1];
                            V[k][i + 1] = s * V[k][i] + c * h;
                            V[k][i]     = c * V[k][i] - s * h;
                        }
                    }
                    p    = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                    // 40 is the EISPACK safety cap; N <= 12 needs ~2 per value
                } while(abs(e[l]) > eps * tst1 && iter < 40);
            }
            d[l] += f;
            e[l] = 0.0;
        }

#pragma unroll
        for(int i = 0; i < N; ++i)
            eigen_values(i) = d[i];
#pragma unroll
        for(int i = 0; i < N; ++i)
#pragma unroll
            for(int j = 0; j < N; ++j)
                eigen_vectors(i, j) = V[i][j];
    }

    namespace details
    {
        template <typename T, int N>
        UIPC_GENERIC void find_maxValue_diagOff(const Eigen::Matrix<T, N, N>& M,
                                                int&                          p,
                                                int&                          q,
                                                T& max_value)
        {
            max_value = -1;
            for(int i = 0; i < N; ++i)
            {
                for(int j = i + 1; j < N; ++j)
                {
                    if(abs(M(i, j)) > max_value)
                    {
                        max_value = abs(M(i, j));
                        p         = i;
                        q         = j;
                    }
                }
            }
        }

        template <typename T, int N>
        UIPC_GENERIC T calc_sumDiagOff(const Eigen::Matrix<T, N, N>& M)
        {
            T sum = 0.0f;
            for(int i = 0; i < N; ++i)
            {
                for(int j = i + 1; j < N; ++j)
                {
                    sum += abs(M(i, j));
                }
            }
            return sum;
        }

        template <typename T, int N>
        UIPC_GENERIC void sort_eigensystem_optimized(Eigen::Vector<T, N>& eigen_values,
                                                     Eigen::Matrix<T, N, N>& eigen_vectors,
                                                     bool ascending = false)
        {
            // create index array
            int indices[N];
            for(int i = 0; i < N; i++)
            {
                indices[i] = i;
            }

            // sort the indices
            for(int i = 0; i < N - 1; i++)
            {
                for(int j = 0; j < N - i - 1; j++)
                {
                    bool should_swap =
                        ascending ?
                            (eigen_values(indices[j]) > eigen_values(indices[j + 1])) :
                            (eigen_values(indices[j]) < eigen_values(indices[j + 1]));

                    if(should_swap)
                    {
                        int temp       = indices[j];
                        indices[j]     = indices[j + 1];
                        indices[j + 1] = temp;
                    }
                }
            }

            // reorder the data by the indices
            Eigen::Vector<T, N>    sorted_values;
            Eigen::Matrix<T, N, N> sorted_vectors;

            for(int i = 0; i < N; i++)
            {
                sorted_values(i)      = eigen_values(indices[i]);
                sorted_vectors.col(i) = eigen_vectors.col(indices[i]);
            }

            // copy back
            eigen_values  = sorted_values;
            eigen_vectors = sorted_vectors;
        }

        template <typename T, int N>
        UIPC_GENERIC void jacobi_rotate(Eigen::Matrix<T, N, N>& M,
                                        Eigen::Matrix<T, N, N>& E,
                                        int                     p,
                                        int                     q)
        {
            if(std::abs(M(p, q)) < 1e-12)
                return;
            T tau = (M(q, q) - M(p, p)) / (2.0 * M(p, q));
            T t;
            if(tau >= 0)
            {
                t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
            }
            else
            {
                t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
            }
            T c = 1.0 / std::sqrt(1.0 + t * t);
            T s = t * c;

            // optimiza ratation
            for(int i = 0; i < N; i++)
            {
                if(i != p && i != q)
                {
                    T Mip   = M(i, p);
                    T Miq   = M(i, q);
                    M(p, i) = M(i, p) = c * Mip - s * Miq;
                    M(q, i) = M(i, q) = s * Mip + c * Miq;
                }

                // update eigen vector
                T Eip   = E(i, p);
                T Eiq   = E(i, q);
                E(i, p) = c * Eip - s * Eiq;
                E(i, q) = s * Eip + c * Eiq;
            }

            // update diagonal and pq
            T Mpp = M(p, p);
            T Mqq = M(q, q);
            T Mpq = M(p, q);

            M(p, p) = c * c * Mpp + s * s * Mqq - 2 * c * s * Mpq;
            M(q, q) = s * s * Mpp + c * c * Mqq + 2 * c * s * Mpq;
            M(p, q) = M(q, p) = 0;
        }
    }  // namespace details

    /**
     * @brief calculate the Eigen System of a symmetric matrix
     */
    template <typename T, int N>
    UIPC_GENERIC void evd_jacobi(const Eigen::Matrix<T, N, N>& M,
                                 Eigen::Vector<T, N>&          eigen_values,
                                 Eigen::Matrix<T, N, N>&       eigen_vectors)
    {
        auto symmetrix = M;
        eigen_vectors.setIdentity();
        while(details::calc_sumDiagOff(symmetrix) > 1e-6)
        {
            int q = -1, p = -1;
            T   max_value;
            details::find_maxValue_diagOff(symmetrix, p, q, max_value);
            details::jacobi_rotate(symmetrix, eigen_vectors, p, q);
        }
        eigen_values = symmetrix.diagonal();
        details::sort_eigensystem_optimized(eigen_values, eigen_vectors, true);
    }
}  // namespace eigen
}  // namespace uipc::backend::cuda_tool