#pragma once
#include <algorithm/givens.hpp>
#include <type_define.h>

namespace uipc::backend::cuda
{
namespace math
{
    template <typename T>
    UIPC_GENERIC constexpr void swap(T& a, T& b)
    {
        auto s = a;
        a      = b;
        b      = s;
    }

    template <typename T>
    UIPC_GENERIC constexpr void polar_decomposition(const Eigen::Matrix<T, 2, 2>& A,
                                                    Eigen::Matrix<T, 2, 2>& S,
                                                    GivensRotation<T>& R) noexcept
    {
        S = A;
        Eigen::Matrix<T, 2, 1> x{A(0, 0) + A(1, 1), A(1, 0) - A(0, 1)};
        auto                   d = x.norm();
        if(d != 0)
        {
            R.c = x(0) / d;
            R.s = -x(1) / d;
        }
        else
        {
            R.c = 1;
            R.s = 0;
        }
        R.rowRotation(S);
    }

    template <typename T>
    UIPC_GENERIC constexpr void qr_svd2x2(const Eigen::Matrix<T, 2, 2>& A,
                                          Eigen::Matrix<T, 2, 1>&       S,
                                          GivensRotation<T>&            U,
                                          GivensRotation<T>& V) noexcept
    {
        Eigen::Matrix<T, 2, 2> S_sym;
        polar_decomposition(A, S_sym, U);

        T    cosine{}, sine{};
        auto x{S_sym(0, 0)}, y{S_sym(0, 1)}, z{S_sym(1, 1)};
        auto y2 = y * y;

        if(y2 == 0)
        {  // S is already diagonal
            cosine = 1;
            sine   = 0;
            S(0)   = x;
            S(1)   = z;
        }
        else
        {
            auto tau = (T)0.5 * (x - z);
            T    w{std::sqrt(tau * tau + y2)}, t{};
            if(tau > 0)  // tau + w > w > y > 0 ==> division is safe
                t = y / (tau + w);
            else  // tau - w < -w < -y < 0 ==> division is safe
                t = y / (tau - w);
            cosine = 1 / std::sqrt(t * t + (T)1);
            sine   = -t * cosine;

            T c2    = cosine * cosine;
            T _2csy = 2 * cosine * sine * y;
            T s2    = sine * sine;

            S(0) = c2 * x - _2csy + s2 * z;
            S(1) = s2 * x + _2csy + c2 * z;
        }

        // Sorting
        if(S(0) < S(1))
        {
            swap(S(0), S(1));
            V.c = -sine;
            V.s = cosine;
        }
        else
        {
            V.c = cosine;
            V.s = sine;
        }
        U *= V;
    }

    template <typename T>
    UIPC_GENERIC constexpr T wilkinson_shift(const T a1, const T b1, const T a2) noexcept
    {
        T d     = (T)0.5 * (a1 - a2);
        T bs    = b1 * b1;
        T shift = bs / (std::abs(d) + std::sqrt(d * d + bs));
        // Keep the sign operation in T precision. The standard-library
        // sign-copy overload is unreliable for float in CUDA host/device code;
        // Wilkinson uses sign(0) = +1.
        if(d < (T)0)
            shift = -shift;
        T mu = a2 - shift;
        return mu;
    }

    template <typename T>
    UIPC_GENERIC constexpr void flip_sign(int                     j,
                                          Eigen::Matrix<T, 3, 3>& U,
                                          Eigen::Matrix<T, 3, 1>& S) noexcept
    {
        S(j)     = -S(j);
        U.col(j) = -U.col(j);
    }

    template <int t, typename T>
    UIPC_GENERIC constexpr void sort_sigma(Eigen::Matrix<T, 3, 3>& U,
                                           Eigen::Matrix<T, 3, 1>& sigma,
                                           Eigen::Matrix<T, 3, 3>& V) noexcept
    {
        /// t == 0
        if constexpr(t == 0)
        {
            if(std::abs(sigma(1)) >= std::abs(sigma(2)))
            {
                if(sigma(1) < 0)
                {
                    flip_sign(1, U, sigma);
                    flip_sign(2, U, sigma);
                }
                return;
            }

            if(sigma(2) < 0)
            {
                flip_sign(1, U, sigma);
                flip_sign(2, U, sigma);
            }

            swap(sigma(1), sigma(2));
            U.col(1).swap(U.col(2));
            V.col(1).swap(V.col(2));

            if(sigma(1) > sigma(0))
            {
                swap(sigma(0), sigma(1));
                U.col(0).swap(U.col(1));
                V.col(0).swap(V.col(1));
            }
            else
            {
                U.col(2) = -U.col(2);
                V.col(2) = -V.col(2);
            }
        }
        /// t == 1
        else if constexpr(t == 1)
        {
            if(std::abs(sigma(0)) >= sigma(1))
            {
                if(sigma(0) < 0)
                {
                    flip_sign(0, U, sigma);
                    flip_sign(2, U, sigma);
                }
                return;
            }

            swap(sigma(0), sigma(1));
            U.col(0).swap(U.col(1));
            V.col(0).swap(V.col(1));

            if(std::abs(sigma(1)) < std::abs(sigma(2)))
            {
                swap(sigma(1), sigma(2));
                U.col(1).swap(U.col(2));
                V.col(1).swap(V.col(2));
            }
            else
            {
                U.col(1) = -U.col(1);
                V.col(1) = -V.col(1);
            }

            if(sigma(1) < 0)
            {
                flip_sign(1, U, sigma);
                flip_sign(2, U, sigma);
            }
        }
    }

    template <int t, typename T>
    UIPC_GENERIC constexpr void process(Eigen::Matrix<T, 3, 3>& B,
                                        Eigen::Matrix<T, 3, 3>& U,
                                        Eigen::Matrix<T, 3, 1>& S,
                                        Eigen::Matrix<T, 3, 3>& V) noexcept
    {
        GivensRotation<T> u{0, 1};
        GivensRotation<T> v{0, 1};
        constexpr int     other   = (t == 1) ? 0 : 2;
        S(other)                  = B(other, other);
        Eigen::Matrix<T, 2, 2> B_ = B.template block<2, 2>(t, t);
        Eigen::Matrix<T, 2, 1> S_;
        qr_svd2x2(B_, S_, u, v);
        S(t)     = S_(0);
        S(t + 1) = S_(1);

        u.rowi += t;
        u.rowk += t;
        v.rowi += t;
        v.rowk += t;
        u.columnRotation(U);
        v.columnRotation(V);
    }


    template <typename T>
    UIPC_GENERIC constexpr void qr_svd(const Eigen::Matrix<T, 3, 3>& A,
                                       Eigen::Matrix<T, 3, 1>&       S,
                                       Eigen::Matrix<T, 3, 3>&       U,
                                       Eigen::Matrix<T, 3, 3>&       V) noexcept
    {
        U.setIdentity();
        V.setIdentity();

        Eigen::Matrix<T, 3, 3> B = A;

        upper_bidiagonalize(B, U, V);

        GivensRotation<T> r{0, 1};

        T alpha[3] = {B(0, 0), B(1, 1), B(2, 2)};
        T beta[2]  = {B(0, 1), B(1, 2)};
        T gamma[2] = {alpha[0] * beta[0], alpha[1] * beta[1]};

        constexpr auto eta = std::numeric_limits<T>::epsilon() * (T)128;
        T              tol = eta
                * std::max((T)0.5
                               * std::sqrt(alpha[0] * alpha[0] + alpha[1] * alpha[1]
                                           + alpha[2] * alpha[2]
                                           + beta[0] * beta[0] + beta[1] * beta[1]),
                           (T)1);

        while(abs(alpha[0]) > tol && abs(alpha[1]) > tol && abs(alpha[2]) > tol
              && abs(beta[0]) > tol && abs(beta[1]) > tol)
        {
            auto mu = wilkinson_shift(alpha[1] * alpha[1] + beta[0] * beta[0],
                                      gamma[1],
                                      alpha[2] * alpha[2] + beta[1] * beta[1]);

            r.computeConventional(alpha[0] * alpha[0] - mu, gamma[0]);
            r.columnRotation(B);
            r.columnRotation(V);
            zero_chasing(B, U, V);

            alpha[0] = B(0, 0);
            alpha[1] = B(1, 1);
            alpha[2] = B(2, 2);
            beta[0]  = B(0, 1);
            beta[1]  = B(1, 2);
            gamma[0] = alpha[0] * beta[0];
            gamma[1] = alpha[1] * beta[1];
        }

        if(std::abs(beta[1]) <= tol)
        {
            process<0>(B, U, S, V);
            sort_sigma<0>(U, S, V);
        }
        else if(std::abs(beta[0]) <= tol)
        {
            process<1>(B, U, S, V);
            sort_sigma<1>(U, S, V);
        }
        else if(std::abs(alpha[1]) <= tol)
        {
            GivensRotation<T> r1(1, 2);
            r1.computeUnconventional(B(1, 2), B(2, 2));
            r1.rowRotation(B);
            r1.columnRotation(U);
            process<0>(B, U, S, V);
            sort_sigma<0>(U, S, V);
        }
        else if(std::abs(alpha[2]) <= tol)
        {
            GivensRotation<T> r1(1, 2);
            r1.computeConventional(B(1, 1), B(1, 2));
            r1.columnRotation(B);
            r1.columnRotation(V);

            GivensRotation<T> r2(0, 2);
            r2.computeConventional(B(0, 0), B(0, 2));
            r2.columnRotation(B);
            r2.columnRotation(V);

            process<0>(B, U, S, V);
            sort_sigma<0>(U, S, V);
        }
        else if(std::abs(alpha[0]) <= tol)
        {
            GivensRotation<T> r1(0, 1);
            r1.computeUnconventional(B(0, 1), B(1, 1));
            r1.rowRotation(B);
            r1.columnRotation(U);

            GivensRotation<T> r2(0, 2);
            r2.computeUnconventional(B(0, 2), B(2, 2));
            r2.rowRotation(B);
            r2.columnRotation(U);

            process<1>(B, U, S, V);
            sort_sigma<1>(U, S, V);
        }
    }
    // ------------------------------------------------------------------
    // Fixed-sweep, branch-free 3x3 SVD (McAdams / Gast-style Jacobi).
    //
    // Replaces the iterative Wilkinson-shift bidiagonal QR of qr_svd() with
    // a fixed number of cyclic Jacobi sweeps on the normal matrix A^T A,
    // followed by a Givens QR of A*V. There is no data-dependent loop, so
    // every lane of a warp executes exactly the same instruction stream.
    //
    // Output convention is identical to qr_svd():
    //   A = U * diag(S) * V^T,  U,V in SO(3) (det = +1),
    //   S(0) >= S(1) >= |S(2)|,  sign(S(2)) = sign(det A).
    // ------------------------------------------------------------------

    template <typename T>
    UIPC_GENERIC constexpr T svd_rsqrt(T x) noexcept
    {
#ifdef __CUDA_ARCH__
        return ::rsqrt(x);
#else
        return T(1) / std::sqrt(x);
#endif
    }

    // One exact Jacobi rotation annihilating the (p,q) entry of a symmetric
    // 3x3 matrix, with the accumulating rotation applied to V's columns p,q.
    // r is the remaining index. Uses 1 sqrt + 1 rsqrt and no division:
    //   tan(2 theta) = -apq / tau,  tau = (app - aqq)/2,  w = hypot(tau, apq)
    //   c = u / g, s = -sign(tau) * apq / g,  u = w + |tau|, g = sqrt(2 w u)
    // which keeps |theta| <= pi/4 (the convergent branch).
    template <typename T>
    UIPC_GENERIC constexpr void jacobi_sym3_rot(T& app,
                                                T& aqq,
                                                T& apq,
                                                T& apr,
                                                T& aqr,
                                                T& v0p,
                                                T& v1p,
                                                T& v2p,
                                                T& v0q,
                                                T& v1q,
                                                T& v2q) noexcept
    {
        const T tau = T(0.5) * (app - aqq);
        const T at  = tau < T(0) ? -tau : tau;
        const T w   = std::sqrt(tau * tau + apq * apq);
        const T u   = w + at;
        const T gi  = svd_rsqrt(T(2) * w * u);

        // The caller normalises A to max|A| = 1, so max|N| <= 3 and w <= 3.
        // Below RotEps the plane is diagonal to far beyond double precision
        // and the rotation is skipped; that also keeps 2*w*u (>= 2*w*w) in
        // the normal range, where rsqrt is exact enough that c*c + s*s == 1
        // and V stays orthogonal. Without this guard a late sweep on a matrix
        // with repeated eigenvalues drives apq into the denormal range and
        // rsqrt(2*w*u) loses V's orthogonality outright (measured: |V^T V - I|
        // = 1.6e-1 at 5 sweeps on three equal singular values).
        constexpr T RotEps = T(1e-150);
        const bool  act    = (w > RotEps);
        const T     sg     = tau < T(0) ? T(-1) : T(1);
        const T     c      = act ? u * gi : T(1);
        const T     s      = act ? -sg * apq * gi : T(0);

        const T cc  = c * c;
        const T ss  = s * s;
        const T cs2 = T(2) * c * s;

        const T npp = cc * app - cs2 * apq + ss * aqq;
        const T nqq = ss * app + cs2 * apq + cc * aqq;
        app         = npp;
        aqq         = nqq;
        apq         = T(0);  // annihilated exactly

        const T t1 = apr;
        const T t2 = aqr;
        apr        = c * t1 - s * t2;
        aqr        = s * t1 + c * t2;

        // V <- V * J
        T a, b;
        a   = v0p;
        b   = v0q;
        v0p = c * a - s * b;
        v0q = s * a + c * b;
        a   = v1p;
        b   = v1q;
        v1p = c * a - s * b;
        v1q = s * a + c * b;
        a   = v2p;
        b   = v2q;
        v2p = c * a - s * b;
        v2q = s * a + c * b;
    }

    // Givens rotation zeroing b in [c -s; s c] * [a; b], pivot becomes +hypot.
    // rsqrt-based (1 sqrt + 1 rsqrt) variant of GivensRotation::computeConventional.
    template <typename T>
    UIPC_GENERIC constexpr void svd_givens(const T a, const T b, T& c, T& s) noexcept
    {
        const T d = a * a + b * b;
        if(d > T(0))
        {
            const T t = svd_rsqrt(d);
            c         = a * t;
            s         = -b * t;
        }
        else
        {
            c = T(1);
            s = T(0);
        }
    }

    template <int NSweeps, typename T>
    UIPC_GENERIC constexpr void qr_svd_fixed(const Eigen::Matrix<T, 3, 3>& A,
                                             Eigen::Matrix<T, 3, 1>&       S,
                                             Eigen::Matrix<T, 3, 3>&       U,
                                             Eigen::Matrix<T, 3, 3>& V) noexcept
    {
        // ---- 0. scale A by a power of two so that max|M| is in [0.5, 1].
        // Exact (no rounding), and it makes the whole routine scale invariant:
        // forming A^T A otherwise overflows above |A| ~ 1e154 and underflows
        // the Jacobi pivot below |A| ~ 1e-162.
        T amax = T(0);
#pragma unroll
        for(int i = 0; i < 3; ++i)
#pragma unroll
            for(int j = 0; j < 3; ++j)
            {
                const T a = A(i, j) < T(0) ? -A(i, j) : A(i, j);
                amax      = a > amax ? a : amax;
            }
        int     aexp  = 0;
        const T dummy = (amax > T(0)) ? std::frexp(amax, &aexp) : T(0);
        (void)dummy;
        const T scl  = std::ldexp(T(1), -aexp);   // M = A * scl, max|M| in [0.5,1)
        const T iscl = std::ldexp(T(1), aexp);

        Eigen::Matrix<T, 3, 3> M;
#pragma unroll
        for(int i = 0; i < 3; ++i)
#pragma unroll
            for(int j = 0; j < 3; ++j)
                M(i, j) = A(i, j) * scl;

        // ---- 1. normal matrix N = M^T M (symmetric, 6 unique entries) ----
        T n00 = M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0) + M(2, 0) * M(2, 0);
        T n11 = M(0, 1) * M(0, 1) + M(1, 1) * M(1, 1) + M(2, 1) * M(2, 1);
        T n22 = M(0, 2) * M(0, 2) + M(1, 2) * M(1, 2) + M(2, 2) * M(2, 2);
        T n01 = M(0, 0) * M(0, 1) + M(1, 0) * M(1, 1) + M(2, 0) * M(2, 1);
        T n02 = M(0, 0) * M(0, 2) + M(1, 0) * M(1, 2) + M(2, 0) * M(2, 2);
        T n12 = M(0, 1) * M(0, 2) + M(1, 1) * M(1, 2) + M(2, 1) * M(2, 2);

        // ---- 2. NSweeps cyclic Jacobi sweeps; V accumulates the rotations ----
        T v00 = T(1), v01 = T(0), v02 = T(0);
        T v10 = T(0), v11 = T(1), v12 = T(0);
        T v20 = T(0), v21 = T(0), v22 = T(1);

#pragma unroll
        for(int sweep = 0; sweep < NSweeps; ++sweep)
        {
            // (p,q,r) = (0,1,2)
            jacobi_sym3_rot(n00, n11, n01, n02, n12, v00, v10, v20, v01, v11, v21);
            // (p,q,r) = (0,2,1)
            jacobi_sym3_rot(n00, n22, n02, n01, n12, v00, v10, v20, v02, v12, v22);
            // (p,q,r) = (1,2,0)
            jacobi_sym3_rot(n11, n22, n12, n01, n02, v01, v11, v21, v02, v12, v22);
        }

        // ---- 3. sort the eigenvalues of N descending, permuting V's columns.
        // A 3-element sorting network; each swap flips det(V), so an odd
        // permutation is repaired by negating the last column (which only
        // flips the sign of S(2), whose sign is set by det(A) anyway).
        bool odd = false;
        {
            T d0 = n00, d1 = n11, d2 = n22;
            if(d0 < d1)
            {
                T t = d0;
                d0  = d1;
                d1  = t;
                t = v00; v00 = v01; v01 = t;
                t = v10; v10 = v11; v11 = t;
                t = v20; v20 = v21; v21 = t;
                odd = !odd;
            }
            if(d1 < d2)
            {
                T t = d1;
                d1  = d2;
                d2  = t;
                t = v01; v01 = v02; v02 = t;
                t = v11; v11 = v12; v12 = t;
                t = v21; v21 = v22; v22 = t;
                odd = !odd;
            }
            if(d0 < d1)
            {
                T t = d0;
                d0  = d1;
                d1  = t;
                t = v00; v00 = v01; v01 = t;
                t = v10; v10 = v11; v11 = t;
                t = v20; v20 = v21; v21 = t;
                odd = !odd;
            }
        }
        if(odd)
        {
            v02 = -v02;
            v12 = -v12;
            v22 = -v22;
        }

        V(0, 0) = v00; V(0, 1) = v01; V(0, 2) = v02;
        V(1, 0) = v10; V(1, 1) = v11; V(1, 2) = v12;
        V(2, 0) = v20; V(2, 1) = v21; V(2, 2) = v22;

        // ---- 4. B = A*V has orthogonal columns; a 3-Givens QR of it gives
        // U (a product of rotations, det +1) and the signed singular values.
        Eigen::Matrix<T, 3, 3> B = M * V;

        U.setIdentity();

        T c, s;
        // zero B(2,0) on rows (1,2)
        svd_givens(B(1, 0), B(2, 0), c, s);
        {
#pragma unroll
            for(int j = 0; j < 3; ++j)
            {
                const T t1 = B(1, j), t2 = B(2, j);
                B(1, j) = c * t1 - s * t2;
                B(2, j) = s * t1 + c * t2;
            }
#pragma unroll
            for(int i = 0; i < 3; ++i)
            {
                const T t1 = U(i, 1), t2 = U(i, 2);
                U(i, 1) = c * t1 - s * t2;
                U(i, 2) = s * t1 + c * t2;
            }
        }
        // zero B(1,0) on rows (0,1)
        svd_givens(B(0, 0), B(1, 0), c, s);
        {
#pragma unroll
            for(int j = 0; j < 3; ++j)
            {
                const T t1 = B(0, j), t2 = B(1, j);
                B(0, j) = c * t1 - s * t2;
                B(1, j) = s * t1 + c * t2;
            }
#pragma unroll
            for(int i = 0; i < 3; ++i)
            {
                const T t1 = U(i, 0), t2 = U(i, 1);
                U(i, 0) = c * t1 - s * t2;
                U(i, 1) = s * t1 + c * t2;
            }
        }
        // zero B(2,1) on rows (1,2)
        svd_givens(B(1, 1), B(2, 1), c, s);
        {
            const T t1 = B(1, 2), t2 = B(2, 2);
            B(1, 2) = c * t1 - s * t2;
            B(2, 2) = s * t1 + c * t2;
            const T u1 = B(1, 1), u2 = B(2, 1);
            B(1, 1) = c * u1 - s * u2;
#pragma unroll
            for(int i = 0; i < 3; ++i)
            {
                const T t3 = U(i, 1), t4 = U(i, 2);
                U(i, 1) = c * t3 - s * t4;
                U(i, 2) = s * t3 + c * t4;
            }
        }

        // undo the power-of-two scaling (exact)
        S(0) = B(0, 0) * iscl;  // >= 0 by construction
        S(1) = B(1, 1) * iscl;  // >= 0 by construction
        S(2) = B(2, 2) * iscl;  // signed: sign(det A)
    }

}  // namespace math

}  // namespace uipc::backend::cuda
