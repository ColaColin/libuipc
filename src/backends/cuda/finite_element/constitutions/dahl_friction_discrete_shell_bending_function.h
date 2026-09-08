#pragma once
#include <type_define.h>
#include <finite_element/constitutions/discrete_shell_bending_reference.h>
#include <utils/dihedral_angle.h>
#include <cmath>

namespace uipc::backend::cuda
{
namespace sym::dahl_friction_discrete_shell_bending
{
    // Works in both host (g++/MSVC) and device (nvcc) translation units.
    using std::isfinite;
    // Dahl-style internal-friction bending on the discrete-shell dihedral hinge.
    //
    // Per-edge state (committed at the last accepted frame):
    //   theta_c : committed dihedral angle
    //   F_c     : committed friction moment, |F_c| <= M_e by construction
    //
    // Per-edge parameters:
    //   kappa : elastic bending stiffness [N*m]  (identical to DiscreteShellBending)
    //   M_e   : friction saturation moment  = moment_per_length * L0  [N*m]
    //   ell_e : friction transition angle [rad] (mesh independent material constant)
    //   sigma_e = M_e / ell_e is the initial friction slope [N*m/rad]
    //
    // Anchored incremental friction potential (evaluate every trial from the
    // SAME committed state; commit once per accepted frame in do_update_state):
    //   d  = wrap(theta - theta_c), s = sign(d), x = |d| / ell_e
    //   W(d; F_c)  = F_c*d + (M_e - s*F_c)*ell_e*(x - (1 - exp(-x)))
    //   dW/dd      = s*M_e + (F_c - s*M_e)*exp(-x)          (= F_new)
    //   d2W/dd2    = ((M_e - s*F_c) / ell_e) * exp(-x)      (>= 0)
    // At d == 0 we use s = 0, which yields the exact one-sided limits
    // W = 0, dW/dd = F_c and the average tangent d2W/dd2 = M_e/ell_e.
    //
    // Total response (elastic part matches sym::discrete_shell_bending exactly,
    // so M_e = 0 degenerates to DiscreteShellBending):
    //   P(theta)    = kappa*w*del^2 + W,        del = wrap(theta - theta_bar)
    //   dP/dtheta   = 2*kappa*w*del + F_new
    //   d2P/dtheta2 = 2*kappa*w + d2W/dd2
    // with w = L0 / h_bar.
    template <typename T>
    inline UIPC_GENERIC constexpr T pi()
    {
        return static_cast<T>(3.14159265358979323846264338327950288);
    }

    template <typename T>
    inline UIPC_GENERIC constexpr T dihedral_guard_eps()
    {
        return static_cast<T>(1e-12);
    }

    template <typename T>
    inline UIPC_GENERIC bool is_finite_scalar(T v)
    {
        return isfinite(v);
    }

    template <typename T>
    inline UIPC_GENERIC T abs_value(T v)
    {
        return v < T(0) ? -v : v;
    }

    template <typename T>
    inline UIPC_GENERIC T max_value(T a, T b)
    {
        return a < b ? b : a;
    }

    template <typename T>
    inline UIPC_GENERIC T min_value(T a, T b)
    {
        return a < b ? a : b;
    }

    template <typename T>
    inline UIPC_GENERIC T sign_value(T v)
    {
        return v > T(0) ? T(1) : (v < T(0) ? T(-1) : T(0));
    }

    template <typename T>
    inline UIPC_GENERIC bool is_finite_vec3(const Eigen::Matrix<T, 3, 1>& v)
    {
        return is_finite_scalar(v[0]) && is_finite_scalar(v[1])
               && is_finite_scalar(v[2]);
    }

    template <typename T>
    inline UIPC_GENERIC T wrap_angle(T angle)
    {
        constexpr T Pi    = pi<T>();
        constexpr T TwoPi = static_cast<T>(2.0) * Pi;

        while(angle > Pi)
            angle -= TwoPi;

        while(angle < -Pi)
            angle += TwoPi;

        return angle;
    }

    template <typename T>
    inline UIPC_GENERIC T angle_delta(T theta, T theta_ref)
    {
        return wrap_angle(theta - theta_ref);
    }

    template <typename T>
    inline UIPC_GENERIC bool safe_dihedral_angle(const Eigen::Matrix<T, 3, 1>& v0,
                                                 const Eigen::Matrix<T, 3, 1>& v1,
                                                 const Eigen::Matrix<T, 3, 1>& v2,
                                                 const Eigen::Matrix<T, 3, 1>& v3,
                                                 T& theta)
    {
        if(!is_finite_vec3(v0) || !is_finite_vec3(v1) || !is_finite_vec3(v2)
           || !is_finite_vec3(v3))
            return false;

        const Eigen::Matrix<T, 3, 1> n1 = (v1 - v0).cross(v2 - v0);
        const Eigen::Matrix<T, 3, 1> n2 = (v2 - v3).cross(v1 - v3);

        const T n1_sq = n1.squaredNorm();
        const T n2_sq = n2.squaredNorm();
        const T eps   = dihedral_guard_eps<T>();

        if(!is_finite_scalar(n1_sq) || !is_finite_scalar(n2_sq) || n1_sq <= eps || n2_sq <= eps)
            return false;

        const T denom = sqrt(n1_sq * n2_sq);
        if(!is_finite_scalar(denom) || denom <= eps)
            return false;

        T cos_theta = n1.dot(n2) / denom;
        if(!is_finite_scalar(cos_theta))
            return false;

        cos_theta = cos_theta < T(-1) ? T(-1) : cos_theta;
        cos_theta = cos_theta > T(1) ? T(1) : cos_theta;
        theta     = acos(cos_theta);
        if(!is_finite_scalar(theta))
            return false;

        if(n2.cross(n1).dot(v1 - v2) < 0)
            theta = -theta;

        return is_finite_scalar(theta);
    }

    // x - (1 - exp(-x)) evaluated without cancellation for tiny x.
    template <typename T>
    inline UIPC_GENERIC T remainder_term(T x)
    {
        if(x < T(1e-3))
            return x * x / T(2) - x * x * x / T(6) + x * x * x * x / T(24)
                   - x * x * x * x * x / T(120) + x * x * x * x * x * x / T(720);
        return x + expm1(-x);
    }

    // Validity check for the per-edge friction configuration.
    template <typename T>
    inline UIPC_GENERIC bool valid_friction_inputs(T M_e, T ell_e)
    {
        return is_finite_scalar(M_e) && is_finite_scalar(ell_e) && M_e >= T(0)
               && (M_e == T(0) || ell_e > T(0));
    }

    // Response of the anchored friction potential at increment d from committed
    // state F0. Returns false on invalid inputs (caller contributes nothing).
    template <typename T>
    inline UIPC_GENERIC bool friction_response(T d, T F0, T M_e, T ell_e, T& W, T& dWdd, T& ddWdd)
    {
        W     = T(0);
        dWdd  = T(0);
        ddWdd = T(0);

        if(!is_finite_scalar(d) || !is_finite_scalar(F0)
           || !valid_friction_inputs(M_e, ell_e))
            return false;

        const T s = sign_value(d);
        const T a = abs_value(d);
        const T x = a / ell_e;

        if(!is_finite_scalar(x))
            return false;

        const T exp_neg_x = exp(-x);
        const T rem       = remainder_term(x);

        W     = F0 * d + (M_e - s * F0) * ell_e * rem;
        dWdd  = s * M_e + (F0 - s * M_e) * exp_neg_x;
        ddWdd = (M_e - s * F0) / ell_e * exp_neg_x;

        return is_finite_scalar(W) && is_finite_scalar(dWdd) && is_finite_scalar(ddWdd);
    }

    // Friction state update for a committed angular increment d from F0.
    // Exact exponential update; a convex combination of F0 and s*M_e, hence
    // the state stays in [-M_e, M_e]. Returns false on invalid inputs.
    template <typename T>
    inline UIPC_GENERIC bool commit_friction_state(T d, T F0, T M_e, T ell_e, T& F_new)
    {
        F_new = F0;

        if(!is_finite_scalar(d) || !is_finite_scalar(F0)
           || !valid_friction_inputs(M_e, ell_e))
            return false;

        const T s = sign_value(d);
        const T x = abs_value(d) / ell_e;

        if(!is_finite_scalar(x))
            return false;

        F_new = s * M_e + (F0 - s * M_e) * exp(-x);

        // Guard against rounding pushing the state marginally outside the box.
        F_new = min_value(max_value(F_new, -M_e), M_e);

        return is_finite_scalar(F_new);
    }

    inline UIPC_GENERIC void compute_constants(Float&         L0,
                                               Float&         h_bar,
                                               Float&         theta_bar,
                                               Float&         V_bar,
                                               const Vector3& x0_bar,
                                               const Vector3& x1_bar,
                                               const Vector3& x2_bar,
                                               const Vector3& x3_bar,
                                               Float          thickness0,
                                               Float          thickness1,
                                               Float          thickness2,
                                               Float          thickness3)
    {
        detail::compute_discrete_shell_bending_reference(
            L0, h_bar, theta_bar, V_bar, x0_bar, x1_bar, x2_bar, x3_bar);
    }

    inline UIPC_GENERIC Float E(const Vector3& x0,
                                const Vector3& x1,
                                const Vector3& x2,
                                const Vector3& x3,
                                Float          L0,
                                Float          h_bar,
                                Float          theta_bar,
                                Float          kappa,
                                Float          M_e,
                                Float          ell_e,
                                Float          theta_commit,
                                Float          F_commit)
    {
        namespace DFDSB = sym::dahl_friction_discrete_shell_bending;
        Float theta     = 0.0;
        if(!DFDSB::safe_dihedral_angle(x0, x1, x2, x3, theta))
            return 0.0;

        const Float del = DFDSB::angle_delta(theta, theta_bar);
        const Float d   = DFDSB::angle_delta(theta, theta_commit);

        Float W     = 0.0;
        Float dWdd  = 0.0;
        Float ddWdd = 0.0;
        if(!DFDSB::friction_response(d, F_commit, M_e, ell_e, W, dWdd, ddWdd))
            return 0.0;

        const Float w = L0 / h_bar;
        return kappa * w * del * del + W;
    }

    inline UIPC_GENERIC void dEdx(Vector12&      G,
                                  const Vector3& x0,
                                  const Vector3& x1,
                                  const Vector3& x2,
                                  const Vector3& x3,
                                  Float          L0,
                                  Float          h_bar,
                                  Float          theta_bar,
                                  Float          kappa,
                                  Float          M_e,
                                  Float          ell_e,
                                  Float          theta_commit,
                                  Float          F_commit)
    {
        namespace DFDSB = sym::dahl_friction_discrete_shell_bending;
        Float theta     = 0.0;
        if(!DFDSB::safe_dihedral_angle(x0, x1, x2, x3, theta))
        {
            G.setZero();
            return;
        }

        const Float del = DFDSB::angle_delta(theta, theta_bar);
        const Float d   = DFDSB::angle_delta(theta, theta_commit);

        Float W     = 0.0;
        Float dWdd  = 0.0;
        Float ddWdd = 0.0;
        if(!DFDSB::friction_response(d, F_commit, M_e, ell_e, W, dWdd, ddWdd))
        {
            G.setZero();
            return;
        }

        const Float w        = L0 / h_bar;
        const Float dEdtheta = 2.0 * kappa * w * del + dWdd;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = dEdtheta * dthetadx;
    }

    inline UIPC_GENERIC void ddEddx(Matrix12x12&   H,
                                    const Vector3& x0,
                                    const Vector3& x1,
                                    const Vector3& x2,
                                    const Vector3& x3,
                                    Float          L0,
                                    Float          h_bar,
                                    Float          theta_bar,
                                    Float          kappa,
                                    Float          M_e,
                                    Float          ell_e,
                                    Float          theta_commit,
                                    Float          F_commit)
    {
        namespace DFDSB = sym::dahl_friction_discrete_shell_bending;
        Float theta     = 0.0;
        if(!DFDSB::safe_dihedral_angle(x0, x1, x2, x3, theta))
        {
            H.setZero();
            return;
        }

        const Float del = DFDSB::angle_delta(theta, theta_bar);
        const Float d   = DFDSB::angle_delta(theta, theta_commit);

        Float W     = 0.0;
        Float dWdd  = 0.0;
        Float ddWdd = 0.0;
        if(!DFDSB::friction_response(d, F_commit, M_e, ell_e, W, dWdd, ddWdd))
        {
            H.setZero();
            return;
        }

        const Float w          = L0 / h_bar;
        const Float dEdtheta   = 2.0 * kappa * w * del + dWdd;
        const Float ddEddtheta = 2.0 * kappa * w + ddWdd;

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Matrix12x12 ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
    }
}  // namespace sym::dahl_friction_discrete_shell_bending
}  // namespace uipc::backend::cuda
