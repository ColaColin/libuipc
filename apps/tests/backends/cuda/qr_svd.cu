#include <app/app.h>
#include <cuda_tool/cuda_tool.h>
#include <algorithm/qr_svd.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

namespace
{
__global__ void wilkinson_shift_float_kernel(uipc::backend::cuda_tool::BufferView<float> result)
{
    namespace math = uipc::backend::cuda::math;
    result(0)      = math::wilkinson_shift(-0.0f, 1.0f, 0.0f);
    result(1)      = math::wilkinson_shift(2.0f, 1.0f, 0.0f);
    result(2)      = math::wilkinson_shift(0.0f, 1.0f, 2.0f);
}
}  // namespace

TEST_CASE("float Wilkinson shift is stable on GPU", "[cuda][qr_svd]")
{
    using uipc::backend::cuda_tool::DeviceBuffer;

    DeviceBuffer<float> device_result(3);
    wilkinson_shift_float_kernel<<<1, 1>>>(device_result.view());

    std::vector<float> result(3);
    device_result.copy_to(result);

    const float root_two = std::sqrt(2.0f);
    CHECK(result[0] == Catch::Approx(-1.0f));
    CHECK(result[1] == Catch::Approx(1.0f - root_two).margin(1e-6));
    CHECK(result[2] == Catch::Approx(1.0f + root_two).margin(1e-6));

    CHECK(uipc::backend::cuda::math::wilkinson_shift(-0.0f, 1.0f, 0.0f)
          == Catch::Approx(-1.0f));
}

// s23: randomised device-side verifier for the fixed-sweep Jacobi SVD
// (math::qr_svd_fixed) against the shipped iterative QR-SVD (math::qr_svd).
// Runs the *real device* functions - the fixed-sweep path calls ::rsqrt on
// device and 1/sqrt on host, so a host-only check would not cover it.
namespace
{
using uipc::Float;
using M3 = Eigen::Matrix<Float, 3, 3>;
using V3 = Eigen::Matrix<Float, 3, 1>;

// per-sample metrics: 0 new recon, 1 old recon, 2 |V^T V - I|_new,
// 3 |U^T U - I|_new, 4 |det V - 1|_new, 5 ordering violation,
// 6 sign-of-S2 error, 7 |S_new - S_old|
constexpr int NMetric = 8;

__global__ void svd_compare_kernel(uipc::backend::cuda_tool::CBufferView<M3> Fs,
                                   uipc::backend::cuda_tool::BufferView<Float> out,
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    namespace math = uipc::backend::cuda::math;

    const M3 F = Fs(i);
    M3       U, V, Uo, Vo;
    V3       S, So;
    math::qr_svd_fixed<4>(F, S, U, V);
    math::qr_svd(F, So, Uo, Vo);

    Float sc = F.cwiseAbs().maxCoeff();
    if(!(sc > 0))
        sc = 1;

    out(i * NMetric + 0) = (U * S.asDiagonal() * V.transpose() - F).cwiseAbs().maxCoeff() / sc;
    out(i * NMetric + 1) = (Uo * So.asDiagonal() * Vo.transpose() - F).cwiseAbs().maxCoeff() / sc;
    out(i * NMetric + 2) = (V.transpose() * V - M3::Identity()).cwiseAbs().maxCoeff();
    out(i * NMetric + 3) = (U.transpose() * U - M3::Identity()).cwiseAbs().maxCoeff();
    out(i * NMetric + 4) = abs(V.determinant() - Float(1));

    Float ord = S(1) - S(0);
    Float o2  = abs(S(2)) - S(1);
    if(o2 > ord)
        ord = o2;
    out(i * NMetric + 5) = (ord > 0 ? ord : Float(0)) / sc;

    const Float detF = F.determinant();
    Float       se   = 0;
    if(abs(detF) > Float(1e-10) * sc * sc * sc)
        se = (signbit(S(2)) == signbit(detF)) ? Float(0) : Float(1);
    out(i * NMetric + 6) = se;

    out(i * NMetric + 7) = (S - So).cwiseAbs().maxCoeff() / sc;
}
}  // namespace

TEST_CASE("fixed-sweep 3x3 SVD matches the iterative QR-SVD on GPU", "[cuda][qr_svd]")
{
    using uipc::backend::cuda_tool::DeviceBuffer;

    // 2^17 samples per family, 12 families = 1.57e6 randomised inputs, covering
    // near-identity, large deformation, reflections (det < 0), repeated and
    // zero singular values, and ill-conditioned F.
    constexpr int N = 1 << 17;

    std::mt19937_64                        rng(20260912);
    std::normal_distribution<double>       g(0, 1);
    std::uniform_real_distribution<double> uni(0, 1);

    auto randM = [&](M3& M)
    {
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                M(i, j) = g(rng);
    };
    auto randR = [&]()
    {
        M3 A;
        randM(A);
        Eigen::HouseholderQR<M3> qr(A);
        M3                       R = qr.householderQ();
        if(R.determinant() < 0)
            R.col(2) = -R.col(2);
        return R;
    };

    struct Family
    {
        const char* name;
        // recon tolerance for the *new* path, relative to max|F|
        double tol;
    };
    const Family fams[] = {
        {"I + 1e-6*N", 1e-13},
        {"I + 1e-3*N", 1e-13},
        {"I + 1e-1*N", 1e-13},
        {"full random N(0,1)", 1e-13},
        {"R*(I + 1e-2*N)", 1e-13},
        {"reflection (det < 0)", 1e-13},
        {"two equal singular values", 1e-13},
        {"three equal singular values", 1e-13},
        {"rank 2 (s2 = 0)", 1e-13},
        {"rank 1 (s1 = s2 = 0)", 1e-13},
        {"near-zero det (s2 ~ 1e-12)", 1e-13},
        {"cond(F) = 1e3", 1e-12},
    };

    std::vector<M3>    h_F(N);
    std::vector<Float> h_out(size_t(N) * NMetric);
    DeviceBuffer<M3>    d_F;
    DeviceBuffer<Float> d_out;

    for(int f = 0; f < int(sizeof(fams) / sizeof(fams[0])); ++f)
    {
        for(int k = 0; k < N; ++k)
        {
            M3 F;
            V3 sv;
            switch(f)
            {
                case 0: randM(F); F = M3::Identity() + 1e-6 * F; break;
                case 1: randM(F); F = M3::Identity() + 1e-3 * F; break;
                case 2: randM(F); F = M3::Identity() + 1e-1 * F; break;
                case 3: randM(F); break;
                case 4: { M3 Nn; randM(Nn); F = randR() * (M3::Identity() + 1e-2 * Nn); } break;
                case 5: randM(F); if(F.determinant() > 0) F.col(0) = -F.col(0); break;
                case 6: { double a = std::exp(g(rng)), b = std::exp(g(rng));
                          sv << std::max(a, b), std::min(a, b), std::min(a, b);
                          F = randR() * sv.asDiagonal() * randR().transpose(); } break;
                case 7: { double a = std::exp(g(rng)); sv << a, a, a;
                          F = randR() * sv.asDiagonal() * randR().transpose(); } break;
                case 8: sv << 1.0, 0.5, 0.0; F = randR() * sv.asDiagonal() * randR().transpose(); break;
                case 9: sv << 1.0, 0.0, 0.0; F = randR() * sv.asDiagonal() * randR().transpose(); break;
                case 10: sv << 1.0, 0.5, 1e-12 * uni(rng);
                         F = randR() * sv.asDiagonal() * randR().transpose(); break;
                case 11: sv << 1.0, 1.0 / std::sqrt(1e3), 1.0 / 1e3;
                         F = randR() * sv.asDiagonal() * randR().transpose(); break;
            }
            h_F[k] = F;
        }
        d_F.copy_from(h_F.data(), N);
        d_out.resize(size_t(N) * NMetric);

        constexpr int block = 128;
        svd_compare_kernel<<<(N + block - 1) / block, block>>>(d_F.cview(), d_out.view(), N);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        d_out.copy_to(h_out);

        double m[NMetric] = {0};
        for(int k = 0; k < N; ++k)
            for(int j = 0; j < NMetric; ++j)
                m[j] = std::max(m[j], double(h_out[size_t(k) * NMetric + j]));

        INFO("family: " << fams[f].name << "  new_recon=" << m[0] << " old_recon=" << m[1]
                        << " VtV=" << m[2] << " UtU=" << m[3] << " detV=" << m[4]
                        << " ord=" << m[5] << " sign=" << m[6] << " dS=" << m[7]);
        CHECK(m[0] <= fams[f].tol);  // reconstruction
        CHECK(m[2] <= 1e-13);        // V orthogonal
        CHECK(m[3] <= 1e-13);        // U orthogonal
        CHECK(m[4] <= 1e-13);        // det V == +1
        CHECK(m[5] <= 1e-13);        // S0 >= S1 >= |S2|
        CHECK(m[6] == 0.0);          // sign(S2) == sign(det F)
        CHECK(m[7] <= fams[f].tol);  // singular values agree with the old path
    }
}
