#include <affine_body/affine_body_constitution.h>
#include <affine_body/constitutions/ortho_potential_function.h>
#include <utils/make_spd.h>
#include <cuda_tool/spread_launch.h>


namespace uipc::backend::cuda
{
namespace
{
    namespace AOP = sym::abd_ortho_potential;

    __global__ void ortho_potential_compute_energy_kernel(
        cuda_tool::BufferView<Float>     shape_energies,
        cuda_tool::CBufferView<Vector12> qs,
        cuda_tool::CBufferView<Float>    kappas,
        cuda_tool::CBufferView<Float>    volumes,
        Float                            dt,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto& q      = qs(i);
        auto& volume = volumes(i);
        auto  kappa  = kappas(i);
        Float Vdt2   = volume * dt * dt;

        Float E;
        AOP::E(E, kappa, q);

        shape_energies(i) = E * Vdt2;
    }

    __global__ void ortho_potential_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Vector12>   qs,
        cuda_tool::CBufferView<Float>      volumes,
        cuda_tool::BufferView<Vector12>    gradients,
        cuda_tool::BufferView<Matrix12x12> body_hessian,
        cuda_tool::CBufferView<Float>      kappas,
        Float                              dt,
        bool                               gradient_only,
        int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        Matrix12x12 H = Matrix12x12::Zero();
        Vector12    G = Vector12::Zero();

        const auto& q      = qs(i);
        Float       kappa  = kappas(i);
        const auto& volume = volumes(i);

        Float Vdt2 = volume * dt * dt;

        Vector9 G9;
        AOP::dEdq(G9, kappa, q);
        G.segment<9>(3) = G9 * Vdt2;
        gradients(i)    = G;

        if(gradient_only)
            return;

        Matrix9x9 H9x9;
        AOP::ddEddq(H9x9, kappa, q);
        make_spd(H9x9);

        H.block<9, 9>(3, 3) = H9x9 * Vdt2;
        body_hessian(i)     = H;
    }
    // perf round 5 (w3): device-side proof that the spread launch geometry is
    // bit-identical. The same kernel is run a second time with the
    // occupancy-max geometry into scratch buffers and every 64-bit word of
    // both outputs is compared. UIPC_GRID_SPREAD_VERIFY=1 turns it on.
    __global__ void ortho_potential_spread_verify_kernel(
        cuda_tool::CBufferView<Vector12>            a_g,
        cuda_tool::CBufferView<Vector12>            b_g,
        cuda_tool::CBufferView<Matrix12x12>         a_h,
        cuda_tool::CBufferView<Matrix12x12>         b_h,
        bool                                        gradient_only,
        cuda_tool::BufferView<unsigned long long>   counters,
        int                                         n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        unsigned long long mismatch = 0;
        unsigned long long words    = 0;
        const unsigned long long* pa =
            reinterpret_cast<const unsigned long long*>(a_g(i).data());
        const unsigned long long* pb =
            reinterpret_cast<const unsigned long long*>(b_g(i).data());
        for(int k = 0; k < 12; ++k)
            mismatch += (pa[k] != pb[k]) ? 1 : 0;
        words += 12;
        if(!gradient_only)
        {
            const unsigned long long* qa =
                reinterpret_cast<const unsigned long long*>(a_h(i).data());
            const unsigned long long* qb =
                reinterpret_cast<const unsigned long long*>(b_h(i).data());
            for(int k = 0; k < 144; ++k)
                mismatch += (qa[k] != qb[k]) ? 1 : 0;
            words += 144;
        }
        atomicAdd(&counters(0), words);
        if(mismatch)
            atomicAdd(&counters(1), mismatch);
    }
}  // namespace

class OrthoPotential final : public AffineBodyConstitution
{
  public:
    static constexpr U64 ConstitutionUID = 1ull;

    using AffineBodyConstitution::AffineBodyConstitution;

    vector<Float> h_kappas;

    cuda_tool::DeviceBuffer<Float> kappas;

    // perf round 5 (w3): UIPC_GRID_SPREAD_VERIFY=1 -> also run the G/H kernel
    // with the old (occupancy-max) launch geometry into scratch buffers and
    // count mismatching 64-bit words on device.
    bool                                        m_spread_verify = false;
    cuda_tool::DeviceBuffer<Vector12>           m_verify_g;
    cuda_tool::DeviceBuffer<Matrix12x12>        m_verify_h;
    cuda_tool::DeviceBuffer<unsigned long long> m_verify_counters;
    unsigned long long                          m_verify_words     = 0;
    unsigned long long                          m_verify_mismatches = 0;

    virtual void do_build(AffineBodyConstitution::BuildInfo& info) override
    {
        if(const char* e = std::getenv("UIPC_GRID_SPREAD_VERIFY"))
            m_spread_verify = !(e[0] == '0');
        if(m_spread_verify)
            logger::warn("[OrthoSpreadVerify] on: the occupancy-max geometry is the reference");
    }

    ~OrthoPotential() override
    {
        if(m_spread_verify)
        {
            unsigned long long h[2] = {0, 0};
            if(m_verify_counters.size() == 2)
                cudaMemcpy(h, m_verify_counters.data(), sizeof(h), cudaMemcpyDeviceToHost);
            m_verify_words += h[0];
            m_verify_mismatches += h[1];
            std::fprintf(stderr,
                         "[OrthoSpreadVerify] total: %llu output words compared, %llu mismatching\n",
                         m_verify_words,
                         m_verify_mismatches);
        }
    }

    U64 get_uid() const override { return ConstitutionUID; }

    void do_init(AffineBodyDynamics::FilteredInfo& info) override
    {
        using ForEachInfo = AffineBodyDynamics::ForEachInfo;

        // find out constitution coefficients
        h_kappas.resize(info.body_count());
        auto geo_slots = world().scene().geometries();

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            { return sc.instances().find<Float>("kappa")->view(); },
            [&](const ForEachInfo& I, Float kappa)
            {
                auto bodyI      = I.global_index();
                h_kappas[bodyI] = kappa;
            });

        auto async_copy = []<typename T>(span<T> src, cuda_tool::DeviceBuffer<T>& dst)
        {
            cuda_tool::BufferLaunch().resize<T>(dst, src.size());
            cuda_tool::BufferLaunch().copy<T>(dst.view(), src.data());
        };

        async_copy(span{h_kappas}, kappas);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace cuda_tool;

        auto body_count = info.qs().size();

        namespace AOP = sym::abd_ortho_potential;

        auto k = ortho_potential_compute_energy_kernel;
        int  n = (int)body_count;
        if(n > 0)
            k<<<cuda_tool::spread_grid_dim(n, k), cuda_tool::spread_block_dim(n, k), 0, nullptr>>>(
                info.energies(), info.qs(), kappas.cview(), info.volumes(), info.dt(), n);
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace cuda_tool;
        auto N             = info.qs().size();
        auto gradient_only = info.gradient_only();

        namespace AOP = sym::abd_ortho_potential;

        auto k = ortho_potential_compute_gradient_hessian_kernel;
        int  n = (int)N;
        if(n > 0)
            k<<<cuda_tool::spread_grid_dim(n, k), cuda_tool::spread_block_dim(n, k), 0, nullptr>>>(
                info.qs(),
                info.volumes(),
                info.gradients(),
                info.hessians(),
                kappas.cview(),
                info.dt(),
                gradient_only,
                n);

        if(m_spread_verify && n > 0)
        {
            m_verify_g.resize(n);
            m_verify_h.resize(n);
            if(m_verify_counters.size() != 2)
            {
                m_verify_counters.resize(2);
                CUDA_TOOL_CHECK(cudaMemset(
                    m_verify_counters.data(), 0, 2 * sizeof(unsigned long long)));
            }
            // the reference: the geometry best_block_dim would have picked
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                info.qs(),
                info.volumes(),
                m_verify_g.view(),
                m_verify_h.view(),
                kappas.cview(),
                info.dt(),
                gradient_only,
                n);
            auto vk = ortho_potential_spread_verify_kernel;
            vk<<<cuda_tool::best_grid_dim(n, vk), cuda_tool::best_block_dim(vk), 0, nullptr>>>(
                info.gradients(),
                m_verify_g.cview(),
                info.hessians(),
                m_verify_h.cview(),
                gradient_only,
                m_verify_counters.view(),
                n);
        }
    }
};

REGISTER_SIM_SYSTEM(OrthoPotential);
}  // namespace uipc::backend::cuda
