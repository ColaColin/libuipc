#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <cuda_tool/cuda_tool.h>
#include <kernel_cout.h>
#include <cstdlib>
#include <cstdio>

namespace uipc::backend::cuda
{
namespace
{
    __global__ void abd_diag_preconditioner_do_assemble_kernel(
        cuda_tool::CBufferView<Matrix12x12> diag_hessian,
        cuda_tool::BufferView<Matrix12x12>  diag_inv,
        int                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        diag_inv(i) = cuda_tool::eigen::inverse(diag_hessian(i));
    }

    // one thread per body, the whole 12x12 FP64 matvec through Eigen
    // (perf round 4 s06: kept as the UIPC_ABD_DIAG_APPLY_LANES=0 path and
    // as the reference of the verify mode)
    __global__ void abd_diag_preconditioner_do_apply_kernel(
        cuda_tool::CDenseVectorView<Float> r,
        cuda_tool::DenseVectorView<Float>  z,
        cuda_tool::CDense<IndexT>          converged,
        cuda_tool::BufferView<Matrix12x12> diag_inv,
        int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        if(*converged != 0)
            return;
        z.segment<12>(i * 12).as_eigen() = diag_inv(i) * r.segment<12>(i * 12).as_eigen();
    }

    // perf round 4 (s06): one lane per output row instead of one thread per
    // body. The old kernel gives one thread the whole 12x12 FP64 matvec
    // (1 920 bodies on the cube wall = 8 blocks of 256 for the whole GPU,
    // every lane striding 1 152 B through its own matrix, 144 uncoalesced
    // loads per thread); here 12 consecutive threads share a body and lane
    // = row, so the 12 loads of a row are coalesced across the lanes and
    // the 12 n threads cover every SM.
    //
    // Bit-identical to the old kernel: Eigen evaluates each coefficient of
    // the fixed-size 12x12 * 12 product as
    // `(M.row(row).transpose().cwiseProduct(v)).sum()` with a completely
    // unrolled redux, i.e. the balanced tree
    //   ((e0 + (e1 + e2)) + (e3 + (e4 + e5))) + ((e6 + (e7 + e8)) + (e9 + (e10 + e11)))
    // and nvcc contracts every leaf pair as fma(m_k, v_k, m_k+1 * v_k+1)
    // and the following add as fma(m_k-1, v_k-1, ...) (decoded from the
    // SASS of the kernel above: per row 4 DMUL, 8 DFMA, 3 DADD). The
    // intrinsics below pin exactly that sequence, independent of the
    // compiler's contraction choices. UIPC_ABD_DIAG_APPLY_VERIFY=1 checks
    // the two kernels against each other bit for bit on the live data.
    __device__ __forceinline__ Float abd_diag_row_dot_eigen_order(const Float* __restrict__ m,
                                                                  const Float* __restrict__ v,
                                                                  int row)
    {
        // m is the column-major 12x12 matrix: element (row, k) at m[k * 12 + row]
        const Float* mr = m + row;
        Float s0 = __fma_rn(mr[0 * 12], v[0], __fma_rn(mr[1 * 12], v[1], __dmul_rn(mr[2 * 12], v[2])));
        Float s3 = __fma_rn(mr[3 * 12], v[3], __fma_rn(mr[4 * 12], v[4], __dmul_rn(mr[5 * 12], v[5])));
        Float s6 = __fma_rn(mr[6 * 12], v[6], __fma_rn(mr[7 * 12], v[7], __dmul_rn(mr[8 * 12], v[8])));
        Float s9 = __fma_rn(mr[9 * 12], v[9], __fma_rn(mr[10 * 12], v[10], __dmul_rn(mr[11 * 12], v[11])));
        return __dadd_rn(__dadd_rn(s0, s3), __dadd_rn(s6, s9));
    }

    constexpr int ABD_DIAG_APPLY_BLOCK = 256;

    __global__ void abd_diag_preconditioner_do_apply_lanes_kernel(
        cuda_tool::CDenseVectorView<Float>  r,
        cuda_tool::DenseVectorView<Float>   z,
        cuda_tool::CDense<IndexT>           converged,
        cuda_tool::CBufferView<Matrix12x12> diag_inv,
        int                                 n)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= n * 12)
            return;
        if(*converged != 0)
            return;
        int body = t / 12;
        int row  = t - body * 12;
        UIPC_KERNEL_ASSERT(body * 12 + 12 <= r.size(),
                           "ABDDiagPreconditioner: r out of range, size=%d, body=%d",
                           r.size(),
                           body);
        const Float* m   = diag_inv(body).data();
        const Float* v   = r.data() + body * 12;
        z(body * 12 + row) = abd_diag_row_dot_eigen_order(m, v, row);
    }

    // verify mode: counters(0) += 1 per apply, counters(1) += number of
    // entries whose bit pattern differs between z and z_ref
    __global__ void abd_diag_preconditioner_verify_kernel(
        cuda_tool::CDenseVectorView<Float>          z,
        cuda_tool::CDenseVectorView<Float>          z_ref,
        cuda_tool::CDense<IndexT>                   converged,
        cuda_tool::BufferView<unsigned long long>   counters,
        int                                         n12)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= n12)
            return;
        if(*converged != 0)
            return;
        if(t == 0)
            atomicAdd(&counters(0), 1ull);
        if(__double_as_longlong(z(t)) != __double_as_longlong(z_ref(t)))
            atomicAdd(&counters(1), 1ull);
    }
}  // namespace

class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    cuda_tool::DeviceBuffer<Matrix12x12> diag_inv;

    // perf/kernels (K13): the per-body 12x12 inverse is a handful of
    // single-thread, latency-bound launches (0.3 ms for the one drum body)
    // that the default stream otherwise serialises in front of the MAS
    // preconditioner assembly. Run it on a side stream forked from the
    // default stream and join in do_finish_assemble() (before the solve).
    // Same kernel, same arithmetic. UIPC_ABD_DIAG_SIDE_STREAM=0 = old path.
    bool         m_side        = true;
    cudaStream_t m_side_stream = nullptr;
    cudaEvent_t  m_fork        = nullptr;
    cudaEvent_t  m_join        = nullptr;

    // perf round 4 (s06): lane-parallel apply (see the kernel comment).
    // UIPC_ABD_DIAG_APPLY_LANES=0 = one thread per body (old);
    // UIPC_ABD_DIAG_APPLY_VERIFY=1 = also run the other kernel into a scratch
    // vector on the same stream and count bit-pattern mismatches on device
    // (read back before every assembly, i.e. outside the PCG graph capture).
    bool                                       m_lanes  = true;
    bool                                       m_verify = false;
    cuda_tool::DeviceBuffer<Float>             m_z_ref;
    cuda_tool::DeviceBuffer<unsigned long long> m_verify_counters;
    unsigned long long                         m_verify_applies    = 0;
    unsigned long long                         m_verify_mismatches = 0;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);

        if(const char* e = std::getenv("UIPC_ABD_DIAG_SIDE_STREAM"))
            m_side = !(e[0] == '0');
        if(m_side)
        {
            CUDA_TOOL_CHECK(cudaStreamCreateWithFlags(&m_side_stream, cudaStreamNonBlocking));
            CUDA_TOOL_CHECK(cudaEventCreateWithFlags(&m_fork, cudaEventDisableTiming));
            CUDA_TOOL_CHECK(cudaEventCreateWithFlags(&m_join, cudaEventDisableTiming));
        }
        if(const char* e = std::getenv("UIPC_ABD_DIAG_APPLY_LANES"))
            m_lanes = !(e[0] == '0');
        if(const char* e = std::getenv("UIPC_ABD_DIAG_APPLY_VERIFY"))
            m_verify = !(e[0] == '0');
        if(m_verify)
            logger::warn("[ABDDiagApplyVerify] on: lanes={} (the other kernel is the reference)",
                         m_lanes);
    }

    ~ABDDiagPreconditioner() override
    {
        if(m_verify)
        {
            // best effort: pick up the applies since the last assembly
            unsigned long long h[2] = {0, 0};
            if(m_verify_counters.size() == 2
               && cudaMemcpy(h, m_verify_counters.data(), sizeof(h), cudaMemcpyDeviceToHost) == cudaSuccess)
            {
                m_verify_applies += h[0];
                m_verify_mismatches += h[1];
            }
            std::fprintf(stderr,
                         "[ABDDiagApplyVerify] total: %llu applies, %llu mismatched entries (lanes=%d)\n",
                         m_verify_applies,
                         m_verify_mismatches,
                         (int)m_lanes);
        }
        // best effort: the CUDA context may already be gone at exit
        if(m_join)
            cudaEventDestroy(m_join);
        if(m_fork)
            cudaEventDestroy(m_fork);
        if(m_side_stream)
            cudaStreamDestroy(m_side_stream);
    }

    virtual void do_finish_assemble() override
    {
        if(m_side && m_pending_join)
        {
            CUDA_TOOL_CHECK(cudaStreamWaitEvent(nullptr, m_join, 0));
            m_pending_join = false;
        }
    }
    bool m_pending_join = false;

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        auto diag_hessian = abd_linear_subsystem->diag_hessian();
        diag_inv.resize(diag_hessian.size());

        int n = (int)diag_inv.size();

        if(m_verify)
        {
            // outside the PCG graph capture: read back and reset the counters
            if(m_verify_counters.size() == 2)
            {
                unsigned long long h[2] = {0, 0};
                CUDA_TOOL_CHECK(cudaMemcpy(h, m_verify_counters.data(), sizeof(h), cudaMemcpyDeviceToHost));
                m_verify_applies += h[0];
                m_verify_mismatches += h[1];
                if(h[1] != 0)
                    logger::warn("[ABDDiagApplyVerify] {} mismatched entries in {} applies since the last assembly (cumulative {} / {} applies)",
                                 h[1],
                                 h[0],
                                 m_verify_mismatches,
                                 m_verify_applies);
                CUDA_TOOL_CHECK(cudaMemset(m_verify_counters.data(), 0, sizeof(h)));
            }
            else
            {
                m_verify_counters.resize(2);
                CUDA_TOOL_CHECK(cudaMemset(m_verify_counters.data(), 0, 2 * sizeof(unsigned long long)));
            }
            m_z_ref.resize((size_t)n * 12);
        }

        if(n > 0)
        {
            cudaStream_t s = nullptr;
            if(m_side)
            {
                s = m_side_stream;
                CUDA_TOOL_CHECK(cudaEventRecord(m_fork, nullptr));
                CUDA_TOOL_CHECK(cudaStreamWaitEvent(s, m_fork, 0));
            }
            auto k = abd_diag_preconditioner_do_assemble_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, s>>>(
                diag_hessian, diag_inv.view(), n);
            if(m_side)
            {
                CUDA_TOOL_CHECK(cudaEventRecord(m_join, s));
                m_pending_join = true;
            }
        }
    }

    void launch_apply(bool                                lanes,
                      cuda_tool::CDenseVectorView<Float>  r,
                      cuda_tool::DenseVectorView<Float>   z,
                      cuda_tool::CDense<IndexT>           converged,
                      int                                 n,
                      cudaStream_t                        stream)
    {
        if(lanes)
        {
            auto k  = abd_diag_preconditioner_do_apply_lanes_kernel;
            int  nt = n * 12;
            k<<<(nt + ABD_DIAG_APPLY_BLOCK - 1) / ABD_DIAG_APPLY_BLOCK, ABD_DIAG_APPLY_BLOCK, 0, stream>>>(
                r, z, converged, diag_inv.cview(), n);
        }
        else
        {
            auto k = abd_diag_preconditioner_do_apply_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, stream>>>(
                r, z, converged, diag_inv.view(), n);
        }
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        auto converged = info.converged();

        int n = (int)diag_inv.size();
        if(n > 0)
        {
            launch_apply(m_lanes, info.r(), info.z(), converged.cviewer(), n, info.stream());

            if(m_verify)
            {
                int nt = n * 12;
                cuda_tool::DenseVectorView<Float> z_ref{m_z_ref.data(), 0, nt, nt};
                launch_apply(!m_lanes, info.r(), z_ref, converged.cviewer(), n, info.stream());
                auto k = abd_diag_preconditioner_verify_kernel;
                k<<<(nt + ABD_DIAG_APPLY_BLOCK - 1) / ABD_DIAG_APPLY_BLOCK, ABD_DIAG_APPLY_BLOCK, 0, info.stream()>>>(
                    info.z().as_const(), z_ref.as_const(), converged.cviewer(), m_verify_counters.view(), nt);
            }
        }
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
