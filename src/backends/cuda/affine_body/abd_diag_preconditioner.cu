#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <cuda_tool/cuda_tool.h>
#include <kernel_cout.h>
#include <cstdlib>

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
    }

    ~ABDDiagPreconditioner() override
    {
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

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        auto converged = info.converged();

        int n = (int)diag_inv.size();
        if(n > 0)
        {
            auto k = abd_diag_preconditioner_do_apply_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, info.stream()>>>(
                info.r(), info.z(), converged.cviewer(), diag_inv.view(), n);
        }
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda
