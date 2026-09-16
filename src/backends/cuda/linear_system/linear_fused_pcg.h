#pragma once
#include <linear_system/iterative_solver.h>
#include <cuda_tool/cuda_tool.h>
#include <cuda_tool/graph.h>
#include <array>

namespace uipc::backend::cuda
{
// R7 s17: the convergence doorbell the GPU writes into pinned host memory so
// the block-replay loop can poll instead of doing a blocking D2H after every
// graph replay. Both fields share one cacheline; `seq` is stored strictly
// after `rz` (single-thread program order + PCIe posted-write order), so a
// host that observes seq >= expected reads the matching rz.
struct PcgPollWord
{
    Float              rz;   // the value fused_pcg_scalar read as rz_new
    unsigned long long seq;  // doorbell: #publishes so far
};

// Fused PCG: keeps dot-product scalars (rz, pAp, rz_new) on device
// to eliminate per-iteration host synchronizations.  The update kernels read
// alpha = rz/pAp and beta = rz_new/rz directly from device memory.
// SpMV and dot(p,Ap) are fused into a single kernel pass.
// Convergence is checked every `check_interval` iterations via a single D2H copy.
//
// CUDA graph replay: the per-iteration kernel chain (spmv_dot -> update_xr ->
// preconditioner -> dot -> converged -> update_p -> swap_rz) launches ~10
// tiny kernels per iteration whose launch gaps dominate the wall time
// (~80us of ~118us per iteration on case2-scale scenes). When
// `linear_system/use_cuda_graph` is on (default), a block of
// `check_interval` iterations is recorded once per (buffer-set, N) and
// replayed as one graph launch; kernels, arguments and ordering are
// identical to the non-graph path, so numerics are unchanged. If capture
// fails (e.g. a preconditioner launches outside the capture stream), the
// solver permanently falls back to the plain loop for that instance.
class LinearFusedPCG : public IterativeSolver
{
  public:
    using IterativeSolver::IterativeSolver;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    using DeviceDenseVector = cuda_tool::DeviceDenseVector<Float>;

    SizeT fused_pcg(cuda_tool::DenseVectorView<Float>  x,
                    cuda_tool::CDenseVectorView<Float> b,
                    SizeT                              max_iter);
    void  check_init_rz_nan_inf(Float rz);
    void  check_iter_rz_nan_inf(Float rz, SizeT k);

    // s13 (UIPC_PCG_AP_ZERO_VERIFY=1): device-side check that Ap is exactly
    // zero where the removed fill<double>(Ap) node used to run, plus its
    // per-solve host report.
    void check_ap_zero(cudaStream_t stream);
    void report_ap_zero();

    // One iteration of the PCG loop body on `stream` (the unit of graph
    // capture and of the uncaptured fallback path).
    void run_iteration(cuda_tool::DenseVectorView<Float> x, cudaStream_t stream, bool timed);

    // Capture `interval` iterations into m_graph (no execution during
    // capture); on any failure disable graph replay for this instance.
    void rebuild_graph(cuda_tool::DenseVectorView<Float>  x,
                       cuda_tool::CDenseVectorView<Float> b,
                       SizeT                              interval,
                       SizeT                              max_iter);
    bool graph_key_matches(cuda_tool::DenseVectorView<Float>  x,
                           cuda_tool::CDenseVectorView<Float> b,
                           SizeT                              interval,
                           SizeT                              max_iter) const;
    void destroy_graph();

    DeviceDenseVector r;
    DeviceDenseVector z;
    DeviceDenseVector p;
    DeviceDenseVector Ap;

    cuda_tool::DeviceVar<Float> d_rz;
    cuda_tool::DeviceVar<Float> d_pAp;
    cuda_tool::DeviceVar<Float> d_rz_new;
    // s11: beta = rz_new / rz, precomputed by the fused scalar kernel
    cuda_tool::DeviceVar<Float>  d_beta;
    cuda_tool::DeviceVar<IndexT> d_converged;
    // rz_tol on device so a captured graph survives rz_tol changes
    cuda_tool::DeviceVar<Float> d_rz_tol;

    // s13 probe accumulator: [0] = non-zero count, [1] = max |Ap| bit pattern
    cuda_tool::DeviceVector<unsigned long long> m_ap_zero_acc;

    // R7 s17 (UIPC_PCG_POLL, default on): pinned zero-copy doorbell written by
    // the graph's fused_pcg_scalar node; the host spins on `seq` instead of a
    // blocking cudaMemcpy D2H after every block replay. m_poll_expected counts
    // the iterations this instance has launched (device counter m_poll_seq
    // counts the publishes executed); both are monotonic for the instance's
    // lifetime, so no reset is needed between solves. Null when the poll is
    // off or the allocation failed -- kernels then skip the publish.
    void*                                    m_poll_host = nullptr;
    void*                                    m_poll_dev  = nullptr;
    cuda_tool::DeviceVar<unsigned long long> m_poll_seq{0ull};
    unsigned long long                       m_poll_expected        = 0;
    bool                                     m_poll_fallback_warned = false;
    bool                                     m_poll_init_done       = false;

    void init_poll();
    // spin until the GPU publishes the launched iterations; returns rz_new.
    // Falls back to the old blocking read (and warns once) if the doorbell
    // shows no progress for 100 ms -- correctness never depends on the poll.
    Float poll_rz_new();

    // R7: last-block ticket for the fused dot + scalar kernel; always 0
    // between launches (the last block resets it).
    cuda_tool::DeviceVar<unsigned int> m_dot_ticket{0u};
    // R7 fold mode 1: this iteration's rz, carried aside by fused_update_xr
    // so the folded update_p kernel can divide by it after d_rz is overwritten.
    cuda_tool::DeviceVar<Float> d_rz_prev;
    // R7 probe (UIPC_PCG_FUSE_DOT_VERIFY=1): shadow scalars written by the
    // fused tail, compared on device against the original node's live ones.
    cuda_tool::DeviceVar<Float>                 m_v_rz;
    cuda_tool::DeviceVar<Float>                 m_v_beta;
    cuda_tool::DeviceVar<IndexT>                m_v_converged;
    cuda_tool::DeviceVar<Float>                 m_v_pAp;
    cuda_tool::DeviceVector<unsigned long long> m_scalar_cmp_acc;

    Float max_iter_ratio  = 2.0;
    Float global_tol_rate = 1e-4;
    Float reserve_ratio   = 1.5;
    SizeT check_interval  = 5;

    // PCG-stall fix (2026-09-15): set by fused_pcg when this solve's rz0
    // check found the preconditioner's action not contract-valid (rz0 < 0,
    // or z non-finite with a finite r); run_iteration then uses z = r and
    // the solve runs the plain-launch path (no graph replay). Reset at the
    // top of every solve.
    bool m_precond_bypass = false;

    // --- CUDA graph state ---
    IndexT m_use_cuda_graph = 1;  // config: linear_system/use_cuda_graph
    // 0 = plain loop, 1 = block replay, 2 = full-GPU while-loop graph
    IndexT                  m_graph_mode = 0;
    cuda_tool::GraphCapture m_graph;
    // validity key: every device pointer baked into the captured kernels
    std::array<const void*, 12> m_graph_ptrs{};
    SizeT                       m_graph_n        = 0;
    SizeT                       m_graph_interval = 0;
    SizeT                       m_graph_max_iter = 0;
    // round6 (s13): the SpMV+dot grid baked into the capture (blocks; 0 = the
    // capacity grid). Part of the key, so a matrix that outgrows it re-captures.
    int m_graph_spmv_grid = -1;

    // --- full-GPU while-loop graph (CUDA >= 12.4) ---
    cuda_tool::GraphWhile        m_while;
    cuda_tool::DeviceVar<IndexT> d_iter;
    std::array<const void*, 12>  m_while_ptrs{};
    SizeT                        m_while_n         = 0;
    SizeT                        m_while_max_iter  = 0;
    int                          m_while_spmv_grid = -1;
    bool while_key_matches(cuda_tool::DenseVectorView<Float>  x,
                           cuda_tool::CDenseVectorView<Float> b,
                           SizeT                              max_iter) const;
    void rebuild_while(cuda_tool::DenseVectorView<Float>  x,
                       cuda_tool::CDenseVectorView<Float> b,
                       SizeT                              max_iter);
    void destroy_while();
};
}  // namespace uipc::backend::cuda
