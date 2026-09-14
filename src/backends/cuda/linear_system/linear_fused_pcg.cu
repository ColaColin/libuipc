#include <linear_system/linear_fused_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <cuda_tool/linear_reduction.h>
#include <uipc/common/timer.h>
#include <cub/warp/warp_reduce.cuh>
#include <cuda_tool/cub.h>
#include <cuda_tool/spread_launch.h>
#include <algorithm>
#include <optional>
namespace uipc::backend::cuda
{
namespace
{
    // s11 env switches for the small per-iteration kernels of the captured PCG
    // block. `fuse` folds fused_update_converged + fused_swap_rz + the two
    // 8-byte accumulator memsets (d_rz_new, d_pAp) into one single-thread
    // scalar kernel plus one store in fused_update_xr; `block` overrides the
    // occupancy-maximising block size of the element-wise vector kernels
    // (cudaOccupancyMaxPotentialBlockSize picks 1024, which leaves half the
    // SMs idle at these vector lengths).
    // NOTE: `fuse` is also read, independently and by the same name, in
    // linear_system/spmv.cu (Spmv::rbk_sym_spmv_dot skips its d_dot memset
    // when it is on). rbk_sym_spmv_dot is called only from this solver.
    struct PcgSmallEnv
    {
        bool fuse  = true;  // UIPC_PCG_FUSE_SCALAR=0 -> old chain
        int  block = 256;   // UIPC_PCG_BLOCK_DIM=0   -> best_block_dim()
        // s13: an existing per-iteration kernel stores 0 back into Ap, so the
        // SpMV's `fill<double>(Ap)` graph node disappears (the vector is
        // already zero when the next SpMV starts). Bit-identical: the SpMV
        // sees the same zeros.  UIPC_PCG_FUSE_AP_ZERO = 0 -> keep the fill
        // node, 1 -> zero in fused_update_xr (which already reads Ap(i)),
        // 2 -> zero in fused_update_p (the last node before the next SpMV, so
        // the zeroed lines are still in L2 when the SpMV reads them).
        int ap_zero = 2;
        // s13 probe: UIPC_PCG_AP_ZERO_VERIFY=1 checks on device, right where
        // the fill node used to be, that every Ap(i) is exactly zero.
        bool ap_zero_verify = false;
        // R7 (REJECTED, default-off): fold the <<<1,1>>> fused_pcg_scalar
        // node into the tail of the r^T z dot reduction. The last block (a
        // ticket atomic + __threadfence, the CUDA threadFenceReduction
        // pattern) reads the completed accumulator and runs exactly the
        // scalar kernel's body. One fewer node per captured PCG iteration.
        // R7 VERDICT: **rejected, default 0** -- the node is not worth what
        // removing it costs. The code and both fold designs stay in tree as
        // measurement arms (and for the 5090 re-test, see below). With fold = 0
        // the shipped kernel chain and every output it produces are the pre-R7
        // ones; the source is not, since fused_update_xr gained a guarded
        // `save_rz_prev` store and d_rz_prev / m_dot_ticket are allocated
        // unconditionally. Note fold is also forced to 0 by UIPC_PCG_FUSE_SCALAR=0.
        //
        // UIPC_PCG_FOLD selects how:
        //   0 = keep the separate node (SHIPPED)
        //   1 = fold it into fused_update_p_beta. Every thread
        //       recomputes the two scalars it needs from the same operands;
        //       block 0 / thread 0 publishes d_converged, the guarded
        //       d_rz <- rz_new and the d_pAp reset. Every address written
        //       there is read by NO thread of that kernel, so this needs no
        //       barrier, no fence and no extra atomic.
        //
        // The fold is GATED ON THE GRID of the kernel it is folded into. The
        // node it removes costs a FIXED ~0.9 us of graph-node overhead per PCG
        // iteration, whatever the problem size; the fold's cost is per block
        // (three scalar loads and one predicated store more per block), so it
        // multiplies by the number of waves. Measured end-to-end, 3 runs per
        // arm, ms per Newton iteration:
        //   rigid-wrecking-balls  grid  27  -1.41 % (3 runs) / -0.83 % (5 runs,
        //                                   OVERLAPPING -- t=1.39, p~0.2)
        //   cube-wall-cloth       grid 140  -0.10 %   (inside scatter)
        //   mas-bunny             grid 225  +0.73 %   (disjoint, a regression)
        //   stiff-gipc-case2      grid 500  +0.59 %   (disjoint, a regression)
        // At the smallest grid in the suite the scope measurement is only
        // -0.0146 ms per Newton iteration (-0.23 % of kernel time): the node
        // is worth 0.0200 ms/it and the fold costs 0.0053 ms/it back. That is
        // below wrecking-balls' own +-1.3 % per-iteration scatter, so it is
        // not a demonstrable end-to-end gain and mode 1 is NOT enabled.
        // UIPC_PCG_FOLD_MAXGRID gates mode 1 on the grid (default = the device
        // SM count, 0 = never gate); it is how the crossover was bracketed.
        //
        // Worth re-testing on the acceptance GPU: the fold's cost is per
        // block and is paid once per WAVE, so at ~170 SMs grid 225 is 1.3
        // waves instead of 6.3 and the cost should fall ~5x, while the node's
        // ~0.93 us is largely fixed. `UIPC_PCG_FOLD=1 UIPC_PCG_FOLD_MAXGRID=0`
        // is the one-env-var arm for that.
        //
        //   2 = fold it into the TAIL OF THE DOT (last-block-done handshake).
        //       Correct, and measured SLOWER than arm A on mas-bunny: the
        //       __threadfence() every block must execute costs more than the
        //       node it removes. Kept as a measurement arm; see the ledger.
        int fold         = 0;
        int fold_maxgrid = -1;  // <0 = device SM count
        // R7 probe: UIPC_PCG_FUSE_DOT_VERIFY=1 runs BOTH paths every
        // iteration -- the fused tail into shadow scalars, then the original
        // <<<1,1>>> node into the live ones -- and compares them on device.
        // The live state is the old path's, so the simulation is unchanged
        // while it is being checked.
        bool fuse_dot_verify = false;
        // R7: UIPC_PCG_FOLD_VERIFY=1 runs the pre-R7 chain (scalar node +
        // fused_update_p_beta) as the reference, snapshots every output,
        // restores the non-idempotent ones (p and Ap are read-modify-write),
        // then runs the folded kernel and compares on device.
        bool fold_verify = false;
        // R7 attribution probe (read only when a fold is on, never shipped):
        // the last-block handshake order. 1 = __threadfence() (membar.gl),
        // 2 = PTX `fence.acq_rel.gpu` (the weakest ordering that is still
        // correct), 0 = **no fence at all** -- incorrect by the CUDA memory
        // model, and present only to price the fence. Never ship 0.
        int dot_fence = 1;
    };
    const PcgSmallEnv& pcg_small_env()
    {
        static const PcgSmallEnv env = []
        {
            PcgSmallEnv e;
            if(const char* s = std::getenv("UIPC_PCG_FUSE_SCALAR"))
                e.fuse = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_PCG_BLOCK_DIM"))
            {
                int b = std::atoi(s);
                e.block = (b >= 0 && b <= 1024 && (b % 32) == 0) ? b : 256;
            }
            if(const char* s = std::getenv("UIPC_PCG_FUSE_AP_ZERO"))
            {
                int v     = std::atoi(s);
                e.ap_zero = (v >= 0 && v <= 2) ? v : 2;
            }
            if(const char* s = std::getenv("UIPC_PCG_AP_ZERO_VERIFY"))
                e.ap_zero_verify = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_PCG_FOLD"))
            {
                int v = std::atoi(s);
                // out of range falls back to the DEFAULT, which is 0 (R7 was
                // rejected). It used to fall back to 1, so UIPC_PCG_FOLD=3
                // silently enabled the rejected fold; the sibling parses of
                // ap_zero and dot_fence both fall back to their own defaults.
                e.fold = (v >= 0 && v <= 2) ? v : 0;
            }
            if(const char* s = std::getenv("UIPC_PCG_FOLD_MAXGRID"))
                e.fold_maxgrid = std::atoi(s);
            if(const char* s = std::getenv("UIPC_PCG_FUSE_DOT_VERIFY"))
                e.fuse_dot_verify = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_PCG_FOLD_VERIFY"))
                e.fold_verify = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_PCG_DOT_FENCE"))
            {
                int v       = std::atoi(s);
                e.dot_fence = (v >= 0 && v <= 2) ? v : 1;
            }
            return e;
        }();
        return env;
    }

    __global__ void fused_dot_kernel(cuda_tool::CDenseVectorView<Float> x,
                                     cuda_tool::CDenseVectorView<Float> y,
                                     cuda_tool::Dense<Float> d_result,
                                     int                     n)
    {
        constexpr int block_dim = 256;
        constexpr int warp_size = 32;
        constexpr int num_warps = block_dim / warp_size;

        using WarpReduce = cub::WarpReduce<Float, warp_size>;
        __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];

        int   i   = blockIdx.x * blockDim.x + threadIdx.x;
        Float val = (i < n) ? x(i) * y(i) : Float(0);

        int   warp_id  = threadIdx.x / warp_size;
        int   lane_id  = threadIdx.x & (warp_size - 1);
        Float warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);

        // two-level reduction: one atomic per block instead of one per warp —
        // ~4k same-address atomic doubles serialize badly on a single counter
        __shared__ Float s_partials[num_warps];
        if(lane_id == 0)
            s_partials[warp_id] = warp_sum;
        __syncthreads();
        if(threadIdx.x < warp_size)
        {
            Float partial =
                (threadIdx.x < num_warps) ? s_partials[threadIdx.x] : Float(0);
            __syncwarp();
            partial = WarpReduce(temp_storage[0]).Sum(partial);
            if(threadIdx.x == 0)
                cuda_tool::atomic_add(d_result.data(), partial);
        }
    }

    // R7: the scalar update that used to be a <<<1, 1>>> kernel of its own,
    // as a device function so that both paths compile from ONE source
    // expression: converged = |rz_new| <= rz_tol, beta = rz_new / rz
    // (pre-swap), the guarded rz <- rz_new, then the reset of the p^T A p
    // accumulator for the next iteration's SpMV.
    __device__ __forceinline__ void pcg_scalar_body(Float        rz_new,
                                                    Float*       d_rz,
                                                    Float*       d_beta,
                                                    IndexT*      d_converged,
                                                    const Float* d_rz_tol,
                                                    Float*       d_pAp)
    {
        Float  rz   = *d_rz;
        IndexT conv = abs(rz_new) <= *d_rz_tol ? 1 : 0;
        *d_converged = conv;
        *d_beta      = rz_new / rz;
        if(conv == 0)
            *d_rz = rz_new;
        *d_pAp = Float(0);
    }

    // R7: fused_dot_kernel with the scalar update appended to the LAST block
    // to finish the reduction. The reduction half is a character-for-character
    // copy of fused_dot_kernel above (deliberately duplicated rather than
    // templated, so the shipped dot keeps its own kernel and its own profile
    // row). The tail is the standard CUDA threadFenceReduction handshake:
    // every block fences after its atomic, then tickets in; the block that
    // draws the last ticket has seen every other block's contribution, so the
    // accumulator it reads is the same completed value the separate <<<1,1>>>
    // node would have read. `d_ticket` is left at 0 for the next launch, which
    // is what makes this safe under graph replay.
    template <int FenceMode>
    __device__ __forceinline__ void pcg_dot_fence()
    {
        if(FenceMode == 1)
            __threadfence();
        else if(FenceMode == 2)
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }

    template <int FenceMode>
    __global__ void fused_dot_scalar_kernel(cuda_tool::CDenseVectorView<Float> x,
                                            cuda_tool::CDenseVectorView<Float> y,
                                            cuda_tool::Dense<Float> d_result,
                                            int                     n,
                                            unsigned int*           d_ticket,
                                            cuda_tool::Dense<Float>  d_rz,
                                            cuda_tool::Dense<Float>  d_beta,
                                            cuda_tool::Dense<IndexT> d_converged,
                                            cuda_tool::CDense<Float> d_rz_tol,
                                            cuda_tool::Dense<Float>  d_pAp,
                                            bool                     verify,
                                            Float*                   v_rz,
                                            Float*                   v_beta,
                                            IndexT*                  v_converged,
                                            Float*                   v_pAp)
    {
        constexpr int block_dim = 256;
        constexpr int warp_size = 32;
        constexpr int num_warps = block_dim / warp_size;

        using WarpReduce = cub::WarpReduce<Float, warp_size>;
        __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];

        int   i   = blockIdx.x * blockDim.x + threadIdx.x;
        Float val = (i < n) ? x(i) * y(i) : Float(0);

        int   warp_id  = threadIdx.x / warp_size;
        int   lane_id  = threadIdx.x & (warp_size - 1);
        Float warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);

        // two-level reduction: one atomic per block instead of one per warp —
        // ~4k same-address atomic doubles serialize badly on a single counter
        __shared__ Float s_partials[num_warps];
        if(lane_id == 0)
            s_partials[warp_id] = warp_sum;
        __syncthreads();
        if(threadIdx.x < warp_size)
        {
            Float partial =
                (threadIdx.x < num_warps) ? s_partials[threadIdx.x] : Float(0);
            __syncwarp();
            partial = WarpReduce(temp_storage[0]).Sum(partial);
            if(threadIdx.x == 0)
                cuda_tool::atomic_add(d_result.data(), partial);
        }

        // --- R7 tail: the last block runs the scalar update ---
        if(threadIdx.x == 0)
        {
            pcg_dot_fence<FenceMode>();
            unsigned int ticket = atomicAdd(d_ticket, 1u);
            if(ticket == gridDim.x - 1)
            {
                *d_ticket = 0;
                // volatile: the accumulator was written by other blocks'
                // atomics, so it must not be served from this SM's L1
                Float rz_new = *static_cast<volatile Float*>(d_result.data());
                if(verify)
                {
                    *v_rz = *d_rz;  // the shadow starts from the live rz
                    pcg_scalar_body(rz_new, v_rz, v_beta, v_converged, d_rz_tol.data(), v_pAp);
                }
                else
                {
                    pcg_scalar_body(rz_new,
                                    d_rz.data(),
                                    d_beta.data(),
                                    d_converged.data(),
                                    d_rz_tol.data(),
                                    d_pAp.data());
                }
            }
        }
    }

    // R7 probe: compare the fused tail's shadow scalars against the values the
    // original <<<1, 1>>> node wrote. [0] = 32-bit words compared,
    // [1] = mismatching words.
    __global__ void pcg_scalar_cmp_kernel(const Float*        rz,
                                          const Float*        beta,
                                          const IndexT*       converged,
                                          const Float*        pAp,
                                          const Float*        v_rz,
                                          const Float*        v_beta,
                                          const IndexT*       v_converged,
                                          const Float*        v_pAp,
                                          unsigned long long* acc)
    {
        const unsigned int* a[4] = {reinterpret_cast<const unsigned int*>(rz),
                                    reinterpret_cast<const unsigned int*>(beta),
                                    reinterpret_cast<const unsigned int*>(pAp),
                                    reinterpret_cast<const unsigned int*>(converged)};
        const unsigned int* b[4] = {reinterpret_cast<const unsigned int*>(v_rz),
                                    reinterpret_cast<const unsigned int*>(v_beta),
                                    reinterpret_cast<const unsigned int*>(v_pAp),
                                    reinterpret_cast<const unsigned int*>(v_converged)};
        const int words[4] = {sizeof(Float) / 4, sizeof(Float) / 4, sizeof(Float) / 4, 1};
        unsigned long long total = 0, bad = 0;
        for(int k = 0; k < 4; ++k)
            for(int w = 0; w < words[k]; ++w)
            {
                ++total;
                if(a[k][w] != b[k][w])
                    ++bad;
            }
        acc[0] += total;
        acc[1] += bad;
    }

    // host-side report for UIPC_PCG_FUSE_DOT_VERIFY=1 (no CUDA call at exit)
    struct PcgScalarVerifyReport
    {
        unsigned long long words = 0, mismatch = 0;
        ~PcgScalarVerifyReport()
        {
            if(words)
                std::fprintf(stderr,
                             "[PcgScalarVerify] %llu output words (32-bit) compared, %llu mismatching\n",
                             words,
                             mismatch);
        }
    };
    PcgScalarVerifyReport g_pcg_scalar_report;

    __global__ void fused_update_xr_kernel(cuda_tool::CDense<Float> d_rz,
                                           cuda_tool::CDense<Float> d_pAp,
                                           cuda_tool::CDense<IndexT> d_converged,
                                           cuda_tool::DenseVectorView<Float>  x,
                                           cuda_tool::CDenseVectorView<Float> p,
                                           cuda_tool::DenseVectorView<Float>  r,
                                           cuda_tool::DenseVectorView<Float>  Ap,
                                           cuda_tool::Dense<Float> d_rz_new_reset,
                                           bool                    reset_rz_new,
                                           bool                    zero_Ap,
                                           cuda_tool::Dense<Float> d_rz_prev_out,
                                           bool                    save_rz_prev,
                                           int                     n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        // s11: zero the r^T z accumulator for this iteration's fused_dot here
        // (nothing reads it between the host check of the previous block and
        // that dot), replacing a graph memset node. Unconditional, exactly
        // like the memset it replaces.
        if(i == 0 && reset_rz_new)
            *d_rz_new_reset = Float(0);
        // R7 fold mode 1: carry this iteration's rz aside for the beta of the
        // folded update_p kernel, which runs after d_rz has been overwritten.
        // Thread 0 only READS d_rz here (every thread does, for alpha) and
        // writes a location nothing in this kernel reads.
        if(i == 0 && save_rz_prev)
            *d_rz_prev_out = *d_rz;
        if(i >= n)
            return;
        if(*d_converged != 0)
        {
            // s13: keep Ap zero on the early-exit path too, so the next SpMV
            // of this block always starts from a zero vector.
            if(zero_Ap)
                Ap(i) = Float(0);
            return;
        }
        Float alpha = *d_rz / *d_pAp;
        x(i) += alpha * p(i);
        Float Ap_i = Ap(i);
        r(i) -= alpha * Ap_i;
        // s13: the accumulator of the *next* SpMV, zeroed here from a line the
        // thread has just read, replacing the fill<double>(Ap) graph node.
        if(zero_Ap)
            Ap(i) = Float(0);
    }

    // s13 probe (UIPC_PCG_AP_ZERO_VERIFY=1): counts the Ap entries that are not
    // exactly zero where the fill node used to run. Launched before the SpMV.
    __global__ void pcg_ap_zero_check_kernel(cuda_tool::CDenseVectorView<Float> Ap,
                                             unsigned long long* out,
                                             int                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        Float v = Ap(i);
        if(v != Float(0))
        {
            atomicAdd(out, 1ull);
            atomicMax(out + 1, (unsigned long long)__double_as_longlong(::fabs(v)));
        }
    }

    __global__ void fused_update_p_kernel(cuda_tool::CDense<Float>  d_rz_new,
                                          cuda_tool::CDense<Float>  d_rz,
                                          cuda_tool::CDense<IndexT> d_converged,
                                          cuda_tool::DenseVectorView<Float>  p,
                                          cuda_tool::CDenseVectorView<Float> z,
                                          cuda_tool::DenseVectorView<Float>  Ap,
                                          bool                               zero_Ap,
                                          int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // s13 mode 2: seed the next SpMV's accumulator from the last node of
        // the iteration (unconditional, exactly like the fill node it
        // replaces).
        if(zero_Ap)
            Ap(i) = Float(0);
        if(*d_converged != 0)
            return;
        Float beta = *d_rz_new / *d_rz;
        p(i)       = z(i) + beta * p(i);
    }

    // s11: p = z + beta * p with beta precomputed by fused_pcg_scalar_kernel.
    // Same value, same expression order as the kernel above.
    __global__ void fused_update_p_beta_kernel(cuda_tool::CDense<Float>  d_beta,
                                               cuda_tool::CDense<IndexT> d_converged,
                                               cuda_tool::DenseVectorView<Float>  p,
                                               cuda_tool::CDenseVectorView<Float> z,
                                               cuda_tool::DenseVectorView<Float>  Ap,
                                               bool                               zero_Ap,
                                               int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // s13 mode 2: see fused_update_p_kernel.
        if(zero_Ap)
            Ap(i) = Float(0);
        if(*d_converged != 0)
            return;
        Float beta = *d_beta;
        p(i)       = z(i) + beta * p(i);
    }

    // R7 fold mode 1: fused_update_p_beta with the <<<1, 1>>> scalar node
    // folded in. The two scalars the node computed are recomputed per thread
    // from exactly the same operands -- conv = |rz_new| <= rz_tol and
    // beta = rz_new / rz_prev, where rz_prev is the value of d_rz that the
    // node would have divided by (carried aside by fused_update_xr earlier in
    // this same iteration, before anything writes d_rz).
    //
    // The three scalars the node *published* are published here by thread 0 of
    // block 0. Each of them is read by NO thread of this kernel, so this needs
    // no barrier and no fence:
    //   d_converged -> read by the preconditioner and fused_update_xr, next it
    //   d_rz        -> read by fused_update_xr (alpha), next iteration
    //   d_pAp       -> the next SpMV's accumulator
    __global__ void fused_update_p_scalar_kernel(cuda_tool::CDense<Float> d_rz_new,
                                                 cuda_tool::CDense<Float> d_rz_prev,
                                                 cuda_tool::CDense<Float> d_rz_tol,
                                                 cuda_tool::Dense<Float>  d_rz,
                                                 cuda_tool::Dense<IndexT> d_converged,
                                                 cuda_tool::Dense<Float>  d_pAp,
                                                 cuda_tool::DenseVectorView<Float>  p,
                                                 cuda_tool::CDenseVectorView<Float> z,
                                                 cuda_tool::DenseVectorView<Float>  Ap,
                                                 bool                               zero_Ap,
                                                 int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        // All three scalars are loaded unconditionally and up front: they are
        // independent addresses, so the block pays ONE memory round trip for
        // them. (Loading rz_prev only on the not-converged path -- which is
        // what the pre-R7 kernel did with d_beta behind the d_converged
        // test -- costs a second, dependent round trip per block: measured
        // +2.3 us per launch at gridDim 225 on mas-bunny.)
        Float  rz_new  = *d_rz_new;
        Float  rz_tol  = *d_rz_tol;
        Float  rz_prev = *d_rz_prev;
        IndexT conv    = abs(rz_new) <= rz_tol ? 1 : 0;
        if(i == 0)
        {
            *d_converged = conv;
            if(conv == 0)
                *d_rz = rz_new;
            *d_pAp = Float(0);
        }
        if(i >= n)
            return;
        // s13 mode 2: see fused_update_p_kernel.
        if(zero_Ap)
            Ap(i) = Float(0);
        if(conv != 0)
            return;
        Float beta = rz_new / rz_prev;
        p(i)       = z(i) + beta * p(i);
    }

    // s11: one single-thread node replacing fused_update_converged +
    // fused_swap_rz + the d_pAp memset. Same expressions in the same order:
    // converged = |rz_new| <= rz_tol, beta = rz_new / rz (pre-swap), then the
    // guarded rz <- rz_new, then the reset of the p^T A p accumulator for the
    // next iteration's SpMV (nothing reads it in between).
    __global__ void fused_pcg_scalar_kernel(cuda_tool::CDense<Float> d_rz_new,
                                            cuda_tool::Dense<Float>  d_rz,
                                            cuda_tool::Dense<Float>  d_beta,
                                            cuda_tool::Dense<IndexT> d_converged,
                                            cuda_tool::CDense<Float> d_rz_tol,
                                            cuda_tool::Dense<Float>  d_pAp)
    {
        // R7: the body now lives in pcg_scalar_body so that this node and the
        // fused dot tail compile from one source expression.
        pcg_scalar_body(*d_rz_new,
                        d_rz.data(),
                        d_beta.data(),
                        d_converged.data(),
                        d_rz_tol.data(),
                        d_pAp.data());
    }

    __global__ void fused_swap_rz_kernel(cuda_tool::CDense<Float>  d_rz_new,
                                         cuda_tool::Dense<Float>   d_rz,
                                         cuda_tool::CDense<IndexT> d_converged)
    {
        if(*d_converged != 0)
            return;
        *d_rz = *d_rz_new;
    }

    __global__ void fused_update_converged_kernel(cuda_tool::CDense<Float> d_rz_new,
                                                  cuda_tool::Dense<IndexT> d_converged,
                                                  cuda_tool::CDense<Float> d_rz_tol,
                                                  int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        Float rz_new = *d_rz_new;
        Float rz_tol = *d_rz_tol;
        *d_converged = abs(rz_new) <= rz_tol ? 1 : 0;
    }

#if CUDA_TOOL_GRAPH_WHILE
    // --- full-GPU while-loop mode kernels (conditional node, CUDA >= 12.4) ---

    // per-launch reset, first node of the setup chain
    __global__ void pcg_while_reset_kernel(cuda_tool::Dense<IndexT> d_converged,
                                           cuda_tool::Dense<IndexT> d_iter,
                                           cuda_tool::Dense<Float>  d_pAp)
    {
        *d_converged = 0;
        *d_iter      = 0;
        // s11: the p^T A p accumulator is reset by the loop body's scalar
        // kernel; seed it here for the first iteration.
        *d_pAp = Float(0);
    }

    // last node of the setup chain: rz_tol = tol_rate * |rz0| on device,
    // and the zero-system early exit (skip the loop body entirely)
    __global__ void pcg_while_setup_kernel(cuda_tool::CDense<Float>   d_rz,
                                           cuda_tool::Dense<Float>    d_rz_tol,
                                           Float                      tol_rate,
                                           cudaGraphConditionalHandle handle)
    {
        Float rz0 = *d_rz;
        UIPC_KERNEL_ASSERT(::isfinite(rz0), "FusedPCG Init: r^T*z = {} is not finite", rz0);
        *d_rz_tol = ::abs(rz0) * tol_rate;
        if(::abs(rz0) == Float{0.0})
            cudaGraphSetConditional(handle, 0u);
    }

    // last node of the loop body: publish convergence and decide whether the
    // WHILE node re-executes the body. `k` counts completed iterations;
    // the plain loop runs at most max_iter-1 of them.
    __global__ void pcg_while_control_kernel(cudaGraphConditionalHandle handle,
                                             cuda_tool::CDense<Float> d_rz_new,
                                             cuda_tool::CDense<Float> d_rz_tol,
                                             cuda_tool::Dense<IndexT> d_converged,
                                             cuda_tool::Dense<IndexT> d_iter,
                                             int max_iter_minus_1)
    {
        Float rz_new = *d_rz_new;
        UIPC_KERNEL_ASSERT(::isfinite(rz_new), "FusedPCG Iter: r^T*z = {} is not finite", rz_new);
        bool converged = ::abs(rz_new) <= *d_rz_tol;
        *d_converged   = converged ? 1 : 0;

        int k     = (int)(*d_iter) + 1;
        *d_iter   = k;
        bool keep = !converged && (k < max_iter_minus_1);
        cudaGraphSetConditional(handle, keep ? 1u : 0u);
    }
#endif
}  // namespace

REGISTER_SIM_SYSTEM(LinearFusedPCG);

void LinearFusedPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto        solver_attr = config.find<std::string>("linear_system/solver");
    std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "fused_pcg")
    {
        throw SimSystemException("LinearFusedPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    global_tol_rate    = tol_rate_attr->view()[0];

    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    if(check_attr)
        check_interval = check_attr->view()[0];

    auto graph_attr = config.find<IndexT>("linear_system/use_cuda_graph");
    if(graph_attr)
        m_use_cuda_graph = graph_attr->view()[0];

    // v1 scope: graph replay is only enabled for the default IPC pipeline.
    // The al-ipc pipeline hits a host-side fail-fast (0xC0000409) during
    // stream capture *in the C++ test binary only* (the equivalent python
    // al-ipc scenes capture and replay fine) — root cause not yet found, so
    // keep al-ipc on the plain launch path until it is. See doc 09.
    auto constitution_attr = config.find<std::string>("contact/constitution");
    if(constitution_attr && constitution_attr->view()[0] != "ipc" && m_use_cuda_graph != 0)
    {
        m_use_cuda_graph = 0;
        logger::info("LinearFusedPCG: contact/constitution != ipc — CUDA graph replay disabled");
    }

    // graph mode: 2 = full-GPU while-loop (CUDA >= 12.4 toolkit+driver),
    // 1 = host-checked block replay, 0 = plain launches.
    // Measured on case2-scale scenes (RTX 5090, ~85 iters/solve): block
    // replay median 203ms/frame vs while-loop 221-226ms — the WHILE node's
    // per-iteration evaluation costs more than the amortized host check
    // every 5 iterations, so 1 is the default; 2 keeps the CPU completely
    // out of the loop if that matters more than raw frame time.
    m_graph_mode = 0;
    if(m_use_cuda_graph == 1)
        m_graph_mode = 1;
#if CUDA_TOOL_GRAPH_WHILE
    else if(m_use_cuda_graph >= 2)
        m_graph_mode = cuda_tool::GraphWhile::runtime_supported() ? 2 : 1;
#endif

    auto dump_attr = config.find<IndexT>("extras/debug/dump_linear_pcg");
    if(dump_attr && dump_attr->view()[0] != 0)
        logger::warn(
            "LinearFusedPCG: extras/debug/dump_linear_pcg is enabled but "
            "fused_pcg does not support PCG vector dumps. "
            "Set linear_system/solver to \"linear_pcg\" to use this feature.");

    if(pcg_small_env().fold_verify)
    {
        // the verifier snapshots on the default stream, which cannot happen
        // inside a stream capture; the captured and plain paths launch the
        // same kernels with the same arguments in the same order.
        m_use_cuda_graph = 0;
        m_graph_mode     = 0;
        logger::warn("LinearFusedPCG: UIPC_PCG_FOLD_VERIFY=1 -> graph replay disabled for this run");
    }

    logger::info("LinearFusedPCG: max_iter_ratio = {}, tol_rate = {}, check_interval = {}, graph_mode = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 check_interval,
                 m_graph_mode);
}

void LinearFusedPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(0);

    auto N = x.size();
    if(r.capacity() < N)
    {
        auto M = reserve_ratio * N;
        r.reserve(M);
        z.reserve(M);
        p.reserve(M);
        Ap.reserve(M);
    }

    r.resize(N);
    z.resize(N);
    p.resize(N);
    Ap.resize(N);

    // s13 probe scratch (allocated outside any graph capture)
    if(pcg_small_env().ap_zero && pcg_small_env().ap_zero_verify && m_ap_zero_acc.size() < 2)
    {
        m_ap_zero_acc.resize(2);
        CUDA_TOOL_CHECK(cudaMemset(m_ap_zero_acc.data(), 0, 2 * sizeof(unsigned long long)));
    }

    // R7: the dot-tail ticket. The last block of every reduction resets it to
    // 0, so this is only ever a belt-and-braces re-seed between solves; it is
    // blocking for the same reason the d_pAp seed below is (it must be ordered
    // against the graph launch stream).
    if(pcg_small_env().fuse && pcg_small_env().fold == 2)
    {
        CUDA_TOOL_CHECK(cudaMemset(m_dot_ticket.data(), 0, sizeof(unsigned int)));
        if(pcg_small_env().fuse_dot_verify && m_scalar_cmp_acc.size() < 2)
        {
            m_scalar_cmp_acc.resize(2);
            CUDA_TOOL_CHECK(cudaMemset(m_scalar_cmp_acc.data(), 0, 2 * sizeof(unsigned long long)));
        }
    }

    auto iter = fused_pcg(x, b, max_iter_ratio * b.size());

    if(pcg_small_env().ap_zero && pcg_small_env().ap_zero_verify)
        report_ap_zero();

    // R7 probe: drain the device comparison counters into the host report
    if(pcg_small_env().fuse && pcg_small_env().fold == 2
       && pcg_small_env().fuse_dot_verify && m_scalar_cmp_acc.size() >= 2)
    {
        std::array<unsigned long long, 2> h{};
        CUDA_TOOL_CHECK(cudaMemcpy(h.data(), m_scalar_cmp_acc.data(), sizeof(h), cudaMemcpyDeviceToHost));
        CUDA_TOOL_CHECK(cudaMemset(m_scalar_cmp_acc.data(), 0, sizeof(h)));
        g_pcg_scalar_report.words += h[0];
        g_pcg_scalar_report.mismatch += h[1];
    }

    info.iter_count(iter);
}

void LinearFusedPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearFusedPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint   = (r_ok && z_bad) ?
                          "preconditioner failed, likely due to inverse matrix calculation failure" :
                          "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    k,
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

// d_result = x^T * y  (device-only CUB warp reduction)
void fused_dot(cuda_tool::CDenseVectorView<Float> x,
               cuda_tool::CDenseVectorView<Float> y,
               cuda_tool::VarView<Float>          d_result,
               cudaStream_t                       stream = nullptr,
               bool                               zero_result = true)
{
    // s11: with the scalar fusion on, the accumulator is already zeroed by the
    // preceding fused_update_xr (d_rz_new) / fused_pcg_scalar_kernel (d_pAp),
    // so this memset node disappears from the captured block.
    if(zero_result)
        cudaMemsetAsync(d_result.data(), 0, sizeof(Float), stream);

    constexpr int block_dim   = 256;
    int           n           = x.size();
    int           block_count = (n + block_dim - 1) / block_dim;

    if(block_count > 0)
    {
        fused_dot_kernel<<<block_count, block_dim, 0, stream>>>(
            x.cviewer(), y.cviewer(), d_result.viewer(), n);
    }
}

// R7: d_result = x^T * y, and the last block of the reduction also performs
// the scalar update that used to be its own <<<1, 1>>> graph node.
void fused_dot_scalar(cuda_tool::CDenseVectorView<Float> x,
                      cuda_tool::CDenseVectorView<Float> y,
                      cuda_tool::VarView<Float>          d_result,
                      unsigned int*                      d_ticket,
                      cuda_tool::VarView<Float>          d_rz,
                      cuda_tool::VarView<Float>          d_beta,
                      cuda_tool::VarView<IndexT>         d_converged,
                      cuda_tool::CVarView<Float>         d_rz_tol,
                      cuda_tool::VarView<Float>          d_pAp,
                      bool                               verify,
                      Float*                             v_rz,
                      Float*                             v_beta,
                      IndexT*                            v_converged,
                      Float*                             v_pAp,
                      cudaStream_t                       stream = nullptr)
{
    constexpr int block_dim   = 256;
    int           n           = x.size();
    int           block_count = (n + block_dim - 1) / block_dim;

    if(block_count > 0)
    {
        switch(pcg_small_env().dot_fence)
        {
            case 0:
                fused_dot_scalar_kernel<0><<<block_count, block_dim, 0, stream>>>(
                    x.cviewer(), y.cviewer(), d_result.viewer(), n, d_ticket,
                    d_rz.viewer(), d_beta.viewer(), d_converged.viewer(),
                    d_rz_tol.cviewer(), d_pAp.viewer(),
                    verify, v_rz, v_beta, v_converged, v_pAp);
                break;
            case 2:
                fused_dot_scalar_kernel<2><<<block_count, block_dim, 0, stream>>>(
                    x.cviewer(), y.cviewer(), d_result.viewer(), n, d_ticket,
                    d_rz.viewer(), d_beta.viewer(), d_converged.viewer(),
                    d_rz_tol.cviewer(), d_pAp.viewer(),
                    verify, v_rz, v_beta, v_converged, v_pAp);
                break;
            default:
                fused_dot_scalar_kernel<1><<<block_count, block_dim, 0, stream>>>(
                    x.cviewer(), y.cviewer(), d_result.viewer(), n, d_ticket,
                    d_rz.viewer(), d_beta.viewer(), d_converged.viewer(),
                    d_rz_tol.cviewer(), d_pAp.viewer(),
                    verify, v_rz, v_beta, v_converged, v_pAp);
                break;
        }
    }
}

// Same as linear_pcg update_xr: alpha = rz/pAp, x += alpha*p, r -= alpha*Ap. Alpha computed on device from d_rz, d_pAp.
void fused_update_xr(cuda_tool::CVarView<Float>         d_rz,
                     cuda_tool::CVarView<Float>         d_pAp,
                     cuda_tool::CVarView<IndexT>        d_converged,
                     cuda_tool::DenseVectorView<Float>  x,
                     cuda_tool::CDenseVectorView<Float> p,
                     cuda_tool::DenseVectorView<Float>  r,
                     cuda_tool::DenseVectorView<Float>  Ap,
                     cuda_tool::VarView<Float>          d_rz_new,
                     bool                               reset_rz_new,
                     bool                               zero_Ap,
                     cuda_tool::VarView<Float>          d_rz_prev,
                     bool                               save_rz_prev,
                     cudaStream_t                       stream = nullptr)
{
    int n = r.size();
    if(n > 0)
    {
        int bd = pcg_small_env().block;
        if(bd <= 0)
            bd = cuda_tool::best_block_dim(fused_update_xr_kernel);
        int gd = (n + bd - 1) / bd;
        fused_update_xr_kernel<<<gd, bd, 0, stream>>>(d_rz.cviewer(),
                                                      d_pAp.cviewer(),
                                                      d_converged.cviewer(),
                                                      x.viewer(),
                                                      p.cviewer(),
                                                      r.viewer(),
                                                      Ap.viewer(),
                                                      d_rz_new.viewer(),
                                                      reset_rz_new,
                                                      zero_Ap,
                                                      d_rz_prev.viewer(),
                                                      save_rz_prev,
                                                      n);
    }
}

// Same as linear_pcg update_p: beta = rz_new/rz, p = z + beta*p.
// Convergence is guarded by d_converged.
void fused_update_p(cuda_tool::CVarView<Float>         d_rz_new,
                    cuda_tool::CVarView<Float>         d_rz,
                    cuda_tool::CVarView<IndexT>        d_converged,
                    cuda_tool::DenseVectorView<Float>  p,
                    cuda_tool::CDenseVectorView<Float> z,
                    cuda_tool::DenseVectorView<Float>  Ap,
                    bool                               zero_Ap,
                    cudaStream_t                       stream = nullptr)
{
    int n = p.size();
    if(n > 0)
    {
        int bd = pcg_small_env().block;
        if(bd <= 0)
            bd = cuda_tool::best_block_dim(fused_update_p_kernel);
        int gd = (n + bd - 1) / bd;
        fused_update_p_kernel<<<gd, bd, 0, stream>>>(d_rz_new.cviewer(),
                                                     d_rz.cviewer(),
                                                     d_converged.cviewer(),
                                                     p.viewer(),
                                                     z.cviewer(),
                                                     Ap.viewer(),
                                                     zero_Ap,
                                                     n);
    }
}

// s11: p = z + beta * p with beta from the fused scalar kernel.
void fused_update_p_beta(cuda_tool::CVarView<Float>         d_beta,
                         cuda_tool::CVarView<IndexT>        d_converged,
                         cuda_tool::DenseVectorView<Float>  p,
                         cuda_tool::CDenseVectorView<Float> z,
                         cuda_tool::DenseVectorView<Float>  Ap,
                         bool                               zero_Ap,
                         cudaStream_t                       stream = nullptr)
{
    int n = p.size();
    if(n > 0)
    {
        int bd = pcg_small_env().block;
        if(bd <= 0)
            bd = cuda_tool::best_block_dim(fused_update_p_beta_kernel);
        int gd = (n + bd - 1) / bd;
        fused_update_p_beta_kernel<<<gd, bd, 0, stream>>>(d_beta.cviewer(),
                                                          d_converged.cviewer(),
                                                          p.viewer(),
                                                          z.cviewer(),
                                                          Ap.viewer(),
                                                          zero_Ap,
                                                          n);
    }
}

// R7 fold mode 1: p = z + beta*p with the scalar node folded in.
void fused_update_p_scalar(cuda_tool::CVarView<Float>         d_rz_new,
                           cuda_tool::CVarView<Float>         d_rz_prev,
                           cuda_tool::CVarView<Float>         d_rz_tol,
                           cuda_tool::VarView<Float>          d_rz,
                           cuda_tool::VarView<IndexT>         d_converged,
                           cuda_tool::VarView<Float>          d_pAp,
                           cuda_tool::DenseVectorView<Float>  p,
                           cuda_tool::CDenseVectorView<Float> z,
                           cuda_tool::DenseVectorView<Float>  Ap,
                           bool                               zero_Ap,
                           cudaStream_t                       stream = nullptr)
{
    int n = p.size();
    if(n > 0)
    {
        int bd = pcg_small_env().block;
        if(bd <= 0)
            bd = cuda_tool::best_block_dim(fused_update_p_scalar_kernel);
        int gd = (n + bd - 1) / bd;
        fused_update_p_scalar_kernel<<<gd, bd, 0, stream>>>(d_rz_new.cviewer(),
                                                            d_rz_prev.cviewer(),
                                                            d_rz_tol.cviewer(),
                                                            d_rz.viewer(),
                                                            d_converged.viewer(),
                                                            d_pAp.viewer(),
                                                            p.viewer(),
                                                            z.cviewer(),
                                                            Ap.viewer(),
                                                            zero_Ap,
                                                            n);
    }
}

// s11: converged + beta + rz swap + d_pAp reset in one single-thread node.
void fused_pcg_scalar(cuda_tool::CVarView<Float> d_rz_new,
                      cuda_tool::VarView<Float>   d_rz,
                      cuda_tool::VarView<Float>   d_beta,
                      cuda_tool::VarView<IndexT>  d_converged,
                      cuda_tool::CVarView<Float>  d_rz_tol,
                      cuda_tool::VarView<Float>   d_pAp,
                      cudaStream_t                stream = nullptr)
{
    fused_pcg_scalar_kernel<<<1, 1, 0, stream>>>(d_rz_new.cviewer(),
                                                 d_rz.viewer(),
                                                 d_beta.viewer(),
                                                 d_converged.viewer(),
                                                 d_rz_tol.cviewer(),
                                                 d_pAp.viewer());
}

// d_rz = d_rz_new when not converged (single-thread write).
void fused_swap_rz(cuda_tool::CVarView<Float>  d_rz_new,
                   cuda_tool::VarView<Float>   d_rz,
                   cuda_tool::CVarView<IndexT> d_converged,
                   cudaStream_t                stream = nullptr)
{
    // single-thread kernel (muda Launch() parity: 1 block x 1 thread)
    fused_swap_rz_kernel<<<1, 1, 0, stream>>>(
        d_rz_new.cviewer(), d_rz.viewer(), d_converged.cviewer());
}

// d_converged = |rz_new| <= rz_tol (single-thread write); rz_tol lives on
// device so a captured CUDA graph survives tolerance changes between solves.
void fused_update_converged(cuda_tool::CVarView<Float> d_rz_new,
                            cuda_tool::VarView<IndexT> d_converged,
                            cuda_tool::CVarView<Float> d_rz_tol,
                            cudaStream_t               stream = nullptr)
{
    int n = 1;
    fused_update_converged_kernel<<<cuda_tool::best_grid_dim(n, fused_update_converged_kernel), cuda_tool::best_block_dim(fused_update_converged_kernel), 0, stream>>>(
        d_rz_new.cviewer(), d_converged.viewer(), d_rz_tol.cviewer(), n);
}

// s13 probe: count the non-zero Ap entries where the fill node used to run.
void LinearFusedPCG::check_ap_zero(cudaStream_t stream)
{
    int n = (int)Ap.size();
    if(n <= 0 || m_ap_zero_acc.size() < 2)
        return;
    constexpr int bd = 256;
    int           gd = (n + bd - 1) / bd;
    pcg_ap_zero_check_kernel<<<gd, bd, 0, stream>>>(Ap.cview(), m_ap_zero_acc.data(), n);
}

void LinearFusedPCG::report_ap_zero()
{
    if(m_ap_zero_acc.size() < 2)
        return;
    std::array<unsigned long long, 2> h{};
    CUDA_TOOL_CHECK(cudaMemcpy(h.data(), m_ap_zero_acc.data(), sizeof(h), cudaMemcpyDeviceToHost));
    double worst = *reinterpret_cast<const double*>(&h[1]);
    logger::info("[PCG Ap-zero verify] frame {} newton {}: non-zero entries seen at SpMV entry = {} (max |Ap| = {:.6e})",
                 engine().frame(),
                 engine().newton_iter(),
                 h[0],
                 worst);
}

// One PCG iteration on `stream`; the unit of both graph capture and the
// uncaptured fallback. Kernels/arguments/order are identical either way.
// `timed` adds the per-iteration "SpMV"/"Apply Preconditioner" Timers —
// plain path only; during graph capture no Timer objects may be created
// (empirically corrupts state in the single-process test suite binary).
void LinearFusedPCG::run_iteration(cuda_tool::DenseVectorView<Float> x, cudaStream_t stream, bool timed)
{
    const bool fuse = pcg_small_env().fuse;
    // R7 (round 5): 0 = the pre-R7 chain and THE SHIPPED DEFAULT; 1 = folded into
    // update_p; 2 = folded into the dot tail. The fold was REJECTED -- 3 runs read
    // -1.41 % disjoint, 5 runs read -0.83 % overlapping (t = 1.39), the scope gate put
    // the truth at -0.23 %, and mas-bunny/case2 showed disjoint regressions. The code is
    // kept only so the RTX 5090 acceptance run can test it as a one-variable arm, where
    // the fold's per-wave cost is ~5x cheaper while the scalar node's ~0.93 us is fixed.
    int fold = fuse ? pcg_small_env().fold : 0;
    if(fold == 1)
    {
        // the grid gate (see PcgSmallEnv::fold). The decision depends only on
        // n, which is part of the graph validity key, so it is stable across
        // capture and replay.
        int bd = pcg_small_env().block;
        if(bd <= 0)
            bd = 256;
        const int gd  = ((int)p.size() + bd - 1) / bd;
        const int cap = pcg_small_env().fold_maxgrid >= 0 ?
                            pcg_small_env().fold_maxgrid :
                            cuda_tool::device_sm_count();
        if(cap > 0 && gd > cap)
            fold = 0;
    }

    // s13 probe: the fill node used to run here; check the invariant it kept.
    if(pcg_small_env().ap_zero && pcg_small_env().ap_zero_verify)
        check_ap_zero(stream);

    // Ap = A * p,  pAp = p^T * Ap
    {
        std::optional<Timer> timer;
        if(timed)
            timer.emplace("SpMV");
        spmv_dot(p.cview(), Ap.view(), d_pAp.view(), stream);
    }

    // alpha = rz / pAp,  x += alpha * p,  r -= alpha * Ap
    // (with the fusion on, thread 0 also zeroes the rz_new accumulator)
    fused_update_xr(d_rz.view(),
                    d_pAp.view(),
                    d_converged.view(),
                    x,
                    p.cview(),
                    r.view(),
                    Ap.view(),
                    d_rz_new.view(),
                    fuse,
                    pcg_small_env().ap_zero == 1,
                    d_rz_prev.view(),
                    fold == 1,
                    stream);

    // z = P^{-1} * r
    {
        std::optional<Timer> timer;
        if(timed)
            timer.emplace("Apply Preconditioner");
        apply_preconditioner(z, r, d_converged.view(), stream);
    }

    const bool fuse_dot   = (fold == 2);
    const bool dot_verify = fuse_dot && pcg_small_env().fuse_dot_verify;

    // rz_new = r^T * z, keep convergence flag on device for preconditioner skip.
    if(fuse_dot)
    {
        // R7: ... and the scalar update in the tail of the same kernel, so the
        // captured iteration has one node fewer.
        fused_dot_scalar(r.cview(),
                         z.cview(),
                         d_rz_new.view(),
                         m_dot_ticket.data(),
                         d_rz.view(),
                         d_beta.view(),
                         d_converged.view(),
                         d_rz_tol.view(),
                         d_pAp.view(),
                         dot_verify,
                         m_v_rz.data(),
                         m_v_beta.data(),
                         m_v_converged.data(),
                         m_v_pAp.data(),
                         stream);
    }
    else
    {
        fused_dot(r.cview(), z.cview(), d_rz_new.view(), stream, !fuse);
    }

    if(fuse)
    {
        // converged + beta + (rz <- rz_new) + d_pAp reset in one node
        if(fold == 0 || dot_verify)
            fused_pcg_scalar(d_rz_new.view(),
                             d_rz.view(),
                             d_beta.view(),
                             d_converged.view(),
                             d_rz_tol.view(),
                             d_pAp.view(),
                             stream);
        if(dot_verify)
            pcg_scalar_cmp_kernel<<<1, 1, 0, stream>>>(d_rz.data(),
                                                       d_beta.data(),
                                                       d_converged.data(),
                                                       d_pAp.data(),
                                                       m_v_rz.data(),
                                                       m_v_beta.data(),
                                                       m_v_converged.data(),
                                                       m_v_pAp.data(),
                                                       m_scalar_cmp_acc.data());
        if(fold == 1)
        {
            const bool zero_Ap = pcg_small_env().ap_zero == 2;
            if(!pcg_small_env().fold_verify)
            {
                fused_update_p_scalar(d_rz_new.view(),
                                      d_rz_prev.view(),
                                      d_rz_tol.view(),
                                      d_rz.view(),
                                      d_converged.view(),
                                      d_pAp.view(),
                                      p.view(),
                                      z.cview(),
                                      Ap.view(),
                                      zero_Ap,
                                      stream);
            }
            else
            {
                // reference = the pre-R7 chain, then restore the
                // read-modify-write outputs and run the folded kernel
                static cuda_tool::SpreadVerifier sv{"LinearFusedPCG::fold_update_p"};
                sv.begin_inout();
                sv.add_in_buffer(p.buffer_view());
                sv.add_in_buffer(Ap.buffer_view());
                sv.add_in(d_rz.data(), sizeof(Float));
                sv.add_in(d_converged.data(), sizeof(IndexT));
                sv.add_in(d_pAp.data(), sizeof(Float));
                sv.save_inputs();

                fused_pcg_scalar(d_rz_new.view(),
                                 d_rz.view(),
                                 d_beta.view(),
                                 d_converged.view(),
                                 d_rz_tol.view(),
                                 d_pAp.view(),
                                 stream);
                fused_update_p_beta(d_beta.view(),
                                    d_converged.view(),
                                    p.view(),
                                    z.cview(),
                                    Ap.view(),
                                    zero_Ap,
                                    stream);

                sv.begin();
                sv.add_buffer(p.buffer_view());
                sv.add_buffer(Ap.buffer_view());
                sv.add(d_rz.data(), sizeof(Float));
                sv.add(d_converged.data(), sizeof(IndexT));
                sv.add(d_pAp.data(), sizeof(Float));
                sv.snapshot();
                sv.restore_inputs();

                fused_update_p_scalar(d_rz_new.view(),
                                      d_rz_prev.view(),
                                      d_rz_tol.view(),
                                      d_rz.view(),
                                      d_converged.view(),
                                      d_pAp.view(),
                                      p.view(),
                                      z.cview(),
                                      Ap.view(),
                                      zero_Ap,
                                      stream);
                sv.compare();
            }
        }
        else
        {
            fused_update_p_beta(d_beta.view(),
                                d_converged.view(),
                                p.view(),
                                z.cview(),
                                Ap.view(),
                                pcg_small_env().ap_zero == 2,
                                stream);
        }
    }
    else
    {
        fused_update_converged(d_rz_new.view(), d_converged.view(), d_rz_tol.view(), stream);

        // p = z + beta * p (skip when abs(rz_new) <= rz_tol), then rz = rz_new.
        fused_update_p(d_rz_new.view(),
                       d_rz.view(),
                       d_converged.view(),
                       p.view(),
                       z.cview(),
                       Ap.view(),
                       pcg_small_env().ap_zero == 2,
                       stream);
        fused_swap_rz(d_rz_new.view(), d_rz.view(), d_converged.view(), stream);
    }
}

// ---------------------------------------------------------------------------
// CUDA graph block replay
// ---------------------------------------------------------------------------

void LinearFusedPCG::destroy_graph()
{
    m_graph.reset_graph();
    m_graph_n         = 0;
    m_graph_spmv_grid = -1;
}

#if CUDA_TOOL_GRAPH_WHILE
void LinearFusedPCG::destroy_while()
{
    m_while.reset_graph();
    m_while_n         = 0;
    m_while_spmv_grid = -1;
}

bool LinearFusedPCG::while_key_matches(cuda_tool::DenseVectorView<Float>  x,
                                       cuda_tool::CDenseVectorView<Float> b,
                                       SizeT max_iter) const
{
    if(!m_while.ready())
        return false;
    auto                        A    = matrix_data_ptrs();
    std::array<const void*, 12> ptrs = {x.data(),
                                        b.data(),
                                        r.buffer_view().data(),
                                        z.buffer_view().data(),
                                        p.buffer_view().data(),
                                        Ap.buffer_view().data(),
                                        A[0],
                                        A[1],
                                        A[2],
                                        d_rz.data(),
                                        d_rz_new.data(),
                                        d_pAp.data()};
    // round6 (s13): as in graph_key_matches -- the SpMV grid is baked in
    return m_while_n == x.size() && m_while_max_iter == max_iter
           && m_while_ptrs == ptrs && m_while_spmv_grid == spmv_grid_key();
}

void LinearFusedPCG::rebuild_while(cuda_tool::DenseVectorView<Float>  x,
                                   cuda_tool::CDenseVectorView<Float> b,
                                   SizeT                              max_iter)
{
    destroy_while();

    auto result = m_while.capture(
        // setup chain: reset -> r=b -> precond -> p=z -> rz = r^T z -> rz_tol
        [&](cudaStream_t stream, cudaGraphConditionalHandle handle)
        {
            pcg_while_reset_kernel<<<1, 1, 0, stream>>>(d_converged.viewer(),
                                                        d_iter.viewer(),
                                                        d_pAp.viewer());
            cuda_tool::BufferLaunch(stream).copy(r.buffer_view(), b.buffer_view());
            if(pcg_small_env().ap_zero)
            {
                // s13: the loop body leaves Ap zero for the next iteration's
                // SpMV; seed it once per launch in the setup chain.
                cuda_tool::BufferLaunch(stream).fill<Float>(Ap.buffer_view(), 0);
            }
            apply_preconditioner(z, r, d_converged.view(), stream);
            cuda_tool::BufferLaunch(stream).copy(p.buffer_view(), z.buffer_view());
            fused_dot(r.cview(), z.cview(), d_rz.view(), stream);
            pcg_while_setup_kernel<<<1, 1, 0, stream>>>(
                d_rz.cviewer(), d_rz_tol.viewer(), global_tol_rate, handle);
        },
        // loop body: one iteration + the keep-going decision
        [&](cudaStream_t stream, cudaGraphConditionalHandle handle)
        {
            run_iteration(x, stream, false);
            pcg_while_control_kernel<<<1, 1, 0, stream>>>(handle,
                                                          d_rz_new.cviewer(),
                                                          d_rz_tol.cviewer(),
                                                          d_converged.viewer(),
                                                          d_iter.viewer(),
                                                          (int)max_iter - 1);
        });

    if(result != cuda_tool::GraphWhile::Result::Ok)
    {
        logger::warn(
            "LinearFusedPCG: while-loop graph capture failed (code {}: {}); "
            "falling back to block replay / plain launches",
            (int)result,
            m_while.failure_detail());
        return;
    }

    auto A           = matrix_data_ptrs();
    m_while_ptrs     = {x.data(),
                        b.data(),
                        r.buffer_view().data(),
                        z.buffer_view().data(),
                        p.buffer_view().data(),
                        Ap.buffer_view().data(),
                        A[0],
                        A[1],
                        A[2],
                        d_rz.data(),
                        d_rz_new.data(),
                        d_pAp.data()};
    m_while_n         = x.size();
    m_while_max_iter  = max_iter;
    m_while_spmv_grid = spmv_grid_key();
    logger::info("LinearFusedPCG: captured full-GPU while-loop graph (n = {})", x.size());
}
#endif

bool LinearFusedPCG::graph_key_matches(cuda_tool::DenseVectorView<Float>  x,
                                       cuda_tool::CDenseVectorView<Float> b,
                                       SizeT interval,
                                       SizeT max_iter) const
{
    if(!m_graph.ready())
        return false;
    auto                        A    = matrix_data_ptrs();
    std::array<const void*, 12> ptrs = {x.data(),
                                        b.data(),
                                        r.buffer_view().data(),
                                        z.buffer_view().data(),
                                        p.buffer_view().data(),
                                        Ap.buffer_view().data(),
                                        A[0],
                                        A[1],
                                        A[2],
                                        d_rz.data(),
                                        d_rz_new.data(),
                                        d_pAp.data()};
    // round6 (s13): the SpMV grid is baked into the capture; a matrix that has
    // grown past what it was sized for must re-capture, not replay.
    return m_graph_n == x.size() && m_graph_interval == interval
           && m_graph_max_iter == max_iter && m_graph_ptrs == ptrs
           && m_graph_spmv_grid == spmv_grid_key();
}

void LinearFusedPCG::rebuild_graph(cuda_tool::DenseVectorView<Float>  x,
                                   cuda_tool::CDenseVectorView<Float> b,
                                   SizeT                              interval,
                                   SizeT                              max_iter)
{
    destroy_graph();

    // recorded, not executed; the block is launched for real right after
    auto result = m_graph.capture(
        [&](cudaStream_t capture_stream)
        {
            for(SizeT i = 0; i < interval; ++i)
                run_iteration(x, capture_stream, false);
        });

    if(result != cuda_tool::GraphCapture::Result::Ok)
    {
        // a callee launched outside the capture stream (e.g. the MAS
        // preconditioner engine) or the runtime rejected the capture
        logger::warn(
            "LinearFusedPCG: CUDA graph capture failed (code {}); "
            "graph replay disabled for this instance",
            (int)result);
        return;
    }

    logger::info("LinearFusedPCG: captured CUDA graph (interval = {}, n = {})",
                 interval,
                 x.size());

    auto A           = matrix_data_ptrs();
    m_graph_ptrs     = {x.data(),
                        b.data(),
                        r.buffer_view().data(),
                        z.buffer_view().data(),
                        p.buffer_view().data(),
                        Ap.buffer_view().data(),
                        A[0],
                        A[1],
                        A[2],
                        d_rz.data(),
                        d_rz_new.data(),
                        d_pAp.data()};
    m_graph_n         = x.size();
    m_graph_interval  = interval;
    m_graph_max_iter  = max_iter;
    m_graph_spmv_grid = spmv_grid_key();
}

SizeT LinearFusedPCG::fused_pcg(cuda_tool::DenseVectorView<Float>  x,
                                cuda_tool::CDenseVectorView<Float> b,
                                SizeT                              max_iter)
{
    Timer pcg_timer{"FusedPCG"};

#if CUDA_TOOL_GRAPH_WHILE
    if(m_graph_mode == 2)
    {
        if(!while_key_matches(x, b, max_iter))
            rebuild_while(x, b, max_iter);

        if(m_while.ready())
        {
            // one launch for the whole solve; zero D2H/H2D inside the loop
            CUDA_TOOL_CHECK(m_while.launch_sync());
            IndexT iters = d_iter;
            if(iters <= 0)
                return 0;
            IndexT converged = d_converged;
            return converged ? (SizeT)iters : max_iter;
        }
        // capture failed: fall through to block replay / plain launches
    }
#endif

    d_converged = 0;

    // r = b - A*x, but x0 = 0 so r = b
    r.buffer_view().copy_from(b.buffer_view());

    // z = P^{-1} * r
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z, r, d_converged.view());
    }

    // p = z
    p = z;

    // rz = r^T * z
    fused_dot(r.cview(), z.cview(), d_rz.view());
    Float rz_host = d_rz;
    check_init_rz_nan_inf(rz_host);
    Float abs_rz0 = std::abs(rz_host);

    if(abs_rz0 == Float{0.0})
        return 0;

    Float rz_tol = global_tol_rate * abs_rz0;
    // synchronous upload: an async copy on the default stream would race with
    // the graph launch stream (blocking streams do not wait for
    // legacy-stream work), letting the converged kernel read a stale/uninit
    // tolerance. (Symptom was dx=0 -> flat line-search energy.)
    CUDA_TOOL_CHECK(cudaMemcpy(d_rz_tol.data(), &rz_tol, sizeof(Float), cudaMemcpyHostToDevice));
    if(pcg_small_env().fuse)
    {
        // s11: the loop body's scalar kernel resets d_pAp for the *next*
        // iteration, so seed it once per solve here (blocking, like the
        // tolerance upload above, to stay ordered against the graph stream).
        CUDA_TOOL_CHECK(cudaMemset(d_pAp.data(), 0, sizeof(Float)));
    }
    if(pcg_small_env().ap_zero)
    {
        // s13: fused_update_xr leaves Ap zero for the *next* SpMV, so the
        // vector is seeded once per solve here (same blocking-memset
        // reasoning as the d_pAp seed above) instead of by a fill node inside
        // every captured iteration.
        CUDA_TOOL_CHECK(cudaMemset(Ap.buffer_view().data(), 0, Ap.size() * sizeof(Float)));
    }
    SizeT effective_check_interval = check_interval > 0 ? check_interval : SizeT{1};

    SizeT total_iters = max_iter > 0 ? max_iter - 1 : 0;
    SizeT iter_done   = 0;
    bool  converged   = false;

    while(true)
    {
        SizeT block = std::min(effective_check_interval, total_iters - iter_done);

        bool graph_block = m_use_cuda_graph && !m_graph.disabled()
                           && block == effective_check_interval;
        if(graph_block)
        {
            if(!graph_key_matches(x, b, effective_check_interval, max_iter))
                rebuild_graph(x, b, effective_check_interval, max_iter);

            if(m_graph.ready())
            {
                // replay; the blocking D2H read of d_rz_new below orders after
                // the graph (blocking launch stream), no explicit wait needed
                m_graph.launch_async();
            }
            else  // capture failed: plain path
            {
                for(SizeT i = 0; i < block; ++i)
                    run_iteration(x, nullptr, true);
            }
        }
        else  // tail block / graph disabled
        {
            for(SizeT i = 0; i < block; ++i)
                run_iteration(x, nullptr, true);
        }
        iter_done += block;

        // host convergence check, same cadence as the plain loop
        Float rz_new_host = d_rz_new;
        check_iter_rz_nan_inf(rz_new_host, iter_done);
        if((std::abs(rz_new_host) / abs_rz0) <= global_tol_rate)
        {
            converged = true;
            break;
        }
        if(iter_done >= total_iters)
            break;
    }

    return converged ? iter_done : max_iter;
}
}  // namespace uipc::backend::cuda
