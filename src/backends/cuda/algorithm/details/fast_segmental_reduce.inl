#include <cub/warp/warp_reduce.cuh>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace uipc::backend::cuda_tool
{
namespace details::fast_segmental_reduce
{
    __host__ __device__ constexpr int b2i(bool b)
    {
        return b ? 1 : 0;
    }

    // perf/round5 (s32): with SegOutInit::CrossWarpOnly the output fill covers
    // only the slots the reduce kernel atomically accumulates into instead of
    // the whole buffer. UIPC_SEG_NARROW_FILL=0 restores the full fill (the
    // A/B switch and the rollback).
    inline bool narrow_fill_enabled()
    {
        static const bool on = []
        {
            const char* e = std::getenv("UIPC_SEG_NARROW_FILL");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    // UIPC_SEG_FILL_POISON=1: poison the whole output with a signalling NaN
    // before the narrow fill runs, so that any slot the narrow fill failed to
    // cover -- a cross-warp segment it did not detect, or a slot no key maps to
    // -- shows up as NaN in the result instead of as a plausible number.
    // Diagnostic only; it makes the narrow path strictly more expensive.
    inline int fill_poison_level()
    {
        static const int lvl = []
        {
            const char* e = std::getenv("UIPC_SEG_FILL_POISON");
            return e ? std::atoi(e) : 0;
        }();
        return lvl;
    }

    inline bool fill_poison_enabled()
    {
        return fill_poison_level() > 0;
    }

    // UIPC_SEG_FILL_POISON=2 additionally counts, after the reduce, how many
    // output slots still hold the poison -- i.e. how many slots the narrow fill
    // did not cover and the kernel never wrote. That is the rig validation:
    // a narrow fill that is complete must report zero.
    inline bool fill_poison_check()
    {
        return fill_poison_level() >= 2;
    }

    template <typename T>
    inline T poison_value()
    {
        if constexpr(std::is_floating_point_v<T>)
            return std::numeric_limits<T>::signaling_NaN();
        else
            return std::numeric_limits<T>::max();
    }

    // perf/round6 (s17): the matrix reduce's warp tree stops at the first
    // level no lane of the warp needs. UIPC_SEG_REDUCE2=0 launches the
    // round-4 (s10) kernel, whose SASS is untouched -- the A/B arm and the
    // rollback.
    inline bool reduce2_enabled()
    {
        static const bool on = []
        {
            const char* e = std::getenv("UIPC_SEG_REDUCE2");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    // perf/round7 (s15): the K-serial re-summation. The k2 warp tree is
    // FP64-issue-bound, not latency-bound: `if(active) v = op(other, v)'
    // compiles to an UNPREDICATED warp-wide DADD plus an FSEL guard, so every
    // executed level costs MN DADD issues on a 1/32-rate pipe no matter how
    // many lanes' results any consumer ever reads (a 32-long segment issues
    // 45 warp-DADDs for 3x3 where 9 would carry the same information). The
    // ks kernel gives each thread K contiguous elements, sums them in
    // registers first (balanced-4 within the window, so the association keeps
    // the k2 tree's error class) -- every lane of every warp-wide DADD
    // usefully active -- and runs the same s17 early-exit tree over the 32
    // per-window partials, cutting the warp-DADD count per element by
    // ~2.1-2.5x at K = 4. This changes the summation order (balanced-4 within
    // a window, pairwise between windows, vs pure pairwise): rounding-level,
    // not bit-identity -- the standalone verifier is the proof instrument.
    // Scoped to the wide blocks (M*N >= 9) by measurement: the window scan's
    // select/move overhead outgrows the FP64-issue saving at 3 entries (the
    // 3x1 doublet class measured +28 %/launch), while the 3x3 classes measure
    // -8.7 % (the scan phase also deletes k2's I2F/DADD/F2I flags reduce and
    // two thirds of its key loads: the ks gather floor is 3.5x below k2's).
    // UIPC_SEGRED_TREE=0 launches the k2 kernel everywhere, whose SASS is
    // main's -- the A/B arm and the rollback.
    inline bool serial_tree_enabled()
    {
        static const bool on = []
        {
            const char* e = std::getenv("UIPC_SEGRED_TREE");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    // elements per thread in the ks kernel (see serial_tree_enabled). A
    // compile-time constant: the window logic is unrolled on it and the
    // launch geometry must not pay a stack frame for it.
    constexpr int kSerialElems = 4;

    // UIPC_SEG_VERIFY=1: after the production launch, re-run the old kernel
    // into a scratch copy and count mismatching 64-bit words, split by whether
    // the slot's segment spans three or more warps (the only slots whose
    // result depends on atomic arrival order in BOTH kernels). Diagnostic.
    inline bool verify_enabled()
    {
        static const bool on = []
        {
            const char* e = std::getenv("UIPC_SEG_VERIFY");
            return e && e[0] == '1';
        }();
        return on;
    }

    // UIPC_SEG_PROBE=1: after the production launch, time two stubs of the
    // new kernel into scratch nothing reads -- Probe=1 no warp tree (gather +
    // store floor), Probe=2 no gather (tree + store floor). Timing only.
    inline bool probe_enabled()
    {
        static const bool on = []
        {
            const char* e = std::getenv("UIPC_SEG_PROBE");
            return e && e[0] == '1';
        }();
        return on;
    }

    // UIPC_SEG_HIST=k: every k-th call, histogram the warps of the launch by
    // the number of tree levels they execute (0..5 = ceil(log2(longest
    // in-warp run))) and print it. Diagnostic.
    inline int hist_every()
    {
        static const int k = []
        {
            const char* e = std::getenv("UIPC_SEG_HIST");
            return e ? std::atoi(e) : 0;
        }();
        return k;
    }
}  // namespace details::fast_segmental_reduce

namespace
{
#ifdef KS_DEBUG
    __device__ double ks_dbg[2048 * 8];  // per-window debug records (s15 bring-up)
#endif

    // get-key functor over a segment-offset buffer (replaces the device lambda)
    struct fast_segmental_reduce_get_offset_key_op
    {
        CBufferView<int> offset;
        __device__ int   operator()(int i) const { return offset(i); }
    };

    // get-value functor over an input buffer (replaces the device lambda)
    template <typename T>
    struct fast_segmental_reduce_get_buffer_value_op
    {
        CBufferView<T> in;
        __device__ T   operator()(int i) const { return in(i); }
    };

    // FastSegmentalReduce::reduce (scalar values)
    template <int BlockSize, int WarpSize, typename T, typename FlagsT, typename GetKeyOp, typename GetValueOp, typename ReduceOp>
    __global__ void fast_segmental_reduce_scalar_kernel(BufferView<T> out,
                                                        size_t        in_size,
                                                        GetKeyOp   get_key_op,
                                                        GetValueOp get_value_op,
                                                        ReduceOp   op)
    {
        using namespace details::fast_segmental_reduce;
        using ValueT             = T;
        using Flags              = FlagsT;
        constexpr int warp_size  = WarpSize;
        constexpr int warp_count = BlockSize / WarpSize;

        using WarpReduceInt = cub::WarpReduce<int, warp_size>;
        using WarpReduceT   = cub::WarpReduce<T, warp_size>;

        __shared__ union
        {
            typename WarpReduceInt::TempStorage index_storage[warp_count];
            typename WarpReduceT::TempStorage   t_storage[warp_count];
        };

        auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
        auto thread_id_in_block = threadIdx.x;
        auto warp_id            = thread_id_in_block / warp_size;
        auto lane_id            = thread_id_in_block & (warp_size - 1);

        int    prev_i = -1;
        int    next_i = -1;
        int    i      = -1;
        Flags  flags;
        ValueT value;
        flags.is_cross_warp = 0;

        if(global_thread_id > 0 && global_thread_id < in_size)
        {
            prev_i = get_key_op(global_thread_id - 1);
        }

        if(global_thread_id < in_size - 1)
        {
            next_i = get_key_op(global_thread_id + 1);
        }

        if(global_thread_id < in_size)
        {
            i              = get_key_op(global_thread_id);
            value          = get_value_op(global_thread_id);
            flags.is_valid = 1;
        }
        else
        {
            i                   = -1;
            value               = ValueT{0};
            flags.is_valid      = 0;
            flags.is_cross_warp = 0;
        }

        if(lane_id == 0)
        {
            flags.is_head       = 1;
            flags.is_cross_warp = b2i(prev_i == i);
        }
        else
        {
            flags.is_head = b2i(prev_i != i);

            if(lane_id == warp_size - 1)
            {
                flags.is_cross_warp = b2i(next_i == i);
            }
        }

        flags.flags = WarpReduceInt(index_storage[warp_id])
                          .HeadSegmentedReduce(flags.flags, flags.is_head, op);

        value = WarpReduceT(t_storage[warp_id]).HeadSegmentedReduce(value, flags.is_head, op);


        if(flags.is_head && flags.is_valid)
        {
            if(flags.is_cross_warp)
            {
                auto& out_value = out(i);
                atomic_add(&out_value, value);
            }
            else
            {
                out(i) = value;
            }
        }
    }

    // FastSegmentalReduce::reduce (Eigen::Matrix values)
    template <int BlockSize, int WarpSize, typename T, int M, int N, typename FlagsT, typename GetKeyOp, typename GetValueOp, typename ReduceOp>
    __global__ void fast_segmental_reduce_matrix_kernel(BufferView<Eigen::Matrix<T, M, N>> out,
                                                        size_t     in_size,
                                                        GetKeyOp   get_key_op,
                                                        GetValueOp get_value_op,
                                                        ReduceOp   op)
    {
        using namespace details::fast_segmental_reduce;
        using Matrix             = Eigen::Matrix<T, M, N>;
        using Flags              = FlagsT;
        constexpr int warp_size  = WarpSize;
        constexpr int warp_count = BlockSize / WarpSize;

        using WarpReduceInt = cub::WarpReduce<int, warp_size>;
        using WarpReduceT   = cub::WarpReduce<T, warp_size>;

        __shared__ union
        {
            typename WarpReduceInt::TempStorage index_storage[warp_count];
            typename WarpReduceT::TempStorage   t_storage[warp_count];
        };

        auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
        auto thread_id_in_block = threadIdx.x;
        auto warp_id            = thread_id_in_block / warp_size;
        auto lane_id            = thread_id_in_block & (warp_size - 1);

        int    prev_i = -1;
        int    next_i = -1;
        int    i      = -1;
        Flags  flags;
        Matrix value;
        flags.is_cross_warp = 0;

        if(global_thread_id > 0 && global_thread_id < in_size)
        {
            prev_i = get_key_op(global_thread_id - 1);
        }

        if(global_thread_id < in_size - 1)
        {
            next_i = get_key_op(global_thread_id + 1);
        }

        if(global_thread_id < in_size)
        {
            i              = get_key_op(global_thread_id);
            value          = get_value_op(global_thread_id);
            flags.is_valid = 1;
        }
        else
        {
            i = -1;
            value.setZero();
            flags.is_valid      = 0;
            flags.is_cross_warp = 0;
        }

        if(lane_id == 0)
        {
            flags.is_head       = 1;
            flags.is_cross_warp = b2i(prev_i == i);
        }
        else
        {
            flags.is_head = b2i(prev_i != i);

            if(lane_id == warp_size - 1)
            {
                flags.is_cross_warp = b2i(next_i == i);
            }
        }

        flags.flags = WarpReduceInt(index_storage[warp_id])
                          .HeadSegmentedReduce(flags.flags, flags.is_head, op);

        for(int j = 0; j < M; j++)
        {
            for(int k = 0; k < N; k++)
            {
                value(j, k) = WarpReduceT(t_storage[warp_id])
                                  .HeadSegmentedReduce(value(j, k), flags.is_head, op);
            }
        }

        if(flags.is_head && flags.is_valid)
        {
            if(flags.is_cross_warp)
            {
                auto& out_value = out(i);
                eigen::atomic_add(out_value, value);
            }
            else
            {
                out(i) = value;
            }
        }
    }

    // perf/round6 (s17): the same reduce with a warp tree that stops early.
    //
    // cub's HeadSegmentedReduce runs five shuffle levels (offsets 1, 2, 4, 8,
    // 16) and at each level every lane executes `if(lane + offset <= last_lane)
    // v = op(other, v)`: the add is predicated per lane, so a level at which
    // NO lane of the warp qualifies still costs the warp one FP64 instruction
    // per matrix entry -- 9 x 16 clocks on a 1/32-rate part -- for nothing.
    // The predicate is monotone in the offset (a lane that fails at 2^d fails
    // at 2^(d+1)), so a warp may stop at the first level where `__any_sync`
    // of the predicate is false, and every lane's sequence of executed adds is
    // exactly the one cub would have executed: same operands, same order,
    // same `add.f64` -- bit-identical to the old kernel for every slot the old
    // kernel stores, and the same multiset of atomic operands for the slots it
    // accumulates. `last_lane` is computed the way cub's SegmentedReduce does
    // (ballot of the head flags, shifted to tail flags, masked to lanes >= the
    // caller, the last lane of the warp forced in). The tail lanes past
    // `in_size` take i = -1 and a zero value, as in the old kernel.
    //
    // Probe: 0 = production; 1 = no tree (gather + store only); 2 = no gather
    // (value synthesised from the key). Probes are timing stubs launched into
    // scratch under UIPC_SEG_PROBE.
    template <int BlockSize, int WarpSize, typename T, int M, int N, typename FlagsT, typename GetKeyOp, typename GetValueOp, typename ReduceOp, int Probe>
    __global__ void fast_segmental_reduce_matrix_k2_kernel(BufferView<Eigen::Matrix<T, M, N>> out,
                                                           size_t   in_size,
                                                           GetKeyOp get_key_op,
                                                           GetValueOp get_value_op,
                                                           ReduceOp op)
    {
        using namespace details::fast_segmental_reduce;
        using Matrix             = Eigen::Matrix<T, M, N>;
        using Flags              = FlagsT;
        constexpr int warp_size  = WarpSize;
        constexpr int warp_count = BlockSize / WarpSize;
        static_assert(warp_size == 32, "the k2 tree assumes a full physical warp");

        using WarpReduceInt = cub::WarpReduce<int, warp_size>;

        __shared__ typename WarpReduceInt::TempStorage index_storage[warp_count];

        auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
        auto thread_id_in_block = threadIdx.x;
        auto warp_id            = thread_id_in_block / warp_size;
        auto lane_id            = thread_id_in_block & (warp_size - 1);

        int    prev_i = -1;
        int    next_i = -1;
        int    i      = -1;
        Flags  flags;
        Matrix value;
        flags.is_cross_warp = 0;

        if(global_thread_id > 0 && global_thread_id < in_size)
        {
            prev_i = get_key_op(global_thread_id - 1);
        }

        if(global_thread_id < in_size - 1)
        {
            next_i = get_key_op(global_thread_id + 1);
        }

        if(global_thread_id < in_size)
        {
            i = get_key_op(global_thread_id);
            if constexpr(Probe == 2)
                value.setConstant(static_cast<T>(i));
            else
                value = get_value_op(global_thread_id);
            flags.is_valid = 1;
        }
        else
        {
            i = -1;
            value.setZero();
            flags.is_valid      = 0;
            flags.is_cross_warp = 0;
        }

        if(lane_id == 0)
        {
            flags.is_head       = 1;
            flags.is_cross_warp = b2i(prev_i == i);
        }
        else
        {
            flags.is_head = b2i(prev_i != i);

            if(lane_id == warp_size - 1)
            {
                flags.is_cross_warp = b2i(next_i == i);
            }
        }

        const unsigned full_mask = 0xffffffffu;

        // cub SegmentedReduce<HEAD_SEGMENTED=true>: last lane of this lane's segment
        unsigned warp_flags = __ballot_sync(full_mask, flags.is_head != 0);
        warp_flags >>= 1;                     // head flags -> tail flags
        warp_flags &= (~0u << lane_id);       // lanes >= this one
        warp_flags |= 1u << (warp_size - 1);  // the warp's last lane
        const int last_lane = __clz(__brev(warp_flags));

        flags.flags = WarpReduceInt(index_storage[warp_id])
                          .HeadSegmentedReduce(flags.flags, flags.is_head, op);

        if constexpr(Probe != 1)
        {
#pragma unroll
            for(int step = 0; step < 5; ++step)
            {
                const int  offset = 1 << step;
                const bool active = (lane_id + offset) <= last_lane;
                if(!__any_sync(full_mask, active))
                    break;  // warp-uniform: no lane needs this level or any above it
#pragma unroll
                for(int j = 0; j < M; j++)
                {
#pragma unroll
                    for(int k = 0; k < N; k++)
                    {
                        T other = __shfl_down_sync(full_mask, value(j, k), offset);
                        if(active)
                            value(j, k) = op(other, value(j, k));
                    }
                }
            }
        }

        if(flags.is_head && flags.is_valid)
        {
            if(flags.is_cross_warp)
            {
                auto& out_value = out(i);
                eigen::atomic_add(out_value, value);
            }
            else
            {
                out(i) = value;
            }
        }
    }

    // perf/round7 (s15): the K-serial matrix reduce. Same contract as the k2
    // kernel -- sorted-by-key input, dense key space when SegOutInit::
    // CrossWarpOnly is used -- but each thread owns K contiguous elements and
    // re-associates the sum (see serial_tree_enabled above for why). The
    // FP64-add *count* per segment is unchanged; what drops is the number of
    // warp-wide DADD *issues*, which is what a 1/32-rate FP64 pipe charges
    // for: the in-window phase keeps all 32 lanes of every warp-wide add on
    // useful work, and the shuffle tree then runs over 32 partials instead of
    // 32 single elements.
    //
    // Window algebra (why the tree sees one value per lane at all). For
    // window w (elements [wK, wK+K)) define
    //   carry_in[w]  = sum of the elements before the window's first segment
    //                  head (0 if the window starts with a head) -- belongs to
    //                  the segment that was running when the window started;
    //   main[w]      = sum of the elements from the window's LAST head to the
    //                  window end (0 if the window holds no head) -- belongs
    //                  to the segment running when the window ends;
    //   interior complete segments (a head and its end both inside the
    //   window) are stored directly: they lie inside one warp span, so they
    //   take the k2 store path, never the atomic one.
    // The segment that spans the w/w+1 window boundary owns both u[w]'s value
    // (it is the last segment of window w, by the definition of main) and
    // carry_in[w+1] (it is running when window w+1 starts) -- so folding
    // carry_in[w+1] into u[w] with one shuffle keeps every segment's data
    // inside exactly one tree range [h, next-head-lane - 1]:
    //   u[w] = (window has a head ? main[w] : carry_in[w]) + carry_in[w+1].
    //
    // The tree-range heads are the lanes whose window contains >= 1 head, and
    // the reduced segment starts at that window's LAST head (last_head_key).
    // Cross-warp behaviour is preserved exactly: only the segment owning the
    // span's last element can cross into the next span (segments are
    // contiguous), lane warp_size-1 sets the spans bit, and the integer flags
    // tree carries it to the range head like k2's cub flags reduce did --
    // including the ragged-tail case, where the last segment must take the
    // atomic path so the narrow fill's tail task still matches (see
    // fast_segmental_reduce_zero_cross_warp_kernel).
    //
    // NOTE on the UIPC_SEG_VERIFY probe with this kernel live: the probe
    // compares against the OLD kernel's summation, so it reports the
    // rounding-level re-association (mismatches outside the >= 3-span atomic
    // class are expected, bounded by the standalone verifier); it is no
    // longer a bit-identity proof. Same for the converter-level
    // UIPC_SEGRED_VERIFY / UIPC_DOUBLET_VERIFY probes.
    template <int BlockSize, int WarpSize, typename T, int M, int N, typename GetKeyOp, typename GetValueOp, typename ReduceOp, int K, int Probe = 0>
    __global__ void fast_segmental_reduce_matrix_ks_kernel(BufferView<Eigen::Matrix<T, M, N>> out,
                                                           size_t   in_size,
                                                           GetKeyOp get_key_op,
                                                           GetValueOp get_value_op,
                                                           ReduceOp op)
    {
        using namespace details::fast_segmental_reduce;
        using Matrix            = Eigen::Matrix<T, M, N>;
        constexpr int warp_size = WarpSize;
        static_assert(warp_size == 32, "the ks tree assumes a full physical warp");
        static_assert(K >= 1, "the ks window needs at least one element");

        auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        auto warp_id = (threadIdx.x / warp_size) * warp_size;  // first thread of my warp
        auto lane_id = threadIdx.x & (warp_size - 1);

        // this thread's window: elements [w0, w0+K)
        const size_t w0 = (size_t)global_thread_id * K;
        // my warp's span: elements [span0, span0 + warp_size*K)
        const size_t span0 = ((size_t)(blockIdx.x * blockDim.x) + warp_id) * K;

        // key of element w0-1 (the segment running when the window starts);
        // -2 = "before the input", so element 0 is always a head (k2's forced
        // is_head at lane 0, window form)
        int prev_key = -2;
        if(w0 > 0 && w0 <= in_size)
            prev_key = get_key_op((int)(w0 - 1));

        Matrix carry_in = Matrix::Zero();  // before the window's first head
        // the open run, accumulated on even/odd run-offsets so the closed
        // partial is a balanced-4 tree for a full window -- the same error
        // class as the k2 tree's pure pairwise association (a serial window
        // would lose ~10-20 % accuracy RMS on cancellation-heavy data).
        // Everything below assigns into named registers (no by-value matrix
        // temporaries: a returned-by-value Matrix costs a 288 B stack frame
        // in this kernel, i.e. local-memory traffic on the hot path).
        Matrix acc_e  = Matrix::Zero();
        Matrix acc_o  = Matrix::Zero();
        bool   have_e = false, have_o = false;
        bool par = false;  // accumulator parity for the NEXT element of the open run
        bool leading     = true;   // the open run is the running-in part
        bool has_head    = false;  // the window contains >= 1 head
        bool any_valid   = false;  // the window contains >= 1 element
        bool had_leading = false;  // a running-in run closed at the first head
        int first_key = -1;  // key of the window's first element (the leading run's key)
        int    last_head_key = -1;              // key of the window's last head
        Matrix closed        = Matrix::Zero();  // the just-closed run's partial

        // close the open run into `closed`
        auto close_run = [&]()
        {
            if(have_e && have_o)
            {
#pragma unroll
                for(int a = 0; a < M; ++a)
#pragma unroll
                    for(int b = 0; b < N; ++b)
                        closed(a, b) = op(acc_e(a, b), acc_o(a, b));
            }
            else if(have_e)
                closed = acc_e;
            else
                closed = acc_o;
        };

#pragma unroll
        for(int j = 0; j < K; ++j)
        {
            const size_t g = w0 + j;
            if(g >= in_size)
                break;
            const int  kj   = get_key_op((int)g);
            const bool head = (kj != prev_key);
            any_valid       = true;
            if(j == 0)
                first_key = kj;
            if(head)
            {
                if(have_e || have_o)  // close the previous run
                {
                    close_run();
                    if(leading)
                    {
                        carry_in    = closed;  // closes the running-in part
                        had_leading = true;
                    }
                    else
                        out(last_head_key) = closed;  // interior complete segment: inside the warp span
                }
                if constexpr(Probe == 2)
                    acc_e.setConstant(static_cast<T>(g));
                else
                    acc_e = get_value_op((int)g);  // run offset 0
                have_e   = true;
                have_o   = false;
                par      = true;  // the next element of this run is offset 1
                leading  = false;
                has_head = true;
                last_head_key = kj;
            }
            else
            {
                Matrix v;
                if constexpr(Probe == 2)
                    v.setConstant(static_cast<T>(g));
                else
                    v = get_value_op((int)g);
                if(par)
                {
                    if(have_o)
                    {
#pragma unroll
                        for(int a = 0; a < M; ++a)
#pragma unroll
                            for(int b = 0; b < N; ++b)
                                acc_o(a, b) = op(acc_o(a, b), v(a, b));
                    }
                    else
                        acc_o = v;
                    have_o = true;
                }
                else
                {
                    if(have_e)
                    {
#pragma unroll
                        for(int a = 0; a < M; ++a)
#pragma unroll
                            for(int b = 0; b < N; ++b)
                                acc_e(a, b) = op(acc_e(a, b), v(a, b));
                    }
                    else
                        acc_e = v;
                    have_e = true;
                }
                par = !par;
            }
            prev_key = kj;
        }
        // close the final run: if the window never saw a head it IS the
        // running-in part; otherwise the partial stays in acc_e/acc_o and is
        // picked up as `main` below
        Matrix main_part = Matrix::Zero();
        if(have_e || have_o)
        {
            close_run();
            if(leading)
                carry_in = closed;
            else
                main_part = closed;
        }

        Matrix u = Matrix::Zero();
        if(has_head)
            u = main_part;
        else
            u = carry_in;

        // The span's left edge, in window form of the per-element kernels'
        // forced is_head at lane 0:
        //  - window 0 holds no head: the whole span-so-far continues the
        //    segment that was running when the span started. Lane 0 becomes a
        //    pseudo tree-head so this span contributes its partial at all --
        //    atomically, because that segment began in an earlier span (a
        //    store would clobber the earlier span's contribution).
        //  - window 0 holds a head: the elements before it (carry_in, always
        //    empty at span0 == 0) belong to a segment headed in the previous
        //    span -- accumulate them directly, like the pseudo-head would.
        const bool pseudo_head = lane_id == 0 && !has_head && any_valid;
        if(pseudo_head)
            last_head_key = first_key;
        if(lane_id == 0 && has_head && had_leading)
        {
            // the leading run's segment is headed in an earlier span: this
            // span contributes its tail atomically (at span0 == 0 the leading
            // run cannot exist, so this never fires on the first span)
            auto& out_value = out(first_key);
            eigen::atomic_add(out_value, carry_in);
        }

        // Fold the NEXT lane's carry_in into this lane's u -- but only when
        // the next window is a tree-range boundary (it holds a head). Then
        // the next segment owns lane t+1's u while our segment's tail is the
        // next window's carry, which no range would otherwise reach. When the
        // next window holds no head, its u already IS that carry and sits
        // inside our range -- folding again would double-count. Lane
        // warp_size-1 takes nothing: its carry belongs to its own range.
        if constexpr(Probe != 1)
        {
            // every lane must issue the shuffle (full-mask contract); the guard
            // is on the USE, never on the shuffle itself
            const int next_has_head = __shfl_down_sync(0xffffffffu, (int)has_head, 1);
            const bool take = lane_id < warp_size - 1 && next_has_head != 0;
#pragma unroll
            for(int j = 0; j < M; j++)
            {
#pragma unroll
                for(int k = 0; k < N; k++)
                {
                    T nc = __shfl_down_sync(0xffffffffu, carry_in(j, k), 1);
                    if(take)
                        u(j, k) = op(u(j, k), nc);
                }
            }
        }

        // flags word reduced alongside the values (integer adds -- k2's cub
        // flags reduce went through plus<double>, i.e. I2F/DADD/F2I per level)
        const bool is_head_lane = has_head || pseudo_head;
        unsigned fl = (is_head_lane ? 1u : 0u) | (any_valid ? (1u << 16) : 0u);
        if(pseudo_head && span0 > 0)
            fl |= (1u << 8);  // the running-in segment began in an earlier span
        if(lane_id == warp_size - 1)
        {
            const size_t span_end = span0 + (size_t)warp_size * K;
            const size_t span_last = (in_size < span_end ? in_size : span_end) - 1;  // in_size > 0 whenever we launch
            bool spans;
            if(span_last + 1 < in_size)
                spans = get_key_op((int)(span_last + 1)) == get_key_op((int)span_last);
            else
                spans = span_end > in_size;  // the ragged-tail quirk: the old kernel's -1 == -1
            if(spans)
                fl |= (1u << 8);
        }

        // the early-exit tree over the 32 per-window partials (s17's design)
        unsigned warp_flags = __ballot_sync(0xffffffffu, is_head_lane);
        warp_flags >>= 1;
        warp_flags &= (~0u << lane_id);
        warp_flags |= 1u << (warp_size - 1);
        const int last_lane = __clz(__brev(warp_flags));

        if constexpr(Probe != 1)
#pragma unroll
            for(int step = 0; step < 5; ++step)
            {
                const int  offset = 1 << step;
                const bool active = (lane_id + offset) <= last_lane;
                if(!__any_sync(0xffffffffu, active))
                    break;
                const unsigned fother = __shfl_down_sync(0xffffffffu, fl, offset);
                if(active)
                    fl += fother;
#pragma unroll
                for(int j = 0; j < M; j++)
                {
#pragma unroll
                    for(int k = 0; k < N; k++)
                    {
                        T other = __shfl_down_sync(0xffffffffu, u(j, k), offset);
                        if(active)
                            u(j, k) = op(other, u(j, k));
                    }
                }
            }

        // fl is the SUM of the range's flag words. Its fields are spaced so
        // they cannot carry into each other (heads <= 1, spans <= 2, valid
        // <= 32 per range), but a shift-only test would alias the neighbouring
        // field -- mask, never shift.
        if(is_head_lane && (fl & 0xFFFF0000u) != 0)
        {
#ifdef KS_DEBUG
            if(global_thread_id < 2048)
            {
                ks_dbg[global_thread_id * 8 + 0] = (double)u(0, 0);
                ks_dbg[global_thread_id * 8 + 1] = (double)last_head_key;
                ks_dbg[global_thread_id * 8 + 2] = (double)fl;
                ks_dbg[global_thread_id * 8 + 3] = (double)last_lane;
                ks_dbg[global_thread_id * 8 + 4] = 1.0;  // stored
                ks_dbg[global_thread_id * 8 + 5] = (double)carry_in(0, 0);
                ks_dbg[global_thread_id * 8 + 6] = (double)span0;
                ks_dbg[global_thread_id * 8 + 7] =
                    (double)(in_size < span0 + (size_t)warp_size * K ?
                                 in_size :
                                 span0 + (size_t)warp_size * K);
            }
#endif
            if((fl & 0xFF00u) != 0)
            {
                auto& out_value = out(last_head_key);
                eigen::atomic_add(out_value, u);
            }
            else
            {
                out(last_head_key) = u;
            }
        }
#ifdef KS_DEBUG
        else if(global_thread_id < 2048)
        {
            ks_dbg[global_thread_id * 8 + 0] = (double)u(0, 0);
            ks_dbg[global_thread_id * 8 + 1] = (double)last_head_key;
            ks_dbg[global_thread_id * 8 + 2] = (double)fl;
            ks_dbg[global_thread_id * 8 + 3] = (double)last_lane;
            ks_dbg[global_thread_id * 8 + 4] = 0.0;  // NOT stored
            ks_dbg[global_thread_id * 8 + 5] = (double)carry_in(0, 0);
            ks_dbg[global_thread_id * 8 + 6] = (double)span0;
            ks_dbg[global_thread_id * 8 + 7] =
                (double)(in_size < span0 + (size_t)warp_size * K ?
                             in_size :
                             span0 + (size_t)warp_size * K);
        }
#endif
    }

    // diagnostic (UIPC_SEG_HIST): per-warp histogram of the tree levels the k2
    // kernel executes = ceil(log2(longest in-warp run of equal keys)).
    template <int WarpSize, typename GetKeyOp>
    __global__ void fast_segmental_reduce_level_hist_kernel(size_t   in_size,
                                                            GetKeyOp get_key_op,
                                                            unsigned int* hist)
    {
        auto g      = blockDim.x * blockIdx.x + threadIdx.x;
        int  lane   = threadIdx.x & (WarpSize - 1);
        int  prev_i = -1, i = -1;
        if(g > 0 && g < in_size)
            prev_i = get_key_op(g - 1);
        if(g < in_size)
            i = get_key_op(g);
        int      is_head = (lane == 0) ? 1 : (prev_i != i ? 1 : 0);
        unsigned wf      = __ballot_sync(0xffffffffu, is_head != 0);
        wf >>= 1;
        wf &= (~0u << lane);
        wf |= 1u << (WarpSize - 1);
        int run = __clz(__brev(wf)) - lane + 1;  // lanes this lane's segment still covers
        for(int o = 16; o > 0; o >>= 1)
            run = max(run, __shfl_xor_sync(0xffffffffu, run, o));
        if(lane == 0)
        {
            int levels = 0;
            while((1 << levels) < run)
                ++levels;
            atomicAdd(hist + levels, 1u);
        }
    }

    // diagnostic (UIPC_SEG_VERIFY): mark the slots whose segment spans >= 3
    // warps -- those accumulate >= 3 atomic operands in BOTH kernels and are
    // arrival-order dependent; every other slot must match bit for bit.
    // SpanElems is the reduce kernel's warp span in elements (32 for the
    // per-element kernels, 32*K for the s15 ks kernel).
    template <int WarpSize, typename GetKeyOp, int SpanElems = WarpSize>
    __global__ void fast_segmental_reduce_mark_multi_warp_kernel(
        size_t in_size, GetKeyOp get_key_op, unsigned char* mark, int n_boundary)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= n_boundary)
            return;
        size_t g = (size_t)(t + 1) * (size_t)SpanElems;
        if(g >= in_size)
            return;
        int key = get_key_op((int)g);
        if(get_key_op((int)(g - 1)) != key)
            return;  // no segment crosses this boundary
        // crosses this boundary: >= 3 warps iff it also crosses the next one.
        // (A segment that ends in a ragged tail has two atomic operands -- the
        // tail's zero lanes are inside the second warp's partial -- and a + b
        // is commutative, so it stays in the strict class.)
        size_t g2 = g + (size_t)SpanElems;
        if(g2 < in_size && get_key_op((int)(g2 - 1)) == key && get_key_op((int)g2) == key)
            mark[key] = 1;
    }

    // diagnostic (UIPC_SEG_FILL_POISON=2): count output slots still poisoned
    // after the reduce -- the slots the narrow fill did not cover and the
    // reduce kernel never wrote.
    template <typename S>
    __device__ inline bool fsr_bits_equal(const S& a, const S& b)
    {
        if constexpr(sizeof(S) == 8)
            return *(const unsigned long long*)&a == *(const unsigned long long*)&b;
        else if constexpr(sizeof(S) == 4)
            return *(const unsigned int*)&a == *(const unsigned int*)&b;
        else
            return a == b;
    }

    template <typename S, int M, int N>
    __global__ void fast_segmental_reduce_compare_kernel(
        CBufferView<Eigen::Matrix<S, M, N>> a,
        CBufferView<Eigen::Matrix<S, M, N>> b,
        const unsigned char*                mark,
        unsigned long long*                 stats)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (int)a.size())
            return;
        const S* pa = a(i).data();
        const S* pb = b(i).data();
        int      c  = 0;
        double   md = 0.0;
        for(int k = 0; k < M * N; ++k)
        {
            if(!fsr_bits_equal(pa[k], pb[k]))
            {
                ++c;
                double d = fabs((double)pa[k] - (double)pb[k]);
                double r = d / fmax(fabs((double)pa[k]), 1e-300);
                md       = fmax(md, r);
            }
        }
        if(c)
        {
            atomicAdd(stats + (mark[i] ? 1 : 0), (unsigned long long)c);
            atomicAdd(stats + 2, 1ull);
            atomicMax(stats + 3, (unsigned long long)__double_as_longlong(md));
        }
        if(mark[i])
            atomicAdd(stats + 4, 1ull);
    }


    template <typename S>
    __global__ void fast_segmental_reduce_poison_count_scalar_kernel(
        CBufferView<S> out, S poison, unsigned long long* stats)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (int)out.size())
            return;
        if(fsr_bits_equal(out(i), poison))
            atomicAdd(stats, 1ull);
    }

    template <typename S, int M, int N>
    __global__ void fast_segmental_reduce_poison_count_matrix_kernel(
        CBufferView<Eigen::Matrix<S, M, N>> out, S poison, unsigned long long* stats)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (int)out.size())
            return;
        const S* v = out(i).data();
        int      c = 0;
        for(int k = 0; k < M * N; ++k)
            if(fsr_bits_equal(v[k], poison))
                ++c;
        if(c == M * N)
        {
            atomicAdd(stats, 1ull);
            atomicMin(stats + 2, (unsigned long long)i);
            atomicMax(stats + 3, (unsigned long long)i);
        }
        else if(c != 0)
            atomicAdd(stats + 1, 1ull);
    }

    // Zero exactly the output slots that `fast_segmental_reduce_matrix_kernel`
    // reaches with `eigen::atomic_add` -- the heads of segments that cross a
    // warp boundary.
    //
    // The input is sorted by key, so a key's elements are contiguous: a segment
    // is either entirely inside one warp (its head lane *stores*, and the old
    // value is irrelevant) or it crosses at least one warp boundary (every warp
    // holding part of it accumulates atomically, and the slot must start at
    // zero). The two cases cannot mix on one slot. Warp boundaries in the
    // reduce kernel's grid are the global thread ids that are multiples of
    // WarpSize, because `lane_id` comes from `threadIdx.x` and `blockDim.x` is
    // a multiple of WarpSize -- so thread `t` here tests boundary
    // `(t + 1) * WarpSize` and zeroes the slot when the keys either side match.
    // A segment spanning three or more warps is found by several threads, which
    // write the same zero to the same slot.
    //
    // perf/round7 (s15): the ks kernel's warp spans are `SpanElems` elements
    // (32*K), not 32 -- pass the span of whichever reduce kernel is about to
    // launch. Extra zeros are harmless (a stored slot overwrites its zero);
    // a missing zero would be a wrong result, so the span must not be larger
    // than the reduce kernel's.
    //
    // The tail is a third case, and it is *not* obvious from the source. When
    // `in_size % WarpSize != 0` the last warp has lanes past the end of the
    // input; those lanes take `i = -1` and `prev_i = -1`, so their `is_head` is
    // 0 and they chain onto the last real segment -- and the one that lands on
    // lane `WarpSize - 1` evaluates `is_cross_warp = b2i(next_i == i)` as
    // `b2i(-1 == -1) = 1`. The last segment therefore takes the *atomic* path
    // even though nothing crosses a warp boundary, and its slot needs the zero
    // like any other accumulated slot. Task `n_boundary` covers it.
    // (Found by UIPC_SEG_FILL_POISON=2, which reported exactly one unwritten
    // slot -- the last -- on rigid-wrecking-balls. This also means the reduce
    // does one pointless atomic per call whenever in_size is not a multiple of
    // the warp size.)
    template <int WarpSize, typename T, typename GetKeyOp>
    __global__ void fast_segmental_reduce_zero_cross_warp_scalar_kernel(
        BufferView<T> out, size_t in_size, GetKeyOp get_key_op, int n_boundary)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t > n_boundary)
            return;

        if(t == n_boundary)  // the tail task
        {
            if(in_size % (size_t)WarpSize != 0)
                out(get_key_op((int)in_size - 1)) = T{0};
            return;
        }

        size_t g = (size_t)(t + 1) * (size_t)WarpSize;
        if(g >= in_size)
            return;

        int prev_key = get_key_op((int)(g - 1));
        int key      = get_key_op((int)g);
        if(prev_key == key)
            out(key) = T{0};
    }

    template <int WarpSize, typename T, int M, int N, typename GetKeyOp, int SpanElems = WarpSize>
    __global__ void fast_segmental_reduce_zero_cross_warp_kernel(
        BufferView<Eigen::Matrix<T, M, N>> out, size_t in_size, GetKeyOp get_key_op, int n_boundary)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t > n_boundary)
            return;

        if(t == n_boundary)  // the tail task
        {
            if(in_size % (size_t)SpanElems != 0)
                out(get_key_op((int)in_size - 1)).setZero();
            return;
        }

        size_t g = (size_t)(t + 1) * (size_t)SpanElems;
        if(g >= in_size)
            return;

        int prev_key = get_key_op((int)(g - 1));
        int key      = get_key_op((int)g);
        if(prev_key == key)
            out(key).setZero();
    }
}  // namespace

template <int BlockSize, int WarpSize>
template <typename T, typename GetKeyOp, typename GetValueOp, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    size_t in_size, BufferView<T> out, GetKeyOp get_key_op, GetValueOp get_value_op, ReduceOp op, SegOutInit init)
{
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");
    static_assert(BlockSize % WarpSize == 0,
                  "the narrow output fill assumes warp boundaries sit at global "
                  "thread ids that are multiples of WarpSize");

    using ValueT  = T;
    namespace fsr = details::fast_segmental_reduce;

    auto          size      = in_size;
    constexpr int block_dim = BlockSize;

    const bool narrow = (init == SegOutInit::CrossWarpOnly) && fsr::narrow_fill_enabled();

    if(!narrow)
    {
        BufferLaunch(this->stream()).fill<ValueT>(out, ValueT{0});
    }
    else
    {
        if(fsr::fill_poison_enabled())
            BufferLaunch(this->stream()).fill<ValueT>(out, fsr::poison_value<T>());

        // tasks 0..n_boundary-1 are the warp boundaries, task n_boundary is the tail
        int n_boundary = (int)((size + WarpSize - 1) / WarpSize) - 1;
        if(n_boundary < 0)
            n_boundary = 0;
        if(size > 0)
        {
            constexpr int fill_block = 256;
            int           n_task     = n_boundary + 1;
            fast_segmental_reduce_zero_cross_warp_scalar_kernel<WarpSize, T, GetKeyOp>
                <<<(n_task + fill_block - 1) / fill_block, fill_block, 0, this->stream()>>>(
                    out, size, get_key_op, n_boundary);
        }
    }

    const bool poison_check = narrow && fsr::fill_poison_check();

    int block_count = (size + block_dim - 1) / block_dim;
    if(block_count > 0)
        fast_segmental_reduce_scalar_kernel<BlockSize, WarpSize, T, Flags>
            <<<block_count, block_dim, 0, this->stream()>>>(out, size, get_key_op, get_value_op, op);

    if(poison_check && out.size() > 0)
    {
        static DeviceBuffer<unsigned long long> stats;
        stats.resize_discard(2);
        unsigned long long zero[2]{0, 0};
        cudaMemcpyAsync(stats.data(), zero, sizeof(zero), cudaMemcpyHostToDevice, this->stream());
        int n = (int)out.size();
        fast_segmental_reduce_poison_count_scalar_kernel<T>
            <<<(n + 255) / 256, 256, 0, this->stream()>>>(
                out.cview(), fsr::poison_value<T>(), stats.data());
        unsigned long long h[2]{};
        cudaMemcpyAsync(h, stats.data(), sizeof(h), cudaMemcpyDeviceToHost, this->stream());
        cudaStreamSynchronize(this->stream());
        std::fprintf(stderr,
                     "[seg-poison] scalar in=%zu out=%d unwritten=%llu\n",
                     (size_t)size,
                     n,
                     h[0]);
    }

    return *this;
}


template <int BlockSize, int WarpSize>
template <typename T, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    CBufferView<int> offset, CBufferView<T> in, BufferView<T> out, ReduceOp op, SegOutInit init)
{
    return reduce(in.size(),
                  out,
                  fast_segmental_reduce_get_offset_key_op{offset},
                  fast_segmental_reduce_get_buffer_value_op<T>{in},
                  op,
                  init);
}


template <int BlockSize, int WarpSize>
template <typename T, int M, int N, typename GetKeyOp, typename GetValueOp, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    size_t                             in_size,
    BufferView<Eigen::Matrix<T, M, N>> out,
    GetKeyOp                           get_key_op,
    GetValueOp                         get_value_op,
    ReduceOp                           op,
    SegOutInit                         init)
{
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");
    static_assert(BlockSize % WarpSize == 0,
                  "the narrow output fill assumes warp boundaries sit at global "
                  "thread ids that are multiples of WarpSize");

    using Matrix  = Eigen::Matrix<T, M, N>;
    namespace fsr = details::fast_segmental_reduce;

    auto          size      = in_size;
    constexpr int block_dim = BlockSize;

    // which reduce kernel runs (and so what a "warp boundary" is in element
    // space): the s15 ks kernel gives each thread fsr::kSerialElems elements,
    // so its spans are WarpSize * kSerialElems wide; k2 and the round-4
    // kernel are per-element, spans of WarpSize. Measured on crease-press
    // (s15 scope A/B + phase probes): the ks re-summation pays on the wide
    // blocks -- 3x3 -8.7 %/launch -- but REGRESSES on the narrow ones (3x1
    // +28 %: the per-element window scan's branchy select/move stream
    // outgrows the FP64-issue saving it buys at 3 entries), so it is scoped
    // to M*N >= 9 and the narrower classes keep the k2 kernel verbatim.
    constexpr bool ks_capable = WarpSize == 32 && M * N >= 9;
    const bool     use_ks =
        ks_capable && fsr::reduce2_enabled() && fsr::serial_tree_enabled();
    const int span = use_ks ? WarpSize * fsr::kSerialElems : WarpSize;

    const bool narrow = (init == SegOutInit::CrossWarpOnly) && fsr::narrow_fill_enabled();

    if(!narrow)
    {
        BufferLaunch(this->stream()).fill<Matrix>(out, Matrix::Zero().eval());
    }
    else
    {
        if(fsr::fill_poison_enabled())
            BufferLaunch(this->stream())
                .fill<Matrix>(out, Matrix::Constant(fsr::poison_value<T>()).eval());

        // tasks 0..n_boundary-1 are the warp-span boundaries, task n_boundary is the tail
        int n_boundary = (int)((size + span - 1) / span) - 1;
        if(n_boundary < 0)
            n_boundary = 0;
        if(size > 0)
        {
            constexpr int fill_block = 256;
            int           n_task     = n_boundary + 1;
            if(use_ks)
                fast_segmental_reduce_zero_cross_warp_kernel<WarpSize, T, M, N, GetKeyOp, WarpSize * fsr::kSerialElems>
                    <<<(n_task + fill_block - 1) / fill_block, fill_block, 0, this->stream()>>>(
                        out, size, get_key_op, n_boundary);
            else
                fast_segmental_reduce_zero_cross_warp_kernel<WarpSize, T, M, N, GetKeyOp>
                    <<<(n_task + fill_block - 1) / fill_block, fill_block, 0, this->stream()>>>(
                        out, size, get_key_op, n_boundary);
        }
    }

    const bool poison_check = narrow && fsr::fill_poison_check();

    int block_count =
        use_ks ? (int)(((size + fsr::kSerialElems - 1) / fsr::kSerialElems + block_dim - 1)
                       / block_dim) :
                 (int)((size + block_dim - 1) / block_dim);
    if(block_count > 0)
    {
        if constexpr(WarpSize == 32)
        {
            if constexpr(M * N >= 9)
            {
                if(fsr::reduce2_enabled() && fsr::serial_tree_enabled())
                    fast_segmental_reduce_matrix_ks_kernel<BlockSize, WarpSize, T, M, N, GetKeyOp, GetValueOp, ReduceOp, fsr::kSerialElems>
                        <<<block_count, block_dim, 0, this->stream()>>>(
                            out, size, get_key_op, get_value_op, op);
                else if(fsr::reduce2_enabled())
                    fast_segmental_reduce_matrix_k2_kernel<BlockSize, WarpSize, T, M, N, Flags, GetKeyOp, GetValueOp, ReduceOp, 0>
                        <<<block_count, block_dim, 0, this->stream()>>>(
                            out, size, get_key_op, get_value_op, op);
                else
                    fast_segmental_reduce_matrix_kernel<BlockSize, WarpSize, T, M, N, Flags>
                        <<<block_count, block_dim, 0, this->stream()>>>(
                            out, size, get_key_op, get_value_op, op);
            }
            else
            {
                if(fsr::reduce2_enabled())
                    fast_segmental_reduce_matrix_k2_kernel<BlockSize, WarpSize, T, M, N, Flags, GetKeyOp, GetValueOp, ReduceOp, 0>
                        <<<block_count, block_dim, 0, this->stream()>>>(
                            out, size, get_key_op, get_value_op, op);
                else
                    fast_segmental_reduce_matrix_kernel<BlockSize, WarpSize, T, M, N, Flags>
                        <<<block_count, block_dim, 0, this->stream()>>>(
                            out, size, get_key_op, get_value_op, op);
            }
        }
        else
        {
            fast_segmental_reduce_matrix_kernel<BlockSize, WarpSize, T, M, N, Flags>
                <<<block_count, block_dim, 0, this->stream()>>>(
                    out, size, get_key_op, get_value_op, op);
        }
    }

    if constexpr(WarpSize == 32)
    {
        // the diagnostics below are per-element kernels -- with the ks kernel
        // live `block_count` counts windows, so they launch their own grid
        const int elem_block_count = (int)((size + block_dim - 1) / block_dim);

        if(block_count > 0 && fsr::hist_every() > 0)
        {
            static int calls = 0;
            if(calls++ % fsr::hist_every() == 0)
            {
                static DeviceBuffer<unsigned int> hist;
                hist.resize_discard(8);
                unsigned int zero[8]{};
                cudaMemcpyAsync(
                    hist.data(), zero, sizeof(zero), cudaMemcpyHostToDevice, this->stream());
                fast_segmental_reduce_level_hist_kernel<WarpSize, GetKeyOp>
                    <<<elem_block_count, block_dim, 0, this->stream()>>>(
                        size, get_key_op, hist.data());
                unsigned int h[8]{};
                cudaMemcpyAsync(h, hist.data(), sizeof(h), cudaMemcpyDeviceToHost, this->stream());
                cudaStreamSynchronize(this->stream());
                std::fprintf(stderr,
                             "[seg-hist] matrix<%d,%d> in=%zu out=%d warps=%u levels0..5= %u %u %u %u %u %u\n",
                             M,
                             N,
                             (size_t)size,
                             (int)out.size(),
                             h[0] + h[1] + h[2] + h[3] + h[4] + h[5],
                             h[0],
                             h[1],
                             h[2],
                             h[3],
                             h[4],
                             h[5]);
            }
        }

        if(block_count > 0 && fsr::probe_enabled())
        {
            static DeviceBuffer<Matrix> scratch;
            scratch.resize_discard(out.size());
            BufferLaunch(this->stream())
                .fill<Matrix>(scratch.view(), Matrix::Zero().eval());
            // ks stubs first (the live kernel's own floors), then the k2 stubs
            // (s17's reference floors) for the same launch
            if(fsr::serial_tree_enabled())
            {
                fast_segmental_reduce_matrix_ks_kernel<BlockSize, WarpSize, T, M, N, GetKeyOp, GetValueOp, ReduceOp, fsr::kSerialElems, 1>
                    <<<block_count, block_dim, 0, this->stream()>>>(
                        scratch.view(), size, get_key_op, get_value_op, op);
                fast_segmental_reduce_matrix_ks_kernel<BlockSize, WarpSize, T, M, N, GetKeyOp, GetValueOp, ReduceOp, fsr::kSerialElems, 2>
                    <<<block_count, block_dim, 0, this->stream()>>>(
                        scratch.view(), size, get_key_op, get_value_op, op);
            }
            fast_segmental_reduce_matrix_k2_kernel<BlockSize, WarpSize, T, M, N, Flags, GetKeyOp, GetValueOp, ReduceOp, 1>
                <<<elem_block_count, block_dim, 0, this->stream()>>>(
                    scratch.view(), size, get_key_op, get_value_op, op);
            fast_segmental_reduce_matrix_k2_kernel<BlockSize, WarpSize, T, M, N, Flags, GetKeyOp, GetValueOp, ReduceOp, 2>
                <<<elem_block_count, block_dim, 0, this->stream()>>>(
                    scratch.view(), size, get_key_op, get_value_op, op);
        }

        if(block_count > 0 && fsr::verify_enabled() && out.size() > 0)
        {
            static DeviceBuffer<Matrix>             ref;
            static DeviceBuffer<unsigned char>      mark;
            static DeviceBuffer<unsigned long long> vstats;
            ref.resize_discard(out.size());
            mark.resize_discard(out.size());
            vstats.resize_discard(5);
            BufferLaunch(this->stream()).fill<Matrix>(ref.view(), Matrix::Zero().eval());
            BufferLaunch(this->stream()).fill<unsigned char>(mark.view(), (unsigned char)0);
            // the reference is the OLD kernel (its SASS is main's), every slot
            // zeroed first so stored and accumulated slots come out as production's
            fast_segmental_reduce_matrix_kernel<BlockSize, WarpSize, T, M, N, Flags>
                <<<elem_block_count, block_dim, 0, this->stream()>>>(
                    ref.view(), size, get_key_op, get_value_op, op);
            // with the ks kernel live this probe compares two different
            // summation orders: mismatches outside the >= 3-span atomic class
            // are the rounding-level re-association, not a defect (the
            // standalone verifier is the proof instrument for s15)
            int n_boundary = (int)((size + span - 1) / span) - 1;
            if(n_boundary > 0)
            {
                if(use_ks)
                    fast_segmental_reduce_mark_multi_warp_kernel<WarpSize, GetKeyOp, WarpSize * fsr::kSerialElems>
                        <<<(n_boundary + 255) / 256, 256, 0, this->stream()>>>(
                            size, get_key_op, mark.data(), n_boundary);
                else
                    fast_segmental_reduce_mark_multi_warp_kernel<WarpSize, GetKeyOp>
                        <<<(n_boundary + 255) / 256, 256, 0, this->stream()>>>(
                            size, get_key_op, mark.data(), n_boundary);
            }
            unsigned long long zero[5]{};
            cudaMemcpyAsync(vstats.data(), zero, sizeof(zero), cudaMemcpyHostToDevice, this->stream());
            int n = (int)out.size();
            fast_segmental_reduce_compare_kernel<T, M, N>
                <<<(n + 255) / 256, 256, 0, this->stream()>>>(
                    out.cview(), ref.cview(), mark.data(), vstats.data());
            unsigned long long h[5]{};
            cudaMemcpyAsync(h, vstats.data(), sizeof(h), cudaMemcpyDeviceToHost, this->stream());
            cudaStreamSynchronize(this->stream());
            double maxrel;
            std::memcpy(&maxrel, &h[3], sizeof(double));
            std::fprintf(stderr,
                         "[seg-verify] matrix<%d,%d> in=%zu out=%d words=%zu mismatch_words=%llu "
                         "(in_multiwarp_slots=%llu, elsewhere=%llu) slots=%llu multiwarp_slots=%llu maxrel=%.3e\n",
                         M,
                         N,
                         (size_t)size,
                         n,
                         (size_t)n * M * N,
                         h[0] + h[1],
                         h[1],
                         h[0],
                         h[2],
                         h[4],
                         maxrel);
        }
    }

    if(poison_check && out.size() > 0)
    {
        static DeviceBuffer<unsigned long long> stats;
        stats.resize_discard(4);
        unsigned long long zero[4]{0, 0, ~0ull, 0};
        cudaMemcpyAsync(stats.data(), zero, sizeof(zero), cudaMemcpyHostToDevice, this->stream());
        int n = (int)out.size();
        fast_segmental_reduce_poison_count_matrix_kernel<T, M, N>
            <<<(n + 255) / 256, 256, 0, this->stream()>>>(
                out.cview(), fsr::poison_value<T>(), stats.data());
        unsigned long long h[4]{};
        cudaMemcpyAsync(h, stats.data(), sizeof(h), cudaMemcpyDeviceToHost, this->stream());
        cudaStreamSynchronize(this->stream());
        std::fprintf(stderr,
                     "[seg-poison] matrix<%d,%d> in=%zu out=%d unwritten=%llu partial=%llu "
                     "first=%lld last=%lld\n",
                     M,
                     N,
                     (size_t)size,
                     n,
                     h[0],
                     h[1],
                     h[0] ? (long long)h[2] : -1,
                     h[0] ? (long long)h[3] : -1);
    }

    return *this;
}

template <int BlockSize, int WarpSize>
template <typename T, int M, int N, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    CBufferView<int>                    offset,
    CBufferView<Eigen::Matrix<T, M, N>> in,
    BufferView<Eigen::Matrix<T, M, N>>  out,
    ReduceOp                            op,
    SegOutInit                          init)
{
    return reduce(in.size(),
                  out,
                  fast_segmental_reduce_get_offset_key_op{offset},
                  fast_segmental_reduce_get_buffer_value_op<Eigen::Matrix<T, M, N>>{in},
                  op,
                  init);
}
}  // namespace uipc::backend::cuda_tool
