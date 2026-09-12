#include <cub/warp/warp_reduce.cuh>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <cstdio>
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

    inline bool fill_poison_enabled() { return fill_poison_level() > 0; }

    // UIPC_SEG_FILL_POISON=2 additionally counts, after the reduce, how many
    // output slots still hold the poison -- i.e. how many slots the narrow fill
    // did not cover and the kernel never wrote. That is the rig validation:
    // a narrow fill that is complete must report zero.
    inline bool fill_poison_check() { return fill_poison_level() >= 2; }

    template <typename T>
    inline T poison_value()
    {
        if constexpr(std::is_floating_point_v<T>)
            return std::numeric_limits<T>::signaling_NaN();
        else
            return std::numeric_limits<T>::max();
    }
}  // namespace details::fast_segmental_reduce

namespace
{
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

    template <int WarpSize, typename T, int M, int N, typename GetKeyOp>
    __global__ void fast_segmental_reduce_zero_cross_warp_kernel(
        BufferView<Eigen::Matrix<T, M, N>> out, size_t in_size, GetKeyOp get_key_op, int n_boundary)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t > n_boundary)
            return;

        if(t == n_boundary)  // the tail task
        {
            if(in_size % (size_t)WarpSize != 0)
                out(get_key_op((int)in_size - 1)).setZero();
            return;
        }

        size_t g = (size_t)(t + 1) * (size_t)WarpSize;
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
    size_t     in_size,
    BufferView<T> out,
    GetKeyOp   get_key_op,
    GetValueOp get_value_op,
    ReduceOp   op,
    SegOutInit init)
{
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");
    static_assert(BlockSize % WarpSize == 0,
                  "the narrow output fill assumes warp boundaries sit at global "
                  "thread ids that are multiples of WarpSize");

    using ValueT = T;
    namespace fsr = details::fast_segmental_reduce;

    auto          size      = in_size;
    constexpr int block_dim = BlockSize;

    const bool narrow =
        (init == SegOutInit::CrossWarpOnly) && fsr::narrow_fill_enabled();

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
    size_t             in_size,
    BufferView<Eigen::Matrix<T, M, N>> out,
    GetKeyOp           get_key_op,
    GetValueOp         get_value_op,
    ReduceOp           op,
    SegOutInit         init)
{
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "FastSegmentalReduce only supports floating point and integral types");
    static_assert(BlockSize % WarpSize == 0,
                  "the narrow output fill assumes warp boundaries sit at global "
                  "thread ids that are multiples of WarpSize");

    using Matrix = Eigen::Matrix<T, M, N>;
    namespace fsr = details::fast_segmental_reduce;

    auto          size      = in_size;
    constexpr int block_dim = BlockSize;

    const bool narrow =
        (init == SegOutInit::CrossWarpOnly) && fsr::narrow_fill_enabled();

    if(!narrow)
    {
        BufferLaunch(this->stream()).fill<Matrix>(out, Matrix::Zero().eval());
    }
    else
    {
        if(fsr::fill_poison_enabled())
            BufferLaunch(this->stream())
                .fill<Matrix>(out, Matrix::Constant(fsr::poison_value<T>()).eval());

        // tasks 0..n_boundary-1 are the warp boundaries, task n_boundary is the tail
        int n_boundary = (int)((size + WarpSize - 1) / WarpSize) - 1;
        if(n_boundary < 0)
            n_boundary = 0;
        if(size > 0)
        {
            constexpr int fill_block = 256;
            int           n_task     = n_boundary + 1;
            fast_segmental_reduce_zero_cross_warp_kernel<WarpSize, T, M, N, GetKeyOp>
                <<<(n_task + fill_block - 1) / fill_block, fill_block, 0, this->stream()>>>(
                    out, size, get_key_op, n_boundary);
        }
    }

    const bool poison_check = narrow && fsr::fill_poison_check();

    int block_count = (size + block_dim - 1) / block_dim;
    if(block_count > 0)
        fast_segmental_reduce_matrix_kernel<BlockSize, WarpSize, T, M, N, Flags>
            <<<block_count, block_dim, 0, this->stream()>>>(out, size, get_key_op, get_value_op, op);

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
