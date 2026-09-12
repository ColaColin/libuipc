#pragma once
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Core>
#include <cub/util_type.cuh>
namespace uipc::backend::cuda_tool
{
// How `reduce` initialises its output buffer before the reduction runs.
//
// The reduce kernel writes a segment's result in one of two ways: a segment
// that lies entirely inside one warp is *stored* by its head lane, a segment
// that crosses a warp boundary is *atomically accumulated* by every warp that
// holds part of it. Only the second kind needs a zero in `out` beforehand.
//
//  - `Full`            zero the whole of `out` (the historical behaviour, and
//                      the only safe choice when `out` has slots that no key
//                      maps to -- those slots are never written at all).
//  - `CrossWarpOnly`   zero only the slots the kernel accumulates into.
//                      **Precondition: every slot of `out` is the key of at
//                      least one input element**, i.e. the caller's key space
//                      is dense over `[0, out.size())`. Slots outside the key
//                      space keep whatever they held.
//
// Which warp reaches a cross-warp segment first is arrival order, not index
// order, so the "first writer stores instead of adding" fold is *not* available
// for that case -- a store would clobber a concurrent contribution. Narrowing
// the fill is what is available.
enum class SegOutInit
{
    Full,
    CrossWarpOnly
};

template <int BlockSize = 128, int WarpSize = 32>
class FastSegmentalReduce : public LaunchBase<FastSegmentalReduce<BlockSize, WarpSize>>
{
    using Base = LaunchBase<FastSegmentalReduce<BlockSize, WarpSize>>;

  public:
    // public on purpose: NVCC requires template argument types of __global__
    // kernels to be publicly accessible (the reduce kernels take FlagsT).
    struct Flags
    {
        union
        {
            struct
            {
                unsigned char is_head;
                unsigned char is_cross_warp;
                unsigned char is_valid;
            };
            unsigned int flags;
        };

        __host__ __device__ void b2i()
        {
            is_head       = is_head ? 1 : 0;
            is_cross_warp = is_cross_warp ? 1 : 0;
            is_valid      = is_valid ? 1 : 0;
        }
    };

  public:
    FastSegmentalReduce(cudaStream_t s = nullptr)
        : Base(s)
    {
    }

    // e.g.
    // when ReduceOp = ::cuda::std::plus
    // dst = [0, 1, 1, 2, 2, 2]
    // in  = [1, 1, 1, 1, 1, 1]
    // out = [1, 2, 3]
    template <typename T, typename ReduceOp = ::cuda::std::plus<T>>
    FastSegmentalReduce& reduce(CBufferView<int> dst,
                                CBufferView<T>   in,
                                BufferView<T>    out,
                                ReduceOp         op   = ReduceOp{},
                                SegOutInit       init = SegOutInit::Full);

    template <typename T, typename GetKeyOp, typename GetValueOp, typename ReduceOp = ::cuda::std::plus<T>>
    FastSegmentalReduce& reduce(size_t        in_size,
                                BufferView<T> out,
                                GetKeyOp      get_key_op,
                                GetValueOp    get_value_op,
                                ReduceOp      op   = ReduceOp{},
                                SegOutInit    init = SegOutInit::Full);


    template <typename T, int M, int N, typename ReduceOp = ::cuda::std::plus<T>>
    FastSegmentalReduce& reduce(CBufferView<int>                    dst,
                                CBufferView<Eigen::Matrix<T, M, N>> in,
                                BufferView<Eigen::Matrix<T, M, N>>  out,
                                ReduceOp   op   = ReduceOp{},
                                SegOutInit init = SegOutInit::Full);

    template <typename T, int M, int N, typename GetKeyOp, typename GetValueOp, typename ReduceOp = ::cuda::std::plus<T>>
    FastSegmentalReduce& reduce(size_t                             in_size,
                                BufferView<Eigen::Matrix<T, M, N>> out,
                                GetKeyOp                           get_key_op,
                                GetValueOp                         get_value_op,
                                ReduceOp   op   = ReduceOp{},
                                SegOutInit init = SegOutInit::Full);
};
}  // namespace uipc::backend::cuda_tool

#include "details/fast_segmental_reduce.inl"
