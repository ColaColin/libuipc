#include <linear_system/spmv.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/util_math.cuh>
#include <cuda_device/bit_operation.h>
#include <cuda_device/builtin.h>
#include <Eigen/Sparse>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <uipc/common/log.h>

namespace uipc::backend::cuda
{
static double __longlong_as_double_host(unsigned long long b)
{
    double d;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}

__host__ __device__ constexpr int b2i(bool b)
{
    return b ? 1 : 0;
}

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

namespace
{
    // y = b * y, shared by sym_spmv / rbk_spmv / rbk_sym_spmv / rbk_sym_spmv_dot
    __global__ void Spmv_scale_y_kernel(Float b, cuda_tool::DenseVectorView<Float> y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        y(i) = b * y(i);
    }

    __global__ void Spmv_sym_spmv_kernel(Float                                a,
                                         cuda_tool::CBCOOMatrixView<Float, 3> A,
                                         cuda_tool::CDenseVectorView<Float>   x,
                                         cuda_tool::DenseVectorView<Float>    y,
                                         int                                  n)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= n)
            return;

        constexpr int N = 3;
        using T         = Float;

        auto&& [i, j, block] = A(index);

        if(i == j)  // diagonal block
        {
            auto seg_x = x.segment<N>(j * N);

            Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
            auto                result = a * block * vec_x;

            auto seg_y = y.segment<N>(i * N);
            seg_y.atomic_add(result.eval());
        }
        else  // off-diagonal block
        {
            // ij-th block
            {
                auto seg_x = x.segment<N>(j * N);

                Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                auto                result = a * block * vec_x;

                auto seg_y = y.segment<N>(i * N);
                seg_y.atomic_add(result.eval());
            }

            // ji-th block
            {
                auto seg_x = x.segment<N>(i * N);

                Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                auto                result = a * block.transpose() * vec_x;

                auto seg_y = y.segment<N>(j * N);
                seg_y.atomic_add(result.eval());
            }
        }
    }

    __global__ void Spmv_rbk_spmv_kernel(Float                                a,
                                         cuda_tool::CBCOOMatrixView<Float, 3> A,
                                         cuda_tool::CDenseVectorView<Float>   x,
                                         cuda_tool::DenseVectorView<Float>    y)
    {
        constexpr int          warp_size = 32;
        constexpr unsigned int warp_mask = ~0u;
        constexpr int          block_dim = 128;
        constexpr int          N         = 3;

        using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
        using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
        using WarpScanInt     = cub::WarpScan<int>;

        auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
        auto thread_id_in_block = threadIdx.x;
        auto warp_id            = thread_id_in_block / warp_size;
        auto lane_id            = thread_id_in_block & (warp_size - 1);

        __shared__ union
        {
            typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
            typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
        };

        int prev_i = -1;
        int next_i = -1;
        int i      = -1;

        Flags   flags;
        Vector3 vec;
        flags.is_cross_warp = 0;


        if(global_thread_id > 0 && global_thread_id < A.triplet_count())
        {
            auto prev_triplet = A(global_thread_id - 1);
            prev_i            = prev_triplet.row_index;
        }

        if(global_thread_id < A.triplet_count() - 1)
        {
            auto next_triplet = A(global_thread_id + 1);
            next_i            = next_triplet.row_index;
        }

        if(global_thread_id < A.triplet_count())
        {
            auto Triplet = A(global_thread_id);
            i            = Triplet.row_index;
            auto j       = Triplet.col_index;

            vec = Triplet.value * x.segment<N>(j * N).as_eigen();

            flags.is_valid = 1;
        }
        else
        {
            i = -1;
            vec.setZero();
            flags.is_valid      = 0;
            flags.is_cross_warp = 0;
        }

        if(lane_id == 0)
        {
            flags.is_head = 1;
            // if this thread is the first thread in the warp
            // check if the previous triplet is in the same row
            // if so, this row crosses the warp boundary, we need use atomic add
            flags.is_cross_warp = b2i(prev_i == i);
        }
        else
        {
            flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined

            if(lane_id == warp_size - 1)
            {
                // if this thread is the last thread in the warp
                // check if the next triplet is in the same row
                // if so, this row crosses the warp boundary, we need use atomic add
                flags.is_cross_warp = b2i(next_i == i);
            }
        }

        flags.flags =
            WarpReduceInt(temp_storage_int[warp_id])
                .HeadSegmentedReduce(flags.flags,
                                     flags.is_head,
                                     [](uint32_t a, uint32_t b) { return a + b; });

        vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(vec.x(),
                                           flags.is_head,
                                           [](Float a, Float b) { return a + b; });

        vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(vec.y(),
                                           flags.is_head,
                                           [](Float a, Float b) { return a + b; });

        vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(vec.z(),
                                           flags.is_head,
                                           [](Float a, Float b) { return a + b; });


        // cub::WARP_SYNC(warp_mask);

        flags.is_head = b2i(flags.is_head && flags.is_valid);

        flags.b2i();
        int is_head_mask = detail::bit_operation::WARP_BALLOT(flags.is_head, warp_mask);
        uint32_t offset = __fns(is_head_mask, 0, lane_id + 1);

        int valid_bit = (offset != ~0u);
        int shuffle_mask = detail::bit_operation::WARP_BALLOT(valid_bit, warp_mask);

        i           = cub::ShuffleIndex<32>(i, offset, shuffle_mask);
        flags.flags = cub::ShuffleIndex<32>(flags.flags, offset, shuffle_mask);
        vec.x()     = cub::ShuffleIndex<32>(vec.x(), offset, shuffle_mask);
        vec.y()     = cub::ShuffleIndex<32>(vec.y(), offset, shuffle_mask);
        vec.z()     = cub::ShuffleIndex<32>(vec.z(), offset, shuffle_mask);

        if(valid_bit && flags.is_head && flags.is_valid)
        {
            auto seg_y  = y.segment<N>(i * N);
            auto result = a * vec;

            if(flags.is_cross_warp)
            {
                seg_y.atomic_add(result.eval());
            }
            else
            {
                seg_y.as_eigen() += result.eval();
            }
        }
    }

    __global__ void Spmv_rbk_sym_spmv_kernel(Float a,
                                             cuda_tool::CBCOOMatrixView<Float, 3> A,
                                             cuda_tool::CDenseVectorView<Float> x,
                                             cuda_tool::DenseVectorView<Float> y,
                                             int n)
    {
        constexpr int warp_size = 32;
        constexpr int block_dim = 256;
        constexpr int N         = 3;

        for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += gridDim.x * blockDim.x)
        {
            using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
            using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
            using WarpScanInt     = cub::WarpScan<int>;

            auto global_thread_id   = idx;
            auto thread_id_in_block = threadIdx.x;
            auto warp_id            = thread_id_in_block / warp_size;
            auto lane_id            = thread_id_in_block & (warp_size - 1);

            __shared__ union
            {
                typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
            };

            int     prev_i = -1;
            int     i      = -1;
            Flags   flags;
            Vector3 vec;

            // In symmtric version, we don't need to check the cross warp
            flags.is_cross_warp = 0;

            // set the previous row index
            if(global_thread_id > 0)
            {
                auto prev_triplet = A(global_thread_id - 1);
                prev_i            = prev_triplet.row_index;
            }

            {
                auto Triplet = A(global_thread_id);
                i            = Triplet.row_index;
                auto j       = Triplet.col_index;

                vec = Triplet.value * x.segment<N>(j * N).as_eigen();

                flags.is_valid = 1;

                if(i != j)  // process lower triangle
                {
                    Vector3 vec_ = a * Triplet.value.transpose()
                                   * x.segment<N>(i * N).as_eigen();

                    y.segment<N>(j * N).atomic_add(vec_);
                }
            }

            if(lane_id == 0)
            {
                flags.is_head = 1;
            }
            else
            {
                flags.is_head = b2i(prev_i != i);  // must be 1 or 0, or the result is undefined
            }


            // ----------------------------------- warp reduce ----------------------------------------------
            vec.x() =
                WarpReduceFloat(temp_storage_float[warp_id])
                    .HeadSegmentedReduce(vec.x(),
                                         flags.is_head,
                                         [](Float a, Float b) { return a + b; });

            vec.y() =
                WarpReduceFloat(temp_storage_float[warp_id])
                    .HeadSegmentedReduce(vec.y(),
                                         flags.is_head,
                                         [](Float a, Float b) { return a + b; });

            vec.z() =
                WarpReduceFloat(temp_storage_float[warp_id])
                    .HeadSegmentedReduce(vec.z(),
                                         flags.is_head,
                                         [](Float a, Float b) { return a + b; });
            // ----------------------------------- warp reduce -----------------------------------------------


            if(flags.is_head)
            {
                auto seg_y  = y.segment<N>(i * N);
                auto result = a * vec;

                // Must use atomic add!
                // Because the same row may be processed by different warps
                seg_y.atomic_add(result.eval());
            }
        }
    }

    __global__ void Spmv_rbk_sym_spmv_dot_kernel(Float a,
                                                 cuda_tool::CBCOOMatrixView<Float, 3> A,
                                                 cuda_tool::CDenseVectorView<Float> x,
                                                 cuda_tool::DenseVectorView<Float> y,
                                                 cuda_tool::Dense<Float> d_dot,
                                                 cuda_tool::CDense<IndexT> d_triplet_count,
                                                 bool skip_idle_blocks)
    {
        // count lives on device: a graph capturing this kernel then stays
        // valid when the matrix nnz changes within the reserved capacity
        const int triplet_count = (int)(*d_triplet_count);
        // perf/round4 (s02): the grid covers the reserved capacity, which is
        // sized from the raw (unreduced) triplet count — ABD contact blocks
        // expand 16x before the reduce, so the capacity can exceed the
        // reduced count by two orders of magnitude (wrecking balls: <= 32k
        // unique triplets, > 4M capacity). A block entirely past the count
        // holds no triplet: leave before the shared-memory reductions
        // instead of running three empty warp reductions, a block barrier
        // and an atomicAdd(+0.0). Uniform per block, so no barrier is
        // skipped by a subset of threads. UIPC_SPMV_SKIP_IDLE_BLOCKS=0 = old.
        if(skip_idle_blocks && (int)(blockIdx.x * blockDim.x) >= triplet_count)
            return;
        constexpr int warp_size = 32;
        constexpr int block_dim = 256;
        constexpr int N         = 3;
        using T                 = Float;

        using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
        using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;

        auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
        auto thread_id_in_block = threadIdx.x;
        auto warp_id            = thread_id_in_block / warp_size;
        auto lane_id            = thread_id_in_block & (warp_size - 1);

        __shared__ union
        {
            typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
            typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
        };

        int     prev_i = -1;
        int     i      = -1;
        Flags   flags;
        Vector3 vec;
        vec.setZero();
        Float dot_local = 0;

        flags.is_cross_warp = 0;
        flags.is_valid      = 0;
        flags.is_head       = 0;

        // raw triplet arrays: the view's A(i) accessor bounds-checks against
        // the count baked at capture time, while the device-side count may be
        // larger (matrix grew within capacity) — read through the pointers
        const int*       rows = A.row_indices().data();
        const int*       cols = A.col_indices().data();
        const Matrix3x3* vals = A.values().data();
        if(global_thread_id < triplet_count)
        {
            if(global_thread_id > 0)
                prev_i = rows[global_thread_id - 1];

            i      = rows[global_thread_id];
            auto j = cols[global_thread_id];

            Eigen::Vector<T, N> x_j = x.segment<N>(j * N).as_eigen();
            vec                     = vals[global_thread_id] * x_j;

            flags.is_valid = 1;

            if(i == j)
            {
                dot_local = a * x_j.dot(vec);
            }
            else
            {
                Eigen::Vector<T, N> x_i = x.segment<N>(i * N).as_eigen();
                dot_local               = 2.0 * a * x_i.dot(vec);

                Vector3 vec_ = a * vals[global_thread_id].transpose() * x_i;
                y.segment<N>(j * N).atomic_add(vec_);
            }

            if(lane_id == 0)
                flags.is_head = 1;
            else
                flags.is_head = b2i(prev_i != i);
        }

        vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(vec.x(),
                                           flags.is_head,
                                           [](Float a, Float b) { return a + b; });

        vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(vec.y(),
                                           flags.is_head,
                                           [](Float a, Float b) { return a + b; });

        vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(vec.z(),
                                           flags.is_head,
                                           [](Float a, Float b) { return a + b; });

        if(flags.is_head && flags.is_valid)
        {
            auto seg_y  = y.segment<N>(i * N);
            auto result = a * vec;
            seg_y.atomic_add(result.eval());
        }

        dot_local = WarpReduceFloat(temp_storage_float[warp_id]).Sum(dot_local);

        // two-level reduction: one atomicAdd per BLOCK, not per warp —
        // ~9k same-address atomic doubles serialized ~20-30us per call
        __shared__ Float s_dot_partials[block_dim / warp_size];
        if(lane_id == 0)
            s_dot_partials[warp_id] = dot_local;
        __syncthreads();
        if(thread_id_in_block < warp_size)
        {
            Float partial = (thread_id_in_block < block_dim / warp_size) ?
                                s_dot_partials[thread_id_in_block] :
                                Float{0};
            __syncwarp();
            partial = WarpReduceFloat(temp_storage_float[0]).Sum(partial);
            if(thread_id_in_block == 0)
                atomicAdd(d_dot.data(), partial);
        }
    }

    // perf/round4 (s07): chunked row-run SpMV + dot.
    // The per-triplet kernel above is FP64-issue-bound on 1/32-rate parts
    // (SASS: ~75 FP64 instructions per triplet — two 3x3 matvecs, the
    // dot, three 5-level segmented warp reductions, the dot reduction and
    // the `a` scaling; 288k triplets x 75 / 141.6 G FP64/s = 152 us on the
    // 2070S, measured 153 us). Here one thread walks C consecutive triplets
    // of the (row, col)-sorted symmetric BCOO: equal-row runs accumulate the
    // row sum with 9 FMAs per block (no cross-lane reduction), a run is
    // flushed with 3 atomics when the row changes, the lower-triangle
    // contribution w = A^T x_i goes out with 3 atomics per off-diagonal
    // block, and the dot uses x^T A x = sum_i x_i.rowsum_i
    // + sum_{i<j} x_j.(A_ij^T x_i), so w is reused (3 FMAs) and the
    // diagonal needs no special case. `a` is applied only when a != 1
    // (exact for a == 1, which the PCG passes). Same matrix, vectors and
    // products; the summation order differs (rounding-level; the old path
    // was already nondeterministic through its atomics). The grid covers
    // the reserved capacity / (256 C) with the s02 idle-block exit and the
    // device-side count, so a captured graph stays valid while the count
    // varies within the capacity. UIPC_SPMV_CHUNK=0 = the kernel above,
    // 1|2|4|8|16 = force C; UIPC_SPMV_VERIFY=1 = also run the old kernel
    // into scratch and accumulate max|dy|, max|y_ref|, max|ddot| on
    // device (reported once per assembly by Spmv::verify_report).
    // Measured (2070S, ms per launch, old -> C = 1 / 2 / 4 / 8 / 16): bunny
    // 0.070 -> 0.053 / 0.065 / 0.077 / 0.076 / 0.110, case2 0.149 -> 0.112 /
    // 0.120 / 0.162 / 0.206 / 0.277 — C = 1 (every block its own run, six
    // atomics per off-diagonal block, perfectly coalesced loads) is the
    // default; longer runs lose more to the strided loads than they save in
    // atomics. Probes at C = 1 (bunny): loads only 31 us, no atomics 43,
    // no transposed part 43, real 53 — the kernel is within 1.7x of the
    // DRAM floor for this storage.
    // MODE (UIPC_SPMV_PROBE timing probes into scratch, never the result):
    // 0 real; 1 no transposed contribution; 2 plain stores instead of
    // atomics; 3 loads only (integer checksum, no FP64)
    template <int C, int MODE = 0>
    __global__ void __launch_bounds__(256)
        Spmv_rbk_sym_spmv_dot_chunked_kernel(Float a,
                                             const int* __restrict__ rows,
                                             const int* __restrict__ cols,
                                             const Matrix3x3* __restrict__ vals,
                                             const Float* __restrict__ x,
                                             Float*        y,
                                             Float*        d_dot,
                                             const IndexT* d_triplet_count)
    {
        constexpr int block_dim = 256;
        constexpr int warp_size = 32;
        using WarpReduceFloat   = cub::WarpReduce<Float, warp_size>;
        __shared__ typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
        __shared__ Float s_dot_partials[block_dim / warp_size];

        const int triplet_count = (int)(*d_triplet_count);

        const bool unit = (a == Float(1));
        auto       acc  = [](Float* p, Float v)
        {
            if constexpr(MODE == 2)
                *p = v;
            else
                atomicAdd(p, v);
        };
        [[maybe_unused]] unsigned long long chk = 0;

        Float dot_local = 0;

        // perf/round5 (s22): grid-stride over *virtual* blocks. The grid used
        // to cover the reserved triplet *capacity*, so that a CUDA graph
        // capturing this launch stays valid when the nnz changes (the count
        // lives on device). But the capacity is sized from the raw, unreduced
        // triplet count, and on the FEM scenes the duplicate reduce is ~7x:
        // stiff-gipc-case2 assembles 1 856 397 raw triplets that reduce to
        // 286 419, so of the 7 977 blocks launched only 1 119 hold a triplet
        // and 6 858 exist only to read the count and exit. A standalone
        // sm_75 probe puts such a block at 1.45 ns, i.e. ~10 us of a 113 us
        // launch, and that is what the measurement recovers.
        // A block now walks virtual blocks `blockIdx.x, +gridDim.x, ...`, so
        // the grid is capped at what the device can hold resident
        // (`resident_blocks`) and depends on neither the count nor the
        // capacity — strictly safer for graph replay than the capacity grid,
        // not less safe. Sizing the grid from the *host-side* count instead
        // is not an option and was measured: the count recorded at capture
        // time goes stale on replay, triplets past it are dropped, and
        // case2's line search then fails to converge from ~Newton 100 on.
        // Per-thread work, coalescing and the per-triplet contributions are
        // unchanged; only the order in which one thread's dot contributions
        // are summed differs (rounding level; y was already nondeterministic
        // through its atomics). With the capacity grid (UIPC_SPMV_GRID_STRIDE=0)
        // every block breaks after one pass, i.e. the old behaviour exactly.
        //
        // s02's early exit, kept verbatim: a block with no triplet at all must
        // leave *before* the block reduction, not fall into it — three empty
        // warp reductions, a barrier and a same-address atomicAdd(+0.0) per
        // idle block cost more than the idle launch itself (measured: letting
        // them fall through takes the capacity-grid path from 112 to 217 us
        // per launch on case2).
        if((int)blockIdx.x * (block_dim * C) >= triplet_count)  // uniform per block
            return;

        for(int vb = (int)blockIdx.x;; vb += (int)gridDim.x)
        {
            const int block_first = vb * (block_dim * C);
            if(block_first >= triplet_count)  // uniform per block: no barrier skipped
                break;

            const int begin = block_first + (int)threadIdx.x * C;

            int   cur_i = -1;
            Float xi0 = 0, xi1 = 0, xi2 = 0;
            Float r0 = 0, r1 = 0, r2 = 0;

            [[maybe_unused]] bool twice = false;  // C == 1 only, see below

            auto flush = [&]()
            {
                // dot += x_i . rowsum_i ; y_i += a * rowsum_i
                Float d = xi0 * r0;
                d       = __fma_rn(xi1, r1, d);
                d       = __fma_rn(xi2, r2, d);
                if constexpr(C == 1)
                    dot_local += twice ? d + d : d;
                else
                    dot_local += d;
                Float* yi = y + 3 * cur_i;
                if(unit)
                {
                    acc(yi + 0, r0);
                    acc(yi + 1, r1);
                    acc(yi + 2, r2);
                }
                else
                {
                    acc(yi + 0, a * r0);
                    acc(yi + 1, a * r1);
                    acc(yi + 2, a * r2);
                }
            };

    #pragma unroll
            for(int k = 0; k < C; ++k)
            {
                const int t = begin + k;
                if(t >= triplet_count)  // sorted: every later t is past the end too
                    break;
                const int i = rows[t];
                const int j = cols[t];

                const Matrix3x3 A = vals[t];  // 9 doubles into registers

                const Float* xj  = x + 3 * j;
                const Float  xj0 = __ldg(xj + 0);
                const Float  xj1 = __ldg(xj + 1);
                const Float  xj2 = __ldg(xj + 2);

                if constexpr(MODE == 3)
                {
                    const Float* xi = x + 3 * i;
                    for(int e = 0; e < 9; ++e)
                        chk ^= (unsigned long long)__double_as_longlong(A.data()[e]);
                    chk ^= (unsigned long long)__double_as_longlong(
                        xj0 + xj1 + xj2 + __ldg(xi) + __ldg(xi + 1) + __ldg(xi + 2));
                    continue;
                }

                if(i != cur_i)
                {
                    if(cur_i >= 0)
                        flush();
                    cur_i = i;
                    r0 = r1 = r2    = 0;
                    const Float* xi = x + 3 * i;
                    xi0             = __ldg(xi + 0);
                    xi1             = __ldg(xi + 1);
                    xi2             = __ldg(xi + 2);
                }

                // rowsum += A * x_j  (9 FMA)
                r0 = __fma_rn(A(0, 0), xj0, r0);
                r0 = __fma_rn(A(0, 1), xj1, r0);
                r0 = __fma_rn(A(0, 2), xj2, r0);
                r1 = __fma_rn(A(1, 0), xj0, r1);
                r1 = __fma_rn(A(1, 1), xj1, r1);
                r1 = __fma_rn(A(1, 2), xj2, r1);
                r2 = __fma_rn(A(2, 0), xj0, r2);
                r2 = __fma_rn(A(2, 1), xj1, r2);
                r2 = __fma_rn(A(2, 2), xj2, r2);

                if(MODE != 1 && i != j)
                {
                    // w = A^T x_i  (3 MUL + 6 FMA); dot += x_j . w ; y_j += a * w
                    Float w0 = A(0, 0) * xi0;
                    w0       = __fma_rn(A(1, 0), xi1, w0);
                    w0       = __fma_rn(A(2, 0), xi2, w0);
                    Float w1 = A(0, 1) * xi0;
                    w1       = __fma_rn(A(1, 1), xi1, w1);
                    w1       = __fma_rn(A(2, 1), xi2, w1);
                    Float w2 = A(0, 2) * xi0;
                    w2       = __fma_rn(A(1, 2), xi1, w2);
                    w2       = __fma_rn(A(2, 2), xi2, w2);

                    if constexpr(C == 1)
                    {
                        // one block per thread: x_j.w == x_i.(A x_j) = x_i.rowsum,
                        // which the flush adds once; add it a second time there
                        // (2 x_i.(A x_j), exact doubling) instead of 3 more FMAs
                        twice = true;
                    }
                    else
                    {
                        dot_local = __fma_rn(xj0, w0, dot_local);
                        dot_local = __fma_rn(xj1, w1, dot_local);
                        dot_local = __fma_rn(xj2, w2, dot_local);
                    }

                    Float* yj = y + 3 * j;
                    if(unit)
                    {
                        acc(yj + 0, w0);
                        acc(yj + 1, w1);
                        acc(yj + 2, w2);
                    }
                    else
                    {
                        acc(yj + 0, a * w0);
                        acc(yj + 1, a * w1);
                        acc(yj + 2, a * w2);
                    }
                }
            }
            if constexpr(MODE == 3)
            {
                y[begin % 3 + 3 * (int)(chk % 7)] = __longlong_as_double((long long)chk);
                continue;
            }
            if(cur_i >= 0)
                flush();
        }  // virtual-block loop

        if constexpr(MODE == 3)
            return;
        if(!unit)
            dot_local *= a;

        // two-level reduction, one atomicAdd per block (as above)
        const int warp_id = threadIdx.x / warp_size;
        const int lane_id = threadIdx.x & (warp_size - 1);
        dot_local = WarpReduceFloat(temp_storage_float[warp_id]).Sum(dot_local);
        if(lane_id == 0)
            s_dot_partials[warp_id] = dot_local;
        __syncthreads();
        if(threadIdx.x < warp_size)
        {
            Float partial = (threadIdx.x < block_dim / warp_size) ?
                                s_dot_partials[threadIdx.x] :
                                Float{0};
            __syncwarp();
            partial = WarpReduceFloat(temp_storage_float[0]).Sum(partial);
            if(threadIdx.x == 0)
                atomicAdd(d_dot, partial);
        }
    }


    // UIPC_SPMV_VERIFY: acc = {max|y - y_ref|, max|y_ref|, max|dot - dot_ref|,
    // max|dot_ref|, launches} as the bit patterns of non-negative doubles
    // (order-preserving as unsigned 64-bit), warp-max first, then atomicMax
    __global__ void Spmv_verify_kernel(const Float*        y,
                                       const Float*        y_ref,
                                       const Float*        dot,
                                       const Float*        dot_ref,
                                       unsigned long long* acc,
                                       int                 n)
    {
        using WarpReduceU = cub::WarpReduce<unsigned long long, 32>;
        __shared__ typename WarpReduceU::TempStorage temp[8];
        const int i  = blockIdx.x * blockDim.x + threadIdx.x;
        auto      ab = [](Float v)
        { return (unsigned long long)__double_as_longlong(fabs(v)); };
        unsigned long long d = 0, m = 0;
        if(i < n)
        {
            d = ab(y[i] - y_ref[i]);
            m = ab(y_ref[i]);
        }
        const int warp_id = threadIdx.x / 32;
        d                 = WarpReduceU(temp[warp_id]).Reduce(d, cub::Max());
        m                 = WarpReduceU(temp[warp_id]).Reduce(m, cub::Max());
        if((threadIdx.x & 31) == 0)
        {
            atomicMax(acc + 0, d);
            atomicMax(acc + 1, m);
        }
        if(i == 0)
        {
            atomicMax(acc + 2, ab(*dot - *dot_ref));
            atomicMax(acc + 3, ab(*dot_ref));
            atomicAdd(acc + 4, 1ull);
        }
    }

    struct SpmvEnv
    {
        int chunk = 1;  // 0 = per-triplet kernel; 1 measured best (see the round-4 record, s07)
        bool verify           = false;
        bool probe            = false;
        bool skip_idle_blocks = true;
        bool pcg_fuse_scalar  = true;
        // s13: same switch as in linear_system/linear_fused_pcg.cu -- when the
        // Ap-zero fusion is on, fused_update_xr has already stored 0 into the
        // whole output vector, so rbk_sym_spmv_dot must not fill it again.
        // rbk_sym_spmv_dot is called only from LinearFusedPCG.
        bool pcg_fuse_ap_zero = true;
        // s22: cap the SpMV+dot grid at what the device can hold resident and
        // let each block grid-stride over virtual blocks, instead of launching
        // one block per 256 triplets of the reserved *capacity*.
        // UIPC_SPMV_GRID_STRIDE=0 = the old capacity-sized grid (with which the
        // kernel's loop breaks after one pass, i.e. the old behaviour).
        bool grid_stride = true;
    };
    // s22: how many blocks of this kernel the device can hold resident.
    // Queried once per C (the occupancy API is a driver call; the result is a
    // property of the kernel and the device, never of the matrix, so a grid
    // built from it is stable across CUDA-graph replays).
    template <int C>
    int resident_blocks()
    {
        static const int n = []
        {
            int device = 0, sm_count = 1, per_sm = 1;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, Spmv_rbk_sym_spmv_dot_chunked_kernel<C, 0>, 256, 0);
            if(per_sm < 1)
                per_sm = 1;
            int waves = 1;  // measured flat 1..16, see the round-5 record (s22)
            if(const char* e = std::getenv("UIPC_SPMV_GRID_WAVES"))
            {
                int w = std::atoi(e);
                if(w > 0)
                    waves = w;
            }
            return sm_count * per_sm * waves;
        }();
        return n;
    }

    const SpmvEnv& spmv_env()
    {
        static const SpmvEnv env = []
        {
            SpmvEnv e;
            if(const char* s = std::getenv("UIPC_SPMV_CHUNK"))
            {
                int c = std::atoi(s);
                e.chunk =
                    (c == 0 || c == 1 || c == 2 || c == 4 || c == 8 || c == 16) ? c : 1;
            }
            if(const char* s = std::getenv("UIPC_SPMV_VERIFY"))
                e.verify = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_SPMV_PROBE"))
                e.probe = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_SPMV_SKIP_IDLE_BLOCKS"))
                e.skip_idle_blocks = !(s[0] == '0');
            // s11: same switch as in linear_system/linear_fused_pcg.cu — when
            // the PCG scalar fusion is on, the p^T A p accumulator is zeroed
            // by the previous iteration's scalar kernel (and once per solve on
            // the host), so rbk_sym_spmv_dot must not memset it again.
            // rbk_sym_spmv_dot is called only from LinearFusedPCG.
            if(const char* s = std::getenv("UIPC_PCG_FUSE_SCALAR"))
                e.pcg_fuse_scalar = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_PCG_FUSE_AP_ZERO"))
                e.pcg_fuse_ap_zero = !(s[0] == '0');
            if(const char* s = std::getenv("UIPC_SPMV_GRID_STRIDE"))
                e.grid_stride = !(s[0] == '0');
            return e;
        }();
        return env;
    }
}  // namespace

void Spmv::sym_spmv(Float                                a,
                    cuda_tool::CBCOOMatrixView<Float, 3> A,
                    cuda_tool::CDenseVectorView<Float>   x,
                    Float                                b,
                    cuda_tool::DenseVectorView<Float>    y)
{
    if(b != 0)
    {
        int n = y.size();
        if(n > 0)
        {
            Spmv_scale_y_kernel<<<cuda_tool::best_grid_dim(n, Spmv_scale_y_kernel), cuda_tool::best_block_dim(Spmv_scale_y_kernel), 0, nullptr>>>(
                b, y, n);
        }
    }
    else
    {
        cuda_tool::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    int triplet_count = A.triplet_count();
    if(triplet_count > 0)
    {
        Spmv_sym_spmv_kernel<<<cuda_tool::best_grid_dim(triplet_count, Spmv_sym_spmv_kernel), cuda_tool::best_block_dim(Spmv_sym_spmv_kernel), 0, nullptr>>>(
            a, A, x, y, triplet_count);
    }
}

void Spmv::rbk_spmv(Float                                a,
                    cuda_tool::CBCOOMatrixView<Float, 3> A,
                    cuda_tool::CDenseVectorView<Float>   x,
                    Float                                b,
                    cuda_tool::DenseVectorView<Float>    y)
{
    if(b != 0)
    {
        int n = y.size();
        if(n > 0)
        {
            Spmv_scale_y_kernel<<<cuda_tool::best_grid_dim(n, Spmv_scale_y_kernel), cuda_tool::best_block_dim(Spmv_scale_y_kernel), 0, nullptr>>>(
                b, y, n);
        }
    }
    else
    {
        cuda_tool::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int block_dim   = 128;
    int           block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    if(block_count > 0)
    {
        Spmv_rbk_spmv_kernel<<<block_count, block_dim, 0, nullptr>>>(a, A, x, y);
    }
}

void Spmv::rbk_sym_spmv(Float                                a,
                        cuda_tool::CBCOOMatrixView<Float, 3> A,
                        cuda_tool::CDenseVectorView<Float>   x,
                        Float                                b,
                        cuda_tool::DenseVectorView<Float>    y)

{
    if(b != 0)
    {
        int n = y.size();
        if(n > 0)
        {
            Spmv_scale_y_kernel<<<cuda_tool::best_grid_dim(n, Spmv_scale_y_kernel), cuda_tool::best_block_dim(Spmv_scale_y_kernel), 0, nullptr>>>(
                b, y, n);
        }
    }
    else
    {
        cuda_tool::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int block_dim   = 256;
    int           block_count = (A.triplet_count() + block_dim - 1) / block_dim;
    int           triplet_count = A.triplet_count();

    if(triplet_count > 0)
    {
        Spmv_rbk_sym_spmv_kernel<<<block_count, block_dim, 0, nullptr>>>(a, A, x, y, triplet_count);
    }
}

void Spmv::rbk_sym_spmv_dot(Float                                a,
                            cuda_tool::CBCOOMatrixView<Float, 3> A,
                            cuda_tool::CDenseVectorView<Float>   x,
                            Float                                b,
                            cuda_tool::DenseVectorView<Float>    y,
                            cuda_tool::VarView<Float>            d_dot,
                            cuda_tool::CDense<IndexT> d_triplet_count,
                            SizeT                     triplet_capacity,
                            cudaStream_t              stream)
{
    if(b != 0)
    {
        int n = y.size();
        if(n > 0)
        {
            Spmv_scale_y_kernel<<<cuda_tool::best_grid_dim(n, Spmv_scale_y_kernel), cuda_tool::best_block_dim(Spmv_scale_y_kernel), 0, stream>>>(
                b, y, n);
        }
    }
    else if(!spmv_env().pcg_fuse_ap_zero)
    {
        cuda_tool::BufferLaunch(stream).fill<Float>(y.buffer_view(), 0);
    }

    const SpmvEnv& env = spmv_env();

    if(!env.pcg_fuse_scalar)
        cudaMemsetAsync(d_dot.data(), 0, sizeof(Float), stream);

    // grid covers the reserved capacity: blocks beyond the current
    // (device-side) count exit with zero work, so the launch shape need not
    // change when the count does
    constexpr int block_dim = 256;

    auto launch_triplet =
        [&](cuda_tool::DenseVectorView<Float> yy, cuda_tool::Dense<Float> dd)
    {
        int block_count = (int)((triplet_capacity + block_dim - 1) / block_dim);
        if(block_count > 0)
            Spmv_rbk_sym_spmv_dot_kernel<<<block_count, block_dim, 0, stream>>>(
                a, A, x, yy, dd, d_triplet_count, env.skip_idle_blocks);
    };

    auto launch_chunked = [&](int C, cuda_tool::DenseVectorView<Float> yy, Float* dd)
    {
        const SizeT per_block = (SizeT)block_dim * (SizeT)C;
        int block_count = (int)((triplet_capacity + per_block - 1) / per_block);
        if(block_count <= 0)
            return;
        const int*       rows = A.row_indices().data();
        const int*       cols = A.col_indices().data();
        const Matrix3x3* vals = A.values().data();
        const Float*     xp   = x.data();
        Float*           yp   = yy.data();
        const IndexT*    cnt  = d_triplet_count.data();
#define UIPC_SPMV_LAUNCH_CHUNKED(CC)                                                       \
    case CC:                                                                               \
    {                                                                                      \
        int bc = env.grid_stride ?                                                         \
                     std::min(block_count, resident_blocks<CC>()) :                        \
                     block_count;                                                          \
        Spmv_rbk_sym_spmv_dot_chunked_kernel<CC>                                           \
            <<<bc, block_dim, 0, stream>>>(a, rows, cols, vals, xp, yp, dd, cnt);          \
        break;                                                                             \
    }
        switch(C)
        {
            default:
                UIPC_SPMV_LAUNCH_CHUNKED(1)
                UIPC_SPMV_LAUNCH_CHUNKED(2)
                UIPC_SPMV_LAUNCH_CHUNKED(4)
                UIPC_SPMV_LAUNCH_CHUNKED(8)
                UIPC_SPMV_LAUNCH_CHUNKED(16)
        }
#undef UIPC_SPMV_LAUNCH_CHUNKED
    };

    if(env.chunk > 0)
        launch_chunked(env.chunk, y, d_dot.data());
    else
        launch_triplet(y, d_dot.viewer());

    // verify: the per-triplet kernel as the reference into scratch on the
    // same stream (captured with the primary), then the max-diff kernel
    if(env.verify && m_verify_acc.size() == 5 && m_verify_y.size() >= (size_t)y.size())
    {
        const int                         n = y.size();
        cuda_tool::DenseVectorView<Float> y_ref{m_verify_y.data(), 0, n, n};
        cuda_tool::BufferLaunch(stream).fill<Float>(y_ref.buffer_view(), 0);
        cudaMemsetAsync(m_verify_dot.data(), 0, sizeof(Float), stream);
        launch_triplet(y_ref, m_verify_dot.viewer());
        if(n > 0)
            Spmv_verify_kernel<<<(n + 255) / 256, 256, 0, stream>>>(y.data(),
                                                                    y_ref.data(),
                                                                    d_dot.data(),
                                                                    m_verify_dot.data(),
                                                                    m_verify_acc.data(),
                                                                    n);
    }

    // timing probes (results discarded): C = 1 variants into the scratch
    if(env.probe && m_verify_y.size() >= (size_t)y.size())
    {
        int block_count = (int)((triplet_capacity + block_dim - 1) / block_dim);
        if(block_count > 0)
        {
            const int*       rows = A.row_indices().data();
            const int*       cols = A.col_indices().data();
            const Matrix3x3* vals = A.values().data();
            const Float*     xp   = x.data();
            Float*           yp   = m_verify_y.data();
            Float*           dd   = m_verify_dot.data();
            const IndexT*    cnt  = d_triplet_count.data();
            Spmv_rbk_sym_spmv_dot_chunked_kernel<1, 1>
                <<<block_count, block_dim, 0, stream>>>(a, rows, cols, vals, xp, yp, dd, cnt);
            Spmv_rbk_sym_spmv_dot_chunked_kernel<1, 2>
                <<<block_count, block_dim, 0, stream>>>(a, rows, cols, vals, xp, yp, dd, cnt);
            Spmv_rbk_sym_spmv_dot_chunked_kernel<1, 3>
                <<<block_count, block_dim, 0, stream>>>(a, rows, cols, vals, xp, yp, dd, cnt);
        }
    }
}

void Spmv::verify_report(SizeT dof_count)
{
    const SpmvEnv& env = spmv_env();
    if(!env.verify && !env.probe)
        return;
    if(env.verify && m_verify_acc.size() == 5)
    {
        unsigned long long h[5];
        CUDA_TOOL_CHECK(cudaMemcpy(h, m_verify_acc.data(), sizeof(h), cudaMemcpyDeviceToHost));
        if(h[4] > 0)
        {
            auto f = [](unsigned long long b)
            { return __longlong_as_double_host(b); };
            double dy = f(h[0]), my = f(h[1]), dd = f(h[2]), md = f(h[3]);
            m_verify_launches += h[4];
            m_verify_max_rel_y = std::max(m_verify_max_rel_y, my > 0 ? dy / my : dy);
            m_verify_max_rel_dot = std::max(m_verify_max_rel_dot, md > 0 ? dd / md : dd);
            logger::warn("[SpmvVerify] chunk={} {} launches since the last assembly: max|dy| {:.3e} / max|y_ref| {:.3e} = {:.3e}; max|ddot| {:.3e} / |dot_ref| {:.3e} = {:.3e}; cumulative {} launches, max rel y {:.3e}, max rel dot {:.3e}",
                         env.chunk,
                         h[4],
                         dy,
                         my,
                         my > 0 ? dy / my : dy,
                         dd,
                         md,
                         md > 0 ? dd / md : dd,
                         m_verify_launches,
                         m_verify_max_rel_y,
                         m_verify_max_rel_dot);
            CUDA_TOOL_CHECK(cudaMemset(m_verify_acc.data(), 0, sizeof(h)));
        }
    }
    else if(env.verify)
    {
        m_verify_acc.resize(5);
        CUDA_TOOL_CHECK(cudaMemset(m_verify_acc.data(), 0, 5 * sizeof(unsigned long long)));
    }
    // the scratch pointers are baked into a captured PCG graph: allocate once,
    // generously, outside any capture (this runs before the solve)
    if(m_verify_y.size() == 0)
    {
        m_verify_y.resize(std::max<SizeT>(2 * dof_count, SizeT{1024}));
    }
    UIPC_ASSERT(m_verify_y.size() >= dof_count,
                "UIPC_SPMV_VERIFY: the DoF count grew past the verify scratch ({} > {}); not supported",
                dof_count,
                m_verify_y.size());
}

void Spmv::cpu_sym_spmv(Float                                a,
                        cuda_tool::CBCOOMatrixView<Float, 3> A_view,
                        cuda_tool::CDenseVectorView<Float>   x,
                        Float                                b,
                        cuda_tool::DenseVectorView<Float>    y)
{
    using BlockMatrix      = Matrix3x3;
    constexpr int BlockDim = 3;

    // Get matrix dimensions
    int total_block_rows  = A_view.total_rows();
    int total_block_cols  = A_view.total_cols();
    int total_scalar_rows = total_block_rows * BlockDim;
    int total_scalar_cols = total_block_cols * BlockDim;

    // Copy triplet data from device to host
    std::vector<int>         row_indices_host(A_view.triplet_count());
    std::vector<int>         col_indices_host(A_view.triplet_count());
    std::vector<BlockMatrix> values_host(A_view.triplet_count());

    A_view.row_indices().copy_to(row_indices_host.data());
    A_view.col_indices().copy_to(col_indices_host.data());
    A_view.values().copy_to(values_host.data());

    // Build Eigen SparseMatrix from symmetric block-sparse representation
    std::vector<Eigen::Triplet<Float>> triplets;
    triplets.reserve(A_view.triplet_count() * BlockDim * BlockDim * 2);  // Reserve for symmetric expansion

    for(size_t t = 0; t < A_view.triplet_count(); ++t)
    {
        int                block_i = row_indices_host[t];
        int                block_j = col_indices_host[t];
        const BlockMatrix& block   = values_host[t];

        // Convert block indices to scalar indices
        int scalar_i_base = block_i * BlockDim;
        int scalar_j_base = block_j * BlockDim;

        // Add all entries from the block (upper triangular part)
        for(int bi = 0; bi < BlockDim; ++bi)
        {
            for(int bj = 0; bj < BlockDim; ++bj)
            {
                Float value    = block(bi, bj);
                int   scalar_i = scalar_i_base + bi;
                int   scalar_j = scalar_j_base + bj;

                triplets.emplace_back(scalar_i, scalar_j, value);
            }
        }

        // Since matrix is symmetric at block level, also add transpose block (lower triangular part)
        if(block_i != block_j)
        {
            BlockMatrix block_transpose = block.transpose();
            for(int bi = 0; bi < BlockDim; ++bi)
            {
                for(int bj = 0; bj < BlockDim; ++bj)
                {
                    Float value  = block_transpose(bi, bj);
                    int scalar_i = scalar_j_base + bi;  // Swapped base indices
                    int scalar_j = scalar_i_base + bj;

                    triplets.emplace_back(scalar_i, scalar_j, value);
                }
            }
        }
    }

    // Build Eigen SparseMatrix
    Eigen::SparseMatrix<Float> A_sparse(total_scalar_rows, total_scalar_cols);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());
    A_sparse.makeCompressed();

    // Copy vectors from device to host
    Eigen::VectorX<Float> x_host(x.size());
    Eigen::VectorX<Float> y_host(y.size());

    x.buffer_view().copy_to(x_host.data());
    y.buffer_view().copy_to(y_host.data());

    // Compute y = a * A * x + b * y using Eigen sparse matrix-vector multiplication
    y_host = a * A_sparse * x_host + b * y_host;

    // Copy result back to device
    y.buffer_view().copy_from(y_host.data());
}
}  // namespace uipc::backend::cuda
