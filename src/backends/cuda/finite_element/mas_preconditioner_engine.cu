#include <finite_element/mas_preconditioner_engine.h>
#include <cstdlib>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <uipc/common/log.h>
#include <cuda_runtime.h>
#include <cub/warp/warp_reduce.cuh>
#include <fmt/format.h>
#include <uipc/common/json.h>
#include <fstream>
#include <vector>
#include <cstring>

namespace uipc::backend::cuda
{
// ============================================================================
// Local constants & type aliases
// ============================================================================
static constexpr int BANKSIZE = MASPreconditionerEngine::BANKSIZE;
static constexpr int DEFAULT_BLOCKSIZE = MASPreconditionerEngine::DEFAULT_BLOCKSIZE;
static constexpr int DEFAULT_WARPNUM = MASPreconditionerEngine::DEFAULT_WARPNUM;
static constexpr int SYM_BLOCK_COUNT = MASPreconditionerEngine::SYM_BLOCK_COUNT;

using ClusterMatrixSym  = MASPreconditionerEngine::ClusterMatrixSym;
using ClusterMatrixSymF = MASPreconditionerEngine::ClusterMatrixSymF;
using LevelTable        = MASPreconditionerEngine::LevelTable;
using Int2              = MASPreconditionerEngine::Int2;

// Device helper: bitmask with bits [0, lane_id) set
UIPC_GENERIC unsigned int lanemask_lt(int lane_id)
{
    return (1U << lane_id) - 1;
}

// Symmetric upper-triangle index for a BANKSIZE x BANKSIZE block
UIPC_GENERIC int sym_index(int row, int col)
{
    return BANKSIZE * row - row * (row + 1) / 2 + col;
}

// Round n up to the nearest multiple of BANKSIZE
constexpr int bank_align(int n)
{
    return (n + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
}

// s09: threads per block of the cluster-matrix inversion sweep kernel
// (4 clusters of 48 columns; 96/144/192/240/288/384 measured 211/207/199/226/224/276 us
// per launch on the bunny's 1 281 clusters, 2070S).
static constexpr int MAS_INVERT_SWEEP_BLOCK = 192;

// ============================================================================
// Named kernels (replacements for the former lambda kernel launches)
// ============================================================================
namespace
{
    __global__ void MASPreconditionerEngine_build_connect_mask_L0_kernel(
        cuda_tool::CBufferView<unsigned int> neighbor_start,
        cuda_tool::BufferView<unsigned int>  neighbor_num,
        cuda_tool::BufferView<unsigned int>  neighbor_list,
        cuda_tool::BufferView<unsigned int>  fine_connect_mask,
        cuda_tool::CBufferView<int>          part_to_real,
        cuda_tool::CBufferView<int>          real_to_part,
        int                                  n)
    {
        using namespace cuda_tool;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid >= n)
            return;

        int bank_id = tid / BANKSIZE;
        int lane_id = tid % BANKSIZE;
        int idx     = part_to_real(tid);

        if(idx < 0)
            return;

        int          num_nbr     = neighbor_num(idx);
        unsigned int connect_msk = (1U << lane_id);
        int          nk          = 0;
        int          start_id    = neighbor_start(idx);

        for(int i = 0; i < num_nbr; i++)
        {
            int nbr_id      = neighbor_list(start_id + i);
            int nbr_part_id = real_to_part(nbr_id);
            if(nbr_part_id < 0)
            {
                continue;
            }
            int nbr_bank_id = nbr_part_id / BANKSIZE;
            if(bank_id == nbr_bank_id)
            {
                unsigned int nbr_lane = nbr_part_id % BANKSIZE;
                connect_msk |= (1U << nbr_lane);
            }
            else
            {
                neighbor_list(start_id + nk) = nbr_id;
                nk++;
            }
        }
        neighbor_num(idx)      = nk;
        fine_connect_mask(idx) = connect_msk;
    }

    __global__ void MASPreconditionerEngine_prepare_prefix_sum_L0_kernel(
        cuda_tool::BufferView<unsigned int> fine_connect_mask,
        cuda_tool::BufferView<int>          prefix_orig,
        cuda_tool::CBufferView<int>         part_to_real,
        int                                 N)
    {
        using namespace cuda_tool;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid >= N)
            return;

        int warp_id       = tid / BANKSIZE;
        int local_warp_id = threadIdx.x / BANKSIZE;
        int lane_id       = tid % BANKSIZE;
        int idx           = part_to_real(tid);

        __shared__ unsigned int cache_mask_s[DEFAULT_BLOCKSIZE];
        __shared__ int          prefix_sum_s[DEFAULT_WARPNUM];
        auto                    cache_mask = make_dense_1d(cache_mask_s);
        auto                    prefix_sum = make_dense_1d(prefix_sum_s);

        if(idx >= 0)
        {
            unsigned int connect_msk = fine_connect_mask(idx);
            if(lane_id == 0)
                prefix_sum(local_warp_id) = 0;
            cache_mask(threadIdx.x) = connect_msk;
            unsigned int visited    = (1U << lane_id);

            while(connect_msk != ~0U)
            {
                unsigned int todo = visited ^ connect_msk;
                if(!todo)
                    break;
                unsigned int next_visit = __ffs(todo) - 1;
                visited |= (1U << next_visit);
                connect_msk |= cache_mask(next_visit + local_warp_id * BANKSIZE);
            }

            fine_connect_mask(idx) = connect_msk;

            unsigned int elected_prefix = __popc(connect_msk & lanemask_lt(lane_id));
            if(elected_prefix == 0)
                atomicAdd(&prefix_sum(local_warp_id), 1);
            if(lane_id == 0)
                prefix_orig(warp_id) = prefix_sum(local_warp_id);
        }
    }

    __global__ void MASPreconditionerEngine_build_level1_kernel(
        cuda_tool::BufferView<Int2>          level_size,
        cuda_tool::BufferView<int>           coarse_table,
        cuda_tool::BufferView<int>           going_next_L0,
        cuda_tool::CBufferView<unsigned int> fine_connect_mask,
        cuda_tool::CBufferView<int>          prefix_sum_orig,
        cuda_tool::CBufferView<int>          prefix_orig,
        cuda_tool::CBufferView<int>          part_to_real,
        int                                  N,
        int                                  level1_begin)
    {
        using namespace cuda_tool;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid >= N)
            return;

        int warp_id       = tid / BANKSIZE;
        int local_warp_id = threadIdx.x / BANKSIZE;
        int lane_id       = tid % BANKSIZE;

        __shared__ unsigned int elected_mask_s[BANKSIZE];
        __shared__ unsigned int lane_prefix_s[BANKSIZE * BANKSIZE];
        auto                    elected_mask = make_dense_1d(elected_mask_s);
        auto                    lane_prefix  = make_dense_1d(lane_prefix_s);

        if(lane_id == 0)
            elected_mask(local_warp_id) = 0;

        if(tid == N - 1)
        {
            level_size(1).x = prefix_sum_orig(warp_id) + prefix_orig(warp_id);
            level_size(1).y = level1_begin;
        }

        int idx = part_to_real(tid);
        if(idx >= 0)
        {
            unsigned int conn_msk = fine_connect_mask(idx);
            unsigned int elected_prefix = __popc(conn_msk & lanemask_lt(lane_id));
            if(elected_prefix == 0)
                atomicOr(&elected_mask(local_warp_id), (1U << lane_id));

            lane_prefix(threadIdx.x) =
                __popc(elected_mask(local_warp_id) & lanemask_lt(lane_id));
            lane_prefix(threadIdx.x) += prefix_sum_orig(warp_id);

            unsigned int elected_lane = __ffs(conn_msk) - 1;
            unsigned int the_lane_prefix =
                lane_prefix(elected_lane + BANKSIZE * local_warp_id);

            coarse_table(idx)  = the_lane_prefix;
            going_next_L0(idx) = the_lane_prefix + level1_begin;
        }
    }

    __global__ void MASPreconditionerEngine_build_connect_mask_Lx_kernel(
        cuda_tool::CBufferView<unsigned int> neighbor_start,
        cuda_tool::BufferView<unsigned int>  neighbor_num,
        cuda_tool::BufferView<unsigned int>  neighbor_list,
        cuda_tool::CBufferView<int>          coarse_table,
        cuda_tool::BufferView<unsigned int>  next_connect_mask,
        cuda_tool::CBufferView<unsigned int> fine_connect_mask,
        cuda_tool::CBufferView<int>          part_to_real,
        int                                  level,
        int                                  vert_num,
        int                                  N)
    {
        using namespace cuda_tool;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid >= N)
            return;

        int local_warp_id = threadIdx.x / BANKSIZE;
        int lane_id       = tid % BANKSIZE;

        __shared__ int cache_msk_s[DEFAULT_BLOCKSIZE];
        auto           cache_msk = make_dense_1d(cache_msk_s);

        int idx = part_to_real(tid);
        if(idx < 0)
            return;

        unsigned int prefix_msk = fine_connect_mask(idx);
        unsigned int conn_msk   = 0;
        int          coarse_idx = coarse_table((level - 1) * vert_num + idx);
        if(coarse_idx < 0 || coarse_idx >= N)
        {
            return;
        }
        int kn       = neighbor_num(idx);
        int nk       = 0;
        int start_id = neighbor_start(idx);

        for(int i = 0; i < kn; i++)
        {
            unsigned int connect = neighbor_list(start_id + i);
            int coarse_connect = coarse_table((level - 1) * vert_num + connect);
            if(coarse_connect < 0 || coarse_connect >= N)
            {
                neighbor_list(start_id + nk) = connect;
                nk++;
                continue;
            }
            if(coarse_idx / BANKSIZE == coarse_connect / BANKSIZE)
            {
                conn_msk |= (1U << (coarse_connect % BANKSIZE));
            }
            else
            {
                neighbor_list(start_id + nk) = connect;
                nk++;
            }
        }
        neighbor_num(idx)      = nk;
        cache_msk(threadIdx.x) = 0;

        if(__popc(prefix_msk) == BANKSIZE)
        {
            atomicOr(&cache_msk(local_warp_id * BANKSIZE), static_cast<int>(conn_msk));
            conn_msk = static_cast<unsigned int>(cache_msk(local_warp_id * BANKSIZE));
        }
        else
        {
            unsigned int elected_lane = __ffs(prefix_msk) - 1;
            if(conn_msk)
                atomicOr(&cache_msk(local_warp_id * BANKSIZE + elected_lane),
                         static_cast<int>(conn_msk));
            conn_msk = static_cast<unsigned int>(
                cache_msk(local_warp_id * BANKSIZE + elected_lane));
        }

        unsigned int elected_prefix = __popc(prefix_msk & lanemask_lt(lane_id));
        if(conn_msk && elected_prefix == 0)
            atomicOr(&next_connect_mask(coarse_idx), conn_msk);
    }

    __global__ void MASPreconditionerEngine_next_level_cluster_kernel(
        cuda_tool::BufferView<unsigned int> next_connect_mask,
        cuda_tool::BufferView<unsigned int> next_prefix,
        int                                 N)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)
            return;

        int local_warp_id = threadIdx.x / BANKSIZE;
        int lane_id       = idx % BANKSIZE;

        __shared__ int          prefix_sum_raw[DEFAULT_WARPNUM];
        __shared__ unsigned int cached_msk_raw[DEFAULT_BLOCKSIZE];
        auto                    prefix_sum_s = make_dense_1d(prefix_sum_raw);
        auto                    cached_msk   = make_dense_1d(cached_msk_raw);

        if(lane_id == 0)
            prefix_sum_s(local_warp_id) = 0;

        unsigned int conn_msk   = (1U << lane_id) | next_connect_mask(idx);
        cached_msk(threadIdx.x) = conn_msk;
        unsigned int visited    = (1U << lane_id);

        while(true)
        {
            unsigned int todo = visited ^ conn_msk;
            if(!todo)
                break;
            unsigned int next_visit = __ffs(todo) - 1;
            visited |= (1U << next_visit);
            conn_msk |= cached_msk(next_visit + local_warp_id * BANKSIZE);
        }

        next_connect_mask(idx)      = conn_msk;
        unsigned int elected_prefix = __popc(conn_msk & lanemask_lt(lane_id));
        if(elected_prefix == 0)
            atomicAdd(&prefix_sum_s(local_warp_id), 1);
        if(lane_id == 0)
            next_prefix(idx / BANKSIZE) = prefix_sum_s(local_warp_id);
    }

    __global__ void MASPreconditionerEngine_prefix_sum_Lx_kernel(
        cuda_tool::BufferView<Int2>          level_size_ptr,
        cuda_tool::CBufferView<unsigned int> next_prefix,
        cuda_tool::CBufferView<unsigned int> next_prefix_sum,
        cuda_tool::BufferView<unsigned int>  next_connect_mask,
        cuda_tool::BufferView<int>           going_next_level,
        int                                  level,
        int                                  next_level_begin,
        int                                  N)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)
            return;

        int warp_id       = idx / BANKSIZE;
        int local_warp_id = threadIdx.x / BANKSIZE;
        int lane_id       = idx % BANKSIZE;

        __shared__ unsigned int elected_mask_s[BANKSIZE];
        __shared__ unsigned int lane_prefix_s[BANKSIZE * BANKSIZE];
        auto                    elected_mask = make_dense_1d(elected_mask_s);
        auto                    lane_prefix  = make_dense_1d(lane_prefix_s);

        if(lane_id == 0)
            elected_mask(local_warp_id) = 0;

        if(idx == N - 1)
        {
            level_size_ptr(level + 1).x = next_prefix_sum(warp_id) + next_prefix(warp_id);
            level_size_ptr(level + 1).y = next_level_begin;
        }

        unsigned int conn_msk       = next_connect_mask(idx);
        unsigned int elected_prefix = __popc(conn_msk & lanemask_lt(lane_id));
        if(elected_prefix == 0)
            atomicOr(&elected_mask(local_warp_id), (1U << lane_id));

        lane_prefix(threadIdx.x) =
            __popc(elected_mask(local_warp_id) & lanemask_lt(lane_id));
        lane_prefix(threadIdx.x) += next_prefix_sum(warp_id);

        unsigned int elected_lane = __ffs(conn_msk) - 1;
        unsigned int the_lane_prefix = lane_prefix(elected_lane + BANKSIZE * local_warp_id);

        next_connect_mask(idx) = the_lane_prefix;
        going_next_level(idx)  = the_lane_prefix + next_level_begin;
    }

    __global__ void MASPreconditionerEngine_compute_next_level_kernel(
        cuda_tool::BufferView<int>           coarse_table,
        cuda_tool::CBufferView<unsigned int> next_connect_mask,
        int                                  level,
        int                                  N)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)
            return;

        int next = coarse_table((level - 1) * N + idx);
        if(next < 0 || next >= N)
        {
            coarse_table(level * N + idx) = -1;
            return;
        }
        coarse_table(level * N + idx) = next_connect_mask(next);
    }

    __global__ void MASPreconditionerEngine_aggregation_kernel_kernel(
        cuda_tool::BufferView<LevelTable> coarse_table,
        cuda_tool::CBufferView<int>       going_next,
        cuda_tool::CBufferView<Int2>      level_size,
        int                               level_num,
        int                               n)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;

        int        current_id = idx;
        LevelTable ctable;

        int first = going_next(current_id);
        if(first >= 0)
        {
            UIPC_KERNEL_ASSERT(first >= level_size(1).y && first < level_size(2).y,
                               "aggregation: going_next[%d]=%d not in level 1 [%d, %d)",
                               current_id,
                               first,
                               level_size(1).y,
                               level_size(2).y);

            current_id      = first;
            ctable.index[0] = first;

            for(int l = 1; l < level_num - 1; l++)
            {
                int next = going_next(current_id);

                UIPC_KERNEL_ASSERT(next >= 0,
                                   "aggregation: partitioned vertex %d has "
                                   "going_next=%d at level %d (expected >= 0)",
                                   idx,
                                   next,
                                   l + 1);

                UIPC_KERNEL_ASSERT(next >= level_size(l + 1).y
                                       && next < level_size(l + 2).y,
                                   "aggregation: going_next[%d]=%d not in level %d [%d, %d)",
                                   current_id,
                                   next,
                                   l + 1,
                                   level_size(l + 1).y,
                                   level_size(l + 2).y);

                current_id      = next;
                ctable.index[l] = next;
            }
        }

        coarse_table(idx) = ctable;
    }


    // ---- s08: warp-aggregated 3x3 FP64 atomic add --------------------------
    // The scatter kernel below adds one 3x3 double block per (triplet, level)
    // into cluster_hess.  At the coarse levels of the MAS hierarchy there are
    // only a handful of clusters (bunny: 75 / 5 / 1 / 1 clusters at levels
    // 1..4), so tens of thousands of threads hammer the same ~136 block
    // addresses with 9 same-address FP64 atomics each.  Lanes of a warp that
    // target the same destination block are combined first (MATCH.ANY +
    // shuffle reduction, Westphal's peer reduction), so only one lane per
    // distinct destination issues the 9 atomics.  Same set of additions,
    // different summation order -> rounding-level (the old path was already
    // nondeterministic through its atomics).
    //
    // Every lane of the warp must call this (a lane with nothing to add passes
    // dst == nullptr); the collectives use the full warp mask.
    __device__ inline void mas_warp_agg_atomic_add_3x3(double* dst, double v[9])
    {
        constexpr unsigned FULL = 0xffffffffu;

        int lane = threadIdx.x & 31;
        // Idle lanes get a per-lane unique key so that they do not form one
        // large peer group and force extra reduction rounds.
        unsigned long long key = dst ? reinterpret_cast<unsigned long long>(dst) :
                                       (unsigned long long)(lane + 1);

        unsigned peers = __match_any_sync(FULL, key);

        int      rel_pos = __popc(peers & ((1u << lane) - 1u));
        bool     leader  = (rel_pos == 0);
        unsigned rest = peers & (0xfffffffeu << lane);  // strictly higher peers
        int      counter = rel_pos;

        while(__any_sync(FULL, rest != 0u))
        {
            int    next = __ffs(rest);  // 1-based lane index, 0 if none left
            double t[9];
#pragma unroll
            for(int k = 0; k < 9; ++k)
                t[k] = __shfl_sync(FULL, v[k], next - 1);
            if(next)
            {
#pragma unroll
                for(int k = 0; k < 9; ++k)
                    v[k] += t[k];
            }
            unsigned done = (unsigned)(counter & 1);
            rest &= ~__ballot_sync(FULL, done);
            counter >>= 1;
        }

        if(dst && leader)
        {
#pragma unroll
            for(int k = 0; k < 9; ++k)
                atomicAdd(dst + k, v[k]);
        }
    }

    // Warp-aggregated variant of the pass-1 scatter.  Structurally identical to
    // the reference kernel below, but no lane ever returns early (an inactive
    // lane carries dst == nullptr) so that the warp stays converged for the
    // aggregation collectives.
    __global__ void MASPreconditionerEngine_scatter_hessian_to_clusters_k1_agg_kernel(
        int                                     offset,
        int                                     level_num,
        cuda_tool::CBufferView<int>             going_next,
        cuda_tool::CBufferView<Int2>            level_size,
        cuda_tool::BufferView<ClusterMatrixSym> cluster_hess,
        cuda_tool::CBufferView<int>             real_to_part,
        cuda_tool::CBufferView<Eigen::Matrix3d> triplet_values,
        cuda_tool::CBufferView<int>             row_ids,
        cuda_tool::CBufferView<int>             col_ids,
        int                                     total_nodes,
        int                                     n)
    {
        using namespace cuda_tool;

        int I = blockIdx.x * blockDim.x + threadIdx.x;

        bool            live     = (I < n);
        int             row_real = 0, col_real = 0;
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

        if(live)
        {
            row_real = row_ids(I) - offset;
            col_real = col_ids(I) - offset;
            H        = triplet_values(I);
            if(row_real < 0 || row_real >= total_nodes || col_real < 0 || col_real >= total_nodes)
                live = false;
        }

        int vert_row = -1, vert_col = -1;
        if(live)
        {
            vert_col = real_to_part(col_real);
            vert_row = real_to_part(row_real);
            if(vert_col < 0 || vert_row < 0)
                live = false;
        }

        const double* Hd = H.data();  // column major: (i,j) -> Hd[3*j + i]

        // ---- level 0 ----
        bool    fine_hit = live && (vert_col / BANKSIZE == vert_row / BANKSIZE);
        double* dst      = nullptr;
        double  v[9];
#pragma unroll
        for(int k = 0; k < 9; ++k)
            v[k] = 0.0;

        if(fine_hit)
        {
            int cluster_id = vert_col / BANKSIZE;
            if(vert_col >= vert_row)
            {
                int si = sym_index(vert_row % BANKSIZE, vert_col % BANKSIZE);
                dst    = cluster_hess(cluster_id).M[si].data();
#pragma unroll
                for(int k = 0; k < 9; ++k)
                    v[k] = Hd[k];
            }
            else
            {
                int si = sym_index(vert_col % BANKSIZE, vert_row % BANKSIZE);
                dst    = cluster_hess(cluster_id).M[si].data();
#pragma unroll
                for(int i = 0; i < 3; ++i)
#pragma unroll
                    for(int j = 0; j < 3; ++j)
                        v[3 * j + i] = Hd[3 * i + j];  // H^T
            }
        }
        mas_warp_agg_atomic_add_3x3(dst, v);

        // ---- coarse levels ----
        bool walk = live && !fine_hit;
        for(int level = 1; level <= level_num - 1; ++level)
        {
            if(walk)
            {
                if(level == 1)
                {
                    vert_col = going_next(col_real);
                    vert_row = going_next(row_real);
                }
                else
                {
                    vert_col = going_next(vert_col);
                    vert_row = going_next(vert_row);
                }
                if(vert_col < 0 || vert_row < 0)
                    walk = false;
            }

            if(walk)
            {
                UIPC_KERNEL_ASSERT(vert_col >= level_size(level).y
                                       && vert_col < level_size(level + 1).y,
                                   "scatter P1: vert_col=%d not in level %d [%d, %d)",
                                   vert_col,
                                   level,
                                   level_size(level).y,
                                   level_size(level + 1).y);
                UIPC_KERNEL_ASSERT(vert_row >= level_size(level).y
                                       && vert_row < level_size(level + 1).y,
                                   "scatter P1: vert_row=%d not in level %d [%d, %d)",
                                   vert_row,
                                   level,
                                   level_size(level).y,
                                   level_size(level + 1).y);
            }

            double* ldst = nullptr;
            double  lv[9];
#pragma unroll
            for(int k = 0; k < 9; ++k)
                lv[k] = 0.0;

            if(walk && vert_col / BANKSIZE == vert_row / BANKSIZE)
            {
                int cluster_id = vert_col / BANKSIZE;
                if(vert_col >= vert_row)
                {
                    int si = sym_index(vert_row % BANKSIZE, vert_col % BANKSIZE);
                    ldst = cluster_hess(cluster_id).M[si].data();
                    if(vert_col == vert_row)
                    {
#pragma unroll
                        for(int i = 0; i < 3; ++i)
#pragma unroll
                            for(int j = 0; j < 3; ++j)
                                lv[3 * j + i] = Hd[3 * j + i] + Hd[3 * i + j];
                    }
                    else
                    {
#pragma unroll
                        for(int k = 0; k < 9; ++k)
                            lv[k] = Hd[k];
                    }
                }
                else
                {
                    int si = sym_index(vert_col % BANKSIZE, vert_row % BANKSIZE);
                    ldst = cluster_hess(cluster_id).M[si].data();
#pragma unroll
                    for(int i = 0; i < 3; ++i)
#pragma unroll
                        for(int j = 0; j < 3; ++j)
                            lv[3 * j + i] = Hd[3 * i + j];  // H^T
                }
            }
            mas_warp_agg_atomic_add_3x3(ldst, lv);
        }
    }

    __global__ void MASPreconditionerEngine_scatter_hessian_to_clusters_k1_kernel(
        int                                     offset,
        int                                     level_num,
        cuda_tool::CBufferView<int>             going_next,
        cuda_tool::CBufferView<Int2>            level_size,
        cuda_tool::BufferView<ClusterMatrixSym> cluster_hess,
        cuda_tool::CBufferView<int>             real_to_part,
        cuda_tool::CBufferView<Eigen::Matrix3d> triplet_values,
        cuda_tool::CBufferView<int>             row_ids,
        cuda_tool::CBufferView<int>             col_ids,
        int                                     total_nodes,
        int                                     n)
    {
        using namespace cuda_tool;

        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;

        int  row_real = row_ids(I) - offset;
        int  col_real = col_ids(I) - offset;
        auto H        = triplet_values(I);

        if(row_real < 0 || row_real >= total_nodes || col_real < 0 || col_real >= total_nodes)
            return;

        int vert_col = real_to_part(col_real);
        int vert_row = real_to_part(row_real);

        if(vert_col < 0 || vert_row < 0)
            return;

        if(vert_col / BANKSIZE == vert_row / BANKSIZE)
        {
            int cluster_id = vert_col / BANKSIZE;
            if(vert_col >= vert_row)
            {
                int si = sym_index(vert_row % BANKSIZE, vert_col % BANKSIZE);
                cuda_tool::eigen::atomic_add(cluster_hess(cluster_id).M[si], H);
            }
            else
            {
                Eigen::Matrix3d Ht = H.transpose();
                int si = sym_index(vert_col % BANKSIZE, vert_row % BANKSIZE);
                cuda_tool::eigen::atomic_add(cluster_hess(cluster_id).M[si], Ht);
            }
        }
        else
        {
            // Walk up ALL levels and add at every level where both endpoints
            // land in the same bank. This implements Galerkin H_L = R_L H R_L^T
            // independently per level (an entry can contribute to L1, L2, L3, ...
            // wherever its ancestors' banks merge).
            int level = 0;
            while(level < level_num - 1)
            {
                level++;
                if(level == 1)
                {
                    vert_col = going_next(col_real);
                    vert_row = going_next(row_real);
                }
                else
                {
                    vert_col = going_next(vert_col);
                    vert_row = going_next(vert_row);
                }
                if(vert_col < 0 || vert_row < 0)
                    return;

                UIPC_KERNEL_ASSERT(vert_col >= level_size(level).y
                                       && vert_col < level_size(level + 1).y,
                                   "scatter P1: vert_col=%d not in level %d [%d, %d)",
                                   vert_col,
                                   level,
                                   level_size(level).y,
                                   level_size(level + 1).y);
                UIPC_KERNEL_ASSERT(vert_row >= level_size(level).y
                                       && vert_row < level_size(level + 1).y,
                                   "scatter P1: vert_row=%d not in level %d [%d, %d)",
                                   vert_row,
                                   level,
                                   level_size(level).y,
                                   level_size(level + 1).y);

                if(vert_col / BANKSIZE == vert_row / BANKSIZE)
                {
                    int cluster_id = vert_col / BANKSIZE;
                    if(vert_col >= vert_row)
                    {
                        int si = sym_index(vert_row % BANKSIZE, vert_col % BANKSIZE);
                        for(int i = 0; i < 3; i++)
                            for(int j = 0; j < 3; j++)
                            {
                                atomicAdd(&(cluster_hess(cluster_id).M[si](i, j)),
                                          H(i, j));
                                if(vert_col == vert_row)
                                    atomicAdd(&(cluster_hess(cluster_id).M[si](i, j)),
                                              H(j, i));
                            }
                    }
                    else
                    {
                        int si = sym_index(vert_col % BANKSIZE, vert_row % BANKSIZE);
                        for(int i = 0; i < 3; i++)
                            for(int j = 0; j < 3; j++)
                                atomicAdd(&(cluster_hess(cluster_id).M[si](i, j)),
                                          H(j, i));
                    }
                    // NO break: keep walking up to coarser levels where
                    // banks may also merge. Each level's Galerkin restriction
                    // is independent (R_L H R_L^T).
                }
            }
        }
    }

    __global__ void MASPreconditionerEngine_scatter_hessian_to_clusters_k2_kernel(
        int                                     level_num,
        cuda_tool::CBufferView<int>             going_next,
        cuda_tool::CBufferView<Int2>            level_size,
        cuda_tool::BufferView<ClusterMatrixSym> cluster_hess,
        cuda_tool::CBufferView<int>             part_to_real,
        cuda_tool::CBufferView<unsigned int>    fine_connect,
        cuda_tool::CBufferView<int>             prefix_orig,
        int                                     map_nodes,
        int                                     n)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;

        int cluster_stride = BANKSIZE * BANKSIZE;
        int cluster_id     = idx / cluster_stride;
        int local_row      = (idx % cluster_stride) / BANKSIZE;
        int local_col      = (idx % cluster_stride) % BANKSIZE;

        int global_row = cluster_id * BANKSIZE + local_row;
        int global_col = cluster_id * BANKSIZE + local_col;
        if(global_row >= map_nodes || global_col >= map_nodes)
            return;

        int rdx = part_to_real(global_row);
        int cdx = part_to_real(global_col);

        __shared__ int prefix;
        if(threadIdx.x == 0)
            prefix = prefix_orig(cluster_id);
        __syncthreads();

        Eigen::Matrix3d mat3;
        if(local_col >= local_row)
        {
            int si = sym_index(local_row, local_col);
            mat3   = cluster_hess(cluster_id).M[si];
        }
        else
        {
            int si = sym_index(local_col, local_row);
            mat3   = cluster_hess(cluster_id).M[si].transpose();
        }

        // Do NOT early-return for invalid lanes — downstream code has warp-
        // scope collectives (cub::WarpReduce, atomicAdd chain) that require
        // every lane in the hw warp to be active. Instead, zero out mat3 so
        // invalid lanes contribute nothing to the reduction.
        bool invalid = (rdx < 0) || (cdx < 0);
        if(invalid)
            mat3.setZero();

        if(prefix == 1)
        {
            // Full 32-lane reduction per hw warp; mat3 for invalid lanes = 0.
            using WarpReduceD = cub::WarpReduce<double>;
            __shared__ typename WarpReduceD::TempStorage temp_reduce_d[BANKSIZE * BANKSIZE / 32];
            int hw_warp = threadIdx.x / 32;

            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 3; j++)
                    mat3(i, j) = WarpReduceD(temp_reduce_d[hw_warp]).Sum(mat3(i, j));

            // Lane 0 of each hw warp writes the warp's sum to coarser levels,
            // but only if its own lane was valid (rdx >= 0).
            if((threadIdx.x & 0x1f) == 0 && !invalid)
            {
                int level   = 0;
                int next_id = going_next(rdx);
                while(next_id >= 0 && level < level_num - 1)
                {
                    level++;

                    UIPC_KERNEL_ASSERT(next_id >= level_size(level).y
                                           && next_id < level_size(level + 1).y,
                                       "scatter P2 diag: next_id=%d not in level %d [%d, %d)",
                                       next_id,
                                       level,
                                       level_size(level).y,
                                       level_size(level + 1).y);

                    int cid = next_id / BANKSIZE;
                    int bv  = next_id % BANKSIZE;
                    int si  = sym_index(bv, bv);
                    for(int i = 0; i < 3; i++)
                        for(int j = 0; j < 3; j++)
                            atomicAdd(&(cluster_hess(cid).M[si](i, j)), mat3(i, j));
                    next_id = going_next(next_id);
                }
            }
        }
        else
        {
            // prefix > 1: each lane walks up independently, no warp collective.
            if(invalid)
                return;

            int level = 1;
            while(level <= level_num - 1)
            {
                rdx = going_next(rdx);
                cdx = going_next(cdx);
                if(rdx < 0 || cdx < 0)
                    return;

                UIPC_KERNEL_ASSERT(rdx >= level_size(level).y
                                       && rdx < level_size(level + 1).y,
                                   "scatter P2: rdx=%d not in level %d [%d, %d)",
                                   rdx,
                                   level,
                                   level_size(level).y,
                                   level_size(level + 1).y);
                UIPC_KERNEL_ASSERT(cdx >= level_size(level).y
                                       && cdx < level_size(level + 1).y,
                                   "scatter P2: cdx=%d not in level %d [%d, %d)",
                                   cdx,
                                   level,
                                   level_size(level).y,
                                   level_size(level + 1).y);

                int cid = cdx / BANKSIZE;
                if(rdx / BANKSIZE == cdx / BANKSIZE)
                {
                    if(cdx >= rdx)
                    {
                        int si = sym_index(rdx % BANKSIZE, cdx % BANKSIZE);
                        for(int i = 0; i < 3; i++)
                            for(int j = 0; j < 3; j++)
                                atomicAdd(&(cluster_hess(cid).M[si](i, j)), mat3(i, j));
                    }
                }
                level++;
            }
        }
    }

    // Kept for A/B (UIPC_MAS_INVERT_SWEEP=0); the sweep kernel below replaces it.
    // NOTE (s09): this kernel has a cross-warp race — after the second
    // __syncthreads() every thread writes `s_mat(col, pivot)` (the pivot column)
    // while thread `pivot` reads and updates that same column in its own row
    // loop. In a standalone harness on random SPD matrices it corrupts ~0.5 % of
    // the cluster inverses non-deterministically; in situ on the benchmark
    // scenes it happens to be benign (UIPC_MAS_INVERT_VERIFY=2 reports
    // bit-identical output run to run). The sweep kernel has no such hazard.
    //
    // The inversion runs in float: the inverses are stored in float anyway
    // (ClusterMatrixSymF, matching GIPC), and float halves the shared-memory
    // footprint per cluster (9.4 KB instead of 18.8 KB), doubling resident
    // blocks per SM. This is a preconditioner - the PCG solve still converges
    // to its double-precision residual tolerance; only the iteration count
    // can change (measured unchanged on the benchmark scenes).
    __global__ void MASPreconditionerEngine_invert_cluster_matrices_kernel(
        cuda_tool::BufferView<ClusterMatrixSymF> cluster_inv,
        cuda_tool::CBufferView<ClusterMatrixSym> cluster_hess,
        int                                      total_threads)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= total_threads)
            return;

        constexpr int MAT_DIM = BANKSIZE * 3;  // 48

        int mat_id       = idx / MAT_DIM;
        int col          = idx % MAT_DIM;
        int block_mat_id = threadIdx.x / MAT_DIM;

        __shared__ float s_mat_raw[32 / BANKSIZE][MAT_DIM][MAT_DIM];
        __shared__ float s_col_raw[32 / BANKSIZE][MAT_DIM];
        auto s_mat = make_dense_2d(&s_mat_raw[block_mat_id][0][0], MAT_DIM, MAT_DIM);
        auto s_col = make_dense_1d(&s_col_raw[block_mat_id][0], MAT_DIM);

        for(int row = 0; row < MAT_DIM; row++)
        {
            int   node_row = row / 3;
            int   node_col = col / 3;
            float v;
            if(node_col >= node_row)
            {
                int si = sym_index(node_row, node_col);
                v = static_cast<float>(cluster_hess(mat_id).M[si](row % 3, col % 3));
            }
            else
            {
                int si = sym_index(node_col, node_row);
                v = static_cast<float>(cluster_hess(mat_id).M[si](col % 3, row % 3));
            }
            s_mat(row, col) = v;
            if(row == col && v == 0.0f)
                s_mat(row, col) = 1.0f;
        }

        for(int pivot = 0; pivot < MAT_DIM; pivot++)
        {
            __syncthreads();
            float pivot_val = s_mat(pivot, pivot);
            s_col(col)      = s_mat(col, pivot);
            __syncthreads();

            s_mat(col, pivot) = (col == pivot) ? 1.0f : 0.0f;

            s_mat(pivot, col) /= pivot_val;

            for(int row = 0; row < MAT_DIM; row++)
            {
                if(row != pivot)
                {
                    float factor = -s_col(row);
                    s_mat(row, col) += factor * s_mat(pivot, col);
                }
            }
        }
        __syncthreads();

        if(col % 3 < 2)
            s_mat(col + 1, col) = s_mat(col, col + 1);
        else
            s_mat(col, col - 2) = s_mat(col - 2, col);
        __syncthreads();

        for(int row = 0; row < MAT_DIM; row++)
        {
            int node_row = row / 3;
            int node_col = col / 3;
            if(node_col >= node_row)
            {
                int si = sym_index(node_row, node_col);
                cluster_inv(mat_id).M[si](row % 3, col % 3) =
                    static_cast<float>(s_mat(row, col));
            }
        }
    }

    // s09: register-column Gauss-Jordan (symmetric sweep) inverse.
    //
    // The old kernel keeps the whole 48x48 working matrix in shared memory
    // (18 816 B per 96-thread block -> 3 blocks = 9 warps per SM) and executes
    // 48 pivots x 48 rows of `s_mat(row,col) += factor * s_mat(pivot,col)` with
    // four shared accesses per FMA. Since every thread owns one *column*, the
    // rank-1 update touches only that thread's own column: the only value it
    // needs from another thread is the pivot column. The cluster matrix is
    // symmetric, and the Gauss-Jordan *sweep* operator
    //     a_pp <- -1/a_pp ; a_ip, a_pj <- a_ip/a_pp, a_pj/a_pp ;
    //     a_ij <- a_ij - a_ip a_pj / a_pp
    // preserves that symmetry (sweeping all pivots yields -A^{-1}), so the
    // pivot column equals the pivot row and every thread can publish its single
    // element A(p,col) instead of one thread publishing 48. That leaves one
    // 48-float shared row per cluster (double buffered, one __syncthreads per
    // pivot) and keeps the column in registers: one shared broadcast load and
    // one FFMA per element.
    //
    // The dynamic index c[p] (read the element on the pivot row, write it back
    // after scaling) is done with a switch over the block-uniform pivot index
    // so that the register array stays a register array (a predicated 48-way
    // select costs 2 extra ops per element; measured 230 vs 210 us).
    //
    // Rounding: the arithmetic per entry is the same (fmaf(-u_i, a_pj/a_pp, a_ij),
    // reciprocal-multiply on the pivot column) but u_i is read from the pivot
    // *row* instead of the pivot *column*; those two differ in the last bits
    // because a*(b/c) != b*(a/c), so the result is rounding-level equal, not
    // bit-identical (measured: max|diff| / max|ref| = 2.7e-7 over 1 281 random
    // SPD clusters, and 1e-7..1e-6 in situ, see UIPC_MAS_INVERT_VERIFY).
#define MAS_GJ_R4(M, b) M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3)
#define MAS_GJ_R16(M, b)                                                       \
    MAS_GJ_R4(M, (b) + 0) MAS_GJ_R4(M, (b) + 4) MAS_GJ_R4(M, (b) + 8)          \
        MAS_GJ_R4(M, (b) + 12)
#define MAS_GJ_R48(M) MAS_GJ_R16(M, 0) MAS_GJ_R16(M, 16) MAS_GJ_R16(M, 32)
#define MAS_GJ_GET(k)                                                          \
    case(k): f = c[(k)]; break;
#define MAS_GJ_SET(k)                                                          \
    case(k): c[(k)] = nv; break;

    __global__ __launch_bounds__(MAS_INVERT_SWEEP_BLOCK, 3) void MASPreconditionerEngine_invert_cluster_matrices_sweep_kernel(
        cuda_tool::BufferView<ClusterMatrixSymF> cluster_inv,
        cuda_tool::CBufferView<ClusterMatrixSym> cluster_hess,
        int                                      total_threads)
    {
        using namespace cuda_tool;

        constexpr int MAT_DIM = BANKSIZE * 3;  // 48

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= total_threads)
            return;

        const int mat_id = idx / MAT_DIM;
        const int col    = idx % MAT_DIM;
        const int a      = col % 3;
        const int nc     = col / 3;
        const int bm     = threadIdx.x / MAT_DIM;

        // one double-buffered pivot row per cluster in the block
        __shared__ float s_row_raw[MAS_INVERT_SWEEP_BLOCK / (BANKSIZE * 3)][2][BANKSIZE * 3];

        float c[MAT_DIM];  // this thread's column of the working matrix
#pragma unroll
        for(int row = 0; row < MAT_DIM; row++)
        {
            const int node_row = row / 3;
            float     v;
            if(nc >= node_row)
                v = static_cast<float>(
                    cluster_hess(mat_id).M[sym_index(node_row, nc)](row % 3, a));
            else
                v = static_cast<float>(
                    cluster_hess(mat_id).M[sym_index(nc, node_row)](a, row % 3));
            if(row == col && v == 0.0f)
                v = 1.0f;
            c[row] = v;
        }

        for(int p = 0; p < MAT_DIM; ++p)
        {
            float f;
            switch(p)
            {
                MAS_GJ_R48(MAS_GJ_GET)
            }

            float* sr = &s_row_raw[bm][p & 1][0];
            sr[col]   = f;
            __syncthreads();

            const float akk = sr[p];
            float       nv;
            if(col == p)
            {
                const float inv = 1.0f / akk;
#pragma unroll
                for(int i = 0; i < MAT_DIM; ++i)
                    c[i] *= inv;
                nv = -inv;
            }
            else
            {
                const float s = f / akk;
#pragma unroll
                for(int i = 0; i < MAT_DIM; ++i)
                    c[i] = fmaf(-sr[i], s, c[i]);
                nv = s;
            }

            switch(p)
            {
                MAS_GJ_R48(MAS_GJ_SET)
            }
        }

        // the sweep produced -A^{-1}; store the upper-triangle blocks, and for
        // the diagonal block mirror the upper entries into the lower ones (the
        // old kernel did the same fix-up through shared memory).
#pragma unroll
        for(int row = 0; row < MAT_DIM; row++)
        {
            const int   node_row = row / 3;
            const int   b        = row % 3;
            const float v        = -c[row];
            if(node_row < nc)
            {
                cluster_inv(mat_id).M[sym_index(node_row, nc)](b, a) = v;
            }
            else if(node_row == nc)
            {
                const int si = sym_index(nc, nc);
                if(b <= a)
                    cluster_inv(mat_id).M[si](b, a) = v;
                if(b < a)
                    cluster_inv(mat_id).M[si](a, b) = v;
            }
        }
    }

#undef MAS_GJ_R4
#undef MAS_GJ_R16
#undef MAS_GJ_R48
#undef MAS_GJ_GET
#undef MAS_GJ_SET

    // s09 verification probe: max |diff| and max |ref| over the cluster inverses.
    __global__ void MASPreconditionerEngine_compare_cluster_inv_kernel(
        cuda_tool::CBufferView<ClusterMatrixSymF> a,
        cuda_tool::CBufferView<ClusterMatrixSymF> b,
        unsigned long long*                       out,  // [0] = max diff, [1] = max ref
        int                                       n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        int          blk = i / (SYM_BLOCK_COUNT * 9);
        int          rem = i % (SYM_BLOCK_COUNT * 9);
        const float* pa  = a(blk).M[rem / 9].data();
        const float* pb  = b(blk).M[rem / 9].data();
        double       va  = pa[rem % 9];
        double       vb  = pb[rem % 9];
        double       d   = fabs(va - vb);
        double       r   = fabs(vb);
        atomicMax(&out[0], (unsigned long long)__double_as_longlong(d));
        atomicMax(&out[1], (unsigned long long)__double_as_longlong(r));
    }

    __global__ void MASPreconditionerEngine_build_multi_level_R_kernel(
        cuda_tool::CDenseVectorView<Float>     R_view,
        cuda_tool::BufferView<Eigen::Vector3f> multi_lr,
        cuda_tool::CBufferView<int>            going_next,
        cuda_tool::CBufferView<Int2>           level_size,
        cuda_tool::CBufferView<int>            prefix_orig,
        cuda_tool::CBufferView<unsigned int>   fine_conn,
        cuda_tool::CBufferView<int>            part_to_real,
        cuda_tool::CDense<IndexT>              converged,
        int                                    level_num,
        int                                    N)
    {
        using namespace cuda_tool;

        if(*converged != 0)
            return;
        int pdx = blockIdx.x * blockDim.x + threadIdx.x;

        bool in_range = (pdx < N);
        int  idx      = in_range ? part_to_real(pdx) : -1;
        bool is_valid = (idx >= 0);

        Eigen::Vector3f r = Eigen::Vector3f::Zero();
        if(is_valid)
        {
            auto seg = R_view.segment<3>(3 * idx);
            r[0]     = static_cast<float>(seg(0));
            r[1]     = static_cast<float>(seg(1));
            r[2]     = static_cast<float>(seg(2));
        }

        int lane_id       = threadIdx.x % BANKSIZE;
        int local_warp_id = threadIdx.x / BANKSIZE;
        int global_warp   = pdx / BANKSIZE;

        if(in_range)
            multi_lr(pdx) = r;

        __shared__ float sum_residual_s[DEFAULT_BLOCKSIZE * 3];
        __shared__ int   prefix_sum_raw[DEFAULT_WARPNUM];
        auto             sum_residual = make_dense_1d(sum_residual_s);
        auto             prefix_sum_v = make_dense_1d(prefix_sum_raw);

        if(lane_id == 0)
            prefix_sum_v(local_warp_id) = in_range ? prefix_orig(global_warp) : 0;

        unsigned int connect_msk = is_valid ? fine_conn(idx) : 0U;

        if(prefix_sum_v(local_warp_id) == 1)
        {
            using WarpReduceF = cub::WarpReduce<float, BANKSIZE>;
            __shared__ typename WarpReduceF::TempStorage temp_reduce_f[DEFAULT_WARPNUM];

            r[0] = WarpReduceF(temp_reduce_f[local_warp_id]).Sum(r[0]);
            r[1] = WarpReduceF(temp_reduce_f[local_warp_id]).Sum(r[1]);
            r[2] = WarpReduceF(temp_reduce_f[local_warp_id]).Sum(r[2]);

            if(lane_id == 0 && is_valid)
            {
                int cur = idx;
                for(int l = 0; l < level_num - 1; l++)
                {
                    cur = going_next(cur);

                    UIPC_KERNEL_ASSERT(cur >= level_size(l + 1).y
                                           && cur < level_size(l + 2).y,
                                       "build_R: cur=%d not in level %d [%d, %d)",
                                       cur,
                                       l + 1,
                                       level_size(l + 1).y,
                                       level_size(l + 2).y);

                    atomicAdd(&(multi_lr(cur)[0]), r[0]);
                    atomicAdd(&(multi_lr(cur)[1]), r[1]);
                    atomicAdd(&(multi_lr(cur)[2]), r[2]);
                }
            }
        }
        else if(is_valid)
        {
            int elected_lane = __ffs(connect_msk) - 1;

            sum_residual(threadIdx.x)                         = 0;
            sum_residual(threadIdx.x + DEFAULT_BLOCKSIZE)     = 0;
            sum_residual(threadIdx.x + 2 * DEFAULT_BLOCKSIZE) = 0;

            atomicAdd(&sum_residual(local_warp_id * BANKSIZE + elected_lane), r[0]);
            atomicAdd(&sum_residual(local_warp_id * BANKSIZE + elected_lane + DEFAULT_BLOCKSIZE),
                      r[1]);
            atomicAdd(&sum_residual(local_warp_id * BANKSIZE + elected_lane + 2 * DEFAULT_BLOCKSIZE),
                      r[2]);

            unsigned int elected_prefix = __popc(connect_msk & lanemask_lt(lane_id));
            if(elected_prefix == 0)
            {
                int cur = idx;
                for(int l = 0; l < level_num - 1; l++)
                {
                    cur = going_next(cur);

                    UIPC_KERNEL_ASSERT(cur >= level_size(l + 1).y
                                           && cur < level_size(l + 2).y,
                                       "build_R: cur=%d not in level %d [%d, %d)",
                                       cur,
                                       l + 1,
                                       level_size(l + 1).y,
                                       level_size(l + 2).y);

                    atomicAdd(&(multi_lr(cur)[0]), sum_residual(threadIdx.x));
                    atomicAdd(&(multi_lr(cur)[1]),
                              sum_residual(threadIdx.x + DEFAULT_BLOCKSIZE));
                    atomicAdd(&(multi_lr(cur)[2]),
                              sum_residual(threadIdx.x + DEFAULT_BLOCKSIZE * 2));
                }
            }
        }
    }

    __global__ void MASPreconditionerEngine_schwarz_local_solve_kernel(
        cuda_tool::CBufferView<ClusterMatrixSymF> cluster_inv,
        cuda_tool::CBufferView<Eigen::Vector3f>   multi_lr,
        cuda_tool::BufferView<float3>             multi_lz,
        cuda_tool::CDense<IndexT>                 converged,
        int                                       N)
    {
        using namespace cuda_tool;

        if(*converged != 0)
            return;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)
            return;

        constexpr int cluster_stride = BANKSIZE * BANKSIZE;

        int cluster_id = idx / cluster_stride;
        int local_row  = (idx % cluster_stride) / BANKSIZE;
        int local_col  = (idx % cluster_stride) % BANKSIZE;

        int vert_row = cluster_id * BANKSIZE + local_row;
        int vert_col = cluster_id * BANKSIZE + local_col;

        __shared__ Eigen::Vector3f s_R_raw[BANKSIZE];
        auto                       s_R = make_dense_1d(s_R_raw);
        if(threadIdx.x < BANKSIZE)
            s_R(threadIdx.x) = multi_lr(vert_col);
        __syncthreads();

        Eigen::Vector3f result;
        if(vert_col >= vert_row)
        {
            int si = sym_index(local_row, local_col);
            result = cluster_inv(cluster_id).M[si] * s_R(local_col);
        }
        else
        {
            int si = sym_index(local_col, local_row);
            result = cluster_inv(cluster_id).M[si].transpose() * s_R(local_col);
        }

        using WarpReduceF = cub::WarpReduce<float, BANKSIZE>;
        __shared__ typename WarpReduceF::TempStorage temp_reduce_f[BANKSIZE];
        int logical_warp = threadIdx.x / BANKSIZE;

        result[0] = WarpReduceF(temp_reduce_f[logical_warp]).Sum(result[0]);
        result[1] = WarpReduceF(temp_reduce_f[logical_warp]).Sum(result[1]);
        result[2] = WarpReduceF(temp_reduce_f[logical_warp]).Sum(result[2]);

        if((threadIdx.x % BANKSIZE) == 0)
        {
            atomicAdd(&(multi_lz(vert_row).x), result[0]);
            atomicAdd(&(multi_lz(vert_row).y), result[1]);
            atomicAdd(&(multi_lz(vert_row).z), result[2]);
        }
    }


    // s08 verification probe: max |a - b| and max |b| over the assembled
    // cluster Hessians, accumulated on device (values are non-negative, so a
    // 64-bit atomicMax on the bit pattern is a valid max for doubles).
    __global__ void MASPreconditionerEngine_compare_cluster_hess_kernel(
        cuda_tool::CBufferView<ClusterMatrixSym> a,
        cuda_tool::CBufferView<ClusterMatrixSym> b,
        unsigned long long* out,  // [0] = max diff, [1] = max ref
        int                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        int           blk = i / (SYM_BLOCK_COUNT * 9);
        int           rem = i % (SYM_BLOCK_COUNT * 9);
        const double* pa  = a(blk).M[rem / 9].data();
        const double* pb  = b(blk).M[rem / 9].data();
        double        va  = pa[rem % 9];
        double        vb  = pb[rem % 9];
        double        d   = fabs(va - vb);
        double        r   = fabs(vb);
        atomicMax(&out[0], (unsigned long long)__double_as_longlong(d));
        atomicMax(&out[1], (unsigned long long)__double_as_longlong(r));
    }

    // ---- s12: atomic-free row-dot local solve --------------------------------
    // The reference kernel above gives one thread to each (row, col) node pair
    // of a cluster (BANKSIZE^2 = 256 threads), forms one 3x3 * vec3 product,
    // reduces 16 of them with a cub::WarpReduce and then issues three float
    // atomicAdds per row.  Here one thread owns one *output scalar* (48 per
    // cluster): it walks the 16 columns of its row, accumulating 16 three-term
    // products in a register, and does one plain store.  No warp reduction, no
    // atomics -- and because every output element is now written exactly once,
    // the separate `multi_level_Z` zero-fill node of the apply disappears too.
    // Different summation order than the cub tree reduce -> rounding-level.
    __global__ void MASPreconditionerEngine_schwarz_local_solve_rowdot_kernel(
        cuda_tool::CBufferView<ClusterMatrixSymF> cluster_inv,
        cuda_tool::CBufferView<Eigen::Vector3f>   multi_lr,
        cuda_tool::BufferView<float3>             multi_lz,
        cuda_tool::CDense<IndexT>                 converged,
        int                                       N)  // 3 * node count
    {
        using namespace cuda_tool;

        if(*converged != 0)
            return;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)
            return;

        constexpr int cluster_scalars = BANKSIZE * 3;

        int cluster_id = idx / cluster_scalars;
        int rem        = idx - cluster_id * cluster_scalars;
        int local_row  = rem / 3;
        int comp       = rem - local_row * 3;

        int vert_row = cluster_id * BANKSIZE + local_row;
        int col_base = cluster_id * BANKSIZE;

        const ClusterMatrixSymF& C = cluster_inv(cluster_id);

        float acc = 0.0f;
        // All lanes of a cluster walk `c` in lockstep, so the residual load is
        // one broadcast per column.
        for(int c = 0; c < BANKSIZE; ++c)
        {
            const Eigen::Vector3f& rv = multi_lr(col_base + c);
            if(c >= local_row)
            {
                const Eigen::Matrix3f& M = C.M[sym_index(local_row, c)];
                acc += M(comp, 0) * rv[0] + M(comp, 1) * rv[1] + M(comp, 2) * rv[2];
            }
            else
            {
                const Eigen::Matrix3f& M = C.M[sym_index(c, local_row)];
                acc += M(0, comp) * rv[0] + M(1, comp) * rv[1] + M(2, comp) * rv[2];
            }
        }

        float3& z = multi_lz(vert_row);
        if(comp == 0)
            z.x = acc;
        else if(comp == 1)
            z.y = acc;
        else
            z.z = acc;
    }

    // s12 verification probe: max |a - b| and max |b| over two float arrays
    // (the multi-level R / Z scratch buffers of the preconditioner apply).
    __global__ void MASPreconditionerEngine_compare_float_kernel(const float* a,
                                                                 const float* b,
                                                                 unsigned long long* out,
                                                                 int                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        double va = a[i];
        double vb = b[i];
        double d  = fabs(va - vb);
        double r  = fabs(vb);
        atomicMax(&out[0], (unsigned long long)__double_as_longlong(d));
        atomicMax(&out[1], (unsigned long long)__double_as_longlong(r));
    }

    // s13 probe (UIPC_MAS_R_TAIL_VERIFY=1): assert on device that the coarse
    // multi_level_R accumulator is exactly zero where the removed fill node
    // used to run, i.e. right before build_multi_level_R.
    __global__ void MASPreconditionerEngine_check_r_tail_zero_kernel(
        cuda_tool::CBufferView<Eigen::Vector3f> r_tail, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        Eigen::Vector3f v = r_tail(i);
        UIPC_KERNEL_ASSERT(v[0] == 0.0f && v[1] == 0.0f && v[2] == 0.0f,
                           "MAS R tail not zero at apply entry: i=%d (%f %f %f)",
                           i,
                           v[0],
                           v[1],
                           v[2]);
    }

    __global__ void MASPreconditionerEngine_collect_final_Z_kernel(
        cuda_tool::DenseVectorView<Float>      Z_view,
        cuda_tool::CBufferView<float3>         multi_lz,
        cuda_tool::CBufferView<LevelTable>     coarse_table,
        cuda_tool::CBufferView<int>            real_to_part,
        cuda_tool::CDense<IndexT>              converged,
        cuda_tool::BufferView<Eigen::Vector3f> r_tail,
        int                                    tail_n,
        int                                    level_num,
        int                                    n)
    {
        using namespace cuda_tool;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // s13: the coarse entries of multi_level_R are an atomic accumulator
        // that must be zero when the *next* build_multi_level_R starts; the
        // threads past the node range zero them here, which removes the
        // separate 15 KB fill node from every captured PCG iteration.
        // Unconditional (the fill node it replaces was unconditional too).
        if(idx >= n)
        {
            int t = idx - n;
            if(t < tail_n)
                r_tail(t) = Eigen::Vector3f::Zero();
            return;
        }

        if(*converged != 0)
            return;
        int rdx = real_to_part(idx);

        // Skip unpartitioned vertices (handled by diagonal fallback)
        if(rdx < 0)
            return;

        // Start with the fine-level solution
        float3 cz = multi_lz(rdx);

        // Pure injection prolongation: sum coarse Z contributions directly,
        // matching StiffGIPC's __collectFinalZ_new (no scaling).
        // The Galerkin coarsening (H_coarse = R H R^T) is consistent with
        // the summation restriction, so no additional scaling is needed.
        LevelTable table = coarse_table(idx);
        for(int l = 1; l < level_num; l++)
        {
            int    node = table.index[l - 1];
            float3 val  = multi_lz(node);
            cz.x += val.x;
            cz.y += val.y;
            cz.z += val.z;
        }

        auto seg = Z_view.segment<3>(3 * idx);
        seg(0)   = static_cast<double>(cz.x);
        seg(1)   = static_cast<double>(cz.y);
        seg(2)   = static_cast<double>(cz.z);
    }
}  // namespace

// ============================================================================
// Phase 1: Initialization
// ============================================================================

void MASPreconditionerEngine::compute_num_levels(int vert_num)
{
    int n_level  = 1;
    int level_sz = bank_align(vert_num);

    while(level_sz > BANKSIZE)
    {
        level_sz /= BANKSIZE;
        n_level++;
        level_sz = bank_align(level_sz);
    }
    n_level++;
    m_level_num = std::min(n_level, MAX_LEVELS);
    m_active_level_num = 0;  // 0 = use full hierarchy (reset on each init_matrix)
}

void MASPreconditionerEngine::init_neighbor(int vert_num,
                                            int total_neighbor_num,
                                            int part_map_size,
                                            span<const unsigned int> h_neighbor_list,
                                            span<const unsigned int> h_neighbor_start,
                                            span<const unsigned int> h_neighbor_num,
                                            span<const int> h_part_to_real,
                                            span<const int> h_real_to_part)
{
    if(vert_num < 1)
        return;

    int max_nodes    = std::max(part_map_size, vert_num);
    int padded_nodes = bank_align(max_nodes);
    compute_num_levels(max_nodes);

    m_total_map_nodes = part_map_size;
    m_total_nodes     = vert_num;

    // Hierarchy buffers
    dense_level.resize(vert_num);
    real_to_part.resize(vert_num);
    coarse_tables.resize(vert_num);
    coarse_space_tables.resize(vert_num * m_level_num);
    level_sizes.resize(m_level_num + 1);
    going_next.resize(padded_nodes * m_level_num);
    prefix_original.resize(max_nodes);
    next_prefixes.resize(max_nodes);
    next_prefix_sums.resize(max_nodes);
    prefix_sum_original.resize(max_nodes);
    fine_connect_masks.resize(max_nodes);
    next_connect_masks.resize(max_nodes);

    // Neighbor buffers
    m_neighbor_list_size = total_neighbor_num;
    neighbor_lists.resize(total_neighbor_num);
    neighbor_starts.resize(vert_num);
    neighbor_nums.resize(vert_num);
    neighbor_lists_init.resize(total_neighbor_num);
    neighbor_nums_init.resize(vert_num);

    // Partition mappings
    part_to_real.resize(part_map_size);

    // Upload host data
    neighbor_lists_init.view().copy_from(h_neighbor_list.data());
    neighbor_starts.view().copy_from(h_neighbor_start.data());
    neighbor_nums_init.view().copy_from(h_neighbor_num.data());
    part_to_real.view().copy_from(h_part_to_real.data());
    real_to_part.view().copy_from(h_real_to_part.data());
}

void MASPreconditionerEngine::init_matrix()
{
    if(m_total_nodes < 1)
        return;

    // Restore neighbor data for initial hierarchy build
    neighbor_lists.view().copy_from(neighbor_lists_init.view());
    neighbor_nums.view().copy_from(neighbor_nums_init.view());

    int total_cluster = static_cast<int>(reorder_realtime() * 1.05);
    int num_blocks    = total_cluster / BANKSIZE;

    cluster_hessians.resize(num_blocks);
    cluster_inverses.resize(num_blocks);
    multi_level_R.resize(total_cluster);
    multi_level_Z.resize(total_cluster);

    m_initialized = true;
}

// ============================================================================
// Hierarchy building
// ============================================================================

int MASPreconditionerEngine::reorder_realtime()
{
    level_sizes.fill(Int2{0, 0});
    coarse_space_tables.fill(-1);
    going_next.fill(-1);

    build_connect_mask_L0();
    prepare_prefix_sum_L0();
    build_level1();

    for(int level = 1; level < m_level_num; level++)
    {
        next_connect_masks.fill(0u);
        build_connect_mask_Lx(level);

        level_sizes.view(level, 1).copy_to(&m_h_level_size);

        next_level_cluster(level);
        prefix_sum_Lx(level);
        compute_next_level(level);
    }

    level_sizes.view(m_level_num, 1).copy_to(&m_h_level_size);

    m_total_num_clusters = m_h_level_size.y;
    aggregation_kernel();

    return m_total_num_clusters;
}

// ---------------------------------------------------------------------------
// Build connectivity mask at level 0 (fine level)
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_connect_mask_L0()
{
    using namespace cuda_tool;
    int N = m_total_map_nodes;
    if(N < 1)
        return;

    auto k = MASPreconditionerEngine_build_connect_mask_L0_kernel;
    k<<<cuda_tool::best_grid_dim(N, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
        neighbor_starts.cview(),
        neighbor_nums.view(),
        neighbor_lists.view(),
        fine_connect_masks.view(),
        part_to_real.cview(),
        real_to_part.cview(),
        N);
}

// ---------------------------------------------------------------------------
// Prepare prefix sum at level 0 (uses shared memory)
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::prepare_prefix_sum_L0()
{
    using namespace cuda_tool;
    int N = m_total_map_nodes;
    if(N < 1)
        return;

    int block_size = DEFAULT_BLOCKSIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    MASPreconditionerEngine_prepare_prefix_sum_L0_kernel<<<num_blocks, block_size, 0, nullptr>>>(
        fine_connect_masks.view(), prefix_original.view(), part_to_real.cview(), N);
}

// ---------------------------------------------------------------------------
// Build coarse level 1 from prefix sums
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_level1()
{
    using namespace cuda_tool;
    int N = m_total_map_nodes;
    if(N < 1)
        return;

    int block_size = BANKSIZE * BANKSIZE;
    int num_blocks = (N + block_size - 1) / block_size;
    int warp_num   = (N + BANKSIZE - 1) / BANKSIZE;

    cuda_tool::DeviceScan().ExclusiveSum(
        prefix_original.data(), prefix_sum_original.data(), warp_num);

    // Cluster matrices are indexed by  node_index / BANKSIZE.
    // Level-0 uses partition indices [0, N),  so its cluster IDs occupy
    // [0, ceil(N / BANKSIZE)).
    //
    // padded_N = align(N, BANKSIZE)
    //   — round N up to a BANKSIZE boundary so that every level-0 bank
    //     is a full block; level-1 indices must start no earlier than this
    //     to avoid sharing a cluster ID with any level-0 bank.
    //
    // max(padded_N, m_total_nodes)
    //   — when unpartitioned vertices exist,
    //     m_total_nodes > m_total_map_nodes is possible.  going_next is
    //     read at real-vertex indices [0, m_total_nodes), so level-1
    //     indices must also exceed m_total_nodes.
    //
    // align(..., BANKSIZE)
    //   — the outer align keeps the level-1 region BANKSIZE-aligned,
    //     which is required by the bank-based cluster addressing used
    //     in all subsequent kernels.
    int padded_N     = bank_align(N);
    int level1_begin = bank_align(std::max(padded_N, m_total_nodes));

    MASPreconditionerEngine_build_level1_kernel<<<num_blocks, block_size, 0, nullptr>>>(
        level_sizes.view(),
        coarse_space_tables.view(),
        going_next.view(0, level1_begin),
        fine_connect_masks.cview(),
        prefix_sum_original.cview(),
        prefix_original.cview(),
        part_to_real.cview(),
        N,
        level1_begin);
}

// ---------------------------------------------------------------------------
// Build connectivity mask at level x (coarsened)
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_connect_mask_Lx(int level)
{
    using namespace cuda_tool;
    int N = m_total_map_nodes;
    if(N < 1)
        return;

    int block_size = DEFAULT_BLOCKSIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    MASPreconditionerEngine_build_connect_mask_Lx_kernel<<<num_blocks, block_size, 0, nullptr>>>(
        neighbor_starts.cview(),
        neighbor_nums.view(),
        neighbor_lists.view(),
        coarse_space_tables.cview(),
        next_connect_masks.view(),
        fine_connect_masks.cview(),
        part_to_real.cview(),
        level,
        m_total_nodes,
        N);
}

// ---------------------------------------------------------------------------
// Cluster connectivity at next level
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::next_level_cluster(int level)
{
    using namespace cuda_tool;
    int N = m_h_level_size.x;
    if(N < 1)
        return;

    int block_size = DEFAULT_BLOCKSIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    MASPreconditionerEngine_next_level_cluster_kernel<<<num_blocks, block_size, 0, nullptr>>>(
        next_connect_masks.view(), next_prefixes.view(), N);
}

// ---------------------------------------------------------------------------
// Prefix sum at level x
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::prefix_sum_Lx(int level)
{
    using namespace cuda_tool;
    int N = m_h_level_size.x;
    if(N < 1)
        return;

    int level_begin       = m_h_level_size.y;
    int level_region_size = bank_align(N);
    int next_level_begin  = level_begin + level_region_size;
    int block_size        = BANKSIZE * BANKSIZE;
    int num_blocks        = (N + block_size - 1) / block_size;
    int warp_num          = (N + BANKSIZE - 1) / BANKSIZE;

    cuda_tool::DeviceScan().ExclusiveSum(next_prefixes.data(), next_prefix_sums.data(), warp_num);

    MASPreconditionerEngine_prefix_sum_Lx_kernel<<<num_blocks, block_size, 0, nullptr>>>(
        level_sizes.view(),
        next_prefixes.cview(),
        next_prefix_sums.cview(),
        next_connect_masks.view(),
        going_next.view(level_begin, level_region_size),
        level,
        next_level_begin,
        N);
}

// ---------------------------------------------------------------------------
// Compute next coarsening level
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::compute_next_level(int level)
{
    using namespace cuda_tool;
    int N = m_total_nodes;
    if(N < 1)
        return;

    auto k = MASPreconditionerEngine_compute_next_level_kernel;
    k<<<cuda_tool::best_grid_dim(N, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
        coarse_space_tables.view(), next_connect_masks.cview(), level, N);
}

// ---------------------------------------------------------------------------
// Build per-node level traversal table
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::aggregation_kernel()
{
    using namespace cuda_tool;
    int N = m_total_nodes;
    if(N < 1 || m_total_num_clusters < 1)
        return;

    int  level_num = m_level_num;
    auto k         = MASPreconditionerEngine_aggregation_kernel_kernel;
    k<<<cuda_tool::best_grid_dim(N, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
        coarse_tables.view(),
        going_next.cview(0, m_total_num_clusters),
        level_sizes.cview(),
        level_num,
        N);
}

// ============================================================================
// Phase 2: Assemble preconditioner
// ============================================================================

void MASPreconditionerEngine::set_preconditioner(cuda_tool::CBufferView<Eigen::Matrix3d> triplet_values,
                                                 cuda_tool::CBufferView<int> row_ids,
                                                 cuda_tool::CBufferView<int> col_ids,
                                                 int dof_offset)
{
    if(m_total_nodes < 1)
        return;

    // The partition adjacency and its multi-level hierarchy are invariant for the
    // lifetime of this engine. init_matrix() builds them once; rebuilding here
    // would repeat scans, kernel launches, and host synchronizations every Newton
    // iteration without changing the result.

    // Resize cluster matrices if needed
    int num_cluster_blocks = m_total_num_clusters / BANKSIZE;
    if(num_cluster_blocks < 1)
        return;

    if(num_cluster_blocks > static_cast<int>(cluster_hessians.size()))
    {
        cluster_hessians.resize(num_cluster_blocks);
        cluster_inverses.resize(num_cluster_blocks);
    }

    // Resize multi-level buffers if needed
    if(m_total_num_clusters > static_cast<int>(multi_level_R.size()))
    {
        multi_level_R.resize(m_total_num_clusters);
        multi_level_Z.resize(m_total_num_clusters);
    }

    // s13: collect_final_Z re-zeroes the coarse multi_level_R accumulator for
    // the *next* apply, so seed it once per Newton iteration here (outside any
    // graph capture) instead of once per PCG iteration inside the captured
    // block.
    if(fuse_r_tail_fill() && m_total_num_clusters > m_total_map_nodes)
    {
        multi_level_R
            .view(m_total_map_nodes, m_total_num_clusters - m_total_map_nodes)
            .fill(Eigen::Vector3f::Zero());
    }

    // perf/kernels (K12): ClusterMatrixSym{} is all zero bytes, so clear the
    // assembly buffer with a memset instead of the generic fill kernel (one
    // 1.7 KB struct per thread, 0.33 ms per Newton iteration on the 2070S).
    // UIPC_MAS_FILL_KERNEL=1 restores the fill kernel (A/B).
    static const bool use_fill_kernel = []
    {
        const char* e = std::getenv("UIPC_MAS_FILL_KERNEL");
        return e && e[0] == '1';
    }();
    if(use_fill_kernel)
        cluster_hessians.view(0, num_cluster_blocks).fill(ClusterMatrixSym{});
    else
        CUDA_TOOL_CHECK(cudaMemsetAsync(
            cluster_hessians.data(), 0, sizeof(ClusterMatrixSym) * num_cluster_blocks, nullptr));

    // Scatter BCOO Hessian blocks into cluster matrices
    scatter_hessian_to_clusters(triplet_values, row_ids, col_ids, dof_offset);

    // s08 verification probe: UIPC_MAS_SCATTER_VERIFY=1 re-runs the *other*
    // scatter path into a scratch buffer, =2 re-runs the *same* one (that is
    // the old/new path's own atomic-order noise), and reports max |diff| and
    // max |ref| over all assembled cluster Hessian entries.
    static const int verify_mode = []
    {
        const char* e = std::getenv("UIPC_MAS_SCATTER_VERIFY");
        return e ? std::atoi(e) : 0;
    }();
    if(verify_mode)
    {
        using namespace cuda_tool;
        if(num_cluster_blocks > static_cast<int>(cluster_hessians_verify.size()))
            cluster_hessians_verify.resize(num_cluster_blocks);
        if(m_verify_stat.size() < 2)
            m_verify_stat.resize(2);
        CUDA_TOOL_CHECK(cudaMemsetAsync(cluster_hessians_verify.data(),
                                        0,
                                        sizeof(ClusterMatrixSym) * num_cluster_blocks,
                                        nullptr));
        bool other = (verify_mode == 2) ? scatter_agg_enabled() : !scatter_agg_enabled();
        scatter_hessian_to_clusters_into(cluster_hessians_verify.view(0, num_cluster_blocks),
                                         other,
                                         triplet_values,
                                         row_ids,
                                         col_ids,
                                         dof_offset);
        CUDA_TOOL_CHECK(cudaMemsetAsync(
            m_verify_stat.data(), 0, sizeof(unsigned long long) * 2, nullptr));
        int  n = num_cluster_blocks * SYM_BLOCK_COUNT * 9;
        auto k = MASPreconditionerEngine_compare_cluster_hess_kernel;
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            cluster_hessians_verify.cview(0, num_cluster_blocks),
            cluster_hessians.cview(0, num_cluster_blocks),
            m_verify_stat.data(),
            n);
        unsigned long long h[2] = {0, 0};
        m_verify_stat.view(0, 2).copy_to(h);
        double dmax = 0.0, rmax = 0.0;
        std::memcpy(&dmax, &h[0], sizeof(double));
        std::memcpy(&rmax, &h[1], sizeof(double));
        ++m_verify_count;
        if(dmax > m_verify_worst_abs)
            m_verify_worst_abs = dmax;
        double rel = (rmax > 0.0) ? dmax / rmax : 0.0;
        if(rel > m_verify_worst_rel)
            m_verify_worst_rel = rel;
        spdlog::info(
            "[MAS scatter verify] mode={} n={} max|diff|={:.6e} max|ref|={:.6e} rel={:.6e} "
            "worst_abs={:.6e} worst_rel={:.6e}",
            verify_mode,
            m_verify_count,
            dmax,
            rmax,
            rel,
            m_verify_worst_abs,
            m_verify_worst_rel);
    }

    // Invert each cluster matrix (Gauss-Jordan)
    invert_cluster_matrices();

    // s09 verification probe: UIPC_MAS_INVERT_VERIFY=1 re-runs the *other*
    // inversion path into a scratch buffer, =2 re-runs the *same* one (that is
    // the path's own run-to-run noise), and reports max |diff| / max |ref| over
    // all cluster inverse entries.
    static const int invert_verify_mode = []
    {
        const char* e = std::getenv("UIPC_MAS_INVERT_VERIFY");
        return e ? std::atoi(e) : 0;
    }();
    if(invert_verify_mode)
    {
        using namespace cuda_tool;
        if(num_cluster_blocks > static_cast<int>(cluster_inverses_verify.size()))
            cluster_inverses_verify.resize(num_cluster_blocks);
        if(m_verify_stat.size() < 2)
            m_verify_stat.resize(2);
        bool other = (invert_verify_mode == 2) ? invert_sweep_enabled() :
                                                 !invert_sweep_enabled();
        invert_cluster_matrices_into(cluster_inverses_verify.view(0, num_cluster_blocks), other);
        CUDA_TOOL_CHECK(cudaMemsetAsync(
            m_verify_stat.data(), 0, sizeof(unsigned long long) * 2, nullptr));
        int  n = num_cluster_blocks * SYM_BLOCK_COUNT * 9;
        auto k = MASPreconditionerEngine_compare_cluster_inv_kernel;
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            cluster_inverses_verify.cview(0, num_cluster_blocks),
            cluster_inverses.cview(0, num_cluster_blocks),
            m_verify_stat.data(),
            n);
        unsigned long long h[2] = {0, 0};
        m_verify_stat.view(0, 2).copy_to(h);
        double dmax = 0.0, rmax = 0.0;
        std::memcpy(&dmax, &h[0], sizeof(double));
        std::memcpy(&rmax, &h[1], sizeof(double));
        ++m_invert_verify_count;
        if(dmax > m_invert_worst_abs)
            m_invert_worst_abs = dmax;
        double rel = (rmax > 0.0) ? dmax / rmax : 0.0;
        if(rel > m_invert_worst_rel)
            m_invert_worst_rel = rel;
        spdlog::info(
            "[MAS invert verify] mode={} n={} max|diff|={:.6e} max|ref|={:.6e} rel={:.6e} "
            "worst_abs={:.6e} worst_rel={:.6e}",
            invert_verify_mode,
            m_invert_verify_count,
            dmax,
            rmax,
            rel,
            m_invert_worst_abs,
            m_invert_worst_rel);
    }
}

// ---------------------------------------------------------------------------
// Scatter BCOO Hessian entries into cluster-level dense matrices
// ---------------------------------------------------------------------------
// s08: warp-aggregated FP64 atomics in the pass-1 scatter. UIPC_MAS_SCATTER_AGG=0 = old.
bool MASPreconditionerEngine::scatter_agg_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_MAS_SCATTER_AGG");
        return !(e && e[0] == '0');
    }();
    return on;
}

void MASPreconditionerEngine::scatter_hessian_to_clusters(
    cuda_tool::CBufferView<Eigen::Matrix3d> triplet_values,
    cuda_tool::CBufferView<int>             row_ids,
    cuda_tool::CBufferView<int>             col_ids,
    int                                     dof_offset)
{
    scatter_hessian_to_clusters_into(
        cluster_hessians.view(), scatter_agg_enabled(), triplet_values, row_ids, col_ids, dof_offset);
}

void MASPreconditionerEngine::scatter_hessian_to_clusters_into(
    cuda_tool::BufferView<ClusterMatrixSym> cluster_hess,
    bool                                    use_agg,
    cuda_tool::CBufferView<Eigen::Matrix3d> triplet_values,
    cuda_tool::CBufferView<int>             row_ids,
    cuda_tool::CBufferView<int>             col_ids,
    int                                     dof_offset)
{
    using namespace cuda_tool;

    int triplet_num = static_cast<int>(triplet_values.size());

    // --- Pass 1: Place each 3x3 block at the finest level where both
    //             row and col belong to the same cluster. ---

    if(triplet_num > 0 && use_agg)
    {
        auto k = MASPreconditionerEngine_scatter_hessian_to_clusters_k1_agg_kernel;
        k<<<cuda_tool::best_grid_dim(triplet_num, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dof_offset,
            m_level_num,
            going_next.cview(0, m_total_num_clusters),
            level_sizes.cview(),
            cluster_hess,
            real_to_part.cview(),
            triplet_values,
            row_ids,
            col_ids,
            m_total_nodes,
            triplet_num);
    }
    else if(triplet_num > 0)
    {
        auto k = MASPreconditionerEngine_scatter_hessian_to_clusters_k1_kernel;
        k<<<cuda_tool::best_grid_dim(triplet_num, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dof_offset,
            m_level_num,
            going_next.cview(0, m_total_num_clusters),
            level_sizes.cview(),
            cluster_hess,
            real_to_part.cview(),
            triplet_values,
            row_ids,
            col_ids,
            m_total_nodes,
            triplet_num);
    }

    // --- Pass 2: Scatter fine-level cluster matrices to coarser levels
    //             using warp-reduction for contiguous partitions. ---

    int total_entries = m_total_map_nodes * BANKSIZE;
    if(total_entries < 1)
        return;
    int thread_num = BANKSIZE * BANKSIZE;
    int block_num  = (total_entries + thread_num - 1) / thread_num;

    MASPreconditionerEngine_scatter_hessian_to_clusters_k2_kernel<<<block_num, thread_num, 0, nullptr>>>(
        m_level_num,
        going_next.cview(0, m_total_num_clusters),
        level_sizes.cview(),
        cluster_hess,
        part_to_real.cview(),
        fine_connect_masks.cview(),
        prefix_original.cview(),
        m_total_map_nodes,
        total_entries);
}

// ---------------------------------------------------------------------------
// Gauss-Jordan inversion of each 48x48 cluster matrix
// ---------------------------------------------------------------------------
// s09: register-column symmetric-sweep inverse. UIPC_MAS_INVERT_SWEEP=0 = old.
bool MASPreconditionerEngine::invert_sweep_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_MAS_INVERT_SWEEP");
        return !(e && e[0] == '0');
    }();
    return on;
}

void MASPreconditionerEngine::invert_cluster_matrices()
{
    invert_cluster_matrices_into(cluster_inverses.view(), invert_sweep_enabled());
}

void MASPreconditionerEngine::invert_cluster_matrices_into(
    cuda_tool::BufferView<ClusterMatrixSymF> cluster_inv, bool use_sweep)
{
    using namespace cuda_tool;
    int total_threads = m_total_num_clusters * 3;  // 48 threads per cluster
    if(total_threads < 1)
        return;

    if(use_sweep)
    {
        int block_size = MAS_INVERT_SWEEP_BLOCK;  // 4 clusters per block
        int num_blocks = (total_threads + block_size - 1) / block_size;
        MASPreconditionerEngine_invert_cluster_matrices_sweep_kernel<<<num_blocks, block_size, 0, nullptr>>>(
            cluster_inv, cluster_hessians.cview(), total_threads);
    }
    else
    {
        int block_size = 32 * 3;  // 96 threads = 2 clusters per block
        int num_blocks = (total_threads + block_size - 1) / block_size;
        MASPreconditionerEngine_invert_cluster_matrices_kernel<<<num_blocks, block_size, 0, nullptr>>>(
            cluster_inv, cluster_hessians.cview(), total_threads);
    }
}

// ============================================================================
// Phase 3: Apply preconditioning  z = M^{-1} r
// ============================================================================

// ---------------------------------------------------------------------------
// Restrict: accumulate residual R from fine to all coarser levels
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::build_multi_level_R(cuda_tool::CDenseVectorView<Float> R,
                                                  cuda_tool::CVarView<IndexT> converged,
                                                  cudaStream_t stream)
{
    using namespace cuda_tool;
    int N = m_total_map_nodes;
    if(N < 1)
        return;

    int block_size = DEFAULT_BLOCKSIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    MASPreconditionerEngine_build_multi_level_R_kernel<<<num_blocks, block_size, 0, stream>>>(
        R,
        multi_level_R.view(),
        going_next.cview(0, m_total_num_clusters),
        level_sizes.cview(),
        prefix_original.cview(),
        fine_connect_masks.cview(),
        part_to_real.cview(),
        converged.cviewer(),
        m_level_num,
        N);
}

// ---------------------------------------------------------------------------
// Local solve: Z = cluster_inverse * R at each level
// ---------------------------------------------------------------------------
// s12: atomic-free row-dot local solve (one thread per output scalar); it also
// makes the multi_level_Z zero-fill node of apply() unnecessary.
// UIPC_MAS_LOCAL_SOLVE_ROWDOT=0 = old (256 threads per cluster, warp reduce + atomics).
bool MASPreconditionerEngine::local_solve_rowdot_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_MAS_LOCAL_SOLVE_ROWDOT");
        return !(e && e[0] == '0');
    }();
    return on;
}

// True when the row-dot solve writes every multi_level_Z entry exactly once, so
// the zero-fill can be skipped. Requires the node count to be a whole number of
// BANKSIZE clusters (it always is: every level region is bank-aligned).
bool MASPreconditionerEngine::z_fill_needed() const
{
    return !(local_solve_rowdot_enabled() && m_total_num_clusters > 0
             && (m_total_num_clusters % BANKSIZE) == 0);
}

void MASPreconditionerEngine::schwarz_local_solve(cuda_tool::CVarView<IndexT> converged,
                                                  cudaStream_t stream)
{
    schwarz_local_solve_into(converged, stream, local_solve_rowdot_enabled());
}

void MASPreconditionerEngine::schwarz_local_solve_into(cuda_tool::CVarView<IndexT> converged,
                                                       cudaStream_t stream,
                                                       bool         use_rowdot)
{
    using namespace cuda_tool;

    if(use_rowdot && m_total_num_clusters >= BANKSIZE
       && (m_total_num_clusters % BANKSIZE) == 0)
    {
        // s13: clusters per block was never swept (s12 fixed it at 2).
        // UIPC_MAS_LOCAL_SOLVE_CPB = 1 / 2 / 4 / 8 / 16.
        static const int cpb = []
        {
            const char* e = std::getenv("UIPC_MAS_LOCAL_SOLVE_CPB");
            int         v = e ? std::atoi(e) : 2;
            return (v == 1 || v == 2 || v == 4 || v == 8 || v == 16) ? v : 2;
        }();
        int N          = m_total_num_clusters * 3;  // one thread per output scalar
        int block_size = BANKSIZE * 3 * cpb;
        int num_blocks = (N + block_size - 1) / block_size;

        MASPreconditionerEngine_schwarz_local_solve_rowdot_kernel<<<num_blocks, block_size, 0, stream>>>(
            cluster_inverses.cview(),
            multi_level_R.cview(),
            multi_level_Z.view(),
            converged.cviewer(),
            N);
        return;
    }

    int N = m_total_num_clusters * BANKSIZE;  // one thread per (cluster, node-pair)
    if(N < 1)
        return;

    int block_size = BANKSIZE * BANKSIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    MASPreconditionerEngine_schwarz_local_solve_kernel<<<num_blocks, block_size, 0, stream>>>(
        cluster_inverses.cview(),
        multi_level_R.cview(),
        multi_level_Z.view(),
        converged.cviewer(),
        N);
}

// ---------------------------------------------------------------------------
// Prolongate: sum Z contributions from all levels for each fine node
// ---------------------------------------------------------------------------
void MASPreconditionerEngine::collect_final_Z(cuda_tool::DenseVectorView<Float> Z,
                                              cuda_tool::CVarView<IndexT> converged,
                                              cudaStream_t stream)
{
    using namespace cuda_tool;
    int N = m_total_nodes;
    if(N < 1)
        return;

    int level_num = (m_active_level_num > 0) ? m_active_level_num : m_level_num;
    auto k        = MASPreconditionerEngine_collect_final_Z_kernel;
    // s11: cudaOccupancyMaxPotentialBlockSize picks 1024 threads per block for
    // this kernel, which on the bunny gives a 19-block grid — half the SMs of a
    // 40-SM part stay idle for a purely gather-bound kernel. 256 spreads the
    // same (per-thread independent, bit-identical) work over all SMs.
    // UIPC_MAS_COLLECT_BLOCK_DIM=0 restores the occupancy heuristic.
    static const int collect_block_dim = []
    {
        const char* e = std::getenv("UIPC_MAS_COLLECT_BLOCK_DIM");
        int         v = e ? std::atoi(e) : 256;
        return (v >= 0 && v <= 1024 && (v % 32) == 0) ? v : 256;
    }();
    int bd = collect_block_dim > 0 ? collect_block_dim : cuda_tool::best_block_dim(k);

    // s13: fold the coarse multi_level_R zero-fill into this kernel's tail
    // threads (see the kernel comment). UIPC_MAS_FUSE_R_TAIL_FILL=0 = old node.
    int tail_off = m_total_map_nodes;
    int tail_n   = fuse_r_tail_fill() && m_total_num_clusters > m_total_map_nodes ?
                       m_total_num_clusters - m_total_map_nodes :
                       0;
    int gd       = (N + tail_n + bd - 1) / bd;
    k<<<gd, bd, 0, stream>>>(
        Z,
        multi_level_Z.cview(),
        coarse_tables.cview(),
        real_to_part.cview(),
        converged.cviewer(),
        tail_n > 0 ? multi_level_R.view(tail_off, tail_n) : multi_level_R.view(0, 0),
        tail_n,
        level_num,
        N);
}

// s13: UIPC_MAS_FUSE_R_TAIL_FILL=0 restores the separate fill node.
bool MASPreconditionerEngine::fuse_r_tail_fill()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_MAS_FUSE_R_TAIL_FILL");
        return !(e && e[0] == '0');
    }();
    return on;
}

// ============================================================================
// Apply: full preconditioning pipeline  z = M^{-1} r
// ============================================================================

void MASPreconditionerEngine::apply(cuda_tool::CDenseVectorView<Float> r,
                                    cuda_tool::DenseVectorView<Float>  z,
                                    cuda_tool::CVarView<IndexT> converged,
                                    cudaStream_t                stream)
{
    if(m_total_nodes < 1)
        return;

    // Ensure multi-level buffers cover all clusters
    if(m_total_num_clusters > static_cast<int>(multi_level_R.size()))
    {
        multi_level_R.resize(m_total_num_clusters);
        multi_level_Z.resize(m_total_num_clusters);
    }

    if(!fuse_r_tail_fill() && m_total_num_clusters > m_total_map_nodes)
    {
        multi_level_R
            .view(m_total_map_nodes, m_total_num_clusters - m_total_map_nodes)
            .fill(Eigen::Vector3f::Zero(), stream);
    }

    // s12: the row-dot local solve writes every Z entry exactly once, so the
    // zero-fill node is only needed for the old atomic-accumulating path.
    if(z_fill_needed())
        multi_level_Z.view(0, m_total_num_clusters).fill(float3{0, 0, 0}, stream);

    static const bool r_tail_verify = []
    {
        const char* e = std::getenv("UIPC_MAS_R_TAIL_VERIFY");
        return e && e[0] != '0';
    }();
    if(r_tail_verify && m_total_num_clusters > m_total_map_nodes)
    {
        int  tn = m_total_num_clusters - m_total_map_nodes;
        auto k  = MASPreconditionerEngine_check_r_tail_zero_kernel;
        k<<<(tn + 255) / 256, 256, 0, stream>>>(
            multi_level_R.cview(m_total_map_nodes, tn), tn);
    }

    // 1. Restrict: accumulate residual down through levels
    build_multi_level_R(r, converged, stream);

    // 2. Local solve: Z = cluster_inverse * R at each level
    schwarz_local_solve(converged, stream);

    // s12 verification probe: UIPC_MAS_APPLY_VERIFY=1 re-runs the restrict +
    // local-solve phases with the *other* code path into the same scratch
    // buffers (the output z has already been written below), =2 re-runs the
    // *same* one -- that is the path's own atomic-order noise -- and reports
    // max |diff| / max |ref| over the multi-level R and Z arrays.
    //
    // The reference has to be snapshotted *here*, between the local solve and
    // collect_final_Z: since s13b, collect_final_Z re-zeroes the coarse tail of
    // multi_level_R for the next apply, so a snapshot taken after it would hold
    // zeros for every coarse entry and the comparison would be meaningless.
    if(apply_verify_mode())
        verify_apply_snapshot(stream);

    // 3. Prolongate: sum Z from all levels back to fine nodes
    collect_final_Z(z, converged, stream);

    if(apply_verify_mode())
        verify_apply(r, converged, stream, apply_verify_mode());
}

int MASPreconditionerEngine::apply_verify_mode()
{
    static const int mode = []
    {
        const char* e = std::getenv("UIPC_MAS_APPLY_VERIFY");
        return e ? std::atoi(e) : 0;
    }();
    return mode;
}

// Copy the production multi-level R / Z of this apply into the probe's
// reference buffers. Called from apply() before collect_final_Z (see there).
void MASPreconditionerEngine::verify_apply_snapshot(cudaStream_t stream)
{
    using namespace cuda_tool;

    // Never inside a captured graph (the probe reads results back to the host).
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(stream, &status) != cudaSuccess
       || status != cudaStreamCaptureStatusNone)
        return;

    int n = m_total_num_clusters;
    if(n < 1)
        return;

    if(static_cast<int>(m_apply_R_verify.size()) < n)
    {
        m_apply_R_verify.resize(n);
        m_apply_Z_verify.resize(n);
    }

    m_apply_R_verify.view(0, n).copy_from(multi_level_R.view(0, n));
    m_apply_Z_verify.view(0, n).copy_from(multi_level_Z.view(0, n));
}

void MASPreconditionerEngine::verify_apply(cuda_tool::CDenseVectorView<Float> r,
                                           cuda_tool::CVarView<IndexT> converged,
                                           cudaStream_t                stream,
                                           int                         mode)
{
    using namespace cuda_tool;

    // Never inside a captured graph (the probe reads results back to the host).
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(stream, &status) != cudaSuccess
       || status != cudaStreamCaptureStatusNone)
        return;

    int n = m_total_num_clusters;
    if(n < 1)
        return;

    // A converged solve leaves every apply kernel a no-op, so the scratch
    // buffers still hold the previous iteration's values (the row-dot path has
    // no zero-fill) -- nothing to compare there.
    IndexT h_converged = 0;
    converged.copy_to(&h_converged, stream);
    if(h_converged != 0)
        return;

    // The reference was taken by verify_apply_snapshot() before collect_final_Z.
    if(static_cast<int>(m_apply_R_verify.size()) < n)
        return;
    if(m_verify_stat.size() < 2)
        m_verify_stat.resize(2);

    // Zero *all* of multi_level_R for the re-run: the fine entries are plain
    // writes of build_multi_level_R, the coarse tail is an atomic accumulator
    // and must start at zero exactly as it does at the entry of a production
    // apply.
    multi_level_R.view(0, n).fill(Eigen::Vector3f::Zero(), stream);
    multi_level_Z.view(0, n).fill(float3{0, 0, 0}, stream);

    bool rowdot = (mode == 2) ? local_solve_rowdot_enabled() : !local_solve_rowdot_enabled();

    build_multi_level_R(r, converged, stream);
    schwarz_local_solve_into(converged, stream, rowdot);

    const char* names[2] = {"R", "Z"};
    const float* newp[2] = {reinterpret_cast<const float*>(multi_level_R.data()),
                            reinterpret_cast<const float*>(multi_level_Z.data())};
    const float* refp[2] = {reinterpret_cast<const float*>(m_apply_R_verify.data()),
                            reinterpret_cast<const float*>(m_apply_Z_verify.data())};

    ++m_apply_verify_count;
    for(int a = 0; a < 2; ++a)
    {
        CUDA_TOOL_CHECK(cudaMemsetAsync(
            m_verify_stat.data(), 0, sizeof(unsigned long long) * 2, stream));
        int  cnt = n * 3;
        auto k   = MASPreconditionerEngine_compare_float_kernel;
        k<<<cuda_tool::best_grid_dim(cnt, k), cuda_tool::best_block_dim(k), 0, stream>>>(
            newp[a], refp[a], m_verify_stat.data(), cnt);
        unsigned long long h[2] = {0, 0};
        m_verify_stat.view(0, 2).copy_to(h);
        double dmax = 0.0, rmax = 0.0;
        std::memcpy(&dmax, &h[0], sizeof(double));
        std::memcpy(&rmax, &h[1], sizeof(double));
        double rel = (rmax > 0.0) ? dmax / rmax : 0.0;
        if(rel > m_apply_worst_rel[a])
            m_apply_worst_rel[a] = rel;
        if(dmax > m_apply_worst_abs[a])
            m_apply_worst_abs[a] = dmax;
        spdlog::info(
            "[MAS apply verify] mode={} arr={} n={} max|diff|={:.6e} max|ref|={:.6e} "
            "rel={:.6e} worst_abs={:.6e} worst_rel={:.6e}",
            mode,
            names[a],
            m_apply_verify_count,
            dmax,
            rmax,
            rel,
            m_apply_worst_abs[a],
            m_apply_worst_rel[a]);
    }

    // Leave the coarse tail in the zeroed state the next production apply
    // expects (with the s13b fold on, collect_final_Z is what normally leaves
    // it that way, and the re-run above has just filled it again). Without
    // this, enabling the probe would perturb the solve.
    if(n > m_total_map_nodes)
    {
        multi_level_R.view(m_total_map_nodes, n - m_total_map_nodes)
            .fill(Eigen::Vector3f::Zero(), stream);
    }
}

void MASPreconditionerEngine::dump_cluster_matrices_debug(std::string_view output_dir,
                                                          SizeT frame,
                                                          SizeT newton_iter)
{
    if(!m_initialized || cluster_hessians.size() == 0)
        return;

    cuda_tool::wait_device();

    // Materialize the path once for the lambdas / std::ofstream consumers below.
    const std::filesystem::path output_dir_path{output_dir};

    const size_t                   nb = cluster_hessians.size();
    std::vector<ClusterMatrixSym>  h_hess(nb);
    std::vector<ClusterMatrixSymF> h_inv(nb);

    cluster_hessians.view(0, nb).copy_to(h_hess.data());
    cluster_inverses.view(0, nb).copy_to(h_inv.data());

    // Block-upper-triangle dump: coordinate real general, (nb*48) × (nb*48) block-diagonal.
    // Each cluster writes SYM_BLOCK_COUNT full 3×3 blocks at their upper-triangle positions.
    // Diagonal blocks are full 3×3 (not scalar-upper-triangle), matching GPU storage exactly.
    // Uses fmt::memory_buffer + FILE* pattern consistent with utils/matrix_market.h.
    constexpr int DIM       = BANKSIZE * 3;
    const int     mat_size  = static_cast<int>(nb) * DIM;
    const int64_t total_nnz = static_cast<int64_t>(nb) * SYM_BLOCK_COUNT * 9;

    auto write_mtx = [&]<typename Scalar>(const std::vector<ClusterMatrixSymT<Scalar>>& clusters,
                                          std::string_view kind)
    {
        auto path = output_dir_path
                    / fmt::format("mas_cluster_{}.f{}.n{}.mtx", kind, frame, newton_iter);
        auto path_str = path.string();

        auto buf = fmt::memory_buffer();

        fmt::format_to(std::back_inserter(buf),
                       "%%MatrixMarket matrix coordinate real general\n"
                       "% MAS cluster {} block-upper-triangle ({} clusters, banksize={})\n"
                       "{} {} {}\n",
                       kind,
                       nb,
                       BANKSIZE,
                       mat_size,
                       mat_size,
                       total_nnz);

        for(size_t c = 0; c < clusters.size(); c++)
        {
            const int base = static_cast<int>(c) * DIM;

            for(int br = 0; br < BANKSIZE; br++)
            {
                for(int bc = br; bc < BANKSIZE; bc++)
                {
                    int         k   = BANKSIZE * br - br * (br + 1) / 2 + bc;
                    const auto& blk = clusters[c].M[k];

                    for(int i = 0; i < 3; i++)
                        for(int j = 0; j < 3; j++)
                            fmt::format_to(std::back_inserter(buf),
                                           "{} {} {:.17g}\n",
                                           base + br * 3 + i + 1,
                                           base + bc * 3 + j + 1,
                                           static_cast<double>(blk(i, j)));
                }
            }
        }

        FILE* fp = std::fopen(path_str.c_str(), "w");
        if(!fp)
        {
            UIPC_WARN_WITH_LOCATION("MAS dump: open {} failed", path_str);
            return;
        }
        std::fwrite(buf.data(), 1, buf.size(), fp);
        std::fclose(fp);
        logger::info("MAS dump: wrote {}", path_str);
    };

    write_mtx(h_hess, "hess");
    write_mtx(h_inv, "inv");

    // Partition metadata as JSON
    {
        auto path = output_dir_path
                    / fmt::format("mas_cluster_meta.f{}.n{}.json", frame, newton_iter);
        std::ofstream out(path);
        if(!out)
        {
            UIPC_WARN_WITH_LOCATION("MAS dump: open {} failed", path.string());
            return;
        }

        Json j;
        j["total_nodes"]        = m_total_nodes;
        j["total_map_nodes"]    = m_total_map_nodes;
        j["total_clusters"]     = m_total_num_clusters;
        j["num_cluster_blocks"] = nb;
        j["banksize"]           = BANKSIZE;
        j["sym_block_count"]    = SYM_BLOCK_COUNT;
        j["frame"]              = frame;
        j["newton_iter"]        = newton_iter;

        if(m_total_map_nodes > 0)
        {
            std::vector<int> h_part_to_real(static_cast<size_t>(m_total_map_nodes));
            part_to_real.view(0, m_total_map_nodes).copy_to(h_part_to_real.data());
            j["part_to_real"] = h_part_to_real;
        }

        // Hierarchy: level_sizes (start position .y, count .x per level)
        {
            std::vector<Int2> h_lvl(static_cast<size_t>(m_level_num + 1));
            level_sizes.view(0, m_level_num + 1).copy_to(h_lvl.data());
            Json jl = Json::array();
            for(int l = 0; l <= m_level_num; l++)
            {
                Json e;
                e["level"]  = l;
                e["count"]  = h_lvl[l].x;
                e["offset"] = h_lvl[l].y;
                jl.push_back(e);
            }
            j["levels"]    = jl;
            j["level_num"] = m_level_num;
        }

        // going_next: maps each "padded position in level X" -> "padded position in level X+1"
        if(m_total_num_clusters > 0)
        {
            std::vector<int> h_gn(static_cast<size_t>(m_total_num_clusters));
            going_next.view(0, m_total_num_clusters).copy_to(h_gn.data());
            j["going_next"] = h_gn;
        }

        // real_to_part: fine real index -> padded fine position (for restriction)
        if(m_total_nodes > 0)
        {
            std::vector<int> h_r2p(static_cast<size_t>(m_total_nodes));
            real_to_part.view(0, m_total_nodes).copy_to(h_r2p.data());
            j["real_to_part"] = h_r2p;
        }

        out << j.dump(4) << '\n';
        logger::info("MAS dump: wrote {}", path.string());
    }
}

}  // namespace uipc::backend::cuda
