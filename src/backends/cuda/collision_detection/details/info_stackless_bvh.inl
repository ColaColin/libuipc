#include <cuda_device/builtin.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <cstdlib>
#include <cstdio>
#include <array>

// Implementation of InfoStacklessBVH.
// All build/sort/reorder functions are identical to InfoStacklessBVH.
// The two traversal functions (stacklessSelf / stacklessOther) are the
// optimized variants: they pre-load per-query bid/cid into shared memory
// ONCE before the traversal loop, eliminating repeated global-memory reads
// of query_bid/query_cid inside the hot node-cull path.

namespace uipc::info_stackless_detail
{
using aabb     = uipc::backend::cuda::AABB;
using node_t   = uipc::backend::cuda::InfoStacklessBVH::Node;
using Vector2i = uipc::Vector2i;
using uint     = uint32_t;
using ullint   = unsigned long long;

constexpr int  K_THREADS         = 256;
constexpr int  K_WARPS           = K_THREADS >> 5;
constexpr int  MAX_RES_PER_BLOCK = 1024;
constexpr int  AABB_BITS         = 15;
constexpr uint AABB_MASK         = 0xFFFFFFFFu >> (32 - AABB_BITS);

struct PlainAABB
{
    float3 _min, _max;
};

UIPC_GENERIC UIPC_INLINE PlainAABB to_plain(const aabb& box)
{
    PlainAABB out;
    out._min = make_float3(box.min().x(), box.min().y(), box.min().z());
    out._max = make_float3(box.max().x(), box.max().y(), box.max().z());
    return out;
}

template <typename T>
UIPC_GENERIC UIPC_INLINE T mm_min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
UIPC_GENERIC UIPC_INLINE T mm_max(T a, T b)
{
    return a > b ? a : b;
}

UIPC_DEVICE UIPC_INLINE float atomic_minf(float* addr, float value)
{
    return (value >= 0) ?
               __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
               __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
}

UIPC_DEVICE UIPC_INLINE float atomic_maxf(float* addr, float value)
{
    return (value >= 0) ?
               __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
               __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

UIPC_GENERIC UIPC_INLINE uint expand_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

UIPC_GENERIC UIPC_INLINE uint morton3D(float x, float y, float z)
{
    x       = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
    y       = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
    z       = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expand_bits((uint)x);
    uint yy = expand_bits((uint)y);
    uint zz = expand_bits((uint)z);
    return xx * 4 + yy * 2 + zz;
}

UIPC_GENERIC UIPC_INLINE Vector2i to_eigen(int2 v)
{
    return Vector2i{v.x, v.y};
}

UIPC_GENERIC UIPC_INLINE int2 ordered_pair(int a, int b)
{
    return (a < b) ? int2{a, b} : int2{b, a};
}

UIPC_GENERIC UIPC_INLINE float3 operator-(const float3& v0, const float3& v1)
{
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

UIPC_GENERIC UIPC_INLINE void safe_copy_to(int2*     shared_res,
                                           int       total_in_block,
                                           Vector2i* global_res,
                                           int       global_idx,
                                           int       max_res)
{
    if(global_idx >= max_res || total_in_block == 0)
        return;
    auto copy_count  = std::min(total_in_block, max_res - global_idx);
    int  full_blocks = (copy_count - 1) / (int)blockDim.x;
    for(int i = 0; i < full_blocks; ++i)
    {
        int offset                      = i * blockDim.x + threadIdx.x;
        global_res[global_idx + offset] = to_eigen(shared_res[offset]);
    }
    int offset = full_blocks * blockDim.x + threadIdx.x;
    if(offset < copy_count)
        global_res[global_idx + offset] = to_eigen(shared_res[offset]);
}
}  // namespace uipc::info_stackless_detail

namespace uipc::backend::cuda
{
using namespace info_stackless_detail;

namespace
{
    __global__ void InfoStacklessBVH_initializeBuildState_kernel(
        int                             num_objs,
        cuda_tool::Dense<AABB>          scene_box,
        cuda_tool::BufferView<uint32_t> flags,
        cuda_tool::BufferView<int>      ext_lca,
        cuda_tool::BufferView<uint32_t> depths,
        cuda_tool::BufferView<int32_t>  unsorted_ids)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx == 0)
            *scene_box = AABB();
        if(idx < num_objs - 1)
            flags(idx) = 0;
        if(idx < num_objs)
        {
            depths(idx)       = 0;
            unsorted_ids(idx) = idx;
        }
        if(idx <= num_objs)
            ext_lca(idx) = -1;
    }

    __global__ void InfoStacklessBVH_initializeQueryState_kernel(
        int num_objs, cuda_tool::Dense<AABB> scene_box, cuda_tool::BufferView<int> unsorted_ids)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx == 0)
            *scene_box = AABB();
        if(idx < num_objs)
            unsorted_ids(idx) = idx;
    }

    __global__ void InfoStacklessBVH_calcMaxBVFromBox_kernel(
        size_t size, cuda_tool::CBufferView<AABB> box, cuda_tool::Dense<AABB> out)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size)
            return;

        __shared__ PlainAABB warp_boxes[K_WARPS];
        auto                 temp     = to_plain(box(idx));
        int                  warp_tid = threadIdx.x & 31;
        int                  warp_id  = threadIdx.x >> 5;

        float minx = temp._min.x, miny = temp._min.y, minz = temp._min.z;
        float maxx = temp._max.x, maxy = temp._max.y, maxz = temp._max.z;
        for(int i = 1; i < 32; i <<= 1)
        {
            minx = mm_min(minx, __shfl_down_sync(0xffffffff, minx, i));
            miny = mm_min(miny, __shfl_down_sync(0xffffffff, miny, i));
            minz = mm_min(minz, __shfl_down_sync(0xffffffff, minz, i));
            maxx = mm_max(maxx, __shfl_down_sync(0xffffffff, maxx, i));
            maxy = mm_max(maxy, __shfl_down_sync(0xffffffff, maxy, i));
            maxz = mm_max(maxz, __shfl_down_sync(0xffffffff, maxz, i));
        }
        if(warp_tid == 0)
        {
            warp_boxes[warp_id]._min = make_float3(minx, miny, minz);
            warp_boxes[warp_id]._max = make_float3(maxx, maxy, maxz);
        }
        __syncthreads();

        int warp_num = (blockIdx.x == gridDim.x - 1) ?
                           ((size - blockIdx.x * blockDim.x + 31) >> 5) :
                           (blockDim.x >> 5);
        if(threadIdx.x >= warp_num)
            return;

        temp = warp_boxes[threadIdx.x];
        minx = temp._min.x;
        miny = temp._min.y;
        minz = temp._min.z;
        maxx = temp._max.x;
        maxy = temp._max.y;
        maxz = temp._max.z;
        for(int i = 1; i < warp_num; i <<= 1)
        {
            minx = mm_min(minx, __shfl_down_sync(0xffffffff, minx, i));
            miny = mm_min(miny, __shfl_down_sync(0xffffffff, miny, i));
            minz = mm_min(minz, __shfl_down_sync(0xffffffff, minz, i));
            maxx = mm_max(maxx, __shfl_down_sync(0xffffffff, maxx, i));
            maxy = mm_max(maxy, __shfl_down_sync(0xffffffff, maxy, i));
            maxz = mm_max(maxz, __shfl_down_sync(0xffffffff, maxz, i));
        }
        if(threadIdx.x == 0)
        {
            atomic_minf(&out->min().x(), minx);
            atomic_minf(&out->min().y(), miny);
            atomic_minf(&out->min().z(), minz);
            atomic_maxf(&out->max().x(), maxx);
            atomic_maxf(&out->max().y(), maxy);
            atomic_maxf(&out->max().z(), maxz);
        }
    }

    __global__ void InfoStacklessBVH_calcMCsFromBox_kernel(cuda_tool::CBufferView<AABB> box,
                                                           cuda_tool::CDense<AABB> scene,
                                                           cuda_tool::BufferView<uint32_t> codes,
                                                           int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        auto   bv        = box(idx);
        auto   center    = bv.center();
        float3 c         = make_float3(center.x(), center.y(), center.z());
        auto   scene_min = scene->min();
        float3 smin = make_float3(scene_min.x(), scene_min.y(), scene_min.z());
        auto   scene_size = scene->sizes();
        float3 off        = c - smin;
        codes(idx)        = morton3D(off.x / scene_size.x(),
                              off.y / scene_size.y(),
                              off.z / scene_size.z());
    }

    __global__ void InfoStacklessBVH_calcInverseMapping_kernel(
        cuda_tool::BufferView<int32_t> map, cuda_tool::BufferView<int32_t> inv, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        inv(map(idx)) = idx;
    }

    __global__ void InfoStacklessBVH_buildPrimitivesFromBox_kernel(
        cuda_tool::BufferView<int>     _prim_idx,
        cuda_tool::BufferView<AABB>    _prim_box,
        cuda_tool::BufferView<int32_t> _prim_map,
        cuda_tool::BufferView<IndexT>  _ext_bid,
        cuda_tool::BufferView<IndexT>  _ext_cid,
        cuda_tool::CBufferView<IndexT> _bids,
        cuda_tool::CBufferView<IndexT> _cids,
        bool                           has_info,
        cuda_tool::CBufferView<AABB>   box,
        int                            n)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        int              idx     = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        int new_idx        = _prim_map(idx);
        _prim_idx(new_idx) = idx;
        _prim_box(new_idx) = box(idx);
        if(has_info)
        {
            _ext_bid(new_idx) = _bids(idx);
            _ext_cid(new_idx) = _cids(idx);
        }
        else
        {
            _ext_bid(new_idx) = invalid;
            _ext_cid(new_idx) = invalid;
        }
    }

    __global__ void InfoStacklessBVH_calcExtNodeSplitMetrics_kernel(
        cuda_tool::BufferView<uint32_t> codes, cuda_tool::BufferView<int> metrics, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        metrics(idx) = idx != n - 1 ? 32 - __clz(codes(idx) ^ codes(idx + 1)) : 33;
    }

    __global__ void InfoStacklessBVH_buildIntNodes_kernel(
        int                             size,
        cuda_tool::BufferView<uint32_t> _depths,
        cuda_tool::BufferView<int>      _lvs_lca,
        cuda_tool::BufferView<int>      _lvs_metric,
        cuda_tool::BufferView<uint32_t> _lvs_par,
        cuda_tool::BufferView<AABB>     _lvs_box,
        cuda_tool::BufferView<IndexT>   _lvs_bid,
        cuda_tool::BufferView<IndexT>   _lvs_cid,
        cuda_tool::BufferView<int>      _tks_lc,
        cuda_tool::BufferView<int>      _tks_rc,
        cuda_tool::BufferView<int>      _tks_range_x,
        cuda_tool::BufferView<int>      _tks_range_y,
        cuda_tool::BufferView<uint32_t> _tks_mark,
        cuda_tool::BufferView<AABB>     _tks_box,
        cuda_tool::BufferView<IndexT>   _tks_bid,
        cuda_tool::BufferView<IndexT>   _tks_cid,
        cuda_tool::BufferView<uint32_t> _flag,
        cuda_tool::BufferView<int>      _tks_par)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        int              idx     = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size)
            return;

        int  l        = idx - 1;
        int  r        = idx;
        bool mark     = (l >= 0) ? (_lvs_metric(l) < _lvs_metric(r)) : false;
        int  cur      = mark ? l : r;
        _lvs_par(idx) = cur;
        if(_flag.total_size() == 0)
            return;

        if(mark)
        {
            _tks_rc(cur)      = idx;
            _tks_range_y(cur) = idx;
            atomicOr(&_tks_mark(cur), 0x00000002);
        }
        else
        {
            _tks_lc(cur)      = idx;
            _tks_range_x(cur) = idx;
            atomicOr(&_tks_mark(cur), 0x00000001);
        }
        __threadfence();

        while(atomicAdd(&_flag(cur), 1) == 1)
        {
            int      chl = _tks_lc(cur);
            int      chr = _tks_rc(cur);
            uint32_t m   = _tks_mark(cur);
            if(m & 1)
                _tks_box(cur) = _lvs_box(chl);
            else
                _tks_box(cur) = _tks_box(chl);
            if(m & 2)
                _tks_box(cur).extend(_lvs_box(chr));
            else
                _tks_box(cur).extend(_tks_box(chr));

            IndexT l_bid  = (m & 1) ? _lvs_bid(chl) : _tks_bid(chl);
            IndexT r_bid  = (m & 2) ? _lvs_bid(chr) : _tks_bid(chr);
            IndexT l_cid  = (m & 1) ? _lvs_cid(chl) : _tks_cid(chl);
            IndexT r_cid  = (m & 2) ? _lvs_cid(chr) : _tks_cid(chr);
            _tks_bid(cur) = (l_bid == r_bid) ? l_bid : invalid;
            _tks_cid(cur) = (l_cid == r_cid) ? l_cid : invalid;

            _tks_mark(cur) &= 0x00000007;
            l               = _tks_range_x(cur) - 1;
            r               = _tks_range_y(cur);
            _lvs_lca(l + 1) = cur;
            _depths(l + 1)++;
            mark = (l >= 0) ? (_lvs_metric(l) < _lvs_metric(r)) : false;
            if(l + 1 == 0 && r == size - 1)
            {
                _tks_par(cur) = -1;
                _tks_mark(cur) &= 0xFFFFFFFB;
                break;
            }

            int par       = mark ? l : r;
            _tks_par(cur) = par;
            if(mark)
            {
                _tks_rc(par)      = cur;
                _tks_range_y(par) = r;
                atomicAnd(&_tks_mark(par), 0xFFFFFFFD);
                _tks_mark(cur) |= 0x00000004;
            }
            else
            {
                _tks_lc(par)      = cur;
                _tks_range_x(par) = l + 1;
                atomicAnd(&_tks_mark(par), 0xFFFFFFFE);
                _tks_mark(cur) &= 0xFFFFFFFB;
            }
            __threadfence();
            cur = par;
        }
    }

    // perf/kernels: refit kernels. Leaf boxes are gathered in the sorted
    // order; internal boxes are recomputed with the same bottom-up climb as
    // the build (flags counter: the second child to arrive continues).
    __global__ void InfoStacklessBVH_refitExtNodes_kernel(cuda_tool::CBufferView<int> _prim_idx,
                                                          cuda_tool::CBufferView<AABB> box,
                                                          cuda_tool::BufferView<AABB> _prim_box,
                                                          int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        _prim_box(idx) = box(_prim_idx(idx));
    }

    __global__ void InfoStacklessBVH_refitIntNodes_kernel(
        int                              size,
        cuda_tool::CBufferView<uint32_t> _lvs_par,
        cuda_tool::CBufferView<AABB>     _lvs_box,
        cuda_tool::CBufferView<int>      _tks_lc,
        cuda_tool::CBufferView<int>      _tks_rc,
        cuda_tool::CBufferView<uint32_t> _tks_mark,
        cuda_tool::CBufferView<int>      _tks_par,
        cuda_tool::BufferView<AABB>      _tks_box,
        cuda_tool::BufferView<uint32_t>  _flag)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size)
            return;
        if(size == 1)
            return;
        int cur = static_cast<int>(_lvs_par(idx));
        __threadfence();
        while(atomicAdd(&_flag(cur), 1) == 1)
        {
            int      chl = _tks_lc(cur);
            int      chr = _tks_rc(cur);
            uint32_t m   = _tks_mark(cur);
            AABB     b   = (m & 1) ? _lvs_box(chl) : _tks_box(chl);
            if(m & 2)
                b.extend(_lvs_box(chr));
            else
                b.extend(_tks_box(chr));
            _tks_box(cur) = b;
            __threadfence();
            int par = _tks_par(cur);
            if(par < 0)
                break;
            cur = par;
        }
    }

    __global__ void InfoStacklessBVH_calcIntNodeOrders_kernel(
        cuda_tool::BufferView<int>      _tks_lc,
        cuda_tool::BufferView<int>      _lcas,
        cuda_tool::BufferView<uint32_t> _depths,
        cuda_tool::BufferView<uint32_t> _offsets,
        cuda_tool::BufferView<int>      _tkMap,
        int                             n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        int node  = _lcas(idx);
        int depth = _depths(idx);
        int id    = _offsets(idx);
        if(node != -1)
        {
            for(; depth--; node = _tks_lc(node))
                _tkMap(node) = id++;
        }
    }

    __global__ void InfoStacklessBVH_updateBvhExtNodeLinks_kernel(
        cuda_tool::BufferView<int>      _map,
        cuda_tool::BufferView<int>      _lcas,
        cuda_tool::BufferView<uint32_t> _pars,
        int                             n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;
        _pars(idx) = _map(_pars(idx));
        int ori    = _lcas(idx);
        _lcas(idx) = (ori != -1) ? (_map(ori) << 1) : (idx << 1 | 1);
    }

    __global__ void InfoStacklessBVH_reorderNode_kernel(
        int                                           int_size,
        cuda_tool::BufferView<int>                    _lvs_lca,
        cuda_tool::BufferView<AABB>                   _lvs_box,
        cuda_tool::BufferView<IndexT>                 _lvs_bid,
        cuda_tool::BufferView<IndexT>                 _lvs_cid,
        cuda_tool::BufferView<int>                    _tk_map,
        cuda_tool::BufferView<int>                    _int_lc,
        cuda_tool::BufferView<uint32_t>               _int_mark,
        cuda_tool::BufferView<int>                    _int_range_y,
        cuda_tool::BufferView<AABB>                   _int_box,
        cuda_tool::BufferView<IndexT>                 _int_bid,
        cuda_tool::BufferView<IndexT>                 _int_cid,
        cuda_tool::BufferView<InfoStacklessBVH::Node> _nodes,
        cuda_tool::BufferView<int>                    _node_range_y,
        int                                           count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= count)
            return;
        InfoStacklessBVH::Node leaf;
        leaf.lc    = -1;
        int escape = _lvs_lca(idx + 1);
        if(escape == -1)
            leaf.escape = -1;
        else
        {
            int b_leaf = escape & 1;
            escape >>= 1;
            leaf.escape = escape + (b_leaf ? int_size : 0);
        }
        leaf.bound                    = _lvs_box(idx);
        leaf.bid                      = _lvs_bid(idx);
        leaf.cid                      = _lvs_cid(idx);
        _nodes(idx + int_size)        = leaf;
        _node_range_y(idx + int_size) = idx;  // K11: a leaf is its own range

        if(idx >= int_size)
            return;

        InfoStacklessBVH::Node n;
        int                    new_id = _tk_map(idx);
        uint32_t               m      = _int_mark(idx);
        n.lc    = (m & 1) ? _int_lc(idx) + int_size : _tk_map(_int_lc(idx));
        n.bound = _int_box(idx);
        int ie  = _lvs_lca(_int_range_y(idx) + 1);
        if(ie == -1)
            n.escape = -1;
        else
        {
            int b_leaf = ie & 1;
            ie >>= 1;
            n.escape = ie + (b_leaf ? int_size : 0);
        }
        n.bid                 = _int_bid(idx);
        n.cid                 = _int_cid(idx);
        _nodes(new_id)        = n;
        _node_range_y(new_id) = _int_range_y(idx);  // K11
    }

    // perf/round4 (s04): the leaf-predicate evaluation of one staged pair,
    // shared by the inline path, the overflow fallback and the filter kernel
    // (same LeafPredInfo as the original traversal for either query type).
    template <bool Self, typename PairPred>
    UIPC_DEVICE UIPC_INLINE bool eval_leaf_pair(int             idx,
                                               int             leaf_raw,
                                               IndexT          qbid,
                                               IndexT          qcid,
                                               IndexT          lbid,
                                               IndexT          lcid,
                                               const PairPred& pair_pred,
                                               int2&           pair)
    {
        if constexpr(Self)
        {
            bool q_first = (idx < leaf_raw);
            pair         = ordered_pair(idx, leaf_raw);
            InfoStacklessBVH::LeafPredInfo info{pair.x,
                                                pair.y,
                                                q_first ? qbid : lbid,
                                                q_first ? qcid : lcid,
                                                q_first ? lbid : qbid,
                                                q_first ? lcid : qcid};
            return pair_pred(info);
        }
        else
        {
            pair = int2{idx, leaf_raw};
            InfoStacklessBVH::LeafPredInfo info{idx, leaf_raw, qbid, qcid, lbid, lcid};
            return pair_pred(info);
        }
    }

    // perf/round4 (s04): second phase — one thread per staged broad pair
    // (query raw index, leaf sorted position); survivors are compacted per
    // block into `res`. Blocks past the staged count exit before the barrier.
    template <bool Self, typename PairPred>
    __global__ void InfoStacklessBVH_pairFilter_kernel(cuda_tool::CBufferView<Vector2i> broad,
                                                       cuda_tool::Dense<int> broadCounter,
                                                       cuda_tool::CBufferView<int> _lvs_idx,
                                                       cuda_tool::CBufferView<IndexT> _lvs_bid,
                                                       cuda_tool::CBufferView<IndexT> _lvs_cid,
                                                       cuda_tool::CBufferView<IndexT> _qbids,
                                                       cuda_tool::CBufferView<IndexT> _qcids,
                                                       bool qhas_info,
                                                       cuda_tool::Dense<int> resCounter,
                                                       cuda_tool::BufferView<Vector2i> res,
                                                       PairPred pair_pred)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        __shared__ int2  s_hits[K_THREADS];
        __shared__ int   s_cnt;
        __shared__ int   s_base;
        const int n     = min(*broadCounter.data(), static_cast<int>(broad.size()));
        const int first = blockIdx.x * blockDim.x;
        if(first >= n)
            return;
        if(threadIdx.x == 0)
            s_cnt = 0;
        __syncthreads();
        const int k = first + threadIdx.x;
        if(k < n)
        {
            Vector2i bp   = broad(k);
            int      idx  = bp.x();
            int      pos  = bp.y();
            IndexT   qbid = qhas_info ? _qbids(idx) : invalid;
            IndexT   qcid = qhas_info ? _qcids(idx) : invalid;
            int2     pair;
            if(eval_leaf_pair<Self>(
                   idx, _lvs_idx(pos), qbid, qcid, _lvs_bid(pos), _lvs_cid(pos), pair_pred, pair))
            {
                int si     = atomicAdd(&s_cnt, 1);
                s_hits[si] = pair;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0)
            s_base = atomicAdd(resCounter.data(), s_cnt);
        __syncthreads();
        safe_copy_to(s_hits, s_cnt, res.data(), s_base, static_cast<int>(res.total_size()));
    }

    // perf/round4 (s04): flush the staged broad pairs of a block to `broad`;
    // the pairs that do not fit are evaluated inline by the block's threads
    // and their hits appended to `res` directly (rare: the capacity grows on
    // the next prepare_query_result).
    template <bool Self, typename PairPred>
    UIPC_DEVICE UIPC_INLINE void flush_broad_pairs(int2* shared_res,
                                                   int   total,
                                                   int   gidx,
                                                   cuda_tool::BufferView<Vector2i> broad,
                                                   cuda_tool::CBufferView<int>     _lvs_idx,
                                                   cuda_tool::CBufferView<IndexT>  _lvs_bid,
                                                   cuda_tool::CBufferView<IndexT>  _lvs_cid,
                                                   cuda_tool::CBufferView<IndexT>  _qbids,
                                                   cuda_tool::CBufferView<IndexT>  _qcids,
                                                   bool                            qhas_info,
                                                   cuda_tool::Dense<int>           resCounter,
                                                   cuda_tool::BufferView<Vector2i> res,
                                                   const PairPred&                 pair_pred)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        const int        cap     = static_cast<int>(broad.total_size());
        const int        fit     = (gidx >= cap) ? 0 : min(total, cap - gidx);
        safe_copy_to(shared_res, fit, broad.data(), gidx, cap);
        const int res_cap = static_cast<int>(res.total_size());
        for(int k = fit + static_cast<int>(threadIdx.x); k < total; k += static_cast<int>(blockDim.x))
        {
            int2   bp   = shared_res[k];
            int    idx  = bp.x;
            int    pos  = bp.y;
            IndexT qbid = qhas_info ? _qbids(idx) : invalid;
            IndexT qcid = qhas_info ? _qcids(idx) : invalid;
            int2   pair;
            if(eval_leaf_pair<Self>(
                   idx, _lvs_idx(pos), qbid, qcid, _lvs_bid(pos), _lvs_cid(pos), pair_pred, pair))
            {
                int g = atomicAdd(resCounter.data(), 1);
                if(g < res_cap)
                    res(g) = to_eigen(pair);
            }
        }
    }

    UIPC_DEVICE UIPC_INLINE unsigned long long warp_sum_ull(unsigned int v)
    {
        unsigned long long s = v;
        for(int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, o);
        return s;
    }

    // perf/round4 (s04): TwoPhase = stage the (query, leaf position) pairs
    // for InfoStacklessBVH_pairFilter_kernel instead of evaluating the leaf
    // predicate here (the predicate ran warp-divergent at the leaves and was
    // 92-95 % of this kernel's time); Stats = probe launch
    // (UIPC_BVH_SELF_STATS=1): same traversal, counts visits, no predicate,
    // no output.
    template <bool TwoPhase, bool Stats, typename NodeCull, typename PairPred>
    __global__ void InfoStacklessBVH_stacklessSelf_kernel(
        int                                           Size,
        cuda_tool::CBufferView<AABB>                  _box,
        int                                           intSize,
        int                                           numObjs,
        cuda_tool::BufferView<int>                    _lvs_idx,
        cuda_tool::CBufferView<IndexT>                _lvs_bid,
        cuda_tool::CBufferView<IndexT>                _lvs_cid,
        cuda_tool::BufferView<InfoStacklessBVH::Node> _nodes,
        cuda_tool::BufferView<int>                    _node_range_y,
        bool                                          range_cull,
        cuda_tool::CBufferView<IndexT>                _bids,
        cuda_tool::CBufferView<IndexT>                _cids,
        bool                                          has_info,
        cuda_tool::Dense<int>                         resCounter,
        cuda_tool::BufferView<Vector2i>               res,
        cuda_tool::Dense<int>                         broadCounter,
        cuda_tool::BufferView<Vector2i>               broad,
        unsigned long long*                           stats,
        NodeCull                                      node_cull,
        PairPred                                      pair_pred)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        int              tid     = blockIdx.x * blockDim.x + threadIdx.x;
        bool             active  = tid < Size;
        int              idx     = -1;
        AABB             bv;
        if(active)
        {
            idx = _lvs_idx(tid);
            bv  = _box(idx);
        }

        // -----------------------------------------------------------------
        // SMem: pre-load query bid/cid once per thread, before hot loop.
        // Shared memory layout (per block, K_THREADS=256):
        //   s_qbid[256]   = 1 KB
        //   s_qcid[256]   = 1 KB
        //   shared_res[1024 * sizeof(int2)] = 8 KB   (existing)
        //   shared_counter, shared_global_idx         (existing)
        // Total: ~10 KB — well within the 48 KB limit.
        // -----------------------------------------------------------------
        __shared__ IndexT s_qbid[K_THREADS];
        __shared__ IndexT s_qcid[K_THREADS];

        s_qbid[threadIdx.x] = (active && has_info) ? _bids(idx) : invalid;
        s_qcid[threadIdx.x] = (active && has_info) ? _cids(idx) : invalid;

        __shared__ int2 shared_res[MAX_RES_PER_BLOCK];
        __shared__ int  shared_counter;
        __shared__ int  shared_global_idx;
        if(threadIdx.x == 0)
            shared_counter = 0;

        int       st       = 0;
        const int max_iter = numObjs * 2;
        // probe counters: visited, range-culled, box-missed, cull-missed,
        // leaf nodes reached, ordered pair tests
        unsigned int c_visit = 0, c_range = 0, c_box = 0, c_cull = 0, c_leaf = 0, c_test = 0;
        while(true)
        {
            // First __syncthreads also ensures SMem pre-loads above are
            // visible to all threads in the block (though each thread
            // only reads its own slot: s_qbid[threadIdx.x]).
            __syncthreads();
            if(active)
            {
                int inner_i = 0;
                for(; inner_i < max_iter; ++inner_i)
                {
                    if(st == -1)
                        break;
                    auto node = _nodes(st);
                    if constexpr(Stats)
                        ++c_visit;
                    // perf/kernels (K11): only pairs with the partner leaf
                    // at a later sorted position are recorded (tid < leaf,
                    // below), so a subtree whose last leaf is <= tid cannot
                    // contribute — skip it without the box tests. The leaf
                    // test below is unchanged, so the pair set is identical.
                    if(range_cull && _node_range_y(st) <= tid)
                    {
                        if constexpr(Stats)
                            ++c_range;
                        st = node.escape;
                        continue;
                    }
                    if(!node.bound.intersects(bv))
                    {
                        if constexpr(Stats)
                            ++c_box;
                        st = node.escape;
                        continue;
                    }
                    if(!node_cull(InfoStacklessBVH::NodePredInfo{
                           idx, s_qbid[threadIdx.x], s_qcid[threadIdx.x], node.bid, node.cid}))
                    {
                        if constexpr(Stats)
                            ++c_cull;
                        st = node.escape;
                        continue;
                    }
                    if(node.lc == -1)
                    {
                        if constexpr(Stats)
                            ++c_leaf;
                        const int pos = st - intSize;
                        if(tid < pos)
                        {
                            if constexpr(Stats)
                            {
                                ++c_test;
                            }
                            else if constexpr(TwoPhase)
                            {
                                int sidx = atomicAdd(&shared_counter, 1);
                                if(sidx >= MAX_RES_PER_BLOCK)
                                    break;
                                shared_res[sidx] = int2{idx, pos};
                            }
                            else
                            {
                                int2 pair;
                                if(eval_leaf_pair<true>(idx,
                                                        _lvs_idx(pos),
                                                        s_qbid[threadIdx.x],
                                                        s_qcid[threadIdx.x],
                                                        node.bid,
                                                        node.cid,
                                                        pair_pred,
                                                        pair))
                                {
                                    int sidx = atomicAdd(&shared_counter, 1);
                                    if(sidx >= MAX_RES_PER_BLOCK)
                                        break;
                                    shared_res[sidx] = pair;
                                }
                            }
                        }
                        st = node.escape;
                    }
                    else
                        st = node.lc;
                }
                UIPC_KERNEL_ASSERT(inner_i < max_iter, "Exceeded max stackless iteration");
            }
            __syncthreads();
            int total = min(shared_counter, MAX_RES_PER_BLOCK);
            if(threadIdx.x == 0)
                shared_global_idx =
                    atomicAdd(TwoPhase ? broadCounter.data() : resCounter.data(), total);
            __syncthreads();
            int gidx = shared_global_idx;
            if(threadIdx.x == 0)
                shared_counter = 0;
            bool done = total < MAX_RES_PER_BLOCK;
            if constexpr(TwoPhase)
                flush_broad_pairs<true>(shared_res,
                                        total,
                                        gidx,
                                        broad,
                                        _lvs_idx,
                                        _lvs_bid,
                                        _lvs_cid,
                                        _bids,
                                        _cids,
                                        has_info,
                                        resCounter,
                                        res,
                                        pair_pred);
            else
                safe_copy_to(shared_res, total, res.data(), gidx, static_cast<int>(res.total_size()));
            if(done)
                break;
        }
        if constexpr(Stats)
        {
            // `done` is block-uniform, so the warp is converged here
            unsigned long long v[6] = {warp_sum_ull(c_visit),
                                       warp_sum_ull(c_range),
                                       warp_sum_ull(c_box),
                                       warp_sum_ull(c_cull),
                                       warp_sum_ull(c_leaf),
                                       warp_sum_ull(c_test)};
            if((threadIdx.x & 31) == 0)
                for(int k = 0; k < 6; ++k)
                    atomicAdd(stats + k, v[k]);
        }
    }

    template <bool TwoPhase, typename NodeCull, typename PairPred>
    __global__ void InfoStacklessBVH_stacklessOther_kernel(
        int                                           Size,
        cuda_tool::CBufferView<AABB>                  _box,
        cuda_tool::CBufferView<int>                   sortedIdx,
        int                                           intSize,
        int                                           numObjs,
        cuda_tool::BufferView<int>                    _lvs_idx,
        cuda_tool::CBufferView<IndexT>                _lvs_bid,
        cuda_tool::CBufferView<IndexT>                _lvs_cid,
        cuda_tool::BufferView<InfoStacklessBVH::Node> _nodes,
        cuda_tool::CBufferView<IndexT>                _qbids,
        cuda_tool::CBufferView<IndexT>                _qcids,
        bool                                          qhas_info,
        cuda_tool::Dense<int>                         resCounter,
        cuda_tool::BufferView<Vector2i>               res,
        cuda_tool::Dense<int>                         broadCounter,
        cuda_tool::BufferView<Vector2i>               broad,
        NodeCull                                      node_cull,
        PairPred                                      pair_pred)
    {
        constexpr IndexT invalid = static_cast<IndexT>(-1);
        int              tid     = blockIdx.x * blockDim.x + threadIdx.x;
        bool             active  = tid < Size;
        int              idx     = -1;
        AABB             bv;
        if(active)
        {
            idx = sortedIdx(tid);
            bv  = _box(idx);
        }

        // -----------------------------------------------------------------
        // SMem: pre-load per-query bid/cid before the traversal loop.
        // -----------------------------------------------------------------
        __shared__ IndexT s_qbid[K_THREADS];
        __shared__ IndexT s_qcid[K_THREADS];

        s_qbid[threadIdx.x] = (active && qhas_info) ? _qbids(idx) : invalid;
        s_qcid[threadIdx.x] = (active && qhas_info) ? _qcids(idx) : invalid;

        __shared__ int2 shared_res[MAX_RES_PER_BLOCK];
        __shared__ int  shared_counter;
        __shared__ int  shared_global_idx;
        if(threadIdx.x == 0)
            shared_counter = 0;

        int       st       = 0;
        const int max_iter = numObjs * 2;
        while(true)
        {
            __syncthreads();
            if(active)
            {
                int inner_i = 0;
                for(; inner_i < max_iter; ++inner_i)
                {
                    if(st == -1)
                        break;
                    auto node = _nodes(st);
                    if(!node.bound.intersects(bv))
                    {
                        st = node.escape;
                        continue;
                    }
                    if(!node_cull(InfoStacklessBVH::NodePredInfo{
                           idx, s_qbid[threadIdx.x], s_qcid[threadIdx.x], node.bid, node.cid}))
                    {
                        st = node.escape;
                        continue;
                    }
                    if(node.lc == -1)
                    {
                        const int pos = st - intSize;
                        if constexpr(TwoPhase)
                        {
                            int sidx = atomicAdd(&shared_counter, 1);
                            if(sidx >= MAX_RES_PER_BLOCK)
                                break;
                            shared_res[sidx] = int2{idx, pos};
                        }
                        else
                        {
                            // query side: SMem pre-loaded; leaf side: node.bid/cid
                            int2 pair;
                            if(eval_leaf_pair<false>(idx,
                                                     _lvs_idx(pos),
                                                     s_qbid[threadIdx.x],
                                                     s_qcid[threadIdx.x],
                                                     node.bid,
                                                     node.cid,
                                                     pair_pred,
                                                     pair))
                            {
                                int sidx = atomicAdd(&shared_counter, 1);
                                if(sidx >= MAX_RES_PER_BLOCK)
                                    break;
                                shared_res[sidx] = pair;
                            }
                        }
                        st = node.escape;
                    }
                    else
                        st = node.lc;
                }
                UIPC_KERNEL_ASSERT(inner_i < max_iter, "Exceeded max stackless iteration");
            }

            __syncthreads();
            int total = min(shared_counter, MAX_RES_PER_BLOCK);
            if(threadIdx.x == 0)
                shared_global_idx = atomicAdd(TwoPhase ? broadCounter.data() : resCounter.data(), total);
            __syncthreads();
            int gidx = shared_global_idx;
            if(threadIdx.x == 0)
                shared_counter = 0;
            __syncthreads();
            bool done = total < MAX_RES_PER_BLOCK;
            if constexpr(TwoPhase)
                flush_broad_pairs<false>(shared_res,
                                         total,
                                         gidx,
                                         broad,
                                         _lvs_idx,
                                         _lvs_bid,
                                         _lvs_cid,
                                         _qbids,
                                         _qcids,
                                         qhas_info,
                                         resCounter,
                                         res,
                                         pair_pred);
            else
                safe_copy_to(shared_res, total, res.data(), gidx, static_cast<int>(res.total_size()));
            if(done)
                break;
        }
    }
}  // namespace

// ---------------------------------------------------------------------------
// Build pipeline — identical to InfoStacklessBVH
// ---------------------------------------------------------------------------

inline void InfoStacklessBVH::Impl::calcMaxBVFromBox(cuda_tool::CBufferView<AABB> aabbs,
                                                     cuda_tool::VarView<AABB> scene_box)
{
    if(aabbs.size() == 0)
        return;

    auto num  = aabbs.size();
    auto grid = (num + K_THREADS - 1) / K_THREADS;

    if(grid > 0)
        InfoStacklessBVH_calcMaxBVFromBox_kernel<<<grid, K_THREADS, 0, nullptr>>>(
            aabbs.size(), aabbs, scene_box.viewer());
}

inline void InfoStacklessBVH::Impl::calcMCsFromBox(cuda_tool::CBufferView<AABB> aabbs,
                                                   cuda_tool::CVarView<AABB> scene_box,
                                                   cuda_tool::BufferView<uint32_t> codes)
{
    auto k = InfoStacklessBVH_calcMCsFromBox_kernel;
    int  n = static_cast<int>(aabbs.size());
    if(n > 0)
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            aabbs, scene_box.viewer(), codes, n);
}

inline void InfoStacklessBVH::Impl::calcInverseMapping()
{
    auto k = InfoStacklessBVH_calcInverseMapping_kernel;
    int  n = static_cast<int>(sorted_id.size());
    if(n > 0)
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            sorted_id.view(), primMap.view(), n);
}

inline void InfoStacklessBVH::Impl::buildPrimitivesFromBox(cuda_tool::CBufferView<AABB> aabbs)
{
    bool has_info = bids.size() == aabbs.size() && cids.size() == aabbs.size();
    auto k        = InfoStacklessBVH_buildPrimitivesFromBox_kernel;
    int  n        = static_cast<int>(aabbs.size());
    if(n > 0)
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            ext_idx.view(),
            ext_aabb.view(),
            primMap.view(),
            ext_bid.view(),
            ext_cid.view(),
            bids,
            cids,
            has_info,
            aabbs,
            n);
}

inline void InfoStacklessBVH::Impl::calcExtNodeSplitMetrics()
{
    auto k = InfoStacklessBVH_calcExtNodeSplitMetrics_kernel;
    int  n = static_cast<int>(sorted_mtcode.size());
    if(n > 0)
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            sorted_mtcode.view(), metric.view(), n);
}

inline void InfoStacklessBVH::Impl::buildIntNodes(int size)
{
    auto grid = (size + 255) / 256;
    if(grid > 0)
        InfoStacklessBVH_buildIntNodes_kernel<<<grid, 256, 0, nullptr>>>(
            size,
            count.view(),
            ext_lca.view(),
            metric.view(),
            ext_par.view(),
            ext_aabb.view(),
            ext_bid.view(),
            ext_cid.view(),
            int_lc.view(),
            int_rc.view(),
            int_range_x.view(),
            int_range_y.view(),
            int_mark.view(),
            int_aabb.view(),
            int_bid.view(),
            int_cid.view(),
            flags.view(),
            int_par.view());
}

inline void InfoStacklessBVH::Impl::calcIntNodeOrders(int size)
{
    auto k = InfoStacklessBVH_calcIntNodeOrders_kernel;
    if(size > 0)
        k<<<cuda_tool::best_grid_dim(size, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            int_lc.view(),
            ext_lca.view(),
            count.view(),
            offsetTable.view(),
            tkMap.view(),
            size);
}

inline void InfoStacklessBVH::Impl::updateBvhExtNodeLinks(int size)
{
    if(flags.size() == 0)
        return;
    auto k = InfoStacklessBVH_updateBvhExtNodeLinks_kernel;
    if(size > 0)
        k<<<cuda_tool::best_grid_dim(size, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            tkMap.view(), ext_lca.view(), ext_par.view(), size);
}

inline void InfoStacklessBVH::Impl::reorderNode(int int_size)
{
    auto k = InfoStacklessBVH_reorderNode_kernel;
    int  n = int_size + 1;
    node_range_y.resize(nodes.size());
    if(n > 0)
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            int_size,
            ext_lca.view(),
            ext_aabb.view(),
            ext_bid.view(),
            ext_cid.view(),
            tkMap.view(),
            int_lc.view(),
            int_mark.view(),
            int_range_y.view(),
            int_aabb.view(),
            int_bid.view(),
            int_cid.view(),
            nodes.view(),
            node_range_y.view(),
            n);
}

inline void InfoStacklessBVH::Impl::propagateInformativeMetadata(int) {}

inline void InfoStacklessBVH::Impl::refitExtNodes(cuda_tool::CBufferView<AABB> aabbs)
{
    auto k = InfoStacklessBVH_refitExtNodes_kernel;
    int  n = static_cast<int>(aabbs.size());
    if(n > 0)
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            ext_idx.view(), aabbs, ext_aabb.view(), n);
}

inline void InfoStacklessBVH::Impl::refitIntNodes(int size)
{
    auto k = InfoStacklessBVH_refitIntNodes_kernel;
    if(size > 1)
        k<<<cuda_tool::best_grid_dim(size, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            size,
            ext_par_orig.view(),
            ext_aabb.view(),
            int_lc.view(),
            int_rc.view(),
            int_mark.view(),
            int_par.view(),
            int_aabb.view(),
            flags.view());
}

inline void InfoStacklessBVH::Impl::refit(cuda_tool::CBufferView<AABB>   aabbs,
                                          cuda_tool::CBufferView<IndexT> _bids,
                                          cuda_tool::CBufferView<IndexT> _cids)
{
    objs          = aabbs;
    bids          = _bids;
    cids          = _cids;
    auto num_objs = aabbs.size();
    if(num_objs == 0)
        return;
    auto num_internal = num_objs - 1;
    if(num_internal > 0)
        cudaMemsetAsync(flags.data(), 0, sizeof(uint32_t) * num_internal, nullptr);
    refitExtNodes(aabbs);
    refitIntNodes(static_cast<int>(num_objs));
    reorderNode(static_cast<int>(num_internal));
}

inline void InfoStacklessBVH::Impl::build(cuda_tool::CBufferView<AABB>   aabbs,
                                          cuda_tool::CBufferView<IndexT> _bids,
                                          cuda_tool::CBufferView<IndexT> _cids)
{
    objs          = aabbs;
    bids          = _bids;
    cids          = _cids;
    auto num_objs = aabbs.size();
    if(num_objs == 0)
        return;

    auto num_internal = num_objs - 1;
    auto num_nodes    = num_objs * 2 - 1;
    mtcode.resize(num_objs);
    sorted_mtcode.resize(num_objs);
    sorted_id.resize(num_objs);
    primMap.resize(num_objs);
    ext_aabb.resize(num_objs);
    ext_idx.resize(num_objs);
    ext_lca.resize(num_objs + 1);
    ext_par.resize(num_objs);
    ext_bid.resize(num_objs);
    ext_cid.resize(num_objs);
    metric.resize(num_objs);
    tkMap.resize(num_objs);
    offsetTable.resize(num_objs);
    count.resize(num_objs);
    flags.resize(num_internal);
    int_lc.resize(num_internal);
    int_rc.resize(num_internal);
    int_par.resize(num_internal);
    int_range_x.resize(num_internal);
    int_range_y.resize(num_internal);
    int_mark.resize(num_internal);
    int_aabb.resize(num_internal);
    int_bid.resize(num_internal);
    int_cid.resize(num_internal);
    nodes.resize(num_nodes);

    auto init = InfoStacklessBVH_initializeBuildState_kernel;
    auto n    = static_cast<int>(num_objs);
    init<<<cuda_tool::best_grid_dim(n + 1, init), cuda_tool::best_block_dim(init), 0, nullptr>>>(
        n,
        scene_box.viewer(),
        flags.view(),
        ext_lca.view(),
        count.view(),
        primMap.view());

    calcMaxBVFromBox(aabbs, scene_box.view());
    calcMCsFromBox(aabbs, scene_box.view(), mtcode.view());
    // The cuda_tool CUB wrappers share persistent scratch per CUDA stream;
    // repeated BVH builds query the required size but do not reallocate it.
    cuda_tool::DeviceRadixSort().SortPairs(
        mtcode.data(), sorted_mtcode.data(), primMap.data(), sorted_id.data(), n);
    calcInverseMapping();
    buildPrimitivesFromBox(aabbs);
    calcExtNodeSplitMetrics();
    buildIntNodes(num_objs);
    cuda_tool::DeviceScan().ExclusiveSum(count.data(), offsetTable.data(), n);
    calcIntNodeOrders(num_objs);
    // keep the leaf parents in build ids for refit() before they are remapped
    ext_par_orig.resize(num_objs);
    cuda_tool::BufferLaunch().copy<uint32_t>(ext_par_orig.view(),
                                             std::as_const(ext_par).view());
    updateBvhExtNodeLinks(num_objs);
    reorderNode(num_internal);
}

// ---------------------------------------------------------------------------
// OPTIMIZED: stacklessSelf
//   Pre-loads query_bid and query_cid for each thread into shared memory
//   before the traversal loop. The node_cull functor receives a NodePredInfo
//   with query_bid/query_cid already filled from SMem — no global reads
//   inside the hot loop.
// ---------------------------------------------------------------------------
template <typename NodeCull, typename PairPred>
void InfoStacklessBVH::Impl::stacklessSelf(NodeCull node_cull, PairPred pair_pred, QueryBuffer& qbuffer)
{
    auto num_query = static_cast<int>(ext_aabb.size());
    auto num_objs  = num_query;
    auto grid      = (num_query + K_THREADS - 1) / K_THREADS;

    bool has_info = bids.size() == (size_t)num_objs && cids.size() == (size_t)num_objs;

    if(grid <= 0)
        return;

    unsigned long long* stats = nullptr;
    if(self_stats)
    {
        self_stat_counters.resize(8);
        cuda_tool::BufferLaunch().fill(self_stat_counters.view(), 0ull);
        stats = self_stat_counters.data();
    }

    auto launch = [&]<bool TwoPhase, bool Stats>()
    {
        InfoStacklessBVH_stacklessSelf_kernel<TwoPhase, Stats, NodeCull, PairPred>
            <<<grid, K_THREADS, 0, nullptr>>>(num_query,
                                              objs,
                                              num_objs - 1,
                                              num_objs,
                                              ext_idx.view(),
                                              ext_bid.view(),
                                              ext_cid.view(),
                                              nodes.view(),
                                              node_range_y.view(),
                                              self_range_cull,
                                              bids,
                                              cids,
                                              has_info,
                                              qbuffer.m_cpNum.viewer(),
                                              qbuffer.m_pairs.view(),
                                              qbuffer.m_broadNum.viewer(),
                                              qbuffer.m_broad.view(),
                                              stats,
                                              node_cull,
                                              pair_pred);
    };
    if(two_phase)
    {
        launch.template operator()<true, false>();
        auto bgrid = static_cast<int>((qbuffer.m_broad.size() + K_THREADS - 1) / K_THREADS);
        InfoStacklessBVH_pairFilter_kernel<true, PairPred>
            <<<bgrid, K_THREADS, 0, nullptr>>>(qbuffer.m_broad.view(),
                                               qbuffer.m_broadNum.viewer(),
                                               ext_idx.view(),
                                               ext_bid.view(),
                                               ext_cid.view(),
                                               bids,
                                               cids,
                                               has_info,
                                               qbuffer.m_cpNum.viewer(),
                                               qbuffer.m_pairs.view(),
                                               pair_pred);
    }
    else
    {
        launch.template operator()<false, false>();
    }
    // perf/round4 (s04): probe — a traversal-only launch (no predicate, no
    // output) after the real one, reporting the visit counters
    if(self_stats)
    {
        launch.template operator()<false, true>();
        std::array<unsigned long long, 8> h{};
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), stats, sizeof(h), cudaMemcpyDeviceToHost);
        std::fprintf(stderr,
                     "UIPC_BVH_SELF_STATS queries=%d visited=%llu range_cull=%llu box_miss=%llu cull_miss=%llu leaf=%llu pair_tests=%llu\n",
                     num_query,
                     h[0],
                     h[1],
                     h[2],
                     h[3],
                     h[4],
                     h[5]);
    }
}

// ---------------------------------------------------------------------------
// OPTIMIZED: stacklessOther
//   Takes explicit query_bids / query_cids buffers and pre-loads them into
//   shared memory. node_cull receives a NodePredInfo with query_bid/query_cid
//   already filled from SMem — no global reads inside the hot loop.
// ---------------------------------------------------------------------------
template <typename NodeCull, typename PairPred>
void InfoStacklessBVH::Impl::stacklessOther(NodeCull node_cull,
                                            PairPred pair_pred,
                                            cuda_tool::CBufferView<AABB> query_aabbs,
                                            cuda_tool::CBufferView<IndexT> query_bids,
                                            cuda_tool::CBufferView<IndexT> query_cids,
                                            cuda_tool::CBufferView<int> query_sorted_id,
                                            QueryBuffer&                qbuffer)
{
    auto num_query = static_cast<int>(query_aabbs.size());
    auto num_objs  = static_cast<int>(ext_aabb.size());
    auto grid      = (num_query + K_THREADS - 1) / K_THREADS;

    bool qhas_info = query_bids.size() == (size_t)num_query
                     && query_cids.size() == (size_t)num_query;

    if(grid <= 0)
        return;
    auto launch = [&]<bool TwoPhase>()
    {
        InfoStacklessBVH_stacklessOther_kernel<TwoPhase, NodeCull, PairPred>
            <<<grid, K_THREADS, 0, nullptr>>>(num_query,
                                              query_aabbs,
                                              query_sorted_id,
                                              num_objs - 1,
                                              num_objs,
                                              ext_idx.view(),
                                              ext_bid.view(),
                                              ext_cid.view(),
                                              nodes.view(),
                                              query_bids,
                                              query_cids,
                                              qhas_info,
                                              qbuffer.m_cpNum.viewer(),
                                              qbuffer.m_pairs.view(),
                                              qbuffer.m_broadNum.viewer(),
                                              qbuffer.m_broad.view(),
                                              node_cull,
                                              pair_pred);
    };
    if(two_phase)  // s04
    {
        launch.template operator()<true>();
        auto bgrid = static_cast<int>((qbuffer.m_broad.size() + K_THREADS - 1) / K_THREADS);
        InfoStacklessBVH_pairFilter_kernel<false, PairPred>
            <<<bgrid, K_THREADS, 0, nullptr>>>(qbuffer.m_broad.view(),
                                               qbuffer.m_broadNum.viewer(),
                                               ext_idx.view(),
                                               ext_bid.view(),
                                               ext_cid.view(),
                                               query_bids,
                                               query_cids,
                                               qhas_info,
                                               qbuffer.m_cpNum.viewer(),
                                               qbuffer.m_pairs.view(),
                                               pair_pred);
    }
    else
    {
        launch.template operator()<false>();
    }
}

// ---------------------------------------------------------------------------
// Public API — same signatures as InfoStacklessBVH
// ---------------------------------------------------------------------------

inline InfoStacklessBVH::InfoStacklessBVH(cuda_tool::Stream& stream) noexcept
{
    (void)stream;
    // perf/kernels (K11)
    const char* e          = std::getenv("UIPC_BVH_SELF_RANGE_CULL");
    m_impl.self_range_cull = !(e && e[0] == '0');
    // perf/round4 (s04)
    const char* tp    = std::getenv("UIPC_BVH_TWO_PHASE");
    m_impl.two_phase  = !(tp && tp[0] == '0');
    const char* st    = std::getenv("UIPC_BVH_SELF_STATS");
    m_impl.self_stats = st && st[0] == '1';
}

inline void InfoStacklessBVH::QueryBuffer::build(cuda_tool::CBufferView<AABB> aabbs)
{
    m_queryMtCode.resize(aabbs.size());
    m_querySortedMtCode.resize(aabbs.size());
    m_queryId.resize(aabbs.size());
    m_querySortedId.resize(aabbs.size());
    auto init = InfoStacklessBVH_initializeQueryState_kernel;
    auto n    = static_cast<int>(aabbs.size());
    if(n > 0)
        init<<<cuda_tool::best_grid_dim(n, init), cuda_tool::best_block_dim(init), 0, nullptr>>>(
            n, m_querySceneBox.viewer(), m_queryId.view());
    Impl::calcMaxBVFromBox(aabbs, m_querySceneBox);
    Impl::calcMCsFromBox(aabbs, m_querySceneBox, m_queryMtCode.view());
    cuda_tool::DeviceRadixSort().SortPairs(m_queryMtCode.data(),
                                           m_querySortedMtCode.data(),
                                           m_queryId.data(),
                                           m_querySortedId.data(),
                                           n);
}

inline void InfoStacklessBVH::build(cuda_tool::CBufferView<AABB>   aabbs,
                                    cuda_tool::CBufferView<IndexT> BIDs,
                                    cuda_tool::CBufferView<IndexT> CIDs)
{
    m_aabbs = aabbs;
    m_BIDs  = BIDs;
    m_CIDs  = CIDs;
    UIPC_ASSERT(m_aabbs.size() == m_BIDs.size(),
                "AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                m_aabbs.size(),
                m_BIDs.size());
    UIPC_ASSERT(m_aabbs.size() == m_CIDs.size(),
                "AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                m_aabbs.size(),
                m_CIDs.size());
    m_impl.build(aabbs, BIDs, CIDs);
}

inline void InfoStacklessBVH::refit(cuda_tool::CBufferView<AABB>   aabbs,
                                    cuda_tool::CBufferView<IndexT> BIDs,
                                    cuda_tool::CBufferView<IndexT> CIDs)
{
    m_aabbs = aabbs;
    m_BIDs  = BIDs;
    m_CIDs  = CIDs;
    m_impl.refit(aabbs, BIDs, CIDs);
}

inline void InfoStacklessBVH::build(cuda_tool::CBufferView<AABB> aabbs)
{
    m_aabbs = aabbs;
    m_BIDs  = {};
    m_CIDs  = {};
    m_impl.build(aabbs, {}, {});
}

inline bool InfoStacklessBVH::prepare_query_result(QueryBuffer& qbuffer, int count)
{
    UIPC_ASSERT(count >= 0, "BVH query count must be non-negative, got %d", count);

    const bool retry = static_cast<size_t>(count) > qbuffer.m_pairs.size();
    if(retry)
    {
        const auto new_size = static_cast<size_t>(count * m_impl.config.reserve_ratio);
        qbuffer.m_pairs.reserve_discard(new_size);
        qbuffer.m_pairs.resize_discard(new_size);
    }
    qbuffer.m_size = count;
    // perf/round4 (s04): the broad (two-phase) stage is complete even when it
    // overflowed (the overflow was evaluated inline); grow it for the next
    // query so the fallback stays rare. One 4-byte D2H at this (already
    // synchronising) point.
    if(m_impl.two_phase)
    {
        int broad_count = qbuffer.m_broadNum;
        if(broad_count > 0 && static_cast<size_t>(broad_count) > qbuffer.m_broad.size())
        {
            const auto new_size = static_cast<size_t>(broad_count * m_impl.config.reserve_ratio);
            qbuffer.m_broad.reserve_discard(new_size);
            qbuffer.m_broad.resize_discard(new_size);
        }
    }
    return retry;
}

template <typename NodePred, typename LeafPred>
inline void InfoStacklessBVH::launch_detect(cuda_tool::CBuffer2DView<IndexT> cmts,
                                            NodePred     np,
                                            LeafPred     lp,
                                            QueryBuffer& qbuffer)
{
    using namespace cuda_tool;
    BufferLaunch().fill(qbuffer.m_cpNum.view(), 0);

    if(m_aabbs.size() == 0)
        return;

    UIPC_ASSERT(m_aabbs.size() == m_BIDs.size(),
                "AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                m_aabbs.size(),
                m_BIDs.size());
    UIPC_ASSERT(m_aabbs.size() == m_CIDs.size(),
                "AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                m_aabbs.size(),
                m_CIDs.size());

    BufferLaunch().fill(qbuffer.m_broadNum.view(), 0);  // s04
    m_impl.stacklessSelf(np, lp, qbuffer);
}

// detect() with NodePred / LeafPred: wrapper passes pre-loaded bid/cid to NodePredInfo
template <typename NodePred, typename LeafPred>
inline void InfoStacklessBVH::detect(cuda_tool::CBuffer2DView<IndexT> cmts,
                                     NodePred                         np,
                                     LeafPred                         lp,
                                     QueryBuffer&                     qbuffer)
{
    launch_detect(cmts, np, lp, qbuffer);
    int h_cp_num = qbuffer.m_cpNum;
    if(prepare_query_result(qbuffer, h_cp_num))
        launch_detect(cmts, np, lp, qbuffer);
}

template <typename NodePred, typename LeafPred>
inline void InfoStacklessBVH::launch_query(cuda_tool::CBufferView<AABB> query_aabbs,
                                           cuda_tool::CBufferView<IndexT> query_BIDs,
                                           cuda_tool::CBufferView<IndexT> query_CIDs,
                                           cuda_tool::CBuffer2DView<IndexT> cmts,
                                           NodePred     np,
                                           LeafPred     lp,
                                           QueryBuffer& qbuffer,
                                           bool         rebuild_query)
{
    using namespace cuda_tool;
    BufferLaunch().fill(qbuffer.m_cpNum.view(), 0);

    if(m_aabbs.size() == 0 || query_aabbs.size() == 0)
        return;

    UIPC_ASSERT(query_aabbs.size() == query_BIDs.size(),
                "Query AABB and BID size mismatch. aabbs=%zu, bids=%zu",
                query_aabbs.size(),
                query_BIDs.size());
    UIPC_ASSERT(query_aabbs.size() == query_CIDs.size(),
                "Query AABB and CID size mismatch. aabbs=%zu, cids=%zu",
                query_aabbs.size(),
                query_CIDs.size());

    if(rebuild_query && !(qbuffer.m_built && qbuffer.m_built_n == query_aabbs.size()))
    {
        qbuffer.build(query_aabbs);
        qbuffer.m_built   = true;
        qbuffer.m_built_n = query_aabbs.size();
    }
    BufferLaunch().fill(qbuffer.m_broadNum.view(), 0);  // s04
    m_impl.stacklessOther(np, lp, query_aabbs, query_BIDs, query_CIDs, qbuffer.m_querySortedId.view(), qbuffer);
}

// query() with NodePred / LeafPred: passes query BIDs/CIDs to stacklessOther for SMem pre-load
template <typename NodePred, typename LeafPred>
inline void InfoStacklessBVH::query(cuda_tool::CBufferView<AABB>   query_aabbs,
                                    cuda_tool::CBufferView<IndexT> query_BIDs,
                                    cuda_tool::CBufferView<IndexT> query_CIDs,
                                    cuda_tool::CBuffer2DView<IndexT> cmts,
                                    NodePred                         np,
                                    LeafPred                         lp,
                                    QueryBuffer&                     qbuffer)
{
    launch_query(query_aabbs, query_BIDs, query_CIDs, cmts, np, lp, qbuffer, true);
    int h_cp_num = qbuffer.m_cpNum;
    if(prepare_query_result(qbuffer, h_cp_num))
        launch_query(query_aabbs, query_BIDs, query_CIDs, cmts, np, lp, qbuffer, false);
}

}  // namespace uipc::backend::cuda
