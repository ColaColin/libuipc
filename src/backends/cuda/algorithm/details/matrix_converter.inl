#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <cub/warp/warp_reduce.cuh>
#include <uipc/common/timer.h>
#include <algorithm/fast_segmental_reduce.h>
#include <cstdlib>
#include <utility>

namespace uipc::backend::cuda
{
namespace
{
    // block value type of DeviceTripletMatrix<T, N> (plain T when N == 1)
    template <typename T, int N>
    using MatrixConverterBlockT = typename cuda_tool::DeviceTripletMatrix<T, N>::ValueT;

    // ---------------------------------------------------------------
    // perf/round4 (s10): fused ge2sym + triplet->BCOO conversion.
    //
    // The old chain moved every 3x3 block three times: ge2sym k1 copied all
    // blocks into a temporary, ge2sym k2 compacted the upper-triangular ones
    // back, and the sort k3 gathered them into sorted order. The fused path
    // keeps the blocks where the assembly left them and carries *indices* only:
    // the ge2sym compaction produces (key, source index) pairs, the radix sort
    // permutes those, and a single gather (or the segmental reduce itself)
    // reads the original block array. Same values in the same order.
    // ---------------------------------------------------------------

    // perf/round5 (s25): drop triplet slots the assembly never wrote.
    //
    // A slot the subsystems did not write carries row = col = -1 (the caller
    // fills the index arrays with -1 before assembly). `row <= col` alone
    // *keeps* such a slot -- -1 <= -1 -- and hands it to the radix sort with a
    // 64-bit key of 0xFFFF.., of which the sort only looks at the low
    // bits(rows*cols-1), i.e. garbage that lands among the real entries. So an
    // unwritten slot has never been survivable, zero-filled values or not.
    //
    // `row >= 0` is the drop that GlobalLinearSystem's "Clear and invalidate
    // previous values" comment always implied, and it is what makes removing
    // that 134 MB zero-fill safe by construction: the value of a slot that was
    // never written is never read. Measured to never fire (0 negative indices
    // in 449 M assembled slots across the four benchmark scenes and the whole
    // sim_case suite), so it is a strict no-op on the observed workloads.
    // UIPC_SKIP_DEAD_FILL=0 restores both halves: the fill and `row <= col`.
    inline bool matrix_converter_drop_invalid()
    {
        static const bool drop = []
        {
            const char* e = std::getenv("UIPC_SKIP_DEAD_FILL");
            return !(e && e[0] == '0');
        }();
        return drop;
    }

    // fused #1: flag the upper-triangular entries (i <= j)
    __global__ void matrix_converter_fused_flag_upper_k1_kernel(
        cuda_tool::CBufferView<int> row_indices,
        cuda_tool::CBufferView<int> col_indices,
        cuda_tool::BufferView<int>  counts,
        bool                        drop_invalid,
        int                         n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        const int* __restrict__ row = row_indices.data();
        const int* __restrict__ col = col_indices.data();
        int* __restrict__ cnt       = counts.data();
        cnt[i] = ((!drop_invalid || row[i] >= 0) && row[i] <= col[i]) ? 1 : 0;
    }

    // fused #2: compact (key, source index) of the upper-triangular entries
    __global__ void matrix_converter_fused_compact_keys_k2_kernel(
        cuda_tool::CBufferView<int>     row_indices,
        cuda_tool::CBufferView<int>     col_indices,
        cuda_tool::CBufferView<int>     counts,
        cuda_tool::CBufferView<int>     offsets,
        cuda_tool::BufferView<uint64_t> keys,
        cuda_tool::BufferView<int>      src_index,
        cuda_tool::Dense<int>           total_count,
        uint64_t                        cols,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        const int* __restrict__ row = row_indices.data();
        const int* __restrict__ col = col_indices.data();
        const int* __restrict__ cnt = counts.data();
        const int* __restrict__ off = offsets.data();
        uint64_t* __restrict__ key  = keys.data();
        int* __restrict__ src       = src_index.data();

        int c = cnt[i];
        int o = off[i];
        if(c != 0)
        {
            key[o] = static_cast<uint64_t>(row[i]) * cols + static_cast<uint64_t>(col[i]);
            src[o] = i;
        }
        if(i == n - 1)
            total_count = o + c;
    }

    // fused #3: gather the blocks into sorted order straight from the source
    template <typename T, int N>
    __global__ void matrix_converter_fused_gather_k3_kernel(
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> src_blocks,
        cuda_tool::CBufferView<int>                         sort_index,
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  dst_blocks,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        const MatrixConverterBlockT<T, N>* __restrict__ src = src_blocks.data();
        const int* __restrict__ perm                        = sort_index.data();
        MatrixConverterBlockT<T, N>* __restrict__ dst       = dst_blocks.data();
        dst[i]                                              = src[perm[i]];
    }

    // value functor for the fused segmental reduce (gather without staging)
    template <typename T, int N>
    struct matrix_converter_permuted_value_op
    {
        const MatrixConverterBlockT<T, N>* src;
        const int*                         perm;
        __device__ MatrixConverterBlockT<T, N> operator()(int i) const
        {
            return src[perm[i]];
        }
    };

    // MatrixConverter::_radix_sort_indices_and_blocks(from, to) #1: hash ij
    __global__ void matrix_converter_radix_sort_indices_and_blocks_k1_kernel(
        cuda_tool::CBufferView<int>     row_indices,
        cuda_tool::CBufferView<int>     col_indices,
        cuda_tool::BufferView<uint64_t> ij_hash,
        cuda_tool::BufferView<int>      sort_index,
        uint64_t                        cols,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // K8: contiguous key (row * cols + col) so only bit_width(rows*cols)
        // radix passes are needed; same lexicographic order as (row, col)
        ij_hash(i) = static_cast<uint64_t>(row_indices(i)) * cols
                     + static_cast<uint64_t>(col_indices(i));
        sort_index(i) = i;
    }

    // MatrixConverter::_radix_sort_indices_and_blocks(from, to) #2: unpack ij hash
    __global__ void matrix_converter_radix_sort_indices_and_blocks_k2_kernel(
        cuda_tool::BufferView<uint64_t>               ij_hash,
        cuda_tool::BufferView<MatrixConverterIntPair> ij_pairs,
        uint64_t                                      cols,
        int                                           n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto hash      = ij_hash(i);
        auto row_index = static_cast<int>(hash / cols);
        auto col_index = static_cast<int>(hash - static_cast<uint64_t>(row_index) * cols);
        ij_pairs(i).x = row_index;
        ij_pairs(i).y = col_index;
    }

    // MatrixConverter::_radix_sort_indices_and_blocks(from, to) #3: sort the block values
    // __restrict__ locals let nvcc batch the random 72-byte gathers (the src
    // and dst buffers are distinct at every call site)
    template <typename T, int N>
    __global__ void matrix_converter_radix_sort_indices_and_blocks_k3_kernel(
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> src_blocks,
        cuda_tool::CBufferView<int>                         sort_index,
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  dst_blocks,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        const MatrixConverterBlockT<T, N>* __restrict__ src = src_blocks.data();
        const int* __restrict__ perm                        = sort_index.data();
        MatrixConverterBlockT<T, N>* __restrict__ dst       = dst_blocks.data();
        dst[i]                                              = src[perm[i]];
    }

    // MatrixConverter::_radix_sort_indices_and_blocks(to) #1: hash ij
    __global__ void matrix_converter_radix_sort_indices_and_blocks_in_place_k1_kernel(
        cuda_tool::CBufferView<int>     row_indices,
        cuda_tool::CBufferView<int>     col_indices,
        cuda_tool::BufferView<uint64_t> ij_hash,
        cuda_tool::BufferView<int>      sort_index,
        uint64_t                        cols,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        ij_hash(i) = uint64_t{row_indices(i)} * cols + uint64_t{col_indices(i)};
        sort_index(i) = i;
    }

    // MatrixConverter::_radix_sort_indices_and_blocks(to) #2: unpack ij hash
    __global__ void matrix_converter_radix_sort_indices_and_blocks_in_place_k2_kernel(
        cuda_tool::BufferView<uint64_t>               ij_hash,
        cuda_tool::BufferView<MatrixConverterIntPair> ij_pairs,
        uint64_t                                      cols,
        int                                           n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto hash      = ij_hash(i);
        auto row_index = int(hash / cols);
        auto col_index = int(hash - uint64_t(row_index) * cols);
        ij_pairs(i).x  = row_index;
        ij_pairs(i).y  = col_index;
    }

    // MatrixConverter::_radix_sort_indices_and_blocks(to) #3: sort the block values
    template <typename T, int N>
    __global__ void matrix_converter_radix_sort_indices_and_blocks_in_place_k3_kernel(
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> src_blocks,
        cuda_tool::CBufferView<int>                         sort_index,
        cuda_tool::CBufferView<MatrixConverterIntPair>      ij_pairs,
        cuda_tool::BufferView<int>                          dst_row,
        cuda_tool::BufferView<int>                          dst_col,
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  dst_blocks,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        const MatrixConverterBlockT<T, N>* __restrict__ src = src_blocks.data();
        const int* __restrict__ perm                        = sort_index.data();
        const MatrixConverterIntPair* __restrict__ ij       = ij_pairs.data();
        int* __restrict__ row                               = dst_row.data();
        int* __restrict__ col                               = dst_col.data();
        MatrixConverterBlockT<T, N>* __restrict__ dst       = dst_blocks.data();
        dst[i]                                              = src[perm[i]];
        row[i]                                              = ij[i].x;
        col[i]                                              = ij[i].y;
    }

    // MatrixConverter::_make_unique_indices(triplet -> bcoo) #1
    __global__ void matrix_converter_make_unique_indices_k1_kernel(
        cuda_tool::BufferView<MatrixConverterIntPair> unique_ij_pairs,
        cuda_tool::BufferView<int>                    row_indices,
        cuda_tool::BufferView<int>                    col_indices,
        int                                           n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        row_indices(i) = unique_ij_pairs(i).x;
        col_indices(i) = unique_ij_pairs(i).y;
    }

    // MatrixConverter::_make_unique_block_warp_reduction #1: mark segment tails
    __global__ void matrix_converter_make_unique_block_warp_reduction_k1_kernel(
        cuda_tool::BufferView<int> sorted_partition,
        cuda_tool::BufferView<int> unique_counts,
        cuda_tool::BufferView<int> offsets,
        int                        n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto offset = offsets(i);
        auto count  = unique_counts(i);

        sorted_partition(offset + count - 1) = 1;
    }

    // MatrixConverter::_calculate_block_offsets #1: scatter run lengths per row
    __global__ void matrix_converter_calculate_block_offsets_k1_kernel(
        cuda_tool::CBufferView<int> unique_indices,
        cuda_tool::BufferView<int>  counts,
        cuda_tool::BufferView<int>  col_counts_per_row,
        int                         n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto row                = unique_indices(i);
        col_counts_per_row(row) = counts(i);
    }

    // MatrixConverter::_make_unique_indices(doublet -> bcoo vector) #1
    __global__ void matrix_converter_make_unique_vector_indices_k1_kernel(
        cuda_tool::BufferView<int> unique_indices, cuda_tool::BufferView<int> dst_indices, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        dst_indices(i) = unique_indices(i);
    }

    // MatrixConverter::_make_unique_segment_warp_reduction #1: mark segment tails
    __global__ void matrix_converter_make_unique_segment_warp_reduction_k1_kernel(
        cuda_tool::BufferView<int> sorted_partition,
        cuda_tool::BufferView<int> unique_counts,
        cuda_tool::BufferView<int> offsets,
        int                        n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto offset = offsets(i);
        auto count  = unique_counts(i);

        sorted_partition(offset + count - 1) = 1;
    }

    // MatrixConverter::ge2sym(bcoo) #1: find the upper triangular part (i <= j)
    template <typename T, int N>
    __global__ void matrix_converter_ge2sym_bcoo_k1_kernel(
        cuda_tool::CBufferView<int>                         row_indices,
        cuda_tool::CBufferView<int>                         col_indices,
        cuda_tool::BufferView<MatrixConverterIntPair>       ij_pairs,
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> blocks,
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  block_temp,
        cuda_tool::BufferView<int>                          counts,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        counts(i)     = row_indices(i) <= col_indices(i) ? 1 : 0;
        ij_pairs(i).x = row_indices(i);
        ij_pairs(i).y = col_indices(i);
        block_temp(i) = blocks(i);
    }

    // MatrixConverter::ge2sym(bcoo) #2: compact the upper triangular part
    template <typename T, int N>
    __global__ void matrix_converter_ge2sym_bcoo_k2_kernel(
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  dst_blocks,
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> src_blocks,
        cuda_tool::CBufferView<MatrixConverterIntPair>      ij_pairs,
        cuda_tool::BufferView<int>                          row_indices,
        cuda_tool::BufferView<int>                          col_indices,
        cuda_tool::CBufferView<int>                         counts,
        cuda_tool::CBufferView<int>                         offsets,
        cuda_tool::Dense<int>                               total_count,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto count  = counts(i);
        auto offset = offsets(i);

        if(count != 0)
        {
            dst_blocks(offset)  = src_blocks(i);
            auto ij             = ij_pairs(i);
            row_indices(offset) = ij.x;
            col_indices(offset) = ij.y;
        }

        if(i == offsets.total_size() - 1)
        {
            total_count = offsets(i) + counts(i);
        }
    }

    // MatrixConverter::ge2sym(triplet) #1: find the upper triangular part (i <= j)
    // The __restrict__ locals are required for nvcc to vectorize the block
    // copy: through the views alone it must assume src/dst may alias and
    // serializes the 72-byte copies (measured 60 GB/s vs ~300 GB/s with
    // restrict on this GPU). All call sites pass distinct buffers.
    template <typename T, int N>
    __global__ void matrix_converter_ge2sym_triplet_k1_kernel(
        cuda_tool::CBufferView<int>                         row_indices,
        cuda_tool::CBufferView<int>                         col_indices,
        cuda_tool::BufferView<MatrixConverterIntPair>       ij_pairs,
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> blocks,
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  block_temp,
        cuda_tool::BufferView<int>                          counts,
        bool                                                drop_invalid,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        const int* __restrict__ row             = row_indices.data();
        const int* __restrict__ col             = col_indices.data();
        MatrixConverterIntPair* __restrict__ ij = ij_pairs.data();
        const MatrixConverterBlockT<T, N>* __restrict__ src = blocks.data();
        MatrixConverterBlockT<T, N>* __restrict__ dst       = block_temp.data();
        int* __restrict__ cnt                               = counts.data();

        // s25: see matrix_converter_drop_invalid
        cnt[i]  = ((!drop_invalid || row[i] >= 0) && row[i] <= col[i]) ? 1 : 0;
        ij[i].x = row[i];
        ij[i].y = col[i];
        dst[i]  = src[i];
    }

    // MatrixConverter::ge2sym(triplet) #2: compact the upper triangular part
    template <typename T, int N>
    __global__ void matrix_converter_ge2sym_triplet_k2_kernel(
        cuda_tool::BufferView<MatrixConverterBlockT<T, N>>  dst_blocks,
        cuda_tool::CBufferView<MatrixConverterBlockT<T, N>> src_blocks,
        cuda_tool::CBufferView<MatrixConverterIntPair>      ij_pairs,
        cuda_tool::BufferView<int>                          row_indices,
        cuda_tool::BufferView<int>                          col_indices,
        cuda_tool::CBufferView<int>                         counts,
        cuda_tool::CBufferView<int>                         offsets,
        cuda_tool::Dense<int>                               total_count,
        int                                                 n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        MatrixConverterBlockT<T, N>* __restrict__ dst       = dst_blocks.data();
        const MatrixConverterBlockT<T, N>* __restrict__ src = src_blocks.data();
        const MatrixConverterIntPair* __restrict__ ij       = ij_pairs.data();
        int* __restrict__ row = row_indices.data();
        int* __restrict__ col = col_indices.data();

        auto count  = counts(i);
        auto offset = offsets(i);

        if(count != 0)
        {
            dst[offset] = src[i];
            row[offset] = ij[i].x;
            col[offset] = ij[i].y;
        }

        if(i == offsets.total_size() - 1)
        {
            total_count = offsets(i) + counts(i);
        }
    }

    // MatrixConverter::sym2ge #1: setup select flag
    __global__ void matrix_converter_sym2ge_k1_kernel(cuda_tool::BufferView<int> flags,
                                                      cuda_tool::CBufferView<int> row_indices,
                                                      cuda_tool::CBufferView<int> col_indices,
                                                      cuda_tool::BufferView<int> partition_index,
                                                      int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        flags(i)           = (row_indices(i) == col_indices(i)) ? 1 : 0;
        partition_index(i) = i;
    }

    // MatrixConverter::sym2ge #2: copy blocks and ij as [ Diag | Upper | Lower ]
    template <typename T, int N>
    __global__ void matrix_converter_sym2ge_k2_kernel(cuda_tool::BCOOMatrixView<T, N> to,
                                                      cuda_tool::CBCOOMatrixView<T, N> from,
                                                      cuda_tool::CBufferView<int> partition_index,
                                                      int diag_count,
                                                      int sym_size,
                                                      int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto index = partition_index(i);
        auto f     = from(index);
        // diag + upper
        to(i).write(f.row_index, f.col_index, f.value);
        if(i >= diag_count)
        {
            // lower
            to(i + sym_size - diag_count)
                .write(f.col_index, f.row_index, f.value.transpose());
        }
    }
}  // namespace

template <typename T, int N>
void MatrixConverter<T, N>::convert(const cuda_tool::DeviceTripletMatrix<T, N>& from,
                                    cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    to.reshape(from.rows(), from.cols());
    to.resize_triplets_discard(from.triplet_count());


    if(to.triplet_count() == 0)
        return;

    _radix_sort_indices_and_blocks(from, to);

    _make_unique_indices(from, to);
    _make_unique_block_warp_reduction(from, to);
}


namespace
{
    // K8: number of radix bits needed for keys in [0, max_key]
    inline int matrix_converter_key_bits(uint64_t max_key)
    {
        int bits = 0;
        while(max_key > 0)
        {
            ++bits;
            max_key >>= 1;
        }
        return bits > 0 ? bits : 1;
    }
}  // namespace
template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_blocks(
    const cuda_tool::DeviceTripletMatrix<T, N>& from, cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    auto src_row_indices = from.row_indices();
    auto src_col_indices = from.col_indices();
    auto src_blocks      = from.values();
    const uint64_t key_rows = static_cast<uint64_t>(from.rows() > 0 ? from.rows() : 1);
    const uint64_t key_cols = static_cast<uint64_t>(from.cols() > 0 ? from.cols() : 1);

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    loose_resize(ij_pairs, src_row_indices.size());


    // hash ij
    int n_hash_ij = (int)src_row_indices.size();
    if(n_hash_ij > 0)
        matrix_converter_radix_sort_indices_and_blocks_k1_kernel<<<(n_hash_ij + 256 - 1) / 256, 256, 0, nullptr>>>(
            src_row_indices,
            src_col_indices,
            ij_hash_input.view(),
            sort_index_input.view(),
            key_cols,
            n_hash_ij);

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size(),
                                0,
                                matrix_converter_key_bits(key_rows * key_cols - 1));

    // set ij_hash back to row_indices and col_indices

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();

    int n_unpack_ij = (int)dst_row_indices.size();
    if(n_unpack_ij > 0)
        matrix_converter_radix_sort_indices_and_blocks_k2_kernel<<<(n_unpack_ij + 256 - 1) / 256, 256, 0, nullptr>>>(
            ij_hash.view(), ij_pairs.view(), key_cols, n_unpack_ij);

    // sort the block values

    {
        loose_resize(blocks_sorted, from.values().size());
        int n_sort_blocks = (int)src_blocks.size();
        if(n_sort_blocks > 0)
            matrix_converter_radix_sort_indices_and_blocks_k3_kernel<T, N>
                <<<(n_sort_blocks + 256 - 1) / 256, 256, 0, nullptr>>>(
                    src_blocks, sort_index.cview(), blocks_sorted.view(), n_sort_blocks);
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_blocks(cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    auto src_row_indices = to.row_indices();
    auto src_col_indices = to.col_indices();
    auto src_blocks      = to.values();
    const uint64_t key_rows = static_cast<uint64_t>(to.rows() > 0 ? to.rows() : 1);
    const uint64_t key_cols = static_cast<uint64_t>(to.cols() > 0 ? to.cols() : 1);

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    loose_resize(ij_pairs, src_row_indices.size());


    // hash ij
    int n_hash_ij = (int)src_row_indices.size();
    if(n_hash_ij > 0)
        matrix_converter_radix_sort_indices_and_blocks_in_place_k1_kernel<<<(n_hash_ij + 256 - 1) / 256, 256, 0, nullptr>>>(
            src_row_indices.cview(),
            src_col_indices.cview(),
            ij_hash_input.view(),
            sort_index_input.view(),
            key_cols,
            n_hash_ij);
    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size(),
                                0,
                                matrix_converter_key_bits(key_rows * key_cols - 1));

    // set ij_hash back to row_indices and col_indices

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();

    int n_unpack_ij = (int)dst_row_indices.size();
    if(n_unpack_ij > 0)
        matrix_converter_radix_sort_indices_and_blocks_in_place_k2_kernel<<<(n_unpack_ij + 256 - 1) / 256, 256, 0, nullptr>>>(
            ij_hash.view(), ij_pairs.view(), key_cols, n_unpack_ij);

    // sort the block values

    {
        loose_resize(blocks_sorted, to.values().size());
        int n_sort_blocks = (int)src_blocks.size();
        if(n_sort_blocks > 0)
            matrix_converter_radix_sort_indices_and_blocks_in_place_k3_kernel<T, N>
                <<<(n_sort_blocks + 256 - 1) / 256, 256, 0, nullptr>>>(
                    src_blocks.cview(),
                    sort_index.cview(),
                    ij_pairs.cview(),
                    to.row_indices(),
                    to.col_indices(),
                    blocks_sorted.view(),
                    n_sort_blocks);

        to.values().copy_from(blocks_sorted);
    }
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices(const cuda_tool::DeviceTripletMatrix<T, N>& from,
                                                 cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    auto row_indices = to.row_indices();
    auto col_indices = to.col_indices();

    loose_resize(unique_ij_pairs, ij_pairs.size());
    loose_resize(unique_counts, ij_pairs.size());


    DeviceRunLengthEncode().Encode(ij_pairs.data(),
                                   unique_ij_pairs.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   ij_pairs.size());

    int h_count = count;

    unique_ij_pairs.resize_discard(h_count);
    unique_counts.resize_discard(h_count);

    loose_resize(offsets, unique_counts.size());

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());


    int n_unique = (int)unique_counts.size();
    if(n_unique > 0)
        matrix_converter_make_unique_indices_k1_kernel<<<(n_unique + 256 - 1) / 256, 256, 0, nullptr>>>(
            unique_ij_pairs.view(), row_indices, col_indices, n_unique);

    to.resize_triplets_discard(h_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_block_warp_reduction(
    const cuda_tool::DeviceTripletMatrix<T, N>& from, cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    loose_resize(sorted_partition_input, ij_pairs.size());
    loose_resize(sorted_partition_output, ij_pairs.size());


    BufferLaunch().fill<int>(sorted_partition_input, 0);

    int n_mark = (int)unique_counts.size();
    if(n_mark > 0)
    {
        auto k = matrix_converter_make_unique_block_warp_reduction_k1_kernel;
        k<<<cuda_tool::best_grid_dim(n_mark, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            sorted_partition_input.view(), unique_counts.view(), offsets.view(), n_mark);
    }

    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input.data(),
                              sorted_partition_output.data(),
                              sorted_partition_input.size());

    auto blocks = to.values();

    FastSegmentalReduce<>().reduce(std::as_const(sorted_partition_output).view(),
                                   std::as_const(blocks_sorted).view(),
                                   blocks);
}

template <typename T, int N>
void MatrixConverter<T, N>::convert(const cuda_tool::DeviceBCOOMatrix<T, N>& from,
                                    cuda_tool::DeviceBSRMatrix<T, N>& to)
{
    // calculate the row offsets
    _calculate_block_offsets(from, to);

    to.resize(from.non_zeros());

    auto vals        = to.values();
    auto col_indices = to.col_indices();

    vals.copy_from(from.values());  // BCOO and BSR have the same block values
    col_indices.copy_from(from.col_indices());  // BCOO and BSR have the same block col indices
}

template <typename T, int N>
void MatrixConverter<T, N>::_calculate_block_offsets(
    const cuda_tool::DeviceBCOOMatrix<T, N>& from, cuda_tool::DeviceBSRMatrix<T, N>& to)
{
    //Timer timer{__FUNCTION__};

    using namespace cuda_tool;
    to.reshape(from.rows(), from.cols());


    auto dst_row_offsets = to.row_offsets();

    loose_resize(col_counts_per_row, dst_row_offsets.size());
    col_counts_per_row.fill(0);

    loose_resize(unique_indices, from.non_zeros());
    loose_resize(unique_counts, from.non_zeros());


    // run length encode the row
    DeviceRunLengthEncode().Encode(from.row_indices().data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   from.non_zeros());
    int h_count = count;

    unique_indices.resize_discard(h_count);
    unique_counts.resize_discard(h_count);

    int n_scatter = (int)unique_counts.size();
    if(n_scatter > 0)
        matrix_converter_calculate_block_offsets_k1_kernel<<<(n_scatter + 256 - 1) / 256, 256, 0, nullptr>>>(
            unique_indices.cview(), unique_counts.view(), col_counts_per_row.view(), n_scatter);

    // calculate the offsets
    DeviceScan().ExclusiveSum(col_counts_per_row.data(),
                              dst_row_offsets.data(),
                              col_counts_per_row.size());
}

//using T         = Float;
//constexpr int N = 3;

template <typename T, int N>
void MatrixConverter<T, N>::convert(const cuda_tool::DeviceDoubletVector<T, N>& from,
                                    cuda_tool::DeviceBCOOVector<T, N>& to)
{
    to.reshape(from.count());
    to.resize_doublets_discard(from.doublet_count());

    if(to.doublet_count() == 0)
        return;

    _radix_sort_indices_and_segments(from, to);
    _make_unique_indices(from, to);
    _make_unique_segment_warp_reduction(from, to);
}

template <typename T, int N>
void MatrixConverter<T, N>::_radix_sort_indices_and_segments(
    const cuda_tool::DeviceDoubletVector<T, N>& from, cuda_tool::DeviceBCOOVector<T, N>& to)
{
    using namespace cuda_tool;

    auto src_indices  = from.indices();
    auto src_segments = from.values();

    loose_resize(indices_sorted, src_indices.size());
    loose_resize(segments_sorted, src_segments.size());

    DeviceRadixSort().SortPairs(src_indices.data(),
                                indices_sorted.data(),
                                src_segments.data(),
                                segments_sorted.data(),
                                src_indices.size(),
                                0,
                                matrix_converter_key_bits(static_cast<uint64_t>(
                                    from.count() > 0 ? from.count() - 1 : 0)));
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_indices(const cuda_tool::DeviceDoubletVector<T, N>& from,
                                                 cuda_tool::DeviceBCOOVector<T, N>& to)
{
    using namespace cuda_tool;

    auto dst_indices  = to.indices();
    auto dst_segments = to.values();
    loose_resize(unique_indices, indices_sorted.size());
    loose_resize(unique_counts, indices_sorted.size());

    DeviceRunLengthEncode().Encode(indices_sorted.data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   indices_sorted.size());

    int h_count = count;

    unique_indices.resize_discard(h_count);
    unique_counts.resize_discard(h_count);

    loose_resize(offsets, unique_counts.size());

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());

    int n_unique = (int)unique_counts.size();
    if(n_unique > 0)
        matrix_converter_make_unique_vector_indices_k1_kernel<<<(n_unique + 256 - 1) / 256, 256, 0, nullptr>>>(
            unique_indices.view(), dst_indices, n_unique);

    to.resize_doublets_discard(h_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::_make_unique_segment_warp_reduction(
    const cuda_tool::DeviceDoubletVector<T, N>& from, cuda_tool::DeviceBCOOVector<T, N>& to)
{
    using namespace cuda_tool;

    loose_resize(sorted_partition_input, indices_sorted.size());
    loose_resize(sorted_partition_output, indices_sorted.size());

    BufferLaunch().fill<int>(sorted_partition_input, 0);

    int n_mark = (int)unique_counts.size();
    if(n_mark > 0)
    {
        auto k = matrix_converter_make_unique_segment_warp_reduction_k1_kernel;
        k<<<cuda_tool::best_grid_dim(n_mark, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            sorted_partition_input.view(), unique_counts.view(), offsets.view(), n_mark);
    }

    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input.data(),
                              sorted_partition_output.data(),
                              sorted_partition_input.size());

    auto segments = to.values();

    FastSegmentalReduce<64, 32>().reduce(std::as_const(sorted_partition_output).view(),
                                         std::as_const(segments_sorted).view(),
                                         segments);
}

template <typename T, int N>
void MatrixConverter<T, N>::ge2sym(cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize(counts, to.non_zeros());
    loose_resize(offsets, to.non_zeros());
    loose_resize(ij_pairs, to.non_zeros());
    loose_resize(block_temp, to.values().size());

    // 0. find the upper triangular part (where i <= j)
    int n_upper = to.non_zeros();
    if(n_upper > 0)
    {
        auto k = matrix_converter_ge2sym_bcoo_k1_kernel<T, N>;
        k<<<cuda_tool::best_grid_dim(n_upper, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            to.row_indices().cview(),
            to.col_indices().cview(),
            ij_pairs.view(),
            to.values().cview(),
            block_temp.view(),
            counts.view(),
            n_upper);
    }

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.values();

    int n_compact = (int)dst_block.size();
    if(n_compact > 0)
    {
        auto k = matrix_converter_ge2sym_bcoo_k2_kernel<T, N>;
        k<<<cuda_tool::best_grid_dim(n_compact, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dst_block,
            block_temp.cview(),
            ij_pairs.cview(),
            to.row_indices(),
            to.col_indices(),
            counts.cview(),
            offsets.cview(),
            count.viewer(),
            n_compact);
    }

    int h_total_count = count;

    to.resize_triplets_discard(h_total_count);
}

template <typename T, int N>
void MatrixConverter<T, N>::ge2sym(cuda_tool::DeviceTripletMatrix<T, N>& to)
{
    using namespace cuda_tool;

    // alias to reuse the memory
    auto& counts     = unique_counts;
    auto& block_temp = blocks_sorted;

    loose_resize(counts, to.triplet_count());
    loose_resize(offsets, to.triplet_count());
    loose_resize(ij_pairs, to.triplet_count());
    loose_resize(block_temp, to.values().size());

    // 0. find the upper triangular part (where i <= j)
    int n_upper = (int)to.triplet_count();
    if(n_upper > 0)
    {
        auto k = matrix_converter_ge2sym_triplet_k1_kernel<T, N>;
        k<<<cuda_tool::best_grid_dim(n_upper, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            to.row_indices().cview(),
            to.col_indices().cview(),
            ij_pairs.view(),
            to.values().cview(),
            block_temp.view(),
            counts.view(),
            matrix_converter_drop_invalid(),
            n_upper);
    }

    // exclusive sum
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), counts.size());

    // set the values
    auto dst_block = to.values();

    int n_compact = (int)dst_block.size();
    if(n_compact > 0)
    {
        auto k = matrix_converter_ge2sym_triplet_k2_kernel<T, N>;
        k<<<cuda_tool::best_grid_dim(n_compact, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dst_block,
            block_temp.cview(),
            ij_pairs.cview(),
            to.row_indices(),
            to.col_indices(),
            counts.cview(),
            offsets.cview(),
            count.viewer(),
            n_compact);
    }

    int h_total_count = count;

    to.resize_triplets_discard(h_total_count);
}


template <typename T, int N>
void MatrixConverter<T, N>::sym2ge(const cuda_tool::DeviceBCOOMatrix<T, N>& from,
                                   cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    auto sym_size = from.non_zeros();

    // alias to reuse the memory
    auto& flags                 = offsets;
    auto& partitioned           = blocks_sorted;
    auto& partition_index_input = sort_index_input;
    auto& partition_index       = sort_index;
    auto& selected_count        = count;
    auto  diag_count            = from.rows();


    loose_resize(flags, sym_size);
    loose_resize(partitioned, sym_size);
    loose_resize(partition_index, sym_size);

    // setup select flag
    if(sym_size > 0)
    {
        auto k = matrix_converter_sym2ge_k1_kernel;
        k<<<cuda_tool::best_grid_dim(sym_size, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            flags.view(),
            from.row_indices(),
            from.col_indices(),
            partition_index_input.view(),
            sym_size);
    }


    cuda_tool::DevicePartition().Flagged(partition_index_input.data(),
                                         flags.data(),
                                         partition_index.data(),
                                         selected_count.data(),
                                         sym_size);


    auto general_bcoo_size = 2 * (sym_size - diag_count) + diag_count;

    to.resize_discard(from.rows(), from.cols(), general_bcoo_size);

    // copy blocks and ij
    // in this sequence:
    // [ Diag | Upper | Lower ]
    //
    if(sym_size > 0)
    {
        auto k = matrix_converter_sym2ge_k2_kernel<T, N>;
        k<<<cuda_tool::best_grid_dim(sym_size, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            to.view(), from.cview(), partition_index.cview(), diag_count, sym_size, sym_size);
    }

    _radix_sort_indices_and_blocks(to);
}

namespace
{
    // perf/round4 (s10) env switch: 0 = old ge2sym + convert chain,
    // 1 = fused ge2sym/sort with a separate gather, 2 = fused + the gather
    // folded into the segmental reduce (default).
    inline int matrix_converter_fused_mode()
    {
        static const int mode = []
        {
            int m = 2;
            if(const char* e = std::getenv("UIPC_CONVERT_FUSED"))
            {
                int v = std::atoi(e);
                if(v >= 0 && v <= 2)
                    m = v;
            }
            return m;
        }();
        return mode;
    }
}  // namespace

template <typename T, int N>
bool MatrixConverter<T, N>::fused_enabled()
{
    return matrix_converter_fused_mode() != 0;
}

template <typename T, int N>
void MatrixConverter<T, N>::convert_sym(const cuda_tool::DeviceTripletMatrix<T, N>& from,
                                        cuda_tool::DeviceBCOOMatrix<T, N>& to)
{
    using namespace cuda_tool;

    to.reshape(from.rows(), from.cols());

    const int n = (int)from.triplet_count();
    if(n == 0)
    {
        to.resize_triplets_discard(0);
        return;
    }

    const uint64_t key_rows = static_cast<uint64_t>(from.rows() > 0 ? from.rows() : 1);
    const uint64_t key_cols = static_cast<uint64_t>(from.cols() > 0 ? from.cols() : 1);

    auto src_row_indices = from.row_indices();
    auto src_col_indices = from.col_indices();
    auto src_blocks      = from.values();

    // alias: the upper-triangular flags live in unique_counts until the RLE
    auto& counts = unique_counts;
    loose_resize(counts, n);
    loose_resize(offsets, n);
    loose_resize(ij_hash_input, n);
    loose_resize(sort_index_input, n);

    // 1. flag the upper triangular part
    matrix_converter_fused_flag_upper_k1_kernel<<<(n + 256 - 1) / 256, 256, 0, nullptr>>>(
        src_row_indices, src_col_indices, counts.view(), matrix_converter_drop_invalid(), n);

    // 2. exclusive sum of the flags
    DeviceScan().ExclusiveSum(counts.data(), offsets.data(), n);

    // 3. compact (key, source index)
    matrix_converter_fused_compact_keys_k2_kernel<<<(n + 256 - 1) / 256, 256, 0, nullptr>>>(
        src_row_indices,
        src_col_indices,
        counts.cview(),
        offsets.cview(),
        ij_hash_input.view(),
        sort_index_input.view(),
        count.viewer(),
        key_cols,
        n);

    const int m = count;  // upper-triangular triplet count (host read-back)

    if(m == 0)
    {
        to.resize_triplets_discard(0);
        return;
    }

    // 4. sort (key, source index) by key
    loose_resize(ij_hash, m);
    loose_resize(sort_index, m);
    loose_resize(ij_pairs, m);

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                m,
                                0,
                                matrix_converter_key_bits(key_rows * key_cols - 1));

    // 5. unpack the sorted keys back to (row, col) pairs
    to.resize_triplets_discard(m);
    matrix_converter_radix_sort_indices_and_blocks_k2_kernel<<<(m + 256 - 1) / 256, 256, 0, nullptr>>>(
        ij_hash.view(), ij_pairs.view(), key_cols, m);

    // 6. run-length encode the sorted (row, col) pairs
    loose_resize(unique_ij_pairs, m);
    loose_resize(unique_counts, m);

    DeviceRunLengthEncode().Encode(
        ij_pairs.data(), unique_ij_pairs.data(), unique_counts.data(), count.data(), m);

    const int h_count = count;

    unique_ij_pairs.resize_discard(h_count);
    unique_counts.resize_discard(h_count);

    loose_resize(offsets, h_count);
    DeviceScan().ExclusiveSum(unique_counts.data(), offsets.data(), h_count);

    matrix_converter_make_unique_indices_k1_kernel<<<(h_count + 256 - 1) / 256, 256, 0, nullptr>>>(
        unique_ij_pairs.view(), to.row_indices(), to.col_indices(), h_count);

    to.resize_triplets_discard(h_count);

    // 7. segment head flags -> segment ids
    loose_resize(sorted_partition_input, m);
    loose_resize(sorted_partition_output, m);

    BufferLaunch().fill<int>(sorted_partition_input.view(), 0);

    if(h_count > 0)
    {
        auto k = matrix_converter_make_unique_block_warp_reduction_k1_kernel;
        k<<<cuda_tool::best_grid_dim(h_count, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            sorted_partition_input.view(), unique_counts.view(), offsets.view(), h_count);
    }

    DeviceScan().ExclusiveSum(
        sorted_partition_input.data(), sorted_partition_output.data(), m);

    // 8. reduce the blocks of each segment
    auto blocks = to.values();

    if(matrix_converter_fused_mode() == 2)
    {
        // gather folded into the reduce: no staged copy of the sorted blocks
        matrix_converter_permuted_value_op<T, N> value_op{src_blocks.data(),
                                                          sort_index.data()};
        FastSegmentalReduce<>().reduce(
            (size_t)m,
            blocks,
            fast_segmental_reduce_get_offset_key_op{
                std::as_const(sorted_partition_output).view()},
            value_op);
    }
    else
    {
        loose_resize(blocks_sorted, m);
        matrix_converter_fused_gather_k3_kernel<T, N>
            <<<(m + 256 - 1) / 256, 256, 0, nullptr>>>(
                src_blocks, sort_index.cview(), blocks_sorted.view(), m);

        FastSegmentalReduce<>().reduce(std::as_const(sorted_partition_output).view(),
                                       std::as_const(blocks_sorted).view(),
                                       blocks);
    }
}

}  // namespace uipc::backend::cuda
