#include <linear_system/global_linear_system.h>
#include <linear_system/diag_linear_subsystem.h>
#include <linear_system/off_diag_linear_subsystem.h>
#include <uipc/common/range.h>
#include <linear_system/iterative_solver.h>
#include <linear_system/global_preconditioner.h>
#include <linear_system/local_preconditioner.h>
#include <fstream>
#include <sim_engine.h>
#include <backends/common/backend_path_tool.h>
#include <Eigen/Sparse>
#include <utils/matrix_market.h>
#include <array>
#include <cstring>
#include <cstdlib>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(GlobalLinearSystem);

SizeT GlobalLinearSystem::dof_count() const
{
    UIPC_ASSERT(m_impl.initialized,
                "GlobalLinearSystem::dof_count() is called before GlobalLinearSystem::init().");
    return m_impl.diag_dof_offsets_counts.total_count();
}

void GlobalLinearSystem::compute_gradient(ComputeGradientInfo& info)
{
    m_impl.compute_gradient(info);
}

void GlobalLinearSystem::do_build()
{
    auto dump_linear_system_attr =
        world().scene().config().find<IndexT>("extras/debug/dump_linear_system");

    m_impl.need_debug_dump =
        dump_linear_system_attr ? dump_linear_system_attr->view()[0] : false;
}

void GlobalLinearSystem::_dump_A_b()
{
    auto path_tool = BackendPathTool(workspace());
    auto output_folder = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto output_path_A = fmt::format("{}A.{}.{}.mtx",
                                     output_folder.string(),
                                     engine().frame(),
                                     engine().newton_iter());
    export_matrix_market(output_path_A, m_impl.bcoo_A.cview());
    logger::info("Dumped global linear system matrix A to {}", output_path_A);

    auto output_path_b = fmt::format("{}b.{}.{}.mtx",
                                     output_folder.string(),
                                     engine().frame(),
                                     engine().newton_iter());
    export_vector_market(output_path_b, m_impl.b.cview());
    logger::info("Dumped global linear system vector b to {}", output_path_b);
}

void GlobalLinearSystem::_dump_x()
{
    auto path_tool = BackendPathTool(workspace());
    auto output_folder = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    export_vector_market(fmt::format("{}x.{}.{}.mtx",
                                     output_folder.string(),
                                     engine().frame(),
                                     engine().newton_iter()),
                         m_impl.x.cview());
}


void GlobalLinearSystem::arm_assembly_prepass()
{
    for(auto&& subsystem : m_impl.diag_subsystems.view())
        subsystem->arm_assemble_prepass();
}

void GlobalLinearSystem::launch_assembly_prepass()
{
    for(auto&& subsystem : m_impl.diag_subsystems.view())
        subsystem->launch_assemble_prepass();
}

void GlobalLinearSystem::solve()
{
    m_impl.last_solve_iterations = 0;
    m_impl.build_linear_system();

    if(m_impl.empty_system) [[unlikely]]
        return;

    logger::info("GlobalLinearSystem has {} DoFs, Unique Triplet Count: {}",
                 m_impl.b.size(),
                 m_impl.bcoo_A.triplet_count());

    if(m_impl.need_debug_dump) [[unlikely]]
        _dump_A_b();

    m_impl.solve_linear_system();

    if(m_impl.need_debug_dump) [[unlikely]]
        _dump_x();

    m_impl.distribute_solution();
}

Float GlobalLinearSystem::diag_norm()
{
    m_impl.build_linear_system();
    return m_impl.diag_norm();
}

Float GlobalLinearSystem::mass_norm()
{
    m_impl.build_linear_system();
    return m_impl.mass_norm();
}

void GlobalLinearSystem::Impl::init()
{
    // 1) Init all diag subsystems and off-diag subsystems

    auto diag_subsystem_view     = diag_subsystems.view();
    auto off_diag_subsystem_view = off_diag_subsystems.view();

    {
        // Sort the diag subsystems by their UIDs to ensure the order is consistent
        // ref: https://github.com/spiriMirror/libuipc/issues/271
        std::ranges::sort(diag_subsystem_view,
                          [](const DiagLinearSubsystem* a, const DiagLinearSubsystem* b)
                          { return a->uid() < b->uid(); });
        std::ranges::sort(off_diag_subsystem_view,
                          [](const OffDiagLinearSubsystem* a,
                             const OffDiagLinearSubsystem* b) -> bool
                          { return a->uid() < b->uid(); });
    }


    auto total_count = diag_subsystem_view.size() + off_diag_subsystem_view.size();
    subsystem_infos.resize(total_count);

    // Diag System Always Go First
    auto diag_span = span{subsystem_infos}.subspan(0, diag_subsystem_view.size());
    // Off Diag System Always After Diag System
    auto off_diag_span = span{subsystem_infos}.subspan(diag_subsystem_view.size(),
                                                       off_diag_subsystem_view.size());
    {
        auto offset = 0;
        for(auto i : range(diag_span.size()))
        {
            auto& dst_diag                  = diag_span[i];
            dst_diag.is_diag                = true;
            dst_diag.local_index            = i;
            auto index                      = offset + i;
            dst_diag.index                  = index;
            diag_subsystem_view[i]->m_index = index;
        }

        offset += diag_subsystem_view.size();
        for(auto i : range(off_diag_span.size()))
        {
            auto& dst_off_diag       = off_diag_span[i];
            dst_off_diag.is_diag     = false;
            dst_off_diag.local_index = i;
            dst_off_diag.index       = offset + i;
        }

        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
            diag_subsystem->init();

        for(auto&& [i, off_diag_subsystem] : enumerate(off_diag_subsystem_view))
            off_diag_subsystem->init();
    }


    // 2) DoF Offsets/Counts
    {
        diag_dof_offsets_counts.resize(diag_subsystem_view.size());
        auto diag_dof_counts = diag_dof_offsets_counts.counts();
        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
        {
            InitDofExtentInfo info;
            diag_subsystem->report_init_extent(info);
            diag_dof_counts[i] = info.m_dof_count;
        }
        diag_dof_offsets_counts.scan();
        auto diag_dof_offsets = diag_dof_offsets_counts.offsets();
        for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
        {
            InitDofInfo info;
            info.m_dof_offset = diag_dof_offsets[i];
            info.m_dof_count  = diag_dof_counts[i];
            diag_subsystem->receive_init_dof_info(info);
        }
    }
    accuracy_statisfied_flags.resize(diag_subsystem_view.size());

    // 3) Triplet Offsets/Counts
    subsystem_triplet_offsets_counts.resize(total_count);
    off_diag_lr_triplet_counts.resize(off_diag_subsystem_view.size());

    // 4) Preconditioner
    // find out diag systems that don't have preconditioner
    auto local_preconditioner_view = local_preconditioners.view();

    for(auto precond : local_preconditioner_view)
    {
        auto index = precond->m_subsystem->m_index;
        diag_span[index].has_local_preconditioner = true;
    }

    no_precond_diag_subsystem_indices.reserve(diag_span.size());
    for(auto&& [i, diag_info] : enumerate(diag_span))
    {
        if(!diag_info.has_local_preconditioner)
        {
            no_precond_diag_subsystem_indices.push_back(i);
        }
    }

    for(auto precond : local_preconditioner_view)
    {
        precond->init();
    }

    initialized = true;
}

namespace
{
    // perf/round4 (s10) probe: measure how often the triplet sparsity pattern
    // (row/col index arrays of triplet_A after assembly) repeats between
    // consecutive Newton iterations. UIPC_TRIPLET_PATTERN_PROBE=1 enables it.
    __global__ void gls_pattern_probe_compare_kernel(const int* __restrict__ row,
                                                     const int* __restrict__ col,
                                                     const int* __restrict__ cached_row,
                                                     const int* __restrict__ cached_col,
                                                     int* __restrict__ mismatch,
                                                     int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        if(row[i] != cached_row[i] || col[i] != cached_col[i])
        {
            atomicMin(mismatch, i);
            atomicMax(mismatch + 1, i);
            atomicAdd(mismatch + 2, 1);
        }
    }

    struct GlsPatternProbe
    {
        bool                         enabled = false;
        cuda_tool::DeviceBuffer<int> cached_row;
        cuda_tool::DeviceBuffer<int> cached_col;
        cuda_tool::DeviceBuffer<int> mismatch;
        long long                    n_calls        = 0;
        long long                    n_same_count   = 0;
        long long                    n_same_pattern = 0;
        long long                    prev_n         = -1;
    };

    GlsPatternProbe& gls_pattern_probe()
    {
        static GlsPatternProbe probe = []
        {
            GlsPatternProbe p;
            if(const char* e = std::getenv("UIPC_TRIPLET_PATTERN_PROBE"))
                p.enabled = !(e[0] == '0');
            return p;
        }();
        return probe;
    }

    void gls_pattern_probe_check(cuda_tool::DeviceTripletMatrix<Float, 3>& triplet_A)
    {
        auto& p = gls_pattern_probe();
        if(!p.enabled)
            return;
        int n = (int)triplet_A.triplet_count();
        ++p.n_calls;
        bool same = false;
        int  m    = (int)std::min<long long>(n, p.prev_n);
        if(m > 0)
        {
            if(n == p.prev_n)
                ++p.n_same_count;
            p.mismatch.resize_discard(3);
            std::array<int, 3> init{m, -1, 0};
            cudaMemcpyAsync(p.mismatch.data(),
                            init.data(),
                            3 * sizeof(int),
                            cudaMemcpyHostToDevice,
                            cuda_tool::default_stream());
            gls_pattern_probe_compare_kernel<<<(m + 255) / 256, 256>>>(
                triplet_A.row_indices().data(),
                triplet_A.col_indices().data(),
                p.cached_row.data(),
                p.cached_col.data(),
                p.mismatch.data(),
                m);
            std::array<int, 3> h{};
            cudaMemcpyAsync(h.data(),
                            p.mismatch.data(),
                            3 * sizeof(int),
                            cudaMemcpyDeviceToHost,
                            cuda_tool::default_stream());
            cudaStreamSynchronize(cuda_tool::default_stream());
            same = (h[2] == 0) && (n == p.prev_n);
            if(same)
                ++p.n_same_pattern;
            else if(p.n_calls % 10 == 0)
                logger::warn("[pattern-probe] n={} prev={} common={} first_diff={} last_diff={} ndiff={}",
                             n,
                             p.prev_n,
                             m,
                             h[0],
                             h[1],
                             h[2]);
        }
        p.cached_row.resize_discard(n);
        p.cached_col.resize_discard(n);
        if(n > 0)
        {
            cudaMemcpyAsync(p.cached_row.data(),
                            triplet_A.row_indices().data(),
                            n * sizeof(int),
                            cudaMemcpyDeviceToDevice,
                            cuda_tool::default_stream());
            cudaMemcpyAsync(p.cached_col.data(),
                            triplet_A.col_indices().data(),
                            n * sizeof(int),
                            cudaMemcpyDeviceToDevice,
                            cuda_tool::default_stream());
        }
        p.prev_n = n;
        if(p.n_calls % 50 == 0)
            logger::warn("[pattern-probe] calls={} same_count={} same_pattern={} ({:.1f} %) n={}",
                         p.n_calls,
                         p.n_same_count,
                         p.n_same_pattern,
                         100.0 * p.n_same_pattern / p.n_calls,
                         n);
    }
}  // namespace

namespace
{
    // perf/round4 (s10) verifier: re-runs the old ge2sym + convert chain on a copy
    // of the assembled triplets and compares the BCOO output of the primary path
    // bit by bit. UIPC_CONVERT_VERIFY=1; the primary path is selected by
    // UIPC_CONVERT_FUSED (so =0 gives the old path's own atomic-order noise).
    __global__ void gls_convert_verify_kernel(cuda_tool::CBCOOMatrixView<Float, 3> a,
                                              cuda_tool::CBCOOMatrixView<Float, 3> b,
                                              unsigned long long* __restrict__ stats,
                                              int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto ta = a(i);
        auto tb = b(i);
        if(ta.row_index != tb.row_index || ta.col_index != tb.col_index)
            atomicAdd(stats + 0, 1ull);
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
            {
                double             va = ta.value(r, c);
                double             vb = tb.value(r, c);
                unsigned long long ba = __double_as_longlong(va);
                unsigned long long bb = __double_as_longlong(vb);
                if(ba != bb)
                    atomicAdd(stats + 1, 1ull);
                double d = fabs(va - vb);
                atomicMax(stats + 2, __double_as_longlong(d));
                atomicMax(stats + 3, __double_as_longlong(fabs(vb)));
            }
    }

    struct GlsConvertVerify
    {
        bool                                        enabled = false;
        MatrixConverter<Float, 3>                   converter;
        cuda_tool::DeviceTripletMatrix<Float, 3>    scratch_triplet;
        cuda_tool::DeviceBCOOMatrix<Float, 3>       scratch_bcoo;
        cuda_tool::DeviceBuffer<unsigned long long> stats;
        long long                                   calls          = 0;
        long long                                   shape_mismatch = 0;
        long long                                   index_mismatch = 0;
        long long                                   bit_mismatch   = 0;
        double                                      max_diff       = 0.0;
        double                                      max_ref        = 0.0;
    };

    GlsConvertVerify& gls_convert_verify()
    {
        static GlsConvertVerify v = []
        {
            GlsConvertVerify t;
            if(const char* e = std::getenv("UIPC_CONVERT_VERIFY"))
                t.enabled = !(e[0] == '0');
            return t;
        }();
        return v;
    }

    // snapshot of the assembled triplets, taken *before* the primary conversion
    // (the old ge2sym compacts triplet_A in place)
    void gls_convert_verify_snapshot(cuda_tool::DeviceTripletMatrix<Float, 3>& triplet_A)
    {
        auto& v = gls_convert_verify();
        if(!v.enabled)
            return;

        int n = (int)triplet_A.triplet_count();
        v.scratch_triplet.reshape(triplet_A.rows(), triplet_A.cols());
        v.scratch_triplet.resize_triplets_discard(n);
        if(n > 0)
        {
            cudaMemcpyAsync(v.scratch_triplet.values().data(),
                            triplet_A.values().data(),
                            n * sizeof(Matrix3x3),
                            cudaMemcpyDeviceToDevice,
                            cuda_tool::default_stream());
            cudaMemcpyAsync(v.scratch_triplet.row_indices().data(),
                            triplet_A.row_indices().data(),
                            n * sizeof(int),
                            cudaMemcpyDeviceToDevice,
                            cuda_tool::default_stream());
            cudaMemcpyAsync(v.scratch_triplet.col_indices().data(),
                            triplet_A.col_indices().data(),
                            n * sizeof(int),
                            cudaMemcpyDeviceToDevice,
                            cuda_tool::default_stream());
        }
    }

    void gls_convert_verify_check(cuda_tool::DeviceBCOOMatrix<Float, 3>& bcoo_A)
    {
        auto& v = gls_convert_verify();
        if(!v.enabled)
            return;
        ++v.calls;

        v.converter.ge2sym(v.scratch_triplet);
        v.converter.convert(v.scratch_triplet, v.scratch_bcoo);

        if(v.scratch_bcoo.non_zeros() != bcoo_A.non_zeros())
        {
            ++v.shape_mismatch;
            logger::warn("[convert-verify] triplet count {} != reference {}",
                         bcoo_A.non_zeros(),
                         v.scratch_bcoo.non_zeros());
            return;
        }

        int u = bcoo_A.non_zeros();
        v.stats.resize_discard(4);
        cudaMemsetAsync(v.stats.data(), 0, 4 * sizeof(unsigned long long), cuda_tool::default_stream());
        if(u > 0)
            gls_convert_verify_kernel<<<(u + 255) / 256, 256>>>(
                bcoo_A.cview(), v.scratch_bcoo.cview(), v.stats.data(), u);
        std::array<unsigned long long, 4> h{};
        cudaMemcpyAsync(h.data(),
                        v.stats.data(),
                        4 * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost,
                        cuda_tool::default_stream());
        cudaStreamSynchronize(cuda_tool::default_stream());

        v.index_mismatch += (long long)h[0];
        v.bit_mismatch += (long long)h[1];
        double d = 0.0, r = 0.0;
        std::memcpy(&d, &h[2], sizeof(double));
        std::memcpy(&r, &h[3], sizeof(double));
        v.max_diff = std::max(v.max_diff, d);
        v.max_ref  = std::max(v.max_ref, r);

        logger::warn("[convert-verify] calls={} nnz={} shape_mismatch={} index_mismatch={} bit_mismatch={} max|diff|={:.3e} max|ref|={:.3e} rel={:.3e}",
                     v.calls,
                     u,
                     v.shape_mismatch,
                     v.index_mismatch,
                     v.bit_mismatch,
                     v.max_diff,
                     v.max_ref,
                     v.max_ref > 0 ? v.max_diff / v.max_ref : 0.0);
    }
}  // namespace

namespace
{
    // ---------------------------------------------------------------------
    // perf/round5 (s26): is the triplet zero-fill dead work?
    //
    // `_assemble_linear_system` used to open with
    //
    //     // Clear and invalidate previous values
    //     triplet_A.values().fill(Matrix3x3::Zero());   // 134 MB on case2
    //     triplet_A.row_indices().fill(-1);
    //     triplet_A.col_indices().fill(-1);
    //
    // Re-reading the comment (PERF_METHOD section 4): the *invalidate* half was
    // never wired to anything. The converter's only index filter is ge2sym's
    // `row <= col`, and -1 <= -1, so an unwritten slot was **kept**, not
    // dropped -- with a 64-bit key of 0xFFFF.. whose low bits are garbage. So
    // an unwritten slot has never been survivable, zero-filled or not.
    //
    // The *clear* half is dead because `TripletMatrixViewT::Proxy::write()` is
    // the only writer of these arrays and it writes row, col and the whole NxN
    // block together: a slot is either fully written or untouched. There is no
    // accumulate path and no partial-block write.
    //
    // s26 therefore drops the values fill, keeps the (cheap, 4-byte) index
    // fills as the unwritten-slot marker, and adds the `row >= 0` term the
    // comment always implied to the converter's ge2sym filters, so that an
    // unwritten slot is *dropped* instead of corrupting the pattern. That
    // makes removing the values fill safe by construction, not merely by
    // observation. UIPC_SKIP_DEAD_FILL=0 restores the fill.
    inline bool gls_skip_dead_fill()
    {
        static const bool skip = []
        {
            const char* e = std::getenv("UIPC_SKIP_DEAD_FILL");
            return !(e && e[0] == '0');
        }();
        return skip;
    }

    // UIPC_FILL_PROBE=1: poison the value blocks instead of zeroing them, then
    // count after assembly how many slots are still poisoned / still carry a
    // negative index. Diagnostic only; it forces the fill on.
    constexpr uint64_t GlsFillPoisonBits = 0x7FF00DEADBEEF001ull;  // a signalling NaN payload

    inline double gls_fill_poison()
    {
        double d;
        uint64_t b = GlsFillPoisonBits;
        std::memcpy(&d, &b, sizeof(double));
        return d;
    }

    __global__ void gls_fill_probe_kernel(const Matrix3x3* __restrict__ values,
                                          const int* __restrict__ row,
                                          const int* __restrict__ col,
                                          unsigned long long* __restrict__ stats,
                                          int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        if(row[i] < 0 || col[i] < 0)
            atomicAdd(stats + 0, 1ull);
        const double* v      = values[i].data();
        int           n_pois = 0;
        for(int k = 0; k < 9; ++k)
        {
            uint64_t bits = __double_as_longlong(v[k]);
            if(bits == GlsFillPoisonBits)
                ++n_pois;
        }
        if(n_pois == 9)
            atomicAdd(stats + 1, 1ull);  // fully unwritten block
        else if(n_pois != 0)
            atomicAdd(stats + 2, 1ull);  // partially written block
    }

    struct GlsFillProbe
    {
        bool                                        enabled = false;
        // UIPC_FILL_PROBE=2 also checks straight after the fill, before the
        // subsystems run: the rig validation, which must report every slot
        // unwritten. Without it a probe that silently never fires would read
        // exactly like a fill that is dead.
        bool                                        pre_check = false;
        cuda_tool::DeviceBuffer<unsigned long long> stats;
        long long                                   calls          = 0;
        long long                                   triplets       = 0;
        long long                                   neg_index      = 0;
        long long                                   unwritten      = 0;
        long long                                   partial        = 0;
    };

    GlsFillProbe& gls_fill_probe()
    {
        static GlsFillProbe p = []
        {
            GlsFillProbe q;
            if(const char* e = std::getenv("UIPC_FILL_PROBE"))
            {
                q.enabled   = !(e[0] == '0');
                q.pre_check = (e[0] == '2');
            }
            return q;
        }();
        return p;
    }

    // UIPC_BCOO_HASH=<n>: order-independent 64-bit hash of the assembled BCOO
    // (nnz, every row/col index and the raw bits of every value) for the first
    // <n> conversions. The first conversion of a run depends only on the
    // deterministic initial state, so its hash is a direct byte-level A/B of
    // the whole assemble+convert pipeline between two runs. (Later ones are
    // not: the gradient is accumulated with atomics, so the trajectory
    // diverges from the first solve on.)
    __global__ void gls_bcoo_hash_kernel(const Matrix3x3* __restrict__ values,
                                         const int* __restrict__ row,
                                         const int* __restrict__ col,
                                         unsigned long long* __restrict__ out,
                                         int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // per-element mix, then a commutative combine so the hash does not
        // depend on the order blocks happen to land in
        unsigned long long h = 1469598103934665603ull;
        auto               mix = [&](unsigned long long v)
        {
            h ^= v;
            h *= 1099511628211ull;
        };
        mix((unsigned long long)(unsigned)row[i]);
        mix((unsigned long long)(unsigned)col[i]);
        const double* v = values[i].data();
        for(int k = 0; k < 9; ++k)
            mix((unsigned long long)__double_as_longlong(v[k]));
        atomicXor(out, h);
        atomicAdd(out + 1, h);
    }

    struct GlsBcooHash
    {
        int                                         remaining = 0;
        cuda_tool::DeviceBuffer<unsigned long long> out;
    };

    GlsBcooHash& gls_bcoo_hash()
    {
        static GlsBcooHash g = []
        {
            GlsBcooHash q;
            if(const char* e = std::getenv("UIPC_BCOO_HASH"))
                q.remaining = std::atoi(e);
            return q;
        }();
        return g;
    }

    void gls_bcoo_hash_check(cuda_tool::DeviceBCOOMatrix<Float, 3>& bcoo_A)
    {
        auto& g = gls_bcoo_hash();
        if(g.remaining <= 0)
            return;
        --g.remaining;
        int n = (int)bcoo_A.non_zeros();
        if(n <= 0)
            return;
        g.out.resize_discard(2);
        std::array<unsigned long long, 2> init{0, 0};
        cudaMemcpyAsync(g.out.data(),
                        init.data(),
                        init.size() * sizeof(unsigned long long),
                        cudaMemcpyHostToDevice,
                        cuda_tool::default_stream());
        gls_bcoo_hash_kernel<<<(n + 255) / 256, 256, 0, cuda_tool::default_stream()>>>(
            bcoo_A.values().data(),
            bcoo_A.row_indices().data(),
            bcoo_A.col_indices().data(),
            g.out.data(),
            n);
        std::array<unsigned long long, 2> h{};
        cudaMemcpyAsync(h.data(),
                        g.out.data(),
                        h.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost,
                        cuda_tool::default_stream());
        cudaStreamSynchronize(cuda_tool::default_stream());
        logger::warn("[bcoo-hash] nnz={} xor={:#018x} sum={:#018x}", n, h[0], h[1]);
    }

    void gls_fill_probe_check(cuda_tool::DeviceTripletMatrix<Float, 3>& triplet_A)
    {
        auto& p = gls_fill_probe();
        if(!p.enabled)
            return;
        int n = (int)triplet_A.triplet_count();
        if(n <= 0)
            return;
        p.stats.resize_discard(3);
        std::array<unsigned long long, 3> init{0, 0, 0};
        cudaMemcpyAsync(p.stats.data(),
                        init.data(),
                        init.size() * sizeof(unsigned long long),
                        cudaMemcpyHostToDevice,
                        cuda_tool::default_stream());
        gls_fill_probe_kernel<<<(n + 255) / 256, 256, 0, cuda_tool::default_stream()>>>(
            triplet_A.values().data(),
            triplet_A.row_indices().data(),
            triplet_A.col_indices().data(),
            p.stats.data(),
            n);
        std::array<unsigned long long, 3> h{};
        cudaMemcpyAsync(h.data(),
                        p.stats.data(),
                        h.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost,
                        cuda_tool::default_stream());
        cudaStreamSynchronize(cuda_tool::default_stream());
        ++p.calls;
        p.triplets += n;
        p.neg_index += (long long)h[0];
        p.unwritten += (long long)h[1];
        p.partial += (long long)h[2];
        logger::warn("[fill-probe] calls={} n={} total_triplets={} neg_index={} unwritten_blocks={} partial_blocks={}",
                     p.calls,
                     n,
                     p.triplets,
                     p.neg_index,
                     p.unwritten,
                     p.partial);
    }
}  // namespace

void GlobalLinearSystem::Impl::build_linear_system()
{
    Timer timer{"Build Linear System"};
    empty_system = !_update_subsystem_extent();

    if(empty_system) [[unlikely]]
    {
        logger::warn("The global linear system is empty, skip *assembling, *solving and *solution distributing phase.");
        return;
    }

    // perf/round4 (s07): UIPC_SPMV_VERIFY readback of the previous solve
    // (no-op unless the env is set)
    spmver.verify_report(b.size());

    {
        Timer t{"Assemble Subsystems"};
        _assemble_linear_system();
    }

    gls_fill_probe_check(triplet_A);

    gls_pattern_probe_check(triplet_A);

    gls_convert_verify_snapshot(triplet_A);

    {
        Timer t{"Convert To BCOO"};
        if(MatrixConverter<Float, 3>::fused_enabled())
        {
            // perf/round4 (s10): fused ge2sym + triplet->BCOO
            converter.convert_sym(triplet_A, bcoo_A);
        }
        else
        {
            converter.ge2sym(triplet_A);
            converter.convert(triplet_A, bcoo_A);
        }
        gls_convert_verify_check(bcoo_A);
        gls_bcoo_hash_check(bcoo_A);
        // upload the nnz count for graph-stable SpMV launches (async on the
        // default stream; drained before any solve reads it)
        triplet_count_dev = (IndexT)bcoo_A.triplet_count();
    }

    {
        Timer t{"Assemble Preconditioner"};
        _assemble_preconditioner();
    }

    logger::info("GlobalLinearSystem has {} DoFs, Unique Triplet Count: {}",
                 b.size(),
                 bcoo_A.triplet_count());
}

bool GlobalLinearSystem::Impl::_update_subsystem_extent()
{
    bool dof_count_changed     = false;
    bool triplet_count_changed = false;

    auto diag_subsystem_view       = diag_subsystems.view();
    auto off_diag_subsystem_view   = off_diag_subsystems.view();
    auto diag_dof_counts           = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets          = diag_dof_offsets_counts.offsets();
    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(subsystem_info.is_diag)
        {
            auto           dof_i          = subsystem_info.local_index;
            auto           triplet_i      = subsystem_info.index;
            auto&          diag_subsystem = diag_subsystem_view[dof_i];
            DiagExtentInfo info;
            diag_subsystem->report_extent(info);

            dof_count_changed |= diag_dof_counts[dof_i] != info.m_dof_count;
            diag_dof_counts[dof_i] = info.m_dof_count;


            triplet_count_changed |= subsystem_triplet_counts[triplet_i] != info.m_block_count;
            subsystem_triplet_counts[triplet_i] = info.m_block_count;
        }
        else
        {
            auto triplet_i = subsystem_info.index;
            auto& off_diag_subsystem = off_diag_subsystem_view[subsystem_info.local_index];
            OffDiagExtentInfo info;
            off_diag_subsystem->report_extent(info);

            auto total_block_count = info.m_lr_block_count + info.m_rl_block_count;

            triplet_count_changed |= subsystem_triplet_counts[triplet_i] != total_block_count;
            subsystem_triplet_counts[triplet_i] = total_block_count;
            off_diag_lr_triplet_counts[subsystem_info.local_index] =
                SizeT2{info.m_lr_block_count, info.m_rl_block_count};
        }
    }

    SizeT total_dof     = 0;
    SizeT total_triplet = 0;

    if(dof_count_changed)
    {
        diag_dof_offsets_counts.scan();
    }
    total_dof = diag_dof_offsets_counts.total_count();
    if(x.capacity() < total_dof)
    {
        auto reserve_count = total_dof * reserve_ratio;
        x.reserve(reserve_count);
        b.reserve(reserve_count);
    }
    auto blocked_dof = total_dof / DoFBlockSize;
    triplet_A.reshape(blocked_dof, blocked_dof);
    x.resize(total_dof);
    b.resize(total_dof);

    if(triplet_count_changed) [[likely]]
    {
        subsystem_triplet_offsets_counts.scan();
    }
    total_triplet = subsystem_triplet_offsets_counts.total_count();

    if(triplet_A.triplet_capacity() < total_triplet)
    {
        auto reserve_count = total_triplet * reserve_ratio;
        triplet_A.reserve_triplets_discard(reserve_count);
        bcoo_A.reserve_triplets_discard(reserve_count);
    }
    triplet_A.resize_triplets_discard(total_triplet);

    if(total_dof == 0 || total_triplet == 0) [[unlikely]]
    {
        return false;
    }

    return true;
}

void GlobalLinearSystem::Impl::_assemble_linear_system()
{
    auto HA = triplet_A.view();

    // perf/round5 (s26): the value fill is dead work -- see the note above
    // `build_linear_system`. The index fills stay: they are 4 bytes per triplet
    // against 72, and they are what makes an unwritten slot detectable (by the
    // probe) and droppable (by the converter's `row >= 0` filter).
    if(!gls_skip_dead_fill() || gls_fill_probe().enabled)
    {
        triplet_A.values().fill(gls_fill_probe().enabled ?
                                    Matrix3x3::Constant(gls_fill_poison()).eval() :
                                    Matrix3x3::Zero().eval());
    }
    triplet_A.row_indices().fill(-1);
    triplet_A.col_indices().fill(-1);

    if(gls_fill_probe().pre_check)
        gls_fill_probe_check(triplet_A);

    auto B = b.view();
    B.buffer_view().fill(0.0);

    auto diag_subsystem_view     = diag_subsystems.view();
    auto off_diag_subsystem_view = off_diag_subsystems.view();

    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    auto subsystem_triplet_counts  = subsystem_triplet_offsets_counts.counts();
    auto subsystem_triplet_offsets = subsystem_triplet_offsets_counts.offsets();

    for(const auto& subsystem_info : subsystem_infos)
    {
        if(subsystem_info.is_diag)
        {
            auto  dof_i          = subsystem_info.local_index;
            auto  triplet_i      = subsystem_info.index;
            auto& diag_subsystem = diag_subsystem_view[dof_i];

            int  dof_offset         = diag_dof_offsets[dof_i];
            int  dof_count          = diag_dof_counts[dof_i];
            int  blocked_dof_offset = dof_offset / DoFBlockSize;
            int  blocked_dof_count  = dof_count / DoFBlockSize;
            int2 ij_offset          = {blocked_dof_offset, blocked_dof_offset};
            int2 ij_count           = {blocked_dof_count, blocked_dof_count};

            DiagInfo info{this};

            info.m_index     = triplet_i;
            info.m_gradients = B.subview(dof_offset, dof_count);
            info.m_hessians  = HA.subview(subsystem_triplet_offsets[triplet_i],
                                         subsystem_triplet_counts[triplet_i])
                                  .submatrix(ij_offset, ij_count);

            diag_subsystem->assemble(info);
        }
        else
        {
            auto triplet_i   = subsystem_info.index;
            auto local_index = subsystem_info.local_index;
            auto& off_diag_subsystem = off_diag_subsystem_view[subsystem_info.local_index];
            auto& l_diag_index = off_diag_subsystem->m_l->m_index;
            auto& r_diag_index = off_diag_subsystem->m_r->m_index;


            int l_blocked_dof_offset = diag_dof_offsets[l_diag_index] / DoFBlockSize;
            int l_blocked_dof_count = diag_dof_counts[l_diag_index] / DoFBlockSize;

            int r_blocked_dof_offset = diag_dof_offsets[r_diag_index] / DoFBlockSize;
            int r_blocked_dof_count = diag_dof_counts[r_diag_index] / DoFBlockSize;

            auto lr_triplet_offset = subsystem_triplet_offsets[triplet_i];
            auto lr_triplet_count  = off_diag_lr_triplet_counts[local_index].x;
            auto rl_triplet_offset = lr_triplet_offset + lr_triplet_count;
            auto rl_triplet_count  = off_diag_lr_triplet_counts[local_index].y;

            OffDiagInfo info{this};
            info.m_index = triplet_i;

            info.m_lr_hessian =
                HA.subview(lr_triplet_offset, lr_triplet_count)
                    .submatrix(int2{l_blocked_dof_offset, r_blocked_dof_offset},
                               int2{l_blocked_dof_count, r_blocked_dof_count});

            info.m_rl_hessian =
                HA.subview(rl_triplet_offset, rl_triplet_count)
                    .submatrix(int2{r_blocked_dof_offset, l_blocked_dof_offset},
                               int2{r_blocked_dof_count, l_blocked_dof_count});

            // logger::info("rl_offset: {}, lr_offset: {}", rl_triplet_offset, lr_triplet_offset);

            off_diag_subsystem->assemble(info);
        }
    }
}

void GlobalLinearSystem::Impl::_assemble_preconditioner()
{
    if(global_preconditioner)
    {
        GlobalPreconditionerAssemblyInfo info{this};
        global_preconditioner->assemble(info);
    }

    for(auto&& preconditioner : local_preconditioners.view())
    {
        LocalPreconditionerAssemblyInfo info{this, preconditioner->m_subsystem->m_index};
        preconditioner->assemble(info);
    }
    // perf/kernels (K13): join side-stream assembly before the solve
    for(auto&& preconditioner : local_preconditioners.view())
        preconditioner->finish_assemble();
}

void GlobalLinearSystem::Impl::solve_linear_system()
{
    Timer timer{"Solve Linear System"};
    if(iterative_solver)
    {
        SolvingInfo info{this};
        info.m_b = b.cview();
        info.m_x = x.view();
        iterative_solver->solve(info);
        last_solve_iterations = info.m_iter_count;
        logger::info("Iterative linear solver iteration count: {}", info.m_iter_count);
    }
}

void GlobalLinearSystem::Impl::distribute_solution()
{
    auto diag_subsystem_view = diag_subsystems.view();
    auto diag_dof_counts     = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets    = diag_dof_offsets_counts.offsets();

    // distribute the solution to all diag subsystems
    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystems.view()))
    {
        SolutionInfo info{this};
        info.m_solution = x.view().subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_subsystem->retrieve_solution(info);
    }
}

void GlobalLinearSystem::Impl::apply_preconditioner(cuda_tool::DenseVectorView<Float> z,
                                                    cuda_tool::CDenseVectorView<Float> r,
                                                    cuda_tool::CVarView<IndexT> converged,
                                                    cudaStream_t stream)
{
    (void)converged;
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    if(global_preconditioner)
    {
        ApplyPreconditionerInfo info{this};
        info.m_z         = z;
        info.m_r         = r;
        info.m_converged = converged;
        info.m_stream    = stream;
        global_preconditioner->apply(info);
    }

    for(auto& preconditioner : local_preconditioners.view())
    {
        ApplyPreconditionerInfo info{this};
        auto                    index  = preconditioner->m_subsystem->m_index;
        auto                    offset = diag_dof_offsets[index];
        auto                    count  = diag_dof_counts[index];
        info.m_z                       = z.subview(offset, count);
        info.m_r                       = r.subview(offset, count);
        info.m_converged               = converged;
        info.m_stream                  = stream;
        preconditioner->apply(info);
    }

    if(!global_preconditioner)
    {
        // For diag subsystems without local preconditioner, just copy r to z
        for(auto i : no_precond_diag_subsystem_indices)
        {
            auto offset = diag_dof_offsets[i];
            auto count  = diag_dof_counts[i];
            auto z_sub  = z.subview(offset, count);
            auto r_sub  = r.subview(offset, count);
            cuda_tool::BufferLaunch(stream).copy(z_sub.buffer_view(), r_sub.buffer_view());
        }
    }
}

void GlobalLinearSystem::Impl::spmv(Float                              a,
                                    cuda_tool::CDenseVectorView<Float> x,
                                    Float                              b,
                                    cuda_tool::DenseVectorView<Float>  y)
{
    spmver.rbk_sym_spmv(a, bcoo_A.cview(), x, b, y);

    // Just some debug options
    //  * spmver.sym_spmv(a, bcoo_A.cview(), x, b, y);      // Slightly slower
    //  * spmver.cpu_sym_spmv(a, bcoo_A.cview(), x, b, y);  // Much slower
}

void GlobalLinearSystem::Impl::spmv_dot(cuda_tool::CDenseVectorView<Float> x,
                                        cuda_tool::DenseVectorView<Float>  y,
                                        cuda_tool::VarView<Float> d_dot,
                                        cudaStream_t              stream)
{
    spmver.rbk_sym_spmv_dot(1.0,
                            bcoo_A.cview(),
                            x,
                            0.0,
                            y,
                            d_dot,
                            triplet_count_dev.cviewer(),
                            bcoo_A.triplet_capacity(),
                            stream);
}

bool GlobalLinearSystem::Impl::accuracy_statisfied(cuda_tool::DenseVectorView<Float> r)
{
    auto diag_dof_counts  = diag_dof_offsets_counts.counts();
    auto diag_dof_offsets = diag_dof_offsets_counts.offsets();

    for(auto&& [i, diag_subsystems] : enumerate(diag_subsystems.view()))
    {
        AccuracyInfo info{this};
        info.m_r = r.subview(diag_dof_offsets[i], diag_dof_counts[i]);
        diag_subsystems->accuracy_check(info);

        accuracy_statisfied_flags[i] = info.m_statisfied ? 1 : 0;
    }

    return std::ranges::all_of(accuracy_statisfied_flags,
                               [](bool flag) { return flag; });
}

void GlobalLinearSystem::Impl::compute_gradient(ComputeGradientInfo& info)
{
    auto diag_subsystem_view = diag_subsystems.view();

    // report extent first (gradient only mode)
    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
    {
        DiagExtentInfo diag_info;
        diag_info.m_gradient_only   = true;
        diag_info.m_component_flags = info.m_flags;
        diag_subsystem->report_extent(diag_info);
    }

    // assemble gradient only
    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystem_view))
    {
        DiagInfo diag_info{this};
        diag_info.m_index           = diag_subsystem->m_index;
        diag_info.m_gradients       = info.m_gradients;
        diag_info.m_hessians        = TripletMatrixView{};
        diag_info.m_gradient_only   = true;
        diag_info.m_component_flags = info.m_flags;
        diag_subsystem->assemble(diag_info);
    }
}

Float GlobalLinearSystem::Impl::diag_norm()
{
    Float norm = 0;

    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystems.view()))
    {
        DiagNormInfo info(this, diag_subsystem->m_index);
        norm = max(norm, diag_subsystem->diag_norm(info));
    }

    return norm;
}

Float GlobalLinearSystem::Impl::mass_norm()
{
    Float norm = 0;

    for(auto&& [i, diag_subsystem] : enumerate(diag_subsystems.view()))
    {
        DiagNormInfo info(this, diag_subsystem->m_index);
        norm = max(norm, diag_subsystem->mass_norm(info));
    }

    return norm;
}

void GlobalLinearSystem::DiagExtentInfo::extent(SizeT hessian_block_count, SizeT dof_count) noexcept
{
    m_block_count = hessian_block_count;
    UIPC_ASSERT(dof_count % DoFBlockSize == 0,
                "dof_count must be multiple of {}, yours {}.",
                DoFBlockSize,
                dof_count);
    m_dof_count = dof_count;
}

void GlobalLinearSystem::OffDiagExtentInfo::extent(SizeT lr_hessian_block_count,
                                                   SizeT rl_hassian_block_count) noexcept
{
    m_lr_block_count = lr_hessian_block_count;
    m_rl_block_count = rl_hassian_block_count;
}
auto GlobalLinearSystem::AssemblyInfo::A() const -> CBCOOMatrixView
{
    return m_impl->bcoo_A.cview();
}

SizeT GlobalLinearSystem::LocalPreconditionerAssemblyInfo::dof_offset() const
{
    auto diag_dof_offsets = m_impl->diag_dof_offsets_counts.offsets();
    return diag_dof_offsets[m_index];
}

SizeT GlobalLinearSystem::LocalPreconditionerAssemblyInfo::dof_count() const
{
    auto diag_dof_counts = m_impl->diag_dof_offsets_counts.counts();
    return diag_dof_counts[m_index];
}

SizeT GlobalLinearSystem::last_solve_iterations() const noexcept
{
    return m_impl.last_solve_iterations;
}
}  // namespace uipc::backend::cuda

namespace uipc::backend::cuda
{
void GlobalLinearSystem::add_subsystem(DiagLinearSubsystem* subsystem)
{
    check_state(SimEngineState::BuildSystems, "add_subsystem()");
    UIPC_ASSERT(subsystem != nullptr, "The subsystem should not be nullptr.");
    m_impl.diag_subsystems.register_sim_system(*subsystem);
}

void GlobalLinearSystem::add_subsystem(OffDiagLinearSubsystem* subsystem)
{
    check_state(SimEngineState::BuildSystems, "add_subsystem()");
    m_impl.off_diag_subsystems.register_sim_system(*subsystem);
}

void GlobalLinearSystem::add_solver(IterativeSolver* solver)
{
    check_state(SimEngineState::BuildSystems, "add_solver()");
    UIPC_ASSERT(solver != nullptr, "The solver should not be nullptr.");
    m_impl.iterative_solver.register_sim_system(*solver);
}

void GlobalLinearSystem::add_preconditioner(LocalPreconditioner* preconditioner)
{
    check_state(SimEngineState::BuildSystems, "add_preconditioner()");
    UIPC_ASSERT(preconditioner != nullptr, "The preconditioner should not be nullptr.");
    m_impl.local_preconditioners.register_sim_system(*preconditioner);
}

void GlobalLinearSystem::add_preconditioner(GlobalPreconditioner* preconditioner)
{
    check_state(SimEngineState::BuildSystems, "add_preconditioner()");
    UIPC_ASSERT(preconditioner != nullptr, "The preconditioner should not be nullptr.");
    m_impl.global_preconditioner.register_sim_system(*preconditioner);
}

void GlobalLinearSystem::init()
{
    m_impl.init();
}

void GlobalLinearSystem::ComputeGradientInfo::flags(ComponentFlags flags) noexcept
{
    m_flags = flags;
}

void GlobalLinearSystem::ComputeGradientInfo::buffer_view(cuda_tool::DenseVectorView<Float> grad) noexcept
{
    m_gradients = grad;
}
}  // namespace uipc::backend::cuda
