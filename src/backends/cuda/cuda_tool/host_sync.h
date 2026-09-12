#pragma once
#include <cuda_tool/stream.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
#if defined(__linux__)
#define UIPC_D2H_SITE_PROFILE 1
#include <execinfo.h>
#include <dlfcn.h>
#else
#define UIPC_D2H_SITE_PROFILE 0
#endif

// ---------------------------------------------------------------------------
// Blocking device -> host readbacks: the host sync points of the Newton + PCG
// loop (convergence residuals, TOI/CFL/feasible-step reductions, line-search
// energies, candidate/triplet counts).
//
// The old shape of every one of them was
//     cudaMemcpyAsync(<pageable host dst>, dev, n, D2H, s);
//     cudaStreamSynchronize(s);
// A *pageable* destination makes cudaMemcpyAsync synchronous with respect to
// the host: the driver must drain the stream and then stage the bytes through
// its own internal bounce buffer before it returns. Round 4's gap attribution
// (s14) measured 24-30 of these per Newton iteration, each leaving 14-22 us of
// GPU idle -- 1.3-3.0 ms per frame on every scene -- while the transfers
// themselves total ~1.2 us each.
//
// `host_read` is the single funnel for all of them.
//   UIPC_HOST_SYNC_FAST=1 (default): copy into a per-thread *pinned* staging
//       buffer (a true async DMA, no driver staging), sync, then memcpy to the
//       caller's destination. Bytes delivered are identical by construction.
//   UIPC_HOST_SYNC_FAST=0: the old pageable path, byte for byte.
//   UIPC_HOST_SYNC_SPIN=1: additionally request cudaDeviceScheduleSpin.
//   UIPC_D2H_PROFILE=1: count the readbacks and split their wall cost into
//       "drain" (the stream still had queued work -> the GPU was NOT idle) and
//       "stall" (latency measured with the stream already drained -> exactly
//       the GPU-idle gap this step removes).
//   UIPC_HOST_SYNC_VERIFY=1: read every value through BOTH paths and memcmp
//       the bytes; mismatches are counted and reported at exit.
// ---------------------------------------------------------------------------

namespace uipc::backend::cuda_tool
{
struct HostSyncEnv
{
    bool fast    = true;
    bool spin    = false;
    bool profile = false;
    bool verify  = false;
    bool sites   = false;
};

inline const HostSyncEnv& host_sync_env()
{
    static const HostSyncEnv e = []
    {
        HostSyncEnv v;
        auto        on = [](const char* name, bool dflt)
        {
            const char* s = std::getenv(name);
            if(!s || !s[0])
                return dflt;
            return s[0] != '0';
        };
        v.fast    = on("UIPC_HOST_SYNC_FAST", true);
        v.spin    = on("UIPC_HOST_SYNC_SPIN", false);
        v.profile = on("UIPC_D2H_PROFILE", false);
        v.verify  = on("UIPC_HOST_SYNC_VERIFY", false);
#if UIPC_D2H_SITE_PROFILE
        const char* pv = std::getenv("UIPC_D2H_PROFILE");
        v.sites        = pv && pv[0] == '2';
#endif
        return v;
    }();
    return e;
}

struct HostSyncStats
{
    std::uint64_t count      = 0;
    std::uint64_t bytes      = 0;
    std::uint64_t drain_ns   = 0;
    std::uint64_t stall_ns   = 0;
    std::uint64_t big        = 0;  // readbacks too large for the staging buffer
    std::uint64_t verify_n   = 0;
    std::uint64_t verify_bad = 0;
};

inline HostSyncStats& host_sync_stats()
{
    static HostSyncStats s;
    return s;
}

#if UIPC_D2H_SITE_PROFILE
// UIPC_D2H_PROFILE=2: attribute every readback to its call stack (probe only;
// backtrace() costs microseconds, so the wall numbers of a =2 run are not
// comparable with anything -- only the *distribution* is).
struct D2HSiteKey
{
    void* f[6]{};
    bool  operator<(const D2HSiteKey& o) const
    {
        for(int i = 0; i < 6; ++i)
            if(f[i] != o.f[i])
                return f[i] < o.f[i];
        return false;
    }
};
struct D2HSiteVal
{
    std::uint64_t n = 0, ns = 0, bytes = 0;
};
inline std::map<D2HSiteKey, D2HSiteVal>& host_sync_sites()
{
    static std::map<D2HSiteKey, D2HSiteVal> m;
    return m;
}
#endif

// Printed once at process exit when profiling/verifying is on.
inline void host_sync_register_dump()
{
    static bool done = false;
    if(done)
        return;
    done = true;
    std::atexit(
        []
        {
            const auto& s = host_sync_stats();
            if(s.count == 0 && s.verify_n == 0)
                return;
            std::fprintf(stderr,
                         "[d2h] readbacks=%llu bytes=%llu drain=%.3f ms stall=%.3f ms "
                         "(stall/readback=%.2f us) oversize=%llu verify=%llu bad=%llu\n",
                         (unsigned long long)s.count,
                         (unsigned long long)s.bytes,
                         s.drain_ns / 1e6,
                         s.stall_ns / 1e6,
                         s.count ? s.stall_ns / 1e3 / (double)s.count : 0.0,
                         (unsigned long long)s.big,
                         (unsigned long long)s.verify_n,
                         (unsigned long long)s.verify_bad);
#if UIPC_D2H_SITE_PROFILE
            auto& sites = host_sync_sites();
            std::fprintf(stderr, "[d2h-sites] %d stacks\n", (int)sites.size());
            for(const auto& kv : sites)
            {
                std::fprintf(stderr,
                             "[d2h-site] n=%llu stall_us=%llu bytes=%llu |",
                             (unsigned long long)kv.second.n,
                             (unsigned long long)(kv.second.ns / 1000),
                             (unsigned long long)kv.second.bytes);
                for(int k = 0; k < 6; ++k)
                {
                    Dl_info di;
                    if(kv.first.f[k] && dladdr(kv.first.f[k], &di) && di.dli_fbase)
                        std::fprintf(stderr,
                                     " %s+0x%llx",
                                     di.dli_fname ? di.dli_fname : "?",
                                     (unsigned long long)((char*)kv.first.f[k] - (char*)di.dli_fbase));
                    else
                        std::fprintf(stderr, " ?");
                }
                std::fprintf(stderr, "\n");
            }
#endif
        });
}

// Per-thread pinned staging buffer. Small (the solver's readbacks are scalars
// and short vectors); anything larger falls back to the old pageable copy.
inline constexpr std::size_t host_sync_staging_bytes = 1u << 20;

inline void* host_sync_staging()
{
    // Intentionally never freed: a thread_local destructor could run after the
    // CUDA runtime has already torn the context down.
    static thread_local void* p = []() -> void*
    {
        void* q = nullptr;
        if(cudaHostAlloc(&q, host_sync_staging_bytes, cudaHostAllocDefault) != cudaSuccess)
        {
            cudaGetLastError();
            return nullptr;
        }
        return q;
    }();
    return p;
}

// Optional: ask the driver to spin rather than yield in the sync. Must be
// called before the context is created to take effect; failure is harmless.
inline void host_sync_apply_device_flags()
{
    static bool done = false;
    if(done)
        return;
    done = true;
    if(!host_sync_env().spin)
        return;
    if(cudaSetDeviceFlags(cudaDeviceScheduleSpin) != cudaSuccess)
        cudaGetLastError();
}

namespace details
{
    inline std::uint64_t host_now_ns()
    {
        return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    // the old path, verbatim
    inline void host_read_pageable(void* dst, const void* src, std::size_t bytes, cudaStream_t s)
    {
        check_cuda_error(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s),
                         "cudaMemcpyAsync(D2H)",
                         __FILE__,
                         __LINE__);
        check_cuda_error(cudaStreamSynchronize(s), "cudaStreamSynchronize", __FILE__, __LINE__);
    }

    // Any size: a transfer larger than the staging buffer is chunked through
    // it, so a big readback also becomes a pinned DMA.
    inline void host_read_pinned(void*        dst,
                                 const void*  src,
                                 std::size_t  bytes,
                                 cudaStream_t s,
                                 void*        staging,
                                 std::size_t  cap)
    {
        auto*       out = static_cast<unsigned char*>(dst);
        const auto* in  = static_cast<const unsigned char*>(src);
        std::size_t off = 0;
        while(off < bytes)
        {
            const std::size_t n = (bytes - off < cap) ? (bytes - off) : cap;
            check_cuda_error(cudaMemcpyAsync(staging, in + off, n, cudaMemcpyDeviceToHost, s),
                             "cudaMemcpyAsync(D2H,pinned)",
                             __FILE__,
                             __LINE__);
            check_cuda_error(cudaStreamSynchronize(s), "cudaStreamSynchronize", __FILE__, __LINE__);
            std::memcpy(out + off, staging, n);
            off += n;
        }
    }
}  // namespace details

// The single blocking device->host readback used by every cuda_tool view and
// buffer. `dst` is ordinary (pageable) host memory owned by the caller.
inline void host_read(void* dst, const void* src, std::size_t bytes, cudaStream_t s)
{
    const auto& env = host_sync_env();
    if(!env.profile && !env.verify)
    {
        void* staging = env.fast ? host_sync_staging() : nullptr;
        if(staging)
            details::host_read_pinned(dst, src, bytes, s, staging, host_sync_staging_bytes);
        else
            details::host_read_pageable(dst, src, bytes, s);
        return;
    }

    host_sync_register_dump();
    auto& st = host_sync_stats();

    std::uint64_t t0 = 0, t1 = 0, t2 = 0;
    if(env.profile)
    {
        // Separate the two costs: everything up to an empty stream is GPU work
        // that was still queued (not idle); everything after it is the host
        // round trip during which the GPU has nothing to do.
        t0 = details::host_now_ns();
        check_cuda_error(cudaStreamSynchronize(s), "cudaStreamSynchronize", __FILE__, __LINE__);
        t1 = details::host_now_ns();
    }

    void* staging = env.fast ? host_sync_staging() : nullptr;
    if(staging)
        details::host_read_pinned(dst, src, bytes, s, staging, host_sync_staging_bytes);
    else
        details::host_read_pageable(dst, src, bytes, s);

    if(env.profile)
    {
        t2 = details::host_now_ns();
#if UIPC_D2H_SITE_PROFILE
        if(host_sync_env().sites)
        {
            void* bt[10];
            int   nf = backtrace(bt, 10);
            D2HSiteKey key;
            for(int i = 0; i < 6; ++i)
                key.f[i] = (i + 1 < nf) ? bt[i + 1] : nullptr;
            auto& sv = host_sync_sites()[key];
            sv.n++;
            sv.ns += t2 - t1;
            sv.bytes += bytes;
        }
#endif
        st.count++;
        st.bytes += bytes;
        st.drain_ns += t1 - t0;
        st.stall_ns += t2 - t1;
        if(!staging)
            st.big++;
    }

    if(env.verify)
    {
        // read the same bytes through the other path and compare
        static thread_local std::vector<unsigned char> other;
        other.assign(bytes, 0);
        if(staging)
            details::host_read_pageable(other.data(), src, bytes, s);
        else if(void* stg = host_sync_staging())
            details::host_read_pinned(other.data(), src, bytes, s, stg, host_sync_staging_bytes);
        else
            details::host_read_pageable(other.data(), src, bytes, s);
        st.verify_n++;
        if(std::memcmp(other.data(), dst, bytes) != 0)
            st.verify_bad++;
    }
}
}  // namespace uipc::backend::cuda_tool
