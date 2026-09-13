#pragma once
#include <cuda_tool/cuda_tool.h>
#include <cmath>
#include <cstdint>

//ref: https://github.com/ipc-sim/Codim-IPC/tree/main/Library/Math/Distance
namespace uipc::backend::cuda::distance
{
// perf/round6 (s04): optional device-side diagnosis counters for the ACCD
// loop (env UIPC_CCD_STATS=1 in the trajectory filter). Slots:
//   0 = calls, 1 = first-pass early exits, 2 = loop passes, 3 = hits.
// Compiled out entirely unless the Stats template argument is true.
using CCDStatCounter = unsigned long long;

// perf/round6 (s06): optional DCD-inactivity verdict for the candidate
// compaction in the trajectory filter (env UIPC_CCD_COMPACT). When
// `DcdCull` is true the ACCD writes, through `dcd_inactive`, whether the
// pair is provably OUTSIDE the contact activation window
// `d < thickness + d_hat` for EVERY point of the swept step [0, 1] -- so
// `filter_active` would reject it wherever the line search lands, and the
// pair may be dropped from the candidate array the line search re-reads.
//
// The test is
//      d0 > (thickness + d_hat + maxDispMag) * (1 + 1e-9)
// compared in squares, on the SAME `dist2_cur` and `maxDispMag` the ACCD
// has already computed: it costs two adds, two multiplies and a compare,
// and no extra loads. It is conservative because the relative motion of
// the two primitives is bounded by `maxDispMag` per unit t (the ACCD's own
// bound: common translation removed, then the largest vertex speed of each
// primitive added), so d(t) >= d0 - t*maxDispMag >= d0 - maxDispMag for
// t in [0, 1]. The 1e-9 relative slack is seven orders of magnitude above
// the double-precision error of the two quantities.
//
// The verdict is STORED BY THIS FUNCTION, through `dcd_keep`, as 1 = keep /
// 0 = drop, at the point where `dist2_cur` is first available -- not returned
// to the caller. That matters for register pressure: the pointer dies before
// the ACCD loop starts instead of staying live across it, which is what keeps
// the compacting instantiation inside its occupancy bracket. On every path
// that does not reach the test (a zero `maxDispMag`) it stores 1 = keep.

template <typename T>
UIPC_GENERIC bool point_edge_cd_broadphase(const Eigen::Vector<T, 3>& x0,
                                           const Eigen::Vector<T, 3>& x1,
                                           const Eigen::Vector<T, 3>& x2,
                                           T                          dist);

template <typename T>
UIPC_GENERIC bool point_edge_ccd_broadphase(const Eigen::Matrix<T, 2, 1>& p,
                                            const Eigen::Matrix<T, 2, 1>& e0,
                                            const Eigen::Matrix<T, 2, 1>& e1,
                                            const Eigen::Matrix<T, 2, 1>& dp,
                                            const Eigen::Matrix<T, 2, 1>& de0,
                                            const Eigen::Matrix<T, 2, 1>& de1,
                                            T                             dist);

template <typename T>
UIPC_GENERIC bool point_triangle_cd_broadphase(const Eigen::Vector<T, 3>& p,
                                               const Eigen::Vector<T, 3>& t0,
                                               const Eigen::Vector<T, 3>& t1,
                                               const Eigen::Vector<T, 3>& t2,
                                               T                          dist);
template <typename T>
UIPC_GENERIC bool edge_edge_cd_broadphase(const Eigen::Vector<T, 3>& ea0,
                                          const Eigen::Vector<T, 3>& ea1,
                                          const Eigen::Vector<T, 3>& eb0,
                                          const Eigen::Vector<T, 3>& eb1,
                                          T                          dist);

template <typename T>
UIPC_GENERIC bool point_triangle_ccd_broadphase(const Eigen::Vector<T, 3>& p,
                                                const Eigen::Vector<T, 3>& t0,
                                                const Eigen::Vector<T, 3>& t1,
                                                const Eigen::Vector<T, 3>& t2,
                                                const Eigen::Vector<T, 3>& dp,
                                                const Eigen::Vector<T, 3>& dt0,
                                                const Eigen::Vector<T, 3>& dt1,
                                                const Eigen::Vector<T, 3>& dt2,
                                                T dist);

template <typename T>
UIPC_GENERIC bool edge_edge_ccd_broadphase(const Eigen::Vector<T, 3>& ea0,
                                           const Eigen::Vector<T, 3>& ea1,
                                           const Eigen::Vector<T, 3>& eb0,
                                           const Eigen::Vector<T, 3>& eb1,
                                           const Eigen::Vector<T, 3>& dea0,
                                           const Eigen::Vector<T, 3>& dea1,
                                           const Eigen::Vector<T, 3>& deb0,
                                           const Eigen::Vector<T, 3>& deb1,
                                           T                          dist);

template <typename T>
UIPC_GENERIC bool point_edge_ccd_broadphase(const Eigen::Vector<T, 3>& p,
                                            const Eigen::Vector<T, 3>& e0,
                                            const Eigen::Vector<T, 3>& e1,
                                            const Eigen::Vector<T, 3>& dp,
                                            const Eigen::Vector<T, 3>& de0,
                                            const Eigen::Vector<T, 3>& de1,
                                            T                          dist);
template <typename T>
UIPC_GENERIC bool point_point_ccd_broadphase(const Eigen::Vector<T, 3>& p0,
                                             const Eigen::Vector<T, 3>& p1,
                                             const Eigen::Vector<T, 3>& dp0,
                                             const Eigen::Vector<T, 3>& dp1,
                                             T                          dist);

template <typename T, bool EarlyOut = false, bool Stats = false, bool DcdCull = false>
UIPC_GENERIC bool point_triangle_ccd(Eigen::Vector<T, 3> p,
                                     Eigen::Vector<T, 3> t0,
                                     Eigen::Vector<T, 3> t1,
                                     Eigen::Vector<T, 3> t2,
                                     Eigen::Vector<T, 3> dp,
                                     Eigen::Vector<T, 3> dt0,
                                     Eigen::Vector<T, 3> dt1,
                                     Eigen::Vector<T, 3> dt2,
                                     T                   eta,
                                     T                   thickness,
                                     int                 max_iter,
                                     T&                  toc,
                                     CCDStatCounter*     stats = nullptr,
                                     T                   d_hat = T(0),
                                     uint8_t*            dcd_keep = nullptr);

template <typename T, bool EarlyOut = false, bool Stats = false, bool DcdCull = false>
UIPC_GENERIC bool edge_edge_ccd(Eigen::Vector<T, 3> ea0,
                                Eigen::Vector<T, 3> ea1,
                                Eigen::Vector<T, 3> eb0,
                                Eigen::Vector<T, 3> eb1,
                                Eigen::Vector<T, 3> dea0,
                                Eigen::Vector<T, 3> dea1,
                                Eigen::Vector<T, 3> deb0,
                                Eigen::Vector<T, 3> deb1,
                                T                   eta,
                                T                   thickness,
                                int                 max_iter,
                                T&                  toc,
                                CCDStatCounter*     stats = nullptr,
                                T                   d_hat = T(0),
                                uint8_t*            dcd_keep = nullptr);

template <typename T, bool EarlyOut = false, bool Stats = false, bool DcdCull = false>
UIPC_GENERIC bool point_edge_ccd(Eigen::Vector<T, 3> p,
                                 Eigen::Vector<T, 3> e0,
                                 Eigen::Vector<T, 3> e1,
                                 Eigen::Vector<T, 3> dp,
                                 Eigen::Vector<T, 3> de0,
                                 Eigen::Vector<T, 3> de1,
                                 T                   eta,
                                 T                   thickness,
                                 int                 max_iter,
                                 T&                  toc,
                                 CCDStatCounter*     stats = nullptr,
                                 T                   d_hat = T(0),
                                 uint8_t*            dcd_keep = nullptr);
template <typename T, bool EarlyOut = false, bool Stats = false, bool DcdCull = false>
UIPC_GENERIC bool point_point_ccd(Eigen::Vector<T, 3> p0,
                                  Eigen::Vector<T, 3> p1,
                                  Eigen::Vector<T, 3> dp0,
                                  Eigen::Vector<T, 3> dp1,
                                  T                   eta,
                                  T                   thickness,
                                  int                 max_iter,
                                  T&                  toc,
                                  CCDStatCounter*     stats = nullptr,
                                  T                   d_hat = T(0),
                                  uint8_t*            dcd_keep = nullptr);
}  // namespace uipc::backend::cuda::distance

#include "details/ccd.inl"
