#!/usr/bin/env python3
"""Apply the fixed-sweep 3x3 SVD change to a libuipc tree.
usage: apply_patch.py <src_root> <body.hpp> <devtest.cu>"""
import sys, os
root, bodyf, testf = sys.argv[1], sys.argv[2], sys.argv[3]
body = open(bodyf).read()

# ---- 1. algorithm/qr_svd.hpp : add qr_svd_fixed -------------------------
p = os.path.join(root, 'src/backends/cuda/algorithm/qr_svd.hpp')
s = open(p).read()
assert 'qr_svd_fixed' not in s, 'already patched'
anchor = '}  // namespace math'
assert anchor in s
s = s.replace(anchor, body + '\n' + anchor, 1)
open(p, 'w').write(s)
print('patched', p)

# ---- 2. stable_neo_hookean_3d.cu : template on FixedSvd -----------------
p = os.path.join(root, 'src/backends/cuda/finite_element/constitutions/stable_neo_hookean_3d.cu')
s = open(p).read()

old = '''    template <bool HoistStretch>
    __global__ void StableNeoHookean3D_do_compute_gradient_hessian_kernel('''
new = '''    // r5-s01: number of cyclic Jacobi sweeps in the fixed-iteration SVD.
    // 4 is the smallest count that reaches full double precision: over 65 536
    // samples of every deformation regime probed, the worst case needs 4
    // (mean 3.2, median 3, p95 4) - the same shape as the iterative path's
    // trip-count histogram in probe p01, and the same reason a fixed count is
    // affordable here. 3 sweeps leaves 3.5e-6 residual; 5 buys nothing and
    // costs 153 more FP64 instructions.
    constexpr int SnkSvdSweeps = 4;

    template <bool HoistStretch, bool FixedSvd>
    __global__ void StableNeoHookean3D_do_compute_gradient_hessian_kernel('''
assert old in s
s = s.replace(old, new, 1)

old = '''        math::qr_svd(F, S, U, V);

        const Float     evScale = lambda * (J - 1.0) - mu;'''
new = '''        if constexpr(FixedSvd)
            math::qr_svd_fixed<SnkSvdSweeps>(F, S, U, V);
        else
            math::qr_svd(F, S, U, V);

        const Float     evScale = lambda * (J - 1.0) - mu;'''
assert old in s
s = s.replace(old, new, 1)

old = '''    bool m_hoist_stretch = true;

    virtual void do_build(BuildInfo& info) override
    {
        const char* e   = std::getenv("UIPC_SNK1_HOIST_STRETCH");
        m_hoist_stretch = !(e && e[0] == '0');
    }'''
new = '''    bool m_hoist_stretch = true;

    // r5-s01: fixed-sweep branch-free Jacobi SVD instead of the iterative
    // Wilkinson-shift bidiagonal QR (UIPC_QR_SVD_FIXED=0 restores the old
    // path). Rounding-level change, not bit-identical.
    bool m_fixed_svd = true;

    virtual void do_build(BuildInfo& info) override
    {
        const char* e   = std::getenv("UIPC_SNK1_HOIST_STRETCH");
        m_hoist_stretch = !(e && e[0] == '0');

        const char* f = std::getenv("UIPC_QR_SVD_FIXED");
        m_fixed_svd   = !(f && f[0] == '0');
    }'''
assert old in s
s = s.replace(old, new, 1)

old = '''        if(m_hoist_stretch)
            launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<true>);
        else
            launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<false>);'''
new = '''        if(m_hoist_stretch)
        {
            if(m_fixed_svd)
                launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<true, true>);
            else
                launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<true, false>);
        }
        else
        {
            if(m_fixed_svd)
                launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<false, true>);
            else
                launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<false, false>);
        }'''
assert old in s
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print('patched', p)

# ---- 3. device-side randomised verifier as a catch2 test ----------------
p = os.path.join(root, 'apps/tests/backends/cuda/qr_svd.cu')
s = open(p).read()
assert 'qr_svd_fixed' not in s
s = s.replace('#include <cmath>\n#include <vector>',
              '#include <Eigen/Dense>\n#include <cmath>\n#include <vector>\n'
              '#include <random>\n#include <algorithm>', 1)
assert '#include <random>' in s
s = s.rstrip() + '\n\n' + open(testf).read()
open(p, 'w').write(s)
print('patched', p)
