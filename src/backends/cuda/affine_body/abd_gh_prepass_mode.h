#pragma once
#include <cstdlib>

namespace uipc::backend::cuda
{
// round 6 (s10/s11): `UIPC_ABD_GH_PREPASS` -- *where in the host's issue order*
// the ABD body-local kinetic/shape gradient+Hessian is enqueued on its side
// stream. Read once, and shared by the two systems that need it:
// `ABDLinearSubsystem` (arms the fork, joins it, owns the buffers) and
// `IPCSimplexNormalContact` (owns the two call sites inside the K9 contact
// fork), so that the default lives in exactly one place.
//
//   0 = off: the body-local G/H runs in place on the default stream, inside
//       _assemble_kinetic_shape. This is the pre-s10 order and the rollback.
//   1 = enqueued by SimEngine after the whole dytopo-effect phase (s10).
//   2 = enqueued by ABDLinearSubsystem itself, before the contact phase.
//   3 = enqueued between contact part 1's and part 2's launches.
//   4 = enqueued just after contact part 2's launch -- **the default**.
//
// s11 measured all five, one build, interleaved, with a bit-identical null arm
// (ms per Newton iteration vs 0; rigid-wrecking-balls n=30, cube-wall-cloth
// n=10-12, and each scene's null envelope beside it):
//
//                      rwb        cwc       mas-bunny   case2
//     1 (s10)        -4.97 %    +0.02 %      +0.03 %    -0.07 %
//     2              -4.81 %    -1.67 %        --         --
//     3              -1.97 %    -3.11 %        --         --
//     4 (default)    -4.82 %    -2.13 %      +0.09 %    -0.46 %
//     null           -0.70 %    -0.11 %       0.00 %    ...
//
// 1, 2 and 4 are statistically indistinguishable on rwb (Welch p = 0.59-0.95 at
// n=30), so rwb alone cannot choose between them; cube-wall-cloth can, and
// there mode 1 -- the arm that wins rwb -- is worth **nothing**, while 4 is
// worth -2.1 % (p = 4e-05). Mode 4 is the only arm at the top on both, which is
// why it is the default. Mode 3 inverts: it is the best arm on cwc and the
// worst of the four on rwb, because it is issued *ahead of contact part 2* and
// so takes SMs from the kernel that was 92-98 % hidden inside part 1.
//
// **Trap for a future A/B**: modes 3 and 4 live inside the K9 contact split, so
// they only fire when that split is taken (`UIPC_CONTACT_SPLIT=2`, the default,
// and both pair lists non-empty, and not a gradient-only assemble). Otherwise
// SimEngine's backstop hook launches the prepass and the behaviour degrades to
// mode 1. So an A/B of `UIPC_CONTACT_SPLIT` at this default is not measuring the
// split alone -- the `=0` and `=1` arms silently move the prepass to mode 1 as
// well. Pin `UIPC_ABD_GH_PREPASS=1` in both arms of any such sweep.
//
// `UIPC_GRID_SPREAD_VERIFY=1` forces the prepass off: the s20/s24 spread
// verifiers stage their shadow copies and their comparison kernels on the
// default stream and cannot see a launch that ran on a side stream, so leaving
// both on would make that verification silently vacuous.
inline int abd_gh_prepass_mode()
{
    static const int mode = []
    {
        const char* v = std::getenv("UIPC_GRID_SPREAD_VERIFY");
        if(v && v[0] != '0')
            return 0;
        const char* e = std::getenv("UIPC_ABD_GH_PREPASS");
        if(!e)
            return 4;
        int m = std::atoi(e);
        return (m >= 0 && m <= 4) ? m : 4;
    }();
    return mode;
}
}  // namespace uipc::backend::cuda
