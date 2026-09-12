#!/usr/bin/env python3
import sys
H = "/root/work/src/src/backends/cuda/contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h"
K = "/root/work/src/src/backends/cuda/contact_system/contact_models/ipc_simplex_normal_contact.cu"
MACROS = """
// ---- s23 PROBE STUBS (temporary, reverted before commit) ----
#ifndef UIPC_STUB_EE_DISTH
#define UIPC_STUB_EE_DISTH 0
#endif
#ifndef UIPC_STUB_MOLL_MATH
#define UIPC_STUB_MOLL_MATH 0
#endif
#ifndef UIPC_STUB_EE_PROJ
#define UIPC_STUB_EE_PROJ 0
#endif
#ifndef UIPC_STUB_EE_PROJ9
#define UIPC_STUB_EE_PROJ9 0
#endif
#ifndef UIPC_STUB_EE_PROJ5
#define UIPC_STUB_EE_PROJ5 0
#endif
#ifndef UIPC_STUB_PT_ALL
#define UIPC_STUB_PT_ALL 0
#endif
#ifndef UIPC_STUB_PT_PROJ
#define UIPC_STUB_PT_PROJ 0
#endif
#ifndef UIPC_STUB_EE_WRITE
#define UIPC_STUB_EE_WRITE 0
#endif
// -------------------------------------------------------------
"""
h = open(H).read()
assert 'UIPC_STUB' not in h, "header already patched -- git checkout first"
h = h.replace("namespace uipc::backend::cuda\n{", MACROS + "\nnamespace uipc::backend::cuda\n{", 1)
old = """        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);"""
new = """        Matrix12x12 HessD;
#if UIPC_STUB_EE_DISTH
        HessD.setZero();
#else
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);
#endif"""
assert h.count(old) == 1; h = h.replace(old, new, 1)
# mollifier math stub -- anchor on the unique `mollified = true;` line
old2 = """        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \\nabla B = \\frac{\\partial B}{\\partial D} \\nabla D$$
        Vector12 GradB = dBdD * GradD;"""
new2 = """#if UIPC_STUB_MOLL_MATH
        G = dBdD * GradD;
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
        return;
#endif
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \\nabla B = \\frac{\\partial B}{\\partial D} \\nabla D$$
        Vector12 GradB = dBdD * GradD;"""
assert h.count(old2) == 1, h.count(old2); h = h.replace(old2, new2, 1)
open(H,'w').write(h)
print("header patched")

k = open(K).read()
assert 'UIPC_STUB' not in k
oldp = """                    if(warp_reduced)
                    {
                        EE_barrier_make_spd(H, flag, E0, E1, E2, E3);
                    }
"""
newp = """                    if(warp_reduced)
                    {
#if !(UIPC_STUB_EE_PROJ || UIPC_STUB_EE_PROJ5)
                        EE_barrier_make_spd(H, flag, E0, E1, E2, E3);
#endif
                    }
"""
assert k.count(oldp)==1; k = k.replace(oldp,newp,1)
oldp2 = """                    else if(ee_reduced_spd)
                        make_spd_translation_free_4x3(H);
                    else
                        make_spd(H);
"""
newp2 = """#if !(UIPC_STUB_EE_PROJ || UIPC_STUB_EE_PROJ9)
                    else if(ee_reduced_spd)
                        make_spd_translation_free_4x3(H);
                    else
                        make_spd(H);
#else
                    else { }
#endif
"""
assert k.count(oldp2)==1; k = k.replace(oldp2,newp2,1)
oldw = """                    DoubletVectorAssembler DVA{EE_Gs};
                    DVA.segment<4>(i * 4).write(EE, G);
                    TripletMatrixAssembler TMA{EE_Hs};
                    TMA.half_block<4>(i * SimplexNormalContact::EEHalfHessianSize)
                        .write(EE, H);
"""
neww = """#if UIPC_STUB_EE_WRITE
                    if(H(0, 0) == 1.2345e-300 && G(0) == 0.0) { }
#else
                    DoubletVectorAssembler DVA{EE_Gs};
                    DVA.segment<4>(i * 4).write(EE, G);
                    TripletMatrixAssembler TMA{EE_Hs};
                    TMA.half_block<4>(i * SimplexNormalContact::EEHalfHessianSize)
                        .write(EE, H);
#endif
"""
assert k.count(oldw)==1; k = k.replace(oldw,neww,1)
oldpt = """            if(idx < ee_offset)  // PT
            {
                int      i    = idx;"""
newpt = """            if(idx < ee_offset)  // PT
            {
#if UIPC_STUB_PT_ALL
                return;
#else
                int      i    = idx;"""
assert k.count(oldpt)==1; k = k.replace(oldpt,newpt,1)
oldpt2 = """                    TMA.half_block<4>(i * SimplexNormalContact::PTHalfHessianSize)
                        .write(PT, H);
                }
                return;
            }"""
newpt2 = """                    TMA.half_block<4>(i * SimplexNormalContact::PTHalfHessianSize)
                        .write(PT, H);
                }
#endif
                return;
            }"""
assert k.count(oldpt2)==1; k = k.replace(oldpt2,newpt2,1)
oldpp = """                    PT_barrier_make_spd(H, flag, P, T0, T1, T2);"""
newpp = """#if !UIPC_STUB_PT_PROJ
                    PT_barrier_make_spd(H, flag, P, T0, T1, T2);
#endif"""
assert k.count(oldpp)==1; k = k.replace(oldpp,newpp,1)
open(K,'w').write(k)
print("kernel patched")
