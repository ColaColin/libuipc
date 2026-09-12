#!/usr/bin/env python3
"""s29: hoist the register-only rejections of the four leaf predicates ahead of
the scattered id loads and the mask-tabular lookups.  Behind UIPC_BVH_PRED_ORDER.
"""
import re, sys

P = "/root/work/src/src/backends/cuda/collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.cu"
s = open(P).read()
orig = s

# ---------------------------------------------------------------- env + probe
ENV = r'''
    // ---------------------------------------------------------------------
    // s29: the four leaf predicates each end with one or two rejections that
    // read NO global memory (a shared-index test on the two simplices, and a
    // same-body test whose operands are already in registers), placed AFTER
    // the 6-8 scattered contact/subscene id loads and the two mask-tabular
    // lookups that gate every pair.  Hoisting them is a pure reordering of
    // independent side-effect-free `return false` guards, so the surviving
    // pair set is identical.
    //   UIPC_BVH_PRED_ORDER = 1 (default) hoisted, 0 = the pre-s29 order,
    //                         2 = run BOTH orders per pair and count
    //                             disagreements on device.
    //   UIPC_BVH_PRED_STATS = 1 counts, per predicate, how many staged pairs
    //                         each free test would reject.
    // Probe/verify counters, 8 slots per predicate:
    //   0 calls, 1 topo-rejected, 2 bid-rejected (topo passed),
    //   3 verify comparisons, 4 verify mismatches
    // ---------------------------------------------------------------------
    __device__ unsigned long long g_bvh_pred_stats[4 * 8];

    struct BvhPredEnv
    {
        int  order = 1;
        bool stats = false;
    };

    inline const BvhPredEnv& bvh_pred_env()
    {
        static BvhPredEnv env = []
        {
            BvhPredEnv e;
            if(const char* s = std::getenv("UIPC_BVH_PRED_ORDER"))
            {
                int v   = std::atoi(s);
                e.order = (v >= 0 && v <= 2) ? v : 1;
            }
            if(const char* s = std::getenv("UIPC_BVH_PRED_STATS"))
                e.stats = std::atoi(s) != 0;
            return e;
        }();
        return env;
    }

    struct BvhPredStatsReport
    {
        ~BvhPredStatsReport()
        {
            if(!bvh_pred_env().stats && bvh_pred_env().order != 2)
                return;
            unsigned long long h[4 * 8] = {};
            if(cudaMemcpyFromSymbol(h, g_bvh_pred_stats, sizeof(h)) != cudaSuccess)
                return;
            static const char* names[4] = {"AllP_CodimP", "CodimP_AllE", "AllE_AllE", "AllP_AllT"};
            for(int p = 0; p < 4; ++p)
            {
                unsigned long long* a = h + p * 8;
                if(a[0] == 0 && a[3] == 0)
                    continue;
                if(a[0])
                    std::fprintf(stderr,
                                 "[BvhPredStats] %-12s calls=%llu topo_reject=%llu (%.2f%%) "
                                 "bid_reject=%llu (%.2f%%) free_reject=%.2f%%\n",
                                 names[p], a[0], a[1], 100.0 * a[1] / a[0], a[2],
                                 100.0 * a[2] / a[0], 100.0 * (a[1] + a[2]) / a[0]);
                if(a[3])
                    std::fprintf(stderr,
                                 "[BvhPredVerify] %-12s %llu pairs evaluated under both orders, "
                                 "%llu disagreeing\n",
                                 names[p], a[3], a[4]);
            }
        }
    };
    BvhPredStatsReport g_bvh_pred_stats_report;

'''

anchor = "    struct InfoStacklessBVHSimplexTrajectoryFilter_detect_node_pred"
assert s.count(anchor) == 1
s = s.replace(anchor, ENV + anchor, 1)

# ---------------------------------------------------------------- per-predicate
# (struct name, stats slot, the free-test block as it appears today, the text
#  that currently precedes the id loads)
SPECS = [
    # AllP_CodimP: dimensions(V) is a load, so only the bid test is free; the
    # V_is_codim test stays where it is.
    ("AllP_CodimP", 0,
     """            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 P0 = Ps(V);""",
     None,
     """            Vector2i cids = {contact_element_ids(V), contact_element_ids(codimV)};""",
     None),
    ("CodimP_AllE", 1,
     """            if(E[0] == codimV || E[1] == codimV)
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 E0 = Ps(E[0]);""",
     "E[0] == codimV || E[1] == codimV",
     """            Vector3i cids = {contact_element_ids(codimV),""",
     None),
    ("AllE_AllE", 2,
     """            if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1])
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 E0_0 = Ps(E0[0]);""",
     "E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1]",
     """            Vector4i cids = {contact_element_ids(E0[0]),""",
     None),
    ("AllP_AllT", 3,
     """            if(F[0] == V || F[1] == V || F[2] == V)
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 P = Ps(V);""",
     "F[0] == V || F[1] == V || F[2] == V",
     """            Vector4i cids = {contact_element_ids(V),""",
     None),
]

for name, slot, tail_block, topo_expr, load_anchor, _ in SPECS:
    assert s.count(tail_block) == 1, ("tail", name, s.count(tail_block))
    assert s.count(load_anchor) == 1, ("anchor", name, s.count(load_anchor))
    keep = tail_block.rsplit("\n\n", 1)[1]          # the first line after the guards
    topo = topo_expr if topo_expr else "false"
    free_tests = f"""            // s29: register-only rejections, hoisted (see BvhPredEnv).
            const bool topo_reject = ({topo});
            const bool bid_reject =
                !topo_reject && info.bid_i == info.bid_j
                && info.bid_i != static_cast<IndexT>(-1) && !body_self_collision(info.bid_i);
            if(stats)
            {{
                atomicAdd(g_bvh_pred_stats + {slot} * 8 + 0, 1ull);
                if(topo_reject)
                    atomicAdd(g_bvh_pred_stats + {slot} * 8 + 1, 1ull);
                else if(bid_reject)
                    atomicAdd(g_bvh_pred_stats + {slot} * 8 + 2, 1ull);
            }}
            if(Order == 1 && (topo_reject || bid_reject))
                return false;

"""
    # 1. drop the guards from their old position, keep them for Order == 0
    s = s.replace(tail_block,
                  f"""            if(Order == 0 && (topo_reject || bid_reject))
                return false;

{keep}""", 1)
    # 2. insert the free tests just before the first id load
    s = s.replace(load_anchor, free_tests + load_anchor, 1)

    # 3. turn operator() into a templated eval<Order> + dispatcher
    struct_tag = f"InfoStacklessBVHSimplexTrajectoryFilter_detect_{name}_pred"
    sig = "        __device__ bool operator()(InfoStacklessBVH::LeafPredInfo info) const\n        {\n"
    i = s.index(struct_tag)
    j = s.index(sig, i)
    new_sig = (f"""        int  order = 1;
        bool stats = false;

        __device__ bool operator()(InfoStacklessBVH::LeafPredInfo info) const
        {{
            if(order == 2)
            {{
                bool a = eval<0>(info);
                bool b = eval<1>(info);
                atomicAdd(g_bvh_pred_stats + {slot} * 8 + 3, 1ull);
                if(a != b)
                    atomicAdd(g_bvh_pred_stats + {slot} * 8 + 4, 1ull);
                return a;
            }}
            return order == 1 ? eval<1>(info) : eval<0>(info);
        }}

        template <int Order>
        __device__ bool eval(InfoStacklessBVH::LeafPredInfo info) const
        {{
""")
    s = s[:j] + new_sig + s[j + len(sig):]

# ---------------------------------------------------------------- ctor sites
for name in ("AllP_CodimP", "CodimP_AllE", "AllE_AllE", "AllP_AllT"):
    tag = f"InfoStacklessBVHSimplexTrajectoryFilter_detect_{name}_pred{{"
    i = s.index(tag, s.index("auto node_pred"))
    j = s.index("        alpha};", i)
    s = s[:j] + "        alpha,\n        bvh_pred_env().order,\n        bvh_pred_env().stats};" + s[j + len("        alpha};"):]

if "#include <cstdlib>" not in s:
    s = s.replace("#include <sim_engine.h>", "#include <sim_engine.h>\n#include <cstdlib>\n#include <cstdio>", 1)

assert s != orig
open(P, "w").write(s)
print("patched", len(orig), "->", len(s))
