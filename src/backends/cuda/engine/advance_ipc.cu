#include <sim_engine.h>
#include <uipc/common/range.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <line_search/line_searcher.h>
#include <linear_system/global_linear_system.h>
#include <animator/global_animator.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <newton_tolerance/newton_tolerance_manager.h>
#include <time_integrator/time_integrator_manager.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
void SimEngine::advance()
{
    Float alpha     = 1.0;
    Float beta      = 1.0;
    Float ccd_alpha = 1.0;
    Float cfl_alpha = 1.0;

    // ---- DIAGNOSTIC oracle flags (extras/debug/*, both default off) ----
    // [candidate_reuse_oracle] skip the DCD broadphase re-detection for
    // newton_iter > 0 and reuse the previous iteration's candidate buffers
    // as-is: the wall-time ceiling of a perfect learned candidate predictor.
    // NOT exactness-preserving (the reused set is not a certified superset).
    const bool dcd_candidate_reuse = m_candidate_reuse_oracle->view()[0] != 0;
    if(dcd_candidate_reuse && m_current_frame == 0)  // announce once per run
    {
        logger::warn(
            "[candidate_reuse_oracle] ACTIVE: DCD re-detection is "
            "skipped for newton_iter > 0 (diagnostic only)");
    }

    // ---- FEATURE: certified DCD candidate reuse
    // (collision_detection/dcd_candidate_reuse, default off) ----
    // For newton_iter > 0, keep the candidate buffers as filled by the
    // previous iteration's line-search trajectory detection instead of
    // re-running the DCD detection. Certification (why this is exact, not
    // approximate):
    //
    //  1. The line-search detection sweeps each primitive over
    //     [x0, x0 + alpha_detect * dx] and accepts a pair iff the two swept
    //     boxes, inflated per axis by the pair's expand = d_hat + thickness,
    //     overlap on every axis (a conservative relaxation of the exact
    //     distance test).
    //  2. The line search only ever applies steps alpha <= alpha_detect
    //     (TOI/CFL filters and halvings shrink it), and positions change
    //     nowhere else inside a frame. So every vertex's current position
    //     x0 + alpha * dx is an interior point of its swept box.
    //  3. If a pair's exact distance at the current positions is < expand
    //     (i.e. a fresh DCD detection would report it), the closest points
    //     witness a per-axis gap < expand between the current boxes, hence
    //     between the (larger) swept boxes: the pair is in the reused set.
    //  4. Every extra pair of the reused set is >= expand away, outside the
    //     active window D_range = (thickness, thickness + d_hat], so
    //     filter_active drops it: the active candidate set handed to contact
    //     assembly is IDENTICAL to a fresh detection's.
    const bool certified_candidate_reuse = m_dcd_candidate_reuse->view()[0] != 0;
    if(certified_candidate_reuse && m_current_frame == 0)
    {
        logger::info(
            "[dcd_candidate_reuse] ACTIVE: newton_iter > 0 reuses the "
            "certified trajectory candidate set");
    }
    const bool reuse_candidates = dcd_candidate_reuse || certified_candidate_reuse;
    // [dcd_candidate_reuse_verify] re-run the fresh DCD detection at every
    // reused iteration and check the certification invariant (fresh set
    // contained in reused set) on the host; default off, costly.
    const bool reuse_verify =
        reuse_candidates && m_dcd_candidate_reuse_verify->view()[0] != 0;
    // [warm_start_oracle == 2] replay the captured frame-t positions as the
    // initial Newton iterate of frame t: the ceiling of a perfect learned
    // warm start. Changes the trajectory by construction.
    const bool warm_start_replay = m_warm_start_oracle->view()[0] == 2;

    /***************************************************************************************
    *                                  Function Shortcuts
    ***************************************************************************************/

    auto detect_dcd_candidates = [this](SizeT newton_iter)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect DCD Candidates"};
            m_global_trajectory_filter->detect(0.0);
            m_global_trajectory_filter->filter_active();

            // DIAGNOSTIC (extras/debug/dump_candidates): per-pair candidate
            // dump; no-op unless the flag is set.
            if(m_dump_candidates->view()[0])
                m_global_trajectory_filter->dump_dcd_candidates(m_current_frame, newton_iter);
        }
    };

    auto detect_trajectory_candidates = [this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect Trajectory Candidates"};
            m_global_trajectory_filter->detect(alpha);
        }
    };

    auto filter_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter Contact Candidates"};
            m_global_trajectory_filter->filter_active();
        }
    };

    auto record_friction_candidates = [this]
    {
        if(m_global_trajectory_filter && m_friction_enabled)
        {
            m_global_trajectory_filter->record_friction_candidates();
        }
    };

    auto compute_adaptive_kappa = [this]
    {
        if(m_global_contact_manager)
            m_global_contact_manager->compute_adaptive_parameters();
    };

    auto compute_dytopo_effect = [this]
    {
        // compute the dytopo effect gradient and hessian, containing:
        // 1) contact effect from contact pairs
        // 2) other dynamic topo effects, e.g. point picker, vertex stitch ...
        if(m_global_dytopo_effect_manager)
        {
            Timer timer{"Compute DyTopo Effect"};
            m_global_dytopo_effect_manager->compute_dytopo_effect();
        }
    };

    auto cfl_condition = [&cfl_alpha, this](Float alpha)
    {
        if(m_global_contact_manager)
        {
            Timer timer{"Compute CFL Condition"};
            cfl_alpha = m_global_contact_manager->compute_cfl_condition();
            m_frame_last_cfl_alpha = cfl_alpha;
            if(cfl_alpha < alpha)
            {
                logger::info("CFL Filter: {} < {}", cfl_alpha, alpha);
                return cfl_alpha;
            }
        }

        return alpha;
    };

    auto filter_toi = [&ccd_alpha, this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter CCD TOI"};
            ccd_alpha = m_global_trajectory_filter->filter_toi(alpha);
            m_frame_last_ccd_toi = ccd_alpha;
            // ccd_alpha is a fraction of the swept step `alpha`; compose to
            // get the absolute step fraction
            Float absolute_alpha = alpha * ccd_alpha;
            if(absolute_alpha < alpha)
            {
                logger::info("CCD Filter: {} < {}", absolute_alpha, alpha);
                return absolute_alpha;
            }
        }

        return alpha;
    };

    // Stiff-GIPC design: cap the step with the feasible step over the active
    // contact set (each active pair keeps at least 20% of its current gap),
    // so the swept-AABB trajectory candidates generated afterwards are fewer.
    auto feasible_step = [this](Float alpha)
    {
        if(m_global_contact_manager)
        {
            Timer timer{"Compute Feasible Step"};
            Float feasible_alpha = m_global_contact_manager->compute_feasible_step();
            if(feasible_alpha < alpha)
            {
                logger::info("Feasible Step Filter: {} < {}", feasible_alpha, alpha);
                return feasible_alpha;
            }
        }

        return alpha;
    };

    auto compute_energy = [this, filter_dcd_candidates](Float alpha) -> Float
    {
        // Step Forward => x = x_0 + alpha * dx
        m_global_vertex_manager->step_forward(alpha);
        m_line_searcher->step_forward(alpha);

        // Dumps surface after step forward
        if(m_dump_surface->view()[0])
        {
            dump_global_surface();
        }

        // Update the collision pairs
        filter_dcd_candidates();

        // Compute New Energy => E
        return m_line_searcher->compute_energy(false);
    };

    auto compute_animation_substep_ratio = [this](SizeT newton_iter)
    {
        // compute the ratio to the aim position.
        // dst = prev_position + ratio * (position - prev_position)
        if(m_global_animator)
        {
            m_global_animator->compute_substep_ratio(newton_iter);
            logger::info("Animation Substep Ratio: {}", m_global_animator->substep_ratio());
        }
    };

    auto animation_reach_target = [this]()
    {
        if(m_global_animator)
        {
            return m_global_animator->substep_ratio() >= 1.0;
        }
        return true;
    };

    auto convergence_check = [&](SizeT newton_iter) -> bool
    {
        if(!animation_reach_target())
            return false;

        if(m_semi_implicit_enabled)
        {
            // Ref: https://arxiv.org/abs/2512.12151, Algorithm 1.
            // newton/semi_implicit/K_min plays Stiff-GIPC's Kmin role: beta
            // accumulation only starts from this iteration on; it is not a
            // floor on the Newton iteration count (that is newton/min_iter,
            // default 0 = no forced minimum).
            auto k_min = m_semi_implicit_kmin->view()[0];
            auto eps   = m_semi_implicit_beta_tol;
            if(newton_iter >= k_min)
                beta = (1.0 - alpha) * beta;

            if(beta <= eps)
                return true;  // early terminate
        }

        NewtonToleranceManager::ResultInfo result_info;
        result_info.frame(m_current_frame);
        result_info.newton_iter(newton_iter);
        m_newton_tolerance_manager->check(result_info);

        if(!result_info.converged())
            return false;

        // ccd alpha should close to 1.0
        if(ccd_alpha < m_ccd_tol->view()[0])
            return false;

        return true;
    };

    auto update_diff_parm = [this]()
    {
        if(m_global_diff_sim_manager)
        {
            Timer timer{"Update Diff Parm"};
            m_global_diff_sim_manager->update();
        }
    };

    auto check_line_search_iter = [&](SizeT line_search_iter_after_loop, Float E0, Float last_E)
    {
        if(line_search_iter_after_loop >= m_line_searcher->max_iter())
        {
            m_frame_hit_line_search_limit = true;
            // alpha was halved once more after the last failed try,
            // so the last actually tried step length is 2*alpha
            Float alpha_last = 2.0 * alpha;
            Float abs_E0     = std::abs(E0);
            Float rel_E_increase = (last_E - E0) / (abs_E0 > 1e-30 ? abs_E0 : 1e-30);
            auto detail = fmt::format(
                "Line Search Exits with Max Iteration: {} (Frame={}, Newton={}, "
                "alpha_last={}, E0={:.6e}, E_last={:.6e}, rel_E_increase={:.3e}, "
                "ccd_alpha={}, cfl_alpha={})",
                m_line_searcher->max_iter(),
                m_current_frame,
                m_newton_iter,
                alpha_last,
                E0,
                last_E,
                rel_E_increase,
                ccd_alpha,
                cfl_alpha);
            logger::warn("{}", detail);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: " + detail);
            }
        }
    };

    auto check_newton_iter = [this](IndexT newton_iter_after_loop)
    {
        auto newton_max = m_newton_max_iter->view()[0];
        if(newton_iter_after_loop >= newton_max)
        {
            m_frame_hit_newton_limit = true;
            logger::warn("Newton Iteration Exits with Max Iteration: {} (Frame={})",
                         newton_max,
                         m_current_frame);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: Newton Iteration Exits with Max Iteration");
            }
        }
        else
        {
            logger::info("Newton Iteration Converged with Iteration Count: {}, Max: {}",
                         newton_iter_after_loop,
                         newton_max);
        }
    };

    /***************************************************************************************
    *                                  Core Pipeline
    ***************************************************************************************/

    // Abort on exception if the runtime check is enabled for debugging
    constexpr bool AbortOnException = uipc::RUNTIME_CHECK;

    auto pipeline = [&]() noexcept(AbortOnException)
    {
        Timer timer{"Pipeline"};

        ++m_current_frame;
        reset_frame_stats();

        logger::info(R"(>>> Begin Frame: {})", m_current_frame);

        // Rebuild Scene
        {
            Timer timer{"Rebuild Scene"};
            // Trigger the rebuild_scene event, systems register their actions will be called here
            m_state = SimEngineState::RebuildScene;
            {
                event_rebuild_scene();

                // TODO: rebuild the vertex and surface info
                // m_global_vertex_manager->rebuild_vertex_info();
                // m_global_surface_manager->rebuild_surface_info();
            }

            // After the rebuild_scene event, the pending creation or deletion can be solved
            world().scene().solve_pending();

            // Update the diff parms
            update_diff_parm();
        }

        // Simulation:
        {
            Timer timer{"Simulation"};

            // 0. Process External Changes
            m_global_vertex_manager->update_attributes();
            [[maybe_unused]] AABB bbox =
                m_global_vertex_manager->compute_vertex_bounding_box();

            // 1. Record Friction Candidates at the beginning of the frame
            m_global_vertex_manager->record_prev_positions();
            record_friction_candidates();

            // 2. Predict Motion => x_tilde = x + v * dt
            m_state = SimEngineState::PredictMotion;
            // MUST step animation before predicting dof
            // some animation may provide information for DOF prediction
            step_animation_and_external_forces();
            m_time_integrator_manager->predict_dof();

            // DIAGNOSTIC: overwrite the initial Newton iterate with the
            // captured solution of this frame (warm-start oracle)
            if(warm_start_replay)
                oracle_inject_frame();

            // 3. Adaptive Parameter Calculation
            detect_dcd_candidates(0);
            compute_adaptive_kappa();

            // 4. Nonlinear-Newton Iteration
            m_newton_tolerance_manager->pre_newton(m_current_frame);

            auto newton_max_iter = m_newton_max_iter->view()[0];
            auto newton_min_iter = m_newton_min_iter->view()[0];
            beta                 = 1.0;
            IndexT newton_iter   = 0;
            // perf/kernels (K6): the accepted line-search trial of iteration k
            // evaluated the energy at exactly the positions iteration k+1
            // starts from (x0 = x, frozen friction/kinetic targets, same active
            // contact set) -> reuse it as E0 instead of re-evaluating.
            // UIPC_REUSE_E0=0 disables (A/B).
            static const bool reuse_e0 = []
            {
                const char* e = std::getenv("UIPC_REUSE_E0");
                return !(e && e[0] == '0');
            }();
            bool  have_prev_E = false;
            Float prev_E      = 0.0;
            // K6 fix: the animator's substep ratio advances per Newton
            // iteration (dst = prev + ratio * (aim - prev)), which moves the
            // constraint targets and changes the energy at unchanged
            // positions -> the cached trial energy is stale whenever the
            // ratio changed since the previous iteration (found by
            // apps/tests/sim_case/30_fem_animiator_substep: line search
            // failed at frame 1 with E_last/E0 = 59x).
            Float prev_substep_ratio = -1.0;
            for(; newton_iter < newton_max_iter; ++newton_iter)
            {
                Timer timer{"Newton Iteration"};
                m_newton_iter             = newton_iter;
                m_frame_newton_iterations = newton_iter + 1;

                // 1) Compute animation substep ratio
                compute_animation_substep_ratio(newton_iter);
                if(m_global_animator)
                {
                    const Float ratio = m_global_animator->substep_ratio();
                    if(ratio != prev_substep_ratio)
                        have_prev_E = false;  // targets moved: re-evaluate E0
                    prev_substep_ratio = ratio;
                }


                // 2) Build Collision Pairs
                // FEATURE (collision_detection/dcd_candidate_reuse) /
                // DIAGNOSTIC (candidate_reuse_oracle): with reuse active,
                // iterations > 0 keep the previous line-search trajectory
                // candidate set (a certified superset, see the certification
                // comment above) instead of re-running the DCD detection.
                // The active set stays fresh: compute_energy re-runs
                // filter_active at the stepped positions inside the line
                // search.
                if(newton_iter > 0 && !reuse_candidates)
                {
                    detect_dcd_candidates(newton_iter);
                }
                else if(newton_iter > 0 && reuse_verify && m_global_trajectory_filter)
                {
                    // DIAGNOSTIC (extras/debug/dcd_candidate_reuse_verify):
                    // snapshot the reused set, run the fresh detection, check
                    // containment on the host, restore the reused set.
                    m_global_trajectory_filter->snapshot_reused_candidates();
                    m_global_trajectory_filter->detect(0.0);
                    m_global_trajectory_filter->verify_reused_candidates(m_current_frame, newton_iter);
                }


                // perf round 6 (s10): arm the assembly prepass -- record the
                // fork event while the default stream is still *ahead* of the
                // contact phase, so the side stream is not ordered behind
                // contact part 1's join. Nothing is enqueued yet.
                // UIPC_ABD_GH_PREPASS=0 = the old order (no prepass at all).
                m_global_linear_system->arm_assembly_prepass();

                // 3) Compute Dynamic Topo Effect Gradient and Hessian => G:Vector3, H:Matrix3x3
                //    - Contact Effect
                //    - Other DyTopo Effects
                m_state = SimEngineState::ComputeDyTopoEffect;
                compute_dytopo_effect();

                // s10: now that contact part 1 and part 2 have been issued,
                // enqueue the prepass on its side stream. Order matters: the
                // work distributor hands SMs out in submission order, so the
                // big 255-register blocks must be queued first and the prepass
                // fills what they leave.
                //
                // s11: under the default (UIPC_ABD_GH_PREPASS=4) the launch has
                // usually already happened, inside the K9 contact fork -- by
                // the time the host gets here it has been blocked in
                // GlobalDyTopoEffectManager::_distribute's D2H until the
                // contact kernels drained, which is far too late to reach
                // contact part 1's shadow. This call stays as the backstop for
                // the iterations that never reach the contact call site (the
                // fused contact launch, a gradient-only assemble, an empty
                // part) and it is a no-op when the prepass is already in
                // flight.
                m_global_linear_system->launch_assembly_prepass();


                // 4) Solve Global Linear System => dx = A^-1 * b
                m_state = SimEngineState::SolveGlobalLinearSystem;
                {
                    Timer timer{"Solve Global Linear System"};
                    m_global_linear_system->solve();
                    m_frame_linear_solver_iterations +=
                        m_global_linear_system->last_solve_iterations();
                }


                // 5) Collect Vertex Displacements Globally
                m_global_vertex_manager->collect_vertex_displacements();

                // 7) Begin Line Search
                m_state = SimEngineState::LineSearch;
                {
                    Timer timer{"Line Search"};

                    // Reset Alpha
                    alpha = 1.0;

                    // Record Current State x to x_0
                    m_line_searcher->record_start_point();
                    m_global_vertex_manager->record_start_point();

                    if(m_dump_surface->view()[0])
                    {
                        dump_global_surface_pre_ccd(newton_iter);
                    }

                    // Stiff-GIPC design: cap the step with the feasible step
                    // over the active contact set first, then generate the
                    // swept-AABB trajectory candidates over the capped step
                    // (fewer candidates than over the full step)
                    alpha = feasible_step(alpha);

                    detect_trajectory_candidates(alpha);

                    // Compute Current Energy => E_0
                    Float E0 = (reuse_e0 && have_prev_E) ?
                                   prev_E :
                                   m_line_searcher->compute_energy(true);  // initial energy

                    // CCD filter
                    alpha = filter_toi(alpha);

                    // CFL Condition (Stiff-GIPC design: the per-step speed cap
                    // applies only when some trajectory pair actually hits within
                    // this step; in free flight it must not bite)
                    if(ccd_alpha < 1.0)
                        alpha = cfl_condition(alpha);

                    // Line Search Iteration
                    bool  converged        = convergence_check(newton_iter);
                    SizeT line_search_iter = 0;
                    Float last_E           = E0;
                    for(; line_search_iter < m_line_searcher->max_iter(); ++line_search_iter)
                    {
                        Timer timer{"Line Search Iteration"};
                        m_line_search_iter = line_search_iter;
                        ++m_frame_line_search_trials;
                        m_frame_last_line_search_alpha = alpha;

                        // Compute Test Energy:
                        //  * Step Forward => x = x_0 + alpha * dx
                        //  * Compute New Energy => E
                        Float E = compute_energy(alpha);
                        last_E  = E;

                        // To prevent numerical energy (fake-) increasing caused by tiny dx
                        if(converged)
                            break;

                        // Check Energy Decrease
                        // TODO: maybe better condition like Wolfe condition/Armijo condition in the future
                        bool energy_decrease = (E <= E0);

                        // Check Inversion
                        // TODO: Inversion check if needed
                        bool no_inversion = true;

                        bool success = energy_decrease && no_inversion;
                        if(success)
                            break;

                        // If not success, then shrink alpha
                        alpha /= 2;
                    }

                    // Check Line Search Iteration: report warnings or throw exceptions if needed
                    check_line_search_iter(line_search_iter, E0, last_E);
                    // the state now sits at the last trial; its energy is next E0
                    prev_E      = last_E;
                    have_prev_E = m_line_searcher->max_iter() > 0;

                    // newton/min_iter is a pure hard floor (default 0 = off);
                    // the semi-implicit Kmin lives in newton/semi_implicit/K_min
                    bool terminated = converged && (newton_iter >= newton_min_iter);
                    if(terminated)
                    {
                        m_frame_converged = true;
                        break;
                    }
                }
            }

            // 5. Update Velocity => v = (x - x_0) / dt
            m_state = SimEngineState::UpdateVelocity;
            {
                Timer timer{"Update Velocity"};
                m_time_integrator_manager->update_state();
            }

            // DIAGNOSTIC: record the converged positions of this frame
            // (warm-start oracle, capture mode)
            if(m_warm_start_oracle->view()[0] == 1)
                oracle_capture_frame();

            // Check Newton Iteration
            // report warnings or throw exceptions if needed
            check_newton_iter(newton_iter);
            m_frame_completed = true;
        }

        logger::info("<<< End Frame: {}", m_current_frame);
    };

    try
    {
        pipeline();
    }
    catch(const SimEngineException& e)
    {
        logger::error("Engine Advance Error: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
    catch(const std::exception& e)
    {
        UIPC_ASSERT(false, "Unexpected Exception: {}", e.what());
    }
}
}  // namespace uipc::backend::cuda
