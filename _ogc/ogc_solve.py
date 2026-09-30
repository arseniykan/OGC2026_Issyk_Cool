# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Top-level solve orchestration: single-process solve + multi-process
island parallelism (the evaluation server allows 4 CPU cores)."""

import os
import sys
import time
import traceback


def _make_sync_cb(shared, lock):
    """Cross-island sync: publish our best, adopt the global best if better.
    `shared` is a Manager dict with keys 'obj' and 'snap'."""
    def cb(best_obj, best_snap):
        with lock:
            g_obj = shared.get("obj", float("inf"))
            if best_obj < g_obj - 1e-9:
                shared["obj"] = float(best_obj)
                shared["snap"] = best_snap
                return None
            if g_obj < best_obj - 1e-9:
                return g_obj, shared.get("snap")
        return None
    return cb


def _twophase_construct(state, prep, rng, t_end_epoch, log=None):
    """Time-decomposed construction ('hole shapes over time'): deep-solve
    the EARLY half of the horizon first -- construct only blocks released
    before the demand-valley cut and run a dedicated LNS on that small
    subproblem -- then place the late half on the settled landscape. The
    early congestion cascades into everything downstream, so optimizing it
    before late blocks exist attacks the fragmentation process itself."""
    import numpy as np
    from _ogc.ogc_search import construct, LNS

    p = prep
    H = int(max(p.R + p.P))
    dem = np.zeros(H + 1)
    for i in range(p.n):
        a = float(p.ncells0[p.orient_base[i]])
        dem[int(p.R[i]):min(H, int(p.R[i]) + int(p.P[i]))] += a
    lo, hi = int(0.3 * H), max(int(0.3 * H) + 1, int(0.7 * H))
    Tcut = lo + int(np.argmin(dem[lo:hi]))
    early = [i for i in range(p.n) if p.R[i] < Tcut]
    late = [i for i in range(p.n) if p.R[i] >= Tcut]
    if len(early) < 10 or len(late) < 10:
        construct(state, deadline=t_end_epoch)
        return
    key = lambda i: (p.D[i], p.R[i], -p.wl[i])
    construct(state, order=sorted(early, key=key), deadline=t_end_epoch)
    tA = time.time() + 0.30 * max(0.0, t_end_epoch - time.time())
    if tA < t_end_epoch:
        # phase-A LNS: no cross-island sync (adopting a FULL solution would
        # smuggle late blocks into the half-solved state)
        lnsA = LNS(state, rng, tA, log=None, sync_cb=None)
        lnsA.run(T0=max(1.0, 0.5 * p.w1))
    construct(state, order=sorted(late, key=key), deadline=t_end_epoch)
    if log:
        log(f"twophase: Tcut={Tcut} early={len(early)} late={len(late)} "
            f"obj={state.objective():.0f}")


def solve_single(prob_info, t_end_epoch, seed, log=None, state_out=None,
                 shared=None, lock=None, targets=None, jit_deadline=None,
                 day_targets=None, day_resolve=False, assign_lock=None):
    """Construct + LNS until t_end_epoch. Returns (obj, solution_dict).

    jit_deadline (epoch, optional): give up if JIT compilation is not done
    by then -- raises TimeoutError so short-time-limit callers can return
    their guaranteed-feasible fallback instead of overrunning the clock."""
    import threading

    import numpy as np

    from _ogc.ogc_kernels import warmup

    # overlap JIT compilation with instance preprocessing (numba compilation
    # is internally locked, so concurrent first-calls are safe)
    wt = threading.Thread(target=warmup, daemon=True)
    wt.start()

    from _ogc.ogc_prep import Prep
    from _ogc.ogc_state import State
    from _ogc.ogc_search import construct, LNS, polish

    prep = Prep(prob_info)
    if jit_deadline is not None:
        wt.join(timeout=max(0.1, jit_deadline - time.time()))
        if wt.is_alive():
            # a shipped cache that the server silently rejected leaves us
            # mid-compilation with no time to finish: bail out on time
            raise TimeoutError("JIT compilation exceeded the sprint budget")
    else:
        wt.join()
    rng = np.random.default_rng(seed)
    state = State(prep, rng)
    # space-time shadow pricing (global lookahead): placement is charged
    # for consuming area-days that the future demand curve needs
    try:
        _pg = float(os.environ.get("OGC_PRICE", "0"))
    except (TypeError, ValueError):
        _pg = 0.0
    if _pg > 0.0:
        try:
            state.set_price(
                _pg,
                lookahead=int(float(os.environ.get("OGC_PRICE_L", "8"))),
                eta_p=float(os.environ.get("OGC_PRICE_ETA", "0.65")))
        except Exception:
            pass
    # island 3 stays guide-free as a hedge against relaxation misdirection;
    # cross-pollination lets the best of guided/unguided win
    if (targets is not None and len(targets) == prep.n
            and (seed // 1000) % 4 != 3):
        state.guide = list(targets)
        try:
            gw = float(os.environ.get("OGC_GUIDE_W", "10"))
            if not (gw == gw):  # a knob typo ('nan') must not poison pruning
                gw = 10.0
        except (TypeError, ValueError):
            gw = 10.0
        state.guide_w = gw * prep.w3
    # day-level targets ride on the same guided/unguided island split: they
    # only ever apply inside the bay the bay-guide chose, so an island with
    # no guide has no day targets either
    if day_targets is not None and state.guide is not None:
        try:
            tday, tbay = day_targets
            if len(tday) == prep.n and len(tbay) == prep.n:
                state.tday = np.asarray(tday, np.int64)
                state.tbay = np.asarray(tbay, np.int64)
                # the relaxation's optimal sacrifice set: blocks it chose to
                # end past due. Role information for the sacrifice-swap move.
                state.tlate = ((state.tday >= 0)
                               & (state.tday + prep.P > prep.D))
        except Exception:
            pass
    variant = (seed // 1000) % 4
    # locked-assignment island: on loose instances island 3 (the weakest
    # hedge arm there) constructs on the certified-optimal bay assignment
    # and runs the LNS with every insertion HARD-locked to it -- the search
    # then optimises schedule/packing only, chasing the assignment bound.
    # Portfolio selection keeps it as pure insurance.
    # the locked-realization arm rides island 3 normally, but on groups
    # whose SISR dose is island3 (overload) it rides island 1 instead --
    # displacing the proven SISR hedge arm at long TLs is the suspected
    # cause of the prob_27/33/37/38 @1200s losses
    _dose = os.environ.get("OGC_V417_SISR", "island3").strip().lower()
    _lock_variant = 1 if _dose == "island3" else 3
    if assign_lock is not None and variant == _lock_variant:
        import numpy as _np
        # probe every capacity arm with a quick forced construction and
        # lock the most REALIZABLE one (area-based eta overestimates what
        # multi-layer crane rules allow; measurement beats guessing)
        best_arm, best_probe = None, float("inf")
        try:
            from _ogc.ogc_assign import forced_construct
            for arm in assign_lock:
                if time.time() > t_end_epoch - 5.0:
                    break
                out = forced_construct(prep, arm[0],
                                       deadline=min(time.time() + 3.0,
                                                    t_end_epoch),
                                       tday=arm[1])
                if out is not None and out[0] < best_probe:
                    best_probe, best_arm = out[0], arm
        except Exception:
            best_arm = None
        if best_arm is None:
            best_arm = assign_lock[-1]
        lb, ltday = best_arm
        state.lock_bay = _np.asarray(lb, _np.int64)
        state.guide = None
        if ltday is not None:
            # certificate schedule of the capacity-constrained assignment:
            # day targets stagger entries the way the area plan requires
            state.tday = _np.asarray(ltday, _np.int64)
            state.tbay = _np.asarray(lb, _np.int64)
            state.tlate = ((state.tday >= 0)
                           & (state.tday + prep.P > prep.D))
    tp_on = (os.environ.get("OGC_TWOPHASE", "0") == "1" and variant == 2
             and t_end_epoch - time.time() > 240)
    day_seed = (os.environ.get("OGC_SEEDCON", "1") == "1"
                and state.tday is not None and variant == 2
                and state.lock_bay is None and not tp_on)
    tc0 = time.time()
    # BRKGA over construction order (from the guide-free fork): construction
    # quality decides overloaded instances, so islands 0/1 search the
    # insertion order itself; islands 2 (relaxation-seeded) and 3 (SISR
    # hedge) keep their distinct construction arms.
    grp = os.environ.get("OGC_GROUP", "")
    bdef = "1" if grp.startswith("overload") else "0"
    use_brkga = (os.environ.get("OGC_BRKGA", bdef) != "0"
                 and variant in (0, 1) and state.lock_bay is None)
    try:
        bfrac = min(0.5, max(0.0, float(
            os.environ.get("OGC_BRKGA_FRAC", "0.12"))))
        if not (bfrac == bfrac):
            bfrac = 0.12
    except (TypeError, ValueError):
        bfrac = 0.12
    t_brkga = time.time() + bfrac * max(0.0, t_end_epoch - time.time())
    constructed = False
    if use_brkga and t_brkga - time.time() > 3.0:
        empty0 = state.snapshot()
        try:
            from _ogc.ogc_brkga import brkga_construct
            brkga_construct(state, rng, t_brkga, log=log)
            constructed = True
        except Exception:
            import traceback as _tb
            if log:
                log("brkga failed: " + _tb.format_exc()[-300:])
            state.restore(empty0)
    if not constructed:
        if tp_on:
            _twophase_construct(state, prep, rng, t_end_epoch, log=log)
        else:
            construct(state, variant=variant, deadline=t_end_epoch,
                      day_seed=day_seed)
    tc = time.time() - tc0
    if log:
        log(f"construct v{variant} obj={state.objective():.0f} "
            f"tard={state.sum_tard:.0f} pref={state.sum_pref:.0f} t={tc:.1f}s")
    # single-process runs (no islands): a second construction variant is
    # cheap insurance when the first one was fast
    if (shared is None and tc < 0.08 * max(1.0, t_end_epoch - tc0)
            and time.time() + 2 * tc < t_end_epoch):
        alt = State(prep, np.random.default_rng(seed + 7))
        alt.guide = state.guide
        alt.guide_w = state.guide_w
        alt.tday = state.tday
        alt.tbay = state.tbay
        alt.tlate = state.tlate
        construct(alt, variant=2 if variant != 2 else 1,
                  deadline=time.time() + 2 * tc + 1.0)
        if alt.objective() < state.objective():
            state = alt
            state.rng = rng
            if log:
                log(f"construct alt wins: obj={state.objective():.0f}")

    if time.time() < t_end_epoch:
        t0_scale = (0.5, 1.0, 0.25, 2.0)[(seed // 1000) % 4]
        if state.lock_bay is not None:
            # locked-assignment island: the assignment is fixed and near-
            # feasible -- fine-grind the schedule instead of hot exploration
            t0_scale = 0.25
        sync_cb = _make_sync_cb(shared, lock) if shared is not None else None
        # SISR string-removal arm rides the guide-free island 3 (the same
        # hedge slot): guided islands keep the tuned v4.17 search bit-for-bit
        # and cross-pollination arbitrates between the basins per instance.
        # OGC_V417_SISR: "hedge"/"island3" (default) | "all" | "0"/"off";
        # unrecognized values keep the default arm rather than silently
        # disabling it
        _sm = os.environ.get("OGC_V417_SISR", "island3").strip().lower()
        if _sm not in ("0", "off", "hedge", "island3", "all"):
            _sm = "hedge"
        sisr_on = (_sm == "all") or (_sm in ("hedge", "island3")
                                      and variant == 3)
        # full-dose mode mirrors the fork's acceptance portfolio: LAHC on
        # the odd-parity islands (fork ogc_solve.py:159), SA on the rest.
        #
        # This USED to be an unconditional os.environ["OGC_LAHC"] = "1000"
        # guarded by "not already set", justified by "islands are separate
        # processes, so the write is per-island". That holds under fork -- but
        # not when the pool is bypassed (OGC_MP_START=single, or any platform
        # without fork), where it runs in the caller's own process. There the
        # write never came back off: the first instance of an in-process batch
        # that took this branch switched LAHC on for every later instance,
        # permanently, and the "not already set" guard is exactly what made it
        # stick. Scope it to this island instead of mutating the environment.
        lahc_len = 0
        if _sm == "all" and variant in (1, 3) and "OGC_LAHC" not in os.environ:
            lahc_len = 1000
        lns = LNS(state, rng, t_end_epoch, log=log, sync_cb=sync_cb,
                  day_resolve=day_resolve, sisr=sisr_on, lahc=lahc_len)
        # `island` selects which arms run the Self-Tuning Lam acceptance
        # controller (OGC_LAM, default island 2 only -- the cold arm). It is
        # the same index t0_scale is drawn from, so the schedule the controller
        # replaces and the island it replaces it on cannot drift apart.
        lns.run(T0=max(1.0, t0_scale * prep.w1), island=variant)
        # polish only with leftover time (LNS normally uses the full budget)
        if time.time() < t_end_epoch - 1.0:
            polish(state, t_end_epoch)
            if log:
                log(f"polish obj={state.objective():.0f}")

    if state_out is not None:
        state_out.append(state)
    return state.objective(), state.build_operations()


def _worker(args):
    (prob_info, t_end_epoch, seed, shared, lock, targets, day_targets,
     assign_lock) = args
    try:
        # numba threads: keep each worker single-threaded
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        # The worker islands were silent: driver.py builds its logger and hands
        # it only to the in-process island, so three of the four arms could not
        # be observed at all -- which is why no annealing experiment in this
        # repository has ever been able to attribute a result to the island it
        # changed. fork shares fd 1, so this just tags and prints. Gated on
        # OGC_LOG exactly like the main logger, so production is unaffected.
        wlog = None
        if os.environ.get("OGC_LOG"):
            _iid = (seed // 1000) % 4
            wlog = lambda msg: print(f"[ogc w{_iid}] {msg}", flush=True)
        obj, sol = solve_single(prob_info, t_end_epoch, seed, log=wlog,
                                shared=shared, lock=lock, targets=targets,
                                day_targets=day_targets,
                                assign_lock=assign_lock)
        return obj, sol, None
    except Exception:
        return None, None, traceback.format_exc()


def solve_parallel(prob_info, t_end_epoch, n_workers, log=None,
                   hard_deadline=None, targets=None, day_targets=None,
                   day_resolve=False, assign_pack=None):
    """Run n_workers island solvers in separate processes plus one in this
    process; return the best (obj, solution). Never blocks past
    hard_deadline (wall-clock epoch)."""
    import multiprocessing as mp

    # `spawn` re-imports the caller's __main__; competition/test harnesses are
    # not required to carry an `if __name__ == "__main__"` guard, so using a
    # spawn Pool can recurse or crash. On platforms without fork, retain full
    # correctness with the main island instead of depending on caller layout.
    methods = mp.get_all_start_methods()
    force_single = os.environ.get("OGC_MP_START", "").lower() == "single"
    if force_single or "fork" not in methods:
        portable_lock = (list(assign_pack[1])
                         if assign_pack is not None and assign_pack[1]
                         else None)
        return solve_single(prob_info, t_end_epoch, seed=20260711, log=log,
                            targets=targets, day_targets=day_targets,
                            day_resolve=day_resolve,
                            assign_lock=portable_lock)

    if hard_deadline is None:
        hard_deadline = t_end_epoch + 8.0
    results = []
    state_holder = []
    pool = None
    manager = None
    async_res = None
    try:
        ctx = mp.get_context("fork")
        if sys.platform.startswith("linux"):
            # compile everything BEFORE forking so children inherit the JIT
            # state and skip recompilation
            from _ogc.ogc_kernels import warmup
            warmup()
        manager = ctx.Manager()
        shared = manager.dict()
        lock = manager.Lock()
        pool = ctx.Pool(processes=n_workers)
        seeds = [20260711 + 1000 * (w + 1) for w in range(n_workers)]
        a_lock = None
        if assign_pack is not None and assign_pack[1]:
            # pass ALL capacity arms; the locked island probes them by
            # forced construction and locks the most realizable one
            a_lock = list(assign_pack[1])
        # leave margin for result collection in the workers' own deadline
        args = [(prob_info, t_end_epoch, s, shared, lock, targets,
                 day_targets, a_lock)
                for s in seeds]
        async_res = pool.map_async(_worker, args)

        # main process solves too; the one-shot day re-solve runs here only
        # (ortools already imported in this process, identical on fork/spawn)
        obj0, sol0 = solve_single(prob_info, t_end_epoch,
                                  seed=20260711, log=log,
                                  shared=shared, lock=lock, targets=targets,
                                  day_targets=day_targets,
                                  day_resolve=day_resolve,
                                  state_out=state_holder)
        results.append((obj0, sol0))

        remaining = max(0.0, min(t_end_epoch + 8.0, hard_deadline)
                        - time.time())
        try:
            worker_out = async_res.get(timeout=remaining)
            for obj, sol, err in worker_out:
                if obj is not None:
                    results.append((obj, sol))
                elif err and log:
                    log(f"worker error: {err[-300:]}")
        except mp.TimeoutError:
            if log:
                log("worker timeout; using main result")
        pool.terminate()
        pool.join()
    except Exception:
        if log:
            log(f"parallel fallback: {traceback.format_exc()[-300:]}")
        # A main-island-only failure (for example in an optional day re-solve)
        # must not discard workers that already completed successfully.
        if async_res is not None:
            try:
                remaining = max(0.0, hard_deadline - time.time())
                for obj, sol, err in async_res.get(timeout=remaining):
                    if obj is not None:
                        results.append((obj, sol))
                    elif err and log:
                        log(f"worker error: {err[-300:]}")
            except Exception:
                pass
        if not results:
            # Do not start another unbounded construction after the search
            # deadline; let the driver return its cheap guaranteed fallback.
            if time.time() >= t_end_epoch:
                raise
            obj0, sol0 = solve_single(prob_info, t_end_epoch,
                                      seed=20260711, log=log,
                                      targets=targets,
                                      day_targets=day_targets)
            results.append((obj0, sol0))
    finally:
        if pool is not None:
            try:
                pool.terminate()
            except Exception:
                pass
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

    results.sort(key=lambda r: r[0])
    if log:
        log(f"islands: {[f'{r[0]:.0f}' for r in results]}")

    # Assignment endgame (loose instances): forced construction on the exact
    # optimal bay assignment. If it reaches Z1=0 its objective equals the
    # certified lower bound. Priced exactly; adopted iff strictly better.
    try:
        if (assign_pack is not None and results and state_holder
                and hard_deadline - time.time() > 1.0):
            a_bound, arm_targets = assign_pack
            if results[0][0] > a_bound + 1e-6:
                from _ogc.ogc_assign import forced_construct
                for (a_targets, a_tday) in arm_targets:
                    if hard_deadline - time.time() < 1.0:
                        break
                    out = forced_construct(state_holder[0].prep, a_targets,
                                           deadline=hard_deadline - 0.7,
                                           tday=a_tday)
                    if out is not None and out[0] < results[0][0] - 1e-9:
                        if log:
                            log(f"assign endgame: {results[0][0]:.0f} -> "
                                f"{out[0]:.0f} (bound {a_bound:,.0f})")
                        results.insert(0, out)
    except Exception:
        if log:
            log(f"assign endgame failed: {traceback.format_exc()[-200:]}")

    # CP-SAT retime polish on the winning schedule (main process only --
    # ortools inside a forked worker can deadlock). Exact-gated: applied via
    # per-insert exact verification, reverted wholesale on disagreement.
    try:
        if (os.environ.get("OGC_CPRETIME", "1") != "0" and results
                and state_holder
                and hard_deadline - time.time() > 2.0):
            from _ogc.ogc_retime import cp_retime_ops
            rt_deadline = min(hard_deadline - 1.0, time.time() + 3.0)
            out = cp_retime_ops(state_holder[0].prep, results[0][1],
                                rt_deadline, log=log)
            if out is not None and out[0] < results[0][0] - 1e-9:
                if log:
                    log(f"cp_retime: {results[0][0]:.0f} -> {out[0]:.0f}")
                results.insert(0, out)
    except Exception:
        if log:
            log(f"cp_retime failed: {traceback.format_exc()[-200:]}")
    return results[0]
