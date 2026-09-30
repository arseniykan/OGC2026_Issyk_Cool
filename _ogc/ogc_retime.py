# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

# =============================================================================
#  CP-SAT bay retiming at frozen geometry (graft of the _ogc `cpsched` idea
#  onto the day_scan engine).
#
#  With positions and orientations frozen, feasibility couples blocks of one
#  bay only through pairwise interaction. We use a CONSERVATIVE pair test:
#  if ANY layer of one block overlaps ANY layer of the other at the fixed
#  offsets (tri-state exact test; ambiguous/no-triangulation counts as
#  overlap), the pair may not be co-present (NoOverlap of their intervals).
#  Non-interacting pairs are provably free at any relative timing:
#  no same-level pair collides and no crane lift path is obstructed in
#  either direction. Same-day entry/exit boundaries are handled by the
#  checker's EXIT-before-ENTRY replay, which build_operations emits.
#
#  A CP-SAT model then minimises the bay's total tardiness over entry days.
#  The result is applied via exact per-insert verification
#  (State.can_place_exact) and reverted wholesale on any disagreement, so
#  this can never make the solution infeasible or worse.
# =============================================================================

import time

import numpy as np


def _pair_interacts(p, oidx_i, xi, yi, oidx_k, xk, yk):
    """Conservative: True unless every layer pair is certainly disjoint."""
    from _ogc.ogc_kernels import _layers_overlap_exact
    bi = int(p.layer_base[oidx_i])
    ni = int(p.layer_cnt[oidx_i])
    bk = int(p.layer_base[oidx_k])
    nk = int(p.layer_cnt[oidx_k])
    for li in range(ni):
        ri = bi + li
        if not p.tri_ok[ri]:
            return True
        for lk in range(nk):
            rk = bk + lk
            if not p.tri_ok[rk]:
                return True
            r = _layers_overlap_exact(
                ri, float(xi), float(yi), rk, float(xk), float(yk),
                p.tri_ptr, p.tri_cnt, p.tri_x, p.tri_y,
                p.lbx0, p.lby0, p.lbx1, p.lby1, 1e-9)
            if r != 0:
                return True
    return False


def _rebuild_state(prep, sol_ops):
    """Replay an operations dict into a fresh State (day-ascending ENTRY
    commit order, exits attached), mirroring driver._repair."""
    from _ogc.ogc_state import State
    state = State(prep, np.random.default_rng(1))
    ops = sol_ops.get("operations", {})
    info = []
    exits = {}
    for t_str in sorted(ops, key=lambda s: int(s)):
        for op in ops[t_str]:
            if op["type"] == "ENTRY":
                info.append((op["block_id"], op["bay_id"], int(t_str),
                             op["x"], op["y"], op["orient_idx"]))
            else:
                exits[op["block_id"]] = int(t_str)
    for (i, b, t, x, y, o) in info:
        state.insert(i, b, t, o, x, y, t2=exits.get(i))
    return state


def cp_retime_ops(prep, sol_ops, deadline, log=None,
                  per_bay_budget=0.25, max_members=220):
    """Retime tardy bays of an operations dict. Returns (obj, ops) when the
    schedule strictly improved, else None. Never raises (caller wraps)."""
    from ortools.sat.python import cp_model

    if time.time() > deadline - 0.3:
        return None
    state = _rebuild_state(prep, sol_ops)
    p = prep
    TCAP = state.occ0.shape[1] if hasattr(state, "occ0") else 8192

    # per-bay tardiness, worst first
    bay_tard = np.zeros(p.m)
    for b in range(p.m):
        for i in state.bay_blocks[b]:
            bay_tard[b] += max(0, int(state.t2[i]) - int(p.D[i]))
    order = [b for b in np.argsort(-bay_tard) if bay_tard[b] > 0]

    improved_any = False
    for b in order:
        if time.time() > deadline - 0.3:
            break
        members = list(state.bay_blocks[b])
        if len(members) < 2 or len(members) > max_members:
            continue
        cur_t1 = {i: int(state.t1[i]) for i in members}
        cur_t2 = {i: int(state.t2[i]) for i in members}
        makespan = max(cur_t2.values())

        # conservative interaction pairs at frozen offsets
        pairs = []
        oidx = {i: int(p.orient_base[i] + state.oi[i]) for i in members}
        overlong = False
        t_pairs0 = time.time()
        for a in range(len(members)):
            if time.time() - t_pairs0 > 0.6:
                overlong = True
                break
            ia = members[a]
            for c in range(a + 1, len(members)):
                ic = members[c]
                if _pair_interacts(p, oidx[ia], state.px[ia], state.py[ia],
                                   oidx[ic], state.px[ic], state.py[ic]):
                    pairs.append((ia, ic))
        if overlong:
            continue

        model = cp_model.CpModel()
        maxP = max(int(p.P[i]) for i in members)
        ub = min(int(TCAP) - 2, makespan + maxP + 2)
        ev, iv, tv = {}, {}, {}
        for i in members:
            P_i = int(p.P[i])
            hi = max(int(cur_t1[i]), ub - P_i)
            e = model.NewIntVar(int(p.R[i]), hi, f"e{i}")
            x = model.NewIntVar(int(p.R[i]) + P_i, hi + P_i, f"x{i}")
            iv[i] = model.NewIntervalVar(e, P_i, x, f"iv{i}")
            t = model.NewIntVar(0, int(TCAP), f"t{i}")
            model.Add(t >= e + P_i - int(p.D[i]))
            ev[i], tv[i] = e, t
            model.AddHint(e, cur_t1[i])
        for (ia, ic) in pairs:
            model.AddNoOverlap([iv[ia], iv[ic]])
        model.Minimize(sum(tv.values()))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(
            0.05, min(per_bay_budget, deadline - time.time() - 0.2))
        solver.parameters.num_search_workers = 1
        st = solver.Solve(model)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        new_tard = sum(int(solver.Value(tv[i])) for i in members)
        if new_tard >= bay_tard[b] - 1e-9:
            continue

        # apply with exact verification; revert wholesale on any failure
        snap = state.snapshot()
        ok = True
        for i in members:
            state.remove(i)
        for i in sorted(members, key=lambda i: int(solver.Value(ev[i]))):
            e = int(solver.Value(ev[i]))
            x2 = e + int(p.P[i])
            if not state.can_place_exact(i, b, int(state.px[i]),
                                         int(state.py[i]),
                                         int(state.oi[i]), e, x2):
                ok = False
                break
            state.insert(i, b, e, int(state.oi[i]),
                         int(state.px[i]), int(state.py[i]), t2=x2)
        if not ok:
            state.restore(snap)
            continue
        improved_any = True
        bay_tard[b] = new_tard
        if log:
            log(f"cp_retime bay {b}: tard -> {new_tard:.0f}")

    if not improved_any:
        return None
    return state.objective(), state.build_operations()
