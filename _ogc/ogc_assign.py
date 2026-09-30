# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

# =============================================================================
#  Exact bay-assignment endgame for spatially loose instances.
#
#  On the loose half of the training set (peak space-time utilization < 0.45)
#  tardiness Z1 = 0 is easy, so the whole objective is w2*Z2 + w3*Z3 -- a pure
#  assignment problem. The geometry-free optimum of that assignment (blocks ->
#  bays they geometrically fit) is a LOWER BOUND on any schedule with Z1 = 0,
#  and it solves to proven optimality in well under a second with CP-SAT.
#
#  Measured gaps of the tuned LNS vs this bound at 60 s (single rolls):
#  prob_19 +36%, prob_20 +30%, prob_13 +12%, prob_17 +3% -- the searchers
#  under-adhere to the optimal assignment even when guides point at it.
#
#  Exploit in two places:
#   1. use the exact-optimal assignment as guide targets (strictly better
#      than the capacity-priced approximation when capacity is slack);
#   2. after the main solve, build a FORCED construction (every block into
#      its assigned bay, earliest feasible day). If it reaches Z1 = 0 its
#      objective EQUALS the bound = certified optimal. Adopt iff better than
#      the incumbent (priced exactly, so this can never regress).
# =============================================================================

import math
import time


def exact_assignment(prob_info, time_budget=3.0, workers=2,
                     eta=None, nbuck=8):
    """Solve min w2*(max u_j L_j - min u_j L_j) + w3*sum pref_miss exactly.

    With eta=None this is the pure assignment optimum -- a LOWER BOUND on any
    Z1=0 schedule, but often unschedulable (it piles blocks into one bay:
    measured per-bay peaks up to 1.30 under the unconstrained optimum).
    With eta set, hard time-bucketed capacity is added: each block's layer-0
    area*days demand is spread over its on-time span [R, max(D, R+P)) and
    each (bay, bucket) must fit within eta * area * bucket_len. That yields
    assignments the greedy forced construction can realise with Z1=0.

    Returns (targets, bound, proven, sdays) or None. `bound` is in true
    objective units (w2*Z2 + w3*Z3); `proven` is True iff CP-SAT proved
    optimality; `sdays` is the certificate schedule's entry day per block
    (None when eta is None)."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    try:
        bays = prob_info["bays"]
        blocks = prob_info["blocks"]
        m = len(bays)
        n = len(blocks)
        if m < 2 or n == 0:
            return None
        w = prob_info.get("weights", {}) or {}
        w2 = float(w.get("w2", 1.0))
        w3 = float(w.get("w3", 1.0))
        areas = [b["width"] * b["height"] for b in bays]
        avg = sum(areas) / m
        SC = 10000
        u = [max(1, round(avg / a * SC)) for a in areas]
        wl = [int(round(bl["workload"])) for bl in blocks]

        def fits(i, b):
            W = bays[b]["width"]
            H = bays[b]["height"]
            for s in blocks[i]["shape"]:
                xs = [v[0] for lay in s["layers"] for v in lay]
                ys = [v[1] for lay in s["layers"] for v in lay]
                if (math.ceil(-min(xs)) <= math.floor(W - max(xs))
                        and math.ceil(-min(ys)) <= math.floor(H - max(ys))):
                    return True
            return False

        md = cp_model.CpModel()
        x = {}
        for i in range(n):
            row = []
            for b in range(m):
                if fits(i, b):
                    x[i, b] = md.NewBoolVar(f"x{i}_{b}")
                    row.append(x[i, b])
            if not row:
                return None
            md.AddExactlyOne(row)
        cap = sum(wl) * max(u) + 1
        loads = []
        for b in range(m):
            L = md.NewIntVar(0, cap, f"L{b}")
            md.Add(L == sum(wl[i] * u[b] * x[i, b]
                            for i in range(n) if (i, b) in x))
            loads.append(L)
        mx = md.NewIntVar(0, cap, "mx")
        mn = md.NewIntVar(0, cap, "mn")
        md.AddMaxEquality(mx, loads)
        md.AddMinEquality(mn, loads)
        # weights scaled x100 so fractional w2/w3 survive integerisation
        W2 = int(round(w2 * 100))
        W3 = int(round(w3 * 100))
        pen = sum(int(max(blocks[i]["bay_preferences"])
                      - blocks[i]["bay_preferences"][b]) * x[i, b]
                  for (i, b) in x)
        md.Minimize(W2 * (mx - mn) + W3 * SC * pen)

        if eta is not None:
            # Joint assignment + cumulative on-time scheduling: one start var
            # per block with start <= D - P (Z1 = 0 enforced), an OPTIONAL
            # interval per (block, candidate bay), and per-bay AddCumulative
            # with layer-0 area demand against eta*area capacity. A solution
            # is an assignment together with an area-feasibility CERTIFICATE
            # that everything can run on time at packing efficiency eta --
            # the "solve the time-sliced area puzzle first, let geometry
            # realise it" decomposition. (Bucketed capacity variants
            # measurably failed to bind on release peaks; this binds by
            # construction.)
            # layer-0 area systematically understates the binding layer;
            # see _ogc/relaxarea.py (OGC_RELAX_AREA selects the proxy)
            from _ogc import relaxarea

            def _area0(bl):
                return relaxarea.block_area(bl)
            H2 = 1
            starts = []
            for i, bl in enumerate(blocks):
                R = int(bl["release_time"])
                P = max(1, int(bl["processing_time"]))
                lo = R
                hi = max(R, int(bl["due_date"]) - P)  # on time or forced-late floor
                s_i = md.NewIntVar(lo, hi, f"s{i}")
                starts.append(s_i)
                H2 = max(H2, hi + P)
            for b in range(m):
                ivs, dems = [], []
                for i, bl in enumerate(blocks):
                    if (i, b) not in x:
                        continue
                    P = max(1, int(bl["processing_time"]))
                    e_i = md.NewIntVar(0, H2, f"e{i}_{b}")
                    iv = md.NewOptionalIntervalVar(starts[i], P, e_i,
                                                   x[i, b], f"iv{i}_{b}")
                    dem = max(1, int(round(_area0(bl))))
                    ivs.append(iv)
                    dems.append(dem)
                if ivs:
                    md.AddCumulative(ivs, dems, int(eta * areas[b]))

        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = float(time_budget)
        sv.parameters.num_search_workers = int(workers)
        st = sv.Solve(md)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        targets = [0] * n
        for i in range(n):
            for b in range(m):
                if (i, b) in x and sv.Value(x[i, b]):
                    targets[i] = b
                    break
        sdays = None
        if eta is not None:
            sdays = [int(sv.Value(s)) for s in starts]
        # ObjectiveValue = W2*(z2*SC) + W3*SC*z3 = 100*SC*(w2*z2 + w3*z3)
        bound = sv.ObjectiveValue() / (100.0 * SC)
        return targets, float(bound), st == cp_model.OPTIMAL, sdays
    except Exception:
        raise


def forced_construct(prep, targets, deadline, tday=None):
    """Construction with every block forced into its assigned bay. When the
    certificate schedule `tday` is given, insertion follows its start order
    and each scan is lifted to start at the planned day (the plan staggers
    entries deliberately -- greedy-earliest measurably breaks it).
    Returns (obj, ops) or None."""
    import numpy as np
    from _ogc.ogc_state import State

    if time.time() > deadline - 0.2:
        return None
    p = prep
    state = State(p, np.random.default_rng(3))
    use_lift = tday is not None
    if use_lift:
        state.tday = np.asarray(tday, np.int64)
        state.tbay = np.asarray(targets, np.int64)
        order = sorted(range(p.n),
                       key=lambda i: (int(tday[i]), p.D[i], -p.wl[i]))
    else:
        order = sorted(range(p.n), key=lambda i: (p.D[i], p.R[i], -p.wl[i]))
    for i in order:
        b = int(targets[i])
        if not p.block_fits[i, b]:
            return None  # assignment fit model disagrees with prep; abort
        cand = None
        if time.time() < deadline:
            cand = state.best_insertion(i, only_bay=b,
                                        use_dayguide=use_lift)
        if cand is None:
            t = state.day_upper_bound(i, b)
            r = state.try_day(i, b, t)
            if r is None:
                return None
            o, px, py = r
            state.insert(i, b, t, o, px, py)
        else:
            c, bb, t, o, px, py = cand
            state.insert(i, bb, t, o, px, py)
    return state.objective(), state.build_operations()


def cumulative_plan(prob_info, eta, time_budget=15.0, workers=2):
    """Full-objective plan for DENSE instances: joint bay assignment +
    cumulative area scheduling with tardiness ALLOWED, minimising
    w1*sum(tard) + w2*(max-min weighted load) + w3*pref at packing
    efficiency eta.

    Measured motivation (data_test, official TLs): the engine's realised
    solutions behave like eta ~= 0.60, and even at matched eta the
    relaxation's schedule beat the incumbent 2x on prob_4@480 (257k at
    eta 0.70 vs 2.1M incumbent on prob_5-class runs). The plan's
    (targets, sdays) feed the locked-realization island; portfolio
    selection guarantees no regression.

    Returns (targets, plan_obj, proven, sdays) or None."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        raise
    try:
        import math as _m
        bays = prob_info["bays"]
        blocks = prob_info["blocks"]
        m = len(bays)
        n = len(blocks)
        if m < 2 or n == 0:
            return None
        w = prob_info.get("weights", {}) or {}
        w1 = float(w.get("w1", 1.0))
        w2 = float(w.get("w2", 1.0))
        w3 = float(w.get("w3", 1.0))
        areas = [b["width"] * b["height"] for b in bays]
        avg = sum(areas) / m
        SC = 10000
        u = [max(1, round(avg / a * SC)) for a in areas]
        wl = [int(round(bl["workload"])) for bl in blocks]

        def _fits(i, b):
            W = bays[b]["width"]
            H = bays[b]["height"]
            for s in blocks[i]["shape"]:
                xs = [v[0] for lay in s["layers"] for v in lay]
                ys = [v[1] for lay in s["layers"] for v in lay]
                if (_m.ceil(-min(xs)) <= _m.floor(W - max(xs))
                        and _m.ceil(-min(ys)) <= _m.floor(H - max(ys))):
                    return True
            return False

        # same proxy as the assignment model above (_ogc/relaxarea.py)
        from _ogc import relaxarea

        def _area0(bl):
            return relaxarea.block_area(bl)

        md = cp_model.CpModel()
        x = {}
        for i in range(n):
            row = []
            for b in range(m):
                if _fits(i, b):
                    x[i, b] = md.NewBoolVar(f"x{i}_{b}")
                    row.append(x[i, b])
            if not row:
                return None
            md.AddExactlyOne(row)
        H2 = max(int(b["release_time"]) + max(1, int(b["processing_time"]))
                 for b in blocks) + 200
        starts, tards = [], []
        for i, bl in enumerate(blocks):
            R = int(bl["release_time"])
            P = max(1, int(bl["processing_time"]))
            s = md.NewIntVar(R, H2, f"s{i}")
            t = md.NewIntVar(0, H2, f"t{i}")
            md.Add(t >= s + P - int(bl["due_date"]))
            starts.append(s)
            tards.append(t)
        for b in range(m):
            ivs, dems = [], []
            for i, bl in enumerate(blocks):
                if (i, b) not in x:
                    continue
                P = max(1, int(bl["processing_time"]))
                e = md.NewIntVar(0, H2 + P, f"e{i}_{b}")
                iv = md.NewOptionalIntervalVar(starts[i], P, e, x[i, b],
                                               f"iv{i}_{b}")
                ivs.append(iv)
                dems.append(max(1, int(round(_area0(bl)))))
            if ivs:
                md.AddCumulative(ivs, dems, int(eta * areas[b]))
        cap = sum(wl) * max(u) + 1
        loads = []
        for b in range(m):
            L = md.NewIntVar(0, cap, f"L{b}")
            md.Add(L == sum(wl[i] * u[b] * x[i, b]
                            for i in range(n) if (i, b) in x))
            loads.append(L)
        mx = md.NewIntVar(0, cap, "mx")
        mn = md.NewIntVar(0, cap, "mn")
        md.AddMaxEquality(mx, loads)
        md.AddMinEquality(mn, loads)
        pen = sum(int(max(blocks[i]["bay_preferences"])
                      - blocks[i]["bay_preferences"][b]) * x[i, b]
                  for (i, b) in x)
        W1 = int(round(w1 * 100))
        W2 = int(round(w2 * 100))
        W3 = int(round(w3 * 100))
        md.Minimize(W1 * SC * sum(tards) + W2 * (mx - mn) + W3 * SC * pen)

        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = float(time_budget)
        sv.parameters.num_search_workers = int(workers)
        st = sv.Solve(md)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        targets = [0] * n
        for i in range(n):
            for b in range(m):
                if (i, b) in x and sv.Value(x[i, b]):
                    targets[i] = b
                    break
        sdays = [int(sv.Value(s)) for s in starts]
        plan_obj = sv.ObjectiveValue() / (100.0 * SC)
        return targets, float(plan_obj), st == cp_model.OPTIMAL, sdays
    except Exception:
        raise
