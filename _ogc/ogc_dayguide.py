# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Day-level tardiness guidance: given the bay assignment from ogc_guide,
solve one cumulative-capacity scheduling relaxation per bay (blocks as
interval tasks whose demand is their layer-0 footprint area, capacity a
fraction eta of the bay area) minimizing tardiness, and return per-block
target entry days.

Stage-0 evidence (hard-6): the relaxation beats its own EDD list schedule
by 31-45% sum-tardiness on every instance -- it finds non-EDD "sacrifice"
orderings the LNS's EDD-shaped priors rarely roll. The LNS uses the targets
to bias WHEN a block is inserted (scan-start lift in best_insertion, day-
aware reinsertion order, an off-day removal op); the true objective still
decides acceptance, exactly like the bay guide."""

import math

from _ogc import relaxarea
import os


def _edd_schedule(tasks, cap, horizon):
    """Greedy EDD list schedule under cumulative capacity.
    tasks: list of (bid, R, D, P, a). Returns {bid: start} or None."""
    usage = [0] * horizon
    starts = {}
    for (bid, R, D, P, a) in sorted(tasks, key=lambda t: (t[2], t[1], t[0])):
        t = R
        placed = False
        while t + P <= horizon:
            ok = True
            for u in range(t, t + P):
                if usage[u] + a > cap:
                    ok = False
                    t = u + 1
                    break
            if ok:
                for u in range(t, t + P):
                    usage[u] += a
                starts[bid] = t
                placed = True
                break
        if not placed:
            return None
    return starts


def _solve_bay(cp_model, tasks, cap, horizon, budget_s, workers,
               hints=None, stab=0):
    """Min-tardiness cumulative relaxation for one bay. The hint (EDD list
    schedule by default, the caller's incumbent schedule on re-solves) seeds
    the search and doubles as the timeout fallback. When `stab` > 0 a
    |start - hint| stability term keeps re-solved targets close to the
    incumbent. Returns {bid: start} or None when no hint schedule exists."""
    if hints is None:
        hints = _edd_schedule(tasks, cap, horizon)
        if hints is None:
            return None
    BIG = 10000
    md = cp_model.CpModel()
    svars = {}
    ivs, dems, terms = [], [], []
    for (bid, R, D, P, a) in tasks:
        s = md.NewIntVar(R, horizon - P, f"s{bid}")
        e = md.NewIntVar(R + P, horizon, f"e{bid}")
        md.Add(e == s + P)
        iv = md.NewIntervalVar(s, P, e, f"iv{bid}")
        T = md.NewIntVar(0, horizon, f"T{bid}")
        md.Add(T >= e - D)
        svars[bid] = s
        ivs.append(iv)
        dems.append(a)
        # left-justification term (s - R): without it CP-SAT slack emits
        # gratuitously late targets that the scan-start lift would enforce
        terms.append(BIG * T + (s - R))
        h = min(max(int(hints[bid]), R), horizon - P)
        md.AddHint(s, h)
        if stab:
            dv = md.NewIntVar(0, horizon, f"dv{bid}")
            md.Add(dv >= s - h)
            md.Add(dv >= h - s)
            terms.append(stab * dv)
    md.AddCumulative(ivs, dems, cap)
    md.Minimize(sum(terms))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(budget_s)
    sv.parameters.num_search_workers = workers
    status = sv.Solve(md)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # the hint schedule is still a usable target set
        return {bid: min(max(int(hints[bid]), R), horizon - P)
                for (bid, R, D, P, a) in tasks}
    return {bid: sv.Value(v) for bid, v in svars.items()}


def recompute_day_targets(n, bay_tasks, caps, budget_s=2.5, stab=2):
    """One-shot re-solve from the LIVE search state (long time limits,
    main island only): membership comes from the current bays rather than
    the guide assignment, capacity from realized occupancy, hints from the
    incumbent entry days with a stability term.

    bay_tasks: {b: [(i, R, D, P, demand, hint_t1), ...]}; caps: {b: cap}.
    Returns (tday, tbay) plain int lists or None."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    try:
        if not bay_tasks:
            return None
        per_bay = max(0.2, float(budget_s) / len(bay_tasks))
        tday = [-1] * n
        tbay = [-1] * n
        for b, rows in bay_tasks.items():
            tasks = [(i, R, D, P, a) for (i, R, D, P, a, _h) in rows]
            hints = {i: h for (i, _R, _D, _P, _a, h) in rows}
            horizon = max(max(h + P for (_i, _R, _D, P, _a, h) in rows),
                          max(D for (_i, _R, D, _P, _a, _h) in rows),
                          max(R + P for (_i, R, _D, P, _a, _h) in rows)) + 5
            try:
                starts = _solve_bay(cp_model, tasks, caps[b], horizon,
                                    per_bay, workers=1,
                                    hints=hints, stab=stab)
            except Exception:
                raise
            if starts is None:
                continue
            for (i, _R, _D, _P, _a, _h) in rows:
                if i in starts:
                    tday[i] = int(starts[i])
                    tbay[i] = int(b)
        if all(t < 0 for t in tday):
            return None
        return tday, tbay
    except Exception:
        raise


def compute_day_targets(prob_info, targets, time_budget=4.0, workers=4):
    """Per-block target entry days for the bay assignment in `targets`.

    Returns (tday, tbay): plain int lists (pickle-safe for the island
    workers), -1 where no target exists -- or None on any failure. Bays are
    independent, so a failed/timeouted bay degrades to -1 for its blocks
    rather than failing the whole computation."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    try:
        bays = prob_info["bays"]
        blocks = prob_info["blocks"]
        n = len(blocks)
        if targets is None or len(targets) != n:
            return None
        # 0.65 measured best (hard-6 screen: 0.75 emits geometrically
        # unreachable target days, 0.5 degenerates to plain EDD)
        try:
            eta = float(os.environ.get("OGC_DAY_ETA", "0.65"))
            if not (0.4 <= eta <= 0.95):  # NaN also fails into the default
                eta = 0.65
        except (TypeError, ValueError):
            eta = 0.65

        def _area(verts):
            s = 0.0
            k = len(verts)
            for i in range(k):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % k]
                s += x1 * y2 - x2 * y1
            return abs(s) * 0.5

        def demand(block, bay):
            # min layer-0 polygon area over bbox-fitting orientations
            W, H = bay["width"], bay["height"]
            best = None
            for s in block["shape"]:
                xs = [v[0] for lay in s["layers"] for v in lay]
                ys = [v[1] for lay in s["layers"] for v in lay]
                if (math.ceil(-min(xs)) <= math.floor(W - max(xs))
                        and math.ceil(-min(ys)) <= math.floor(H - max(ys))):
                    a = relaxarea.layer_area(s["layers"])
                    if best is None or a < best:
                        best = a
            if best is None:
                best = relaxarea.block_area(block)
            return max(1, int(round(best)))

        by_bay = {}
        for i in range(n):
            b = targets[i]
            if b is not None and 0 <= int(b) < len(bays):
                by_bay.setdefault(int(b), []).append(i)
        if not by_bay:
            return None
        per_bay = max(0.25, float(time_budget) / len(by_bay))

        tday = [-1] * n
        tbay = [-1] * n
        for b, ids in by_bay.items():
            bay = bays[b]
            cap = max(1, int(eta * bay["width"] * bay["height"]))
            tasks = []
            for i in ids:
                bl = blocks[i]
                tasks.append((i, int(bl["release_time"]),
                              int(bl["due_date"]),
                              max(1, int(bl["processing_time"])),
                              min(demand(bl, bay), cap)))
            horizon = (max(t[1] for t in tasks)
                       + sum(t[3] for t in tasks) + 5)
            horizon = max(horizon, max(t[2] for t in tasks) + 5)
            try:
                starts = _solve_bay(cp_model, tasks, cap, horizon,
                                    per_bay, workers)
            except Exception:
                raise
            if starts is None:
                continue
            for i in ids:
                if i in starts:
                    tday[i] = int(starts[i])
                    tbay[i] = b
        if all(t < 0 for t in tday):
            return None
        return tday, tbay
    except Exception:
        raise
