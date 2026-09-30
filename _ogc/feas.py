# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""
feas.py — own exact feasibility checker (mirrors the grader; guide 4.4).

Checks: assignment completeness, release, processing, containment, same-level
collision over co-present blocks, and crane orderability at every ENTRY/EXIT.
This is the LOCAL ground truth (no shapely available in dev).  When shapely /
utils are present at runtime the orchestrator cross-checks against them.
"""
from . import geo, crane
from .model import Solution


def check_solution(prob, sol, require_assigned=True):
    """Return (ok: bool, reason: str). First failure wins."""
    n = prob.n
    # ---- assignment / scheduling ---------------------------------------
    for i in range(n):
        if sol.bay[i] < 0:
            if require_assigned:
                return False, f"block {i} unassigned"
            continue
        if not (0 <= sol.bay[i] < prob.m):
            return False, f"block {i} bad bay {sol.bay[i]}"
        if sol.entry[i] < prob.R[i]:
            return False, f"block {i} entry {sol.entry[i]} < release {prob.R[i]}"
        if sol.exit[i] - sol.entry[i] < prob.P[i]:
            return False, f"block {i} duration < processing {prob.P[i]}"
        if sol.exit[i] <= sol.entry[i]:
            return False, f"block {i} exit<=entry"
        # containment
        og = sol.og(i)
        if not geo.contained_in_bay(og, int(sol.x[i]), int(sol.y[i]),
                                    int(prob.W[sol.bay[i]]), int(prob.H[sol.bay[i]])):
            return False, f"block {i} not contained in bay {sol.bay[i]}"

    # ---- per-bay geometry ----------------------------------------------
    for j in range(prob.m):
        members = [i for i in range(n) if sol.bay[i] == j]
        if not members:
            continue
        # same-level collision on overlapping intervals
        for a_idx in range(len(members)):
            i1 = members[a_idx]
            e1, x1 = int(sol.entry[i1]), int(sol.exit[i1])
            for b_idx in range(a_idx + 1, len(members)):
                i2 = members[b_idx]
                e2, x2 = int(sol.entry[i2]), int(sol.exit[i2])
                # intervals [e,x) overlap ?
                if e1 < x2 and e2 < x1:
                    if geo.same_level_collision(sol.og(i1), int(sol.x[i1]), int(sol.y[i1]),
                                                sol.og(i2), int(sol.x[i2]), int(sol.y[i2])):
                        return False, f"same-level collision blocks {i1},{i2} in bay {j}"

        # crane orderability on every event day
        ev = set()
        for i in members:
            ev.add(int(sol.entry[i]))
            ev.add(int(sol.exit[i]))
        for t in sorted(ev):
            prev = [i for i in members if sol.entry[i] <= t - 1 < sol.exit[i]]
            exiting = [i for i in members if sol.exit[i] == t]
            entering = [i for i in members if sol.entry[i] == t]
            if exiting:
                if crane.order_exits(prob, sol, prev, exiting) is None:
                    return False, f"crane: no feasible EXIT order day {t} bay {j} ({exiting})"
            base = [i for i in prev if i not in set(exiting)]
            if entering:
                if crane.order_entries(prob, sol, base, entering) is None:
                    return False, f"crane: no feasible ENTRY order day {t} bay {j} ({entering})"
    return True, "ok"


def build_ordered_output(prob, sol):
    """
    Emit operations dict with EXITs before ENTRYs each day, each group in a
    crane-feasible order (peeling/insertion).  Falls back to arbitrary order if
    ordering fails (should not happen for a checked-feasible solution).
    """
    n = prob.n
    # gather per-day per-bay events
    days = set()
    for i in range(n):
        if sol.bay[i] < 0:
            continue
        days.add(int(sol.entry[i]))
        days.add(int(sol.exit[i]))
    out = {}
    for t in sorted(days):
        day_ops = []
        # EXITs first, ordered per bay
        exit_ids = []
        for j in range(prob.m):
            members = [i for i in range(n) if sol.bay[i] == j]
            prev = [i for i in members if sol.entry[i] <= t - 1 < sol.exit[i]]
            exiting = [i for i in members if sol.exit[i] == t]
            if exiting:
                order = crane.order_exits(prob, sol, prev, exiting)
                if order is None:
                    order = exiting
                exit_ids.extend(order)
        for i in exit_ids:
            day_ops.append({"type": "EXIT", "block_id": int(i), "bay_id": int(sol.bay[i])})
        # ENTRYs after, ordered per bay
        entry_ids = []
        for j in range(prob.m):
            members = [i for i in range(n) if sol.bay[i] == j]
            prev = [i for i in members if sol.entry[i] <= t - 1 < sol.exit[i]]
            exiting = set(i for i in members if sol.exit[i] == t)
            base = [i for i in prev if i not in exiting]
            entering = [i for i in members if sol.entry[i] == t]
            if entering:
                order = crane.order_entries(prob, sol, base, entering)
                if order is None:
                    order = entering
                entry_ids.extend(order)
        for i in entry_ids:
            og = sol.og(i)
            day_ops.append({"type": "ENTRY", "block_id": int(i), "bay_id": int(sol.bay[i]),
                            "x": int(sol.x[i]), "y": int(sol.y[i]),
                            "orient_idx": int(sol.orient[i])})
        if day_ops:
            out[str(t)] = day_ops
    return {"operations": out}


def parse_output(prob, solution_dict):
    """Parse a grader-format output dict back into a Solution (for cross-checking).
    `orient_idx` is the index into blocks[i]['shape'] == our geoms[i] list index."""
    sol = Solution(prob)
    ops = solution_dict.get("operations", {})
    for day_str, lst in ops.items():
        day = int(day_str)
        for op in lst:
            i = op["block_id"]
            if op["type"] == "ENTRY":
                sol.bay[i] = op["bay_id"]
                sol.x[i] = op["x"]; sol.y[i] = op["y"]
                oi = op["orient_idx"]
                sol.orient[i] = oi if 0 <= oi < prob.O[i] else 0
                sol.entry[i] = day
            else:
                sol.exit[i] = day
    return sol


def try_utils_check(prob_info, solution_dict):
    """If utils.check_feasibility is importable at runtime, call it. Returns
    (available: bool, ok: bool|None)."""
    try:
        import utils  # provided by grader environment
    except Exception:
        return False, None
    try:
        res = utils.check_feasibility(prob_info, solution_dict)
        if isinstance(res, tuple):
            return True, bool(res[0])
        return True, bool(res)
    except Exception:
        return True, False
