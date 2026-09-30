# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Last-resort pure-python solver.

It serializes blocks within each bay, which guarantees crane and collision
feasibility without a dense time array, while greedily minimizing the exact
incremental tardiness/preference/workload objective.  It is intentionally
usable for arbitrarily large legal dates rather than merely crash insurance.
"""

import math


def _bbox(block, oi):
    xs = [v[0] for lay in block["shape"][oi]["layers"] for v in lay]
    ys = [v[1] for lay in block["shape"][oi]["layers"] for v in lay]
    return min(xs), min(ys), max(xs), max(ys)


def _objective_greedy(prob_info):
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    m = len(bays)
    # per-bay list of (entry, exit) windows
    sched = [[] for _ in range(m)]
    loads = [0.0] * m
    ops = {}
    w = prob_info.get("weights", {}) or {}
    w1 = float(w.get("w1", 1.0))
    w2 = float(w.get("w2", 1.0))
    w3 = float(w.get("w3", 1.0))
    areas = [float(b["width"] * b["height"]) for b in bays]
    avg_area = sum(areas) / max(1, m)
    u = [avg_area / max(a, 1e-9) for a in areas]

    def imbalance(ls):
        if m <= 1:
            return 0
        z = [u[b] * ls[b] for b in range(m)]
        return math.floor(max(z) - min(z))

    def earliest_empty(b, release, duration):
        entry = release
        changed = True
        while changed:
            changed = False
            for a, e in sched[b]:
                if entry < e and a < entry + duration:
                    entry = e
                    changed = True
        return entry

    def add_op(t, op):
        ops.setdefault(str(int(t)), []).append(op)

    order = sorted(range(len(blocks)),
                   key=lambda i: (blocks[i]["due_date"],
                                  blocks[i]["release_time"]))
    for i in order:
        blk = blocks[i]
        R = int(blk["release_time"])
        P = max(1, int(blk["processing_time"]))
        best = None
        base_z2 = imbalance(loads)
        smax = max(blk["bay_preferences"])
        for b in range(m):
            W, H = bays[b]["width"], bays[b]["height"]
            for oi in range(len(blk["shape"])):
                x0, y0, x1, y1 = _bbox(blk, oi)
                px_lo = math.ceil(-x0)
                px_hi = math.floor(W - x1)
                py_lo = math.ceil(-y0)
                py_hi = math.floor(H - y1)
                if px_lo > px_hi or py_lo > py_hi:
                    continue
                entry = earliest_empty(b, R, P)
                exit_t = entry + P
                trial = loads.copy()
                trial[b] += float(blk["workload"])
                delta = (w1 * max(0, exit_t - int(blk["due_date"]))
                         + w3 * (smax - blk["bay_preferences"][b])
                         + w2 * (imbalance(trial) - base_z2))
                cand = (delta, exit_t, -blk["bay_preferences"][b], b, oi,
                        entry, px_lo, py_lo)
                if best is None or cand < best:
                    best = cand
                break
        if best is None:
            raise RuntimeError(f"block {i} fits no bay")
        _cost, exit_t, _pref, b, oi, entry, px, py = best
        sched[b].append((entry, exit_t))
        loads[b] += float(blk["workload"])
        add_op(entry, {"type": "ENTRY", "block_id": int(i),
                       "bay_id": int(b), "x": int(px),
                       "y": int(py), "orient_idx": int(oi)})
        add_op(exit_t, {"type": "EXIT", "block_id": int(i),
                        "bay_id": int(b)})
    # EXITs before ENTRYs within each day; emit day keys in numeric order
    # (utils.check_feasibility processes dict insertion order and must see
    # each block's ENTRY before its EXIT)
    out = {}
    for t in sorted(ops, key=int):
        out[t] = sorted(ops[t], key=lambda op: 0 if op["type"] == "EXIT" else 1)
    return {"operations": out}


def _preference_greedy(prob_info):
    """The old robust arm; useful when locally cheap load moves hurt later."""
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    sched = [[] for _ in bays]
    ops = {}

    def add_op(t, op):
        ops.setdefault(str(int(t)), []).append(op)

    for i in sorted(range(len(blocks)),
                    key=lambda j: (blocks[j]["due_date"],
                                   blocks[j]["release_time"])):
        blk = blocks[i]
        release = int(blk["release_time"])
        duration = max(1, int(blk["processing_time"]))
        placed = False
        for b in sorted(range(len(bays)),
                        key=lambda j: -blk["bay_preferences"][j]):
            width, height = bays[b]["width"], bays[b]["height"]
            for oi in range(len(blk["shape"])):
                x0, y0, x1, y1 = _bbox(blk, oi)
                px, py = math.ceil(-x0), math.ceil(-y0)
                if px > math.floor(width - x1) or py > math.floor(height - y1):
                    continue
                entry = release
                changed = True
                while changed:
                    changed = False
                    for start, end in sched[b]:
                        if entry < end and start < entry + duration:
                            entry = end
                            changed = True
                exit_t = entry + duration
                sched[b].append((entry, exit_t))
                add_op(entry, {"type": "ENTRY", "block_id": int(i),
                               "bay_id": int(b), "x": int(px), "y": int(py),
                               "orient_idx": int(oi)})
                add_op(exit_t, {"type": "EXIT", "block_id": int(i),
                                "bay_id": int(b)})
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError(f"block {i} fits no bay")
    return {"operations": {
        t: sorted(ops[t], key=lambda op: 0 if op["type"] == "EXIT" else 1)
        for t in sorted(ops, key=int)
    }}


def _objective(prob_info, solution):
    """Exact contest objective for a structurally feasible fallback."""
    blocks = prob_info["blocks"]
    bays = prob_info["bays"]
    weights = prob_info.get("weights", {}) or {}
    w1 = float(weights.get("w1", 1.0))
    w2 = float(weights.get("w2", 1.0))
    w3 = float(weights.get("w3", 1.0))
    assigned = [None] * len(blocks)
    exits = [None] * len(blocks)
    for day, day_ops in solution["operations"].items():
        for op in day_ops:
            i = int(op["block_id"])
            if op["type"] == "ENTRY":
                assigned[i] = int(op["bay_id"])
            elif op["type"] == "EXIT":
                exits[i] = int(day)
    loads = [0.0] * len(bays)
    tardiness = preference = 0.0
    for i, blk in enumerate(blocks):
        b = assigned[i]
        loads[b] += float(blk["workload"])
        tardiness += max(0, exits[i] - int(blk["due_date"]))
        preference += max(blk["bay_preferences"]) - blk["bay_preferences"][b]
    avg_area = sum(float(b["width"] * b["height"]) for b in bays) / len(bays)
    normalized = [avg_area / float(bays[b]["width"] * bays[b]["height"])
                  * loads[b] for b in range(len(bays))]
    imbalance = 0 if len(bays) == 1 else math.floor(max(normalized) - min(normalized))
    return w1 * tardiness + w2 * imbalance + w3 * preference


def fallback_solve(prob_info):
    # A two-arm deterministic portfolio costs only O(blocks*bays): retain the
    # old preference-first behavior whenever the locally objective-aware arm
    # happens to make a globally worse sequence of greedy choices.
    objective_arm = _objective_greedy(prob_info)
    preference_arm = _preference_greedy(prob_info)
    return min((objective_arm, preference_arm),
               key=lambda sol: _objective(prob_info, sol))
