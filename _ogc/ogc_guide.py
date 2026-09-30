# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Assignment guidance: solve the geometry-free bay-assignment relaxation
(min w2*imbalance + w3*preference penalty + soft space-time capacity) with
CP-SAT and return the optimal target bay per block.

The capacity term prices each bay's area-time overflow at the tardiness cost
it would force (w1 per day of delay ~ w1/bay_area per unit of area-time), so
targets stay attainable on congested instances. The LNS uses the targets to
bias insertion proposals; the true objective still decides acceptance."""

import math
import os


def _poly_area(verts):
    s = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def compute_targets(prob_info, time_budget=6.0, workers=4):
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
        SCALE = 10000
        u = [max(1, round(avg / a * SCALE)) for a in areas]
        wl = [int(round(bl["workload"])) for bl in blocks]

        # geometric fit: block i may go to bay b only if some orientation's
        # bounding box fits the bay at an integer position
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
            any_fit = False
            for b in range(m):
                x[i, b] = md.NewBoolVar(f"x{i}_{b}")
                if not fits(i, b):
                    md.Add(x[i, b] == 0)
                else:
                    any_fit = True
            if not any_fit:
                return None  # malformed instance; no guidance
            md.AddExactlyOne(x[i, b] for b in range(m))
        cap = sum(wl) * max(u)
        loads = []
        for b in range(m):
            L = md.NewIntVar(0, cap, f"L{b}")
            md.Add(L == sum(wl[i] * u[b] * x[i, b] for i in range(n)))
            loads.append(L)
        z = md.NewIntVar(0, cap, "z")
        for a in range(m):
            for b in range(m):
                if a != b:
                    md.Add(z >= loads[a] - loads[b])
        pen = sum(
            int(max(blocks[i]["bay_preferences"])
                - blocks[i]["bay_preferences"][b]) * x[i, b]
            for i in range(n) for b in range(m))

        # soft space-time capacity: overflowing a bay's area*days budget
        # forces roughly overflow/bay_area days of delay, priced at w1
        w1 = float(w.get("w1", 1.0))
        eta = 0.75  # attainable packing efficiency
        horizon = max(int(bl["due_date"]) for bl in blocks)
        demand = [int(round(_poly_area(next((lay for lay in
                                             bl["shape"][0]["layers"]
                                             if lay), []))
                            * max(1, int(bl["processing_time"]))))
                  for bl in blocks]
        # objective in centi-units: scale-then-round so fractional weights
        # survive (int(0.5) would silently delete a whole term)
        W2 = int(round(w2 * 100))
        W3 = int(round(w3 * 100))
        try:
            nbuck = int(os.environ.get("OGC_GBUCKETS", "1"))
        except ValueError:
            nbuck = 1  # a knob typo must not silently disable all guidance
        if nbuck >= 2:
            # time-bucketed capacity: each block's demand is spread over its
            # on-time span [R, max(D, R+P)), so the relaxation sees WHEN the
            # space is needed, not just the horizon total
            spans = []
            H2 = horizon
            for bl in blocks:
                s0 = int(bl["release_time"])
                s1 = max(int(bl["due_date"]),
                         s0 + max(1, int(bl["processing_time"])))
                spans.append((s0, s1))
                H2 = max(H2, s1)
            edges = [round(k * H2 / nbuck) for k in range(nbuck + 1)]
            over_cost = 0
            for b in range(m):
                price = int(round(w1 * 100 * SCALE / areas[b]))
                for k in range(nbuck):
                    e0, e1 = edges[k], edges[k + 1]
                    if e1 <= e0:
                        continue
                    terms = []
                    for i in range(n):
                        s0, s1 = spans[i]
                        olap = min(s1, e1) - max(s0, e0)
                        if olap <= 0:
                            continue
                        d_ik = int(round(demand[i] * olap / (s1 - s0)))
                        if d_ik > 0:
                            terms.append(d_ik * x[i, b])
                    if not terms:
                        continue
                    cap_bk = int(eta * areas[b] * (e1 - e0))
                    ov = md.NewIntVar(0, sum(demand), f"ov{b}_{k}")
                    md.Add(ov >= sum(terms) - cap_bk)
                    over_cost += price * ov
        else:
            over_cost = 0
            for b in range(m):
                cap_b = int(eta * areas[b] * horizon)
                load_at = sum(demand[i] * x[i, b] for i in range(n))
                ov = md.NewIntVar(0, sum(demand), f"ov{b}")
                md.Add(ov >= load_at - cap_b)
                over_cost += int(round(w1 * 100 * SCALE / areas[b])) * ov
        md.Minimize(W2 * z + W3 * SCALE * pen + over_cost)

        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = float(time_budget)
        sv.parameters.num_search_workers = int(workers)
        st = sv.Solve(md)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        targets = [0] * n
        for i in range(n):
            for b in range(m):
                if sv.Value(x[i, b]):
                    targets[i] = b
                    break
        return targets
    except Exception:
        # The caller owns optional fallback and diagnostics; do not erase
        # programming/API/data errors as a normal no-incumbent outcome.
        raise
