# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Global joint (block -> bay x day-bucket) time-indexed MIP via Gurobi
(server provides 13.0.2 per problem statement 4.1; local academic license
gives dev parity). Jointly optimizes true w1 tardiness + w3 preference +
w2 imbalance under aggregate area capacity. Output: the SAME (tday, tbay)
artifact as ogc_dayguide.compute_day_targets, so every consumer works
unchanged. Returns None on ANY failure -- callers fall back to the
two-stage pipeline.

Hardened after the v17 adversarial review (all empirically confirmed):
  - license PROBE before any build work: a >2000-var throwaway model
    catches the pip size-limited license (which otherwise fails only at
    optimize(), after the full build); env-based creation with OutputFlag
    set pre-start silences the stdout banner;
  - the O(n^2) assignment loop and the O(m*K*|x|) capacity loop are gone
    (per-block var lists at creation; one-pass bucket term accumulation);
  - a hard `deadline` is checked between build phases (the Gurobi
    TimeLimit bounds only the solve);
  - a size gate coarsens buckets for big instances and refuses outright
    when even coarse buckets would blow the variable budget;
  - the release bucket is included (start = max(R, k*bw)): the old
    range(ceil(R/bw), K) forced 18-63% of blocks per instance into
    artificially-late targets, poisoning tlate and the scan lift;
  - per-bay demand is clamped to eta*area (parity with ogc_dayguide),
    so one oversized block cannot make the whole model infeasible."""

import math

from _ogc import relaxarea
import os
import time

_VAR_CAP = 120000   # max assignment binaries; buckets coarsen to fit
_MIN_K = 8          # below this many buckets the signal is too coarse


def compute_global_targets(prob_info, time_budget=6.0, bucket_target=40,
                           threads=4, deadline=None):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None
    env = None
    md = None
    try:
        if deadline is None:
            deadline = time.time() + 2.0 * float(time_budget) + 3.0

        # --- license probe: must exceed the pip size-limited license's
        # 2000-var cap, or that failure mode surfaces only after the
        # full model build (measured 3-15s of dead work)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()  # raises without a usable license
        probe = gp.Model(env=env)
        probe.addVars(2001, vtype=GRB.BINARY)
        probe.Params.TimeLimit = 0.5
        probe.optimize()  # raises under the size-limited license
        probe.dispose()
        if time.time() > deadline:
            return None

        bays = prob_info["bays"]
        blocks = prob_info["blocks"]
        w = prob_info.get("weights", {}) or {}
        w1 = float(w.get("w1", 1.0))
        w2 = float(w.get("w2", 1.0))
        w3 = float(w.get("w3", 1.0))
        n, m = len(blocks), len(bays)
        if n == 0 or m == 0:
            return None
        R = [int(b["release_time"]) for b in blocks]
        D = [int(b["due_date"]) for b in blocks]
        P = [max(1, int(b["processing_time"])) for b in blocks]
        wl = [float(b["workload"]) for b in blocks]
        pref = [b.get("bay_preferences", [0] * m) for b in blocks]
        smax = [max(pr) if pr else 0 for pr in pref]
        areas = [b["width"] * b["height"] for b in bays]
        avg = sum(areas) / m
        u = [avg / a for a in areas]
        try:
            eta = float(os.environ.get("OGC_DAY_ETA", "0.65"))
            if not (0.4 <= eta <= 0.95):
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
            return None if best is None else max(1, int(round(best)))

        dem = []
        for i in range(n):
            dem.append([demand(blocks[i], bays[b]) for b in range(m)])
            # clamp: an oversized block must not force model infeasibility
            for b in range(m):
                if dem[i][b] is not None:
                    dem[i][b] = min(dem[i][b],
                                    max(1, int(eta * areas[b])))
        if time.time() > deadline:
            return None

        load = sum((next((d for d in dem[i] if d is not None), 1)) * P[i]
                   for i in range(n))
        cap_total = eta * sum(areas)
        H = int(max(max(D), max(R) + load / cap_total + 2 * max(P)) * 1.25)
        bw = max(1, math.ceil(H / bucket_target))
        K = math.ceil(H / bw)
        # size gate: coarsen buckets for big instances; refuse when even
        # coarse buckets cannot fit the variable budget
        while n * m * K > _VAR_CAP and K > _MIN_K:
            bw *= 2
            K = math.ceil(H / bw)
        if n * m * K > _VAR_CAP:
            return None

        md = gp.Model("gseed", env=env)
        md.Params.OutputFlag = 0
        md.Params.MIPGap = 0.01
        md.Params.Threads = threads
        md.Params.Seed = 20260711 % 2000000000

        # variables: one pass, per-block lists (no dict rescans); the
        # release bucket IS included via s = max(R, k*bw)
        entries = []          # (i, b, s, var)
        per_i = [[] for _ in range(n)]
        for i in range(n):
            for b in range(m):
                if dem[i][b] is None:
                    continue
                for k in range(R[i] // bw, K):
                    s = max(R[i], k * bw)
                    if s + P[i] > H:
                        break
                    cost = (w1 * max(0, s + P[i] - D[i])
                            + w3 * (smax[i] - pref[i][b]))
                    v = md.addVar(vtype=GRB.BINARY, obj=cost)
                    entries.append((i, b, s, v))
                    per_i[i].append(v)
        for i in range(n):
            if not per_i[i]:
                return None  # malformed instance; no guidance
            md.addConstr(gp.quicksum(per_i[i]) == 1)
        if time.time() > deadline:
            return None

        # capacity + loads: single pass over entries
        cap_terms = [[[] for _ in range(K)] for _ in range(m)]
        load_terms = [[] for _ in range(m)]
        for (i, b, s, v) in entries:
            d_ib = dem[i][b]
            kq1 = min(K - 1, (s + P[i] - 1) // bw)
            for kq in range(s // bw, kq1 + 1):
                ov = min((kq + 1) * bw, s + P[i]) - max(kq * bw, s)
                if ov > 0:
                    cap_terms[b][kq].append(d_ib * ov * v)
            load_terms[b].append(wl[i] * v)
        for b in range(m):
            cap = eta * areas[b] * bw
            for kq in range(K):
                if cap_terms[b][kq]:
                    md.addConstr(gp.quicksum(cap_terms[b][kq]) <= cap)
        L = [gp.quicksum(load_terms[b]) for b in range(m)]
        M = md.addVar(lb=0.0, obj=w2)
        for b in range(m):
            for c in range(m):
                if b != c:
                    md.addConstr(M >= u[b] * L[b] - u[c] * L[c])
        if time.time() > deadline:
            return None

        md.Params.TimeLimit = max(0.5, min(float(time_budget),
                                           deadline - time.time()))
        md.ModelSense = GRB.MINIMIZE
        md.optimize()
        if md.SolCount == 0:
            return None
        tday = [-1] * n
        tbay = [-1] * n
        for (i, b, s, v) in entries:
            if v.X > 0.5:
                tday[i] = s
                tbay[i] = b
        if all(t < 0 for t in tday):
            return None
        return tday, tbay
    except Exception:
        raise
    finally:
        try:
            if md is not None:
                md.dispose()
            if env is not None:
                env.dispose()
        except Exception:
            pass
