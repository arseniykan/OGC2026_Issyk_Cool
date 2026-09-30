# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""
model.py — parse prob_info, precompute geometry, hold problem + solution state,
compute the exact objective.  (build step 1 / io_utils in the guide)
"""
import os
import math
import numpy as np
from . import geo
from .geo import SCALE


def _raster_on():
    v = os.environ.get("OGC_RASTER")
    if v is not None:
        return v not in ("0", "false", "False", "")
    try:
        from . import settings
        return getattr(settings, "USE_RASTER", False)
    except Exception:
        return False


_USE_RASTER = _raster_on()

# A5 geometry dedup: (block,orient) OGs frequently share IDENTICAL triangle sets
# (repeated shapes / symmetric orientations). Rasterize ONCE per distinct geometry
# and share the (read-only) masks — turns thousands of rasterizations into ~hundreds.
# Keyed by the exact scaled-triangle bytes, so no false reuse. Process-global (a
# fresh worker process starts empty); grows only with distinct geometries (bounded).
_RASTER_CACHE = {}


def _bounding_circle(all_tris, ubbox):
    """All-levels bounding circle [cx, cy, r] (float, ×SCALE units). Center = bbox
    center; radius = max distance to any triangle vertex (so it bounds every level)."""
    if all_tris.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    cx = 0.5 * (float(ubbox[0]) + float(ubbox[2]))
    cy = 0.5 * (float(ubbox[1]) + float(ubbox[3]))
    v = all_tris.reshape(-1, 2).astype(np.float64)
    dx = v[:, 0] - cx; dy = v[:, 1] - cy
    r = float(np.sqrt((dx * dx + dy * dy).max()))
    return np.array([cx, cy, r], dtype=np.float64)


def _inscribed_poles(tris0, max_poles=2):
    """Up to `max_poles` inscribed circles of the level-0 footprint [cx, cy, r]
    (float, ×SCALE). Uses triangle incircles (largest first) — each is guaranteed
    inside the footprint, so overlap of two blocks' poles proves a real collision."""
    if tris0.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    out = []
    cand = []
    for t in tris0:
        a = t.astype(np.float64)
        # side lengths
        la = np.hypot(a[1, 0] - a[2, 0], a[1, 1] - a[2, 1])
        lb = np.hypot(a[0, 0] - a[2, 0], a[0, 1] - a[2, 1])
        lc = np.hypot(a[0, 0] - a[1, 0], a[0, 1] - a[1, 1])
        s = la + lb + lc
        if s <= 1e-9:
            continue
        icx = (la * a[0, 0] + lb * a[1, 0] + lc * a[2, 0]) / s
        icy = (la * a[0, 1] + lb * a[1, 1] + lc * a[2, 1]) / s
        area = 0.5 * abs((a[1, 0] - a[0, 0]) * (a[2, 1] - a[0, 1])
                         - (a[2, 0] - a[0, 0]) * (a[1, 1] - a[0, 1]))
        r = 2.0 * area / s                    # incircle radius = area / semiperim
        cand.append((r, icx, icy))
    cand.sort(reverse=True)
    for r, icx, icy in cand[:max_poles]:
        out.append([icx, icy, r])
    return np.array(out, dtype=np.float64) if out else np.zeros((0, 3), dtype=np.float64)


class OrientGeom:
    """Precomputed geometry for one (block, orientation). Fields consumed by geo.py."""
    __slots__ = ("block_id", "orient_idx", "K", "tris", "tri_bb", "lbbox", "ubbox",
                 "w", "h", "area", "footprint_area", "layer_area",
                 "all_tris", "all_lvl", "all_bb", "bcircle", "poles",
                 "r_r0", "r_c0", "r_nrows", "r_nwords", "r_wstart",
                 "r_ow", "r_iw")

    def __init__(self, block_id, orient_idx, layers_verts):
        self.block_id = block_id
        self.orient_idx = orient_idx
        self.K = len(layers_verts)
        self.tris = []
        self.tri_bb = []   # per-level (T,4) int64 relative per-triangle bboxes
        lb = []
        self.layer_area = []
        for lv in layers_verts:
            p = geo.scale_poly(lv)
            tri = geo.stack_triangles(geo.triangulate(p))
            self.tris.append(tri)
            if tri.shape[0]:
                bb = np.stack([tri[:, :, 0].min(axis=1), tri[:, :, 1].min(axis=1),
                               tri[:, :, 0].max(axis=1), tri[:, :, 1].max(axis=1)], axis=1)
                self.tri_bb.append(bb.astype(np.int64))
            else:
                self.tri_bb.append(np.zeros((0, 4), dtype=np.int64))
            lb.append(geo.poly_bbox(p))
            # area of this layer (unscaled) = sum triangle areas / 2 / SCALE^2
            a2 = sum(geo.signed_area2(t) for t in geo.triangulate(p)) if len(p) >= 3 else 0
            self.layer_area.append(a2 / 2.0 / (SCALE * SCALE))
        self.lbbox = np.array(lb, dtype=np.int64) if lb else np.zeros((0, 4), np.int64)
        if lb:
            self.ubbox = np.array([min(b[0] for b in lb), min(b[1] for b in lb),
                                   max(b[2] for b in lb), max(b[3] for b in lb)], np.int64)
        else:
            self.ubbox = np.zeros(4, np.int64)
        # unscaled bbox dims (for candidate generation / fit tests)
        self.w = (self.ubbox[2] - self.ubbox[0]) / SCALE
        self.h = (self.ubbox[3] - self.ubbox[1]) / SCALE
        # footprint area = area of level 0 (used as space demand proxy)
        self.footprint_area = self.layer_area[0] if self.layer_area else 0.0
        self.area = self.footprint_area
        # concatenated ALL-levels triangles + per-triangle level ids + bboxes
        # (relative) for the unified level-aware early-exit placement kernel
        at = []; al = []; ab = []
        for l in range(self.K):
            t = self.tris[l]
            if t.shape[0]:
                at.append(t)
                al.append(np.full(t.shape[0], l, dtype=np.int64))
                ab.append(self.tri_bb[l])
        if at:
            self.all_tris = np.concatenate(at, axis=0)
            self.all_lvl = np.concatenate(al)
            self.all_bb = np.concatenate(ab, axis=0)
        else:
            self.all_tris = np.zeros((0, 3, 2), np.int64)
            self.all_lvl = np.zeros(0, np.int64)
            self.all_bb = np.zeros((0, 4), np.int64)
        # ---- CDE fail-fast surrogate (jagua-rs, "Decoupling Geometry" 2024) ----
        # bcircle = an all-levels BOUNDING circle (center, radius) so two blocks
        # whose bounding circles are STRICTLY disjoint provably cannot collide at
        # any level (a CLEAR fast-path); poles = level-0 INSCRIBED circles so two
        # blocks with STRICTLY overlapping poles provably collide (a HIT fast-path).
        # STRICT inequalities keep touching in the AMBIGUOUS zone -> exact SAT
        # (which allows touching). Coordinates are ×SCALE relative to the ref point.
        self.bcircle = _bounding_circle(self.all_tris, self.ubbox)
        self.poles = _inscribed_poles(self.tris[0] if self.K else
                                      np.zeros((0, 3, 2), np.int64))
        # ---- INNER/OUTER raster masks per level (A1), only when enabled ----
        if _USE_RASTER:
            key = (self.all_tris.shape, self.all_tris.tobytes(),
                   self.all_lvl.tobytes())
            cached = _RASTER_CACHE.get(key)
            if cached is not None:                       # A5: reuse identical geometry
                (self.r_r0, self.r_c0, self.r_nrows, self.r_nwords,
                 self.r_wstart, self.r_ow, self.r_iw) = cached
            else:
                rr0 = []; rc0 = []; rnr = []; rnw = []; wst = [0]
                owl = []; iwl = []
                for l in range(self.K):
                    r0, c0, nrows, nwords, ow, iw = geo.rasterize_poly_bits(self.tris[l])
                    rr0.append(r0); rc0.append(c0)
                    rnr.append(nrows); rnw.append(nwords)
                    owl.append(ow); iwl.append(iw)
                    wst.append(wst[-1] + nrows * nwords)
                self.r_r0 = np.array(rr0, dtype=np.int64) if rr0 else np.zeros(0, np.int64)
                self.r_c0 = np.array(rc0, dtype=np.int64) if rc0 else np.zeros(0, np.int64)
                self.r_nrows = np.array(rnr, dtype=np.int64) if rnr else np.zeros(0, np.int64)
                self.r_nwords = np.array(rnw, dtype=np.int64) if rnw else np.zeros(0, np.int64)
                self.r_wstart = np.array(wst, dtype=np.int64)
                self.r_ow = np.concatenate(owl) if owl else np.zeros(0, np.uint64)
                self.r_iw = np.concatenate(iwl) if iwl else np.zeros(0, np.uint64)
                _RASTER_CACHE[key] = (self.r_r0, self.r_c0, self.r_nrows, self.r_nwords,
                                      self.r_wstart, self.r_ow, self.r_iw)
        else:
            self.r_r0 = self.r_c0 = None


class Problem:
    """All precomputed, read-only instance data."""

    def __init__(self, prob_info):
        self.raw = prob_info
        self.name = prob_info.get("name", "instance")
        self.bays = prob_info["bays"]
        self.m = len(self.bays)
        self.W = np.array([b["width"] for b in self.bays], dtype=np.int64)
        self.H = np.array([b["height"] for b in self.bays], dtype=np.int64)
        self.bay_area = (self.W * self.H).astype(np.float64)

        blocks = prob_info["blocks"]
        self.n = len(blocks)
        self.R = np.array([b["release_time"] for b in blocks], dtype=np.int64)
        self.D = np.array([b["due_date"] for b in blocks], dtype=np.int64)
        self.P = np.array([b["processing_time"] for b in blocks], dtype=np.int64)
        self.L = np.array([b["workload"] for b in blocks], dtype=np.float64)

        # preferences (n,m) ; Smax per block ; preference cost matrix (Smax - S_ij)
        self.pref = np.array([b["bay_preferences"] for b in blocks], dtype=np.int64)
        self.Smax = self.pref.max(axis=1)
        self.pref_cost = (self.Smax[:, None] - self.pref).astype(np.int64)  # (n,m) cost of assigning i to j

        # geometry per block: list over orientation of OrientGeom
        self.geoms = []
        self.O = np.zeros(self.n, dtype=np.int64)
        self.K = np.zeros(self.n, dtype=np.int64)
        for i, b in enumerate(blocks):
            og_list = []
            for o in b["shape"]:
                og_list.append(OrientGeom(i, o["orientation"], o["layers"]))
            self.geoms.append(og_list)
            self.O[i] = len(og_list)
            self.K[i] = og_list[0].K if og_list else 0

        # weights
        w = prob_info.get("weights", {"w1": 1, "w2": 1, "w3": 1})
        self.w1 = float(w["w1"]); self.w2 = float(w["w2"]); self.w3 = float(w["w3"])

        # u_j = (avg bay area) / (bay j area); larger bays -> smaller weight
        avg_area = self.bay_area.mean()
        self.u = avg_area / self.bay_area  # (m,)

        # horizon
        self.horizon = int(self.D.max() + self.P.max())

        # per (block) which bays it can possibly fit (some orientation bbox <= bay dims)
        self.fit_bays = []
        # min footprint area over orientations (space demand)
        self.min_area = np.zeros(self.n, dtype=np.float64)
        # smallest bounding box orientation area per block (for candidate ordering)
        for i in range(self.n):
            fits = []
            best_area = None
            for j in range(self.m):
                ok = False
                for og in self.geoms[i]:
                    if og.w <= self.W[j] + 1e-9 and og.h <= self.H[j] + 1e-9:
                        ok = True
                        break
                if ok:
                    fits.append(j)
            for og in self.geoms[i]:
                if best_area is None or og.footprint_area < best_area:
                    best_area = og.footprint_area
            self.fit_bays.append(fits if fits else list(range(self.m)))
            self.min_area[i] = best_area if best_area else 0.0

        # slack
        self.slack = self.D - self.R - self.P

        # ---- objective lower-bound brackets (OBJECTIVE_BOUNDING_PLAN) --------
        # Per-block optimistic FLOORS (analogue of INNER cells): the least each
        # separable term can contribute, independent of placement. Precomputed once.
        #   Z1: earliest-possible exit R+P  ->  Tfloor_i = max(0, R+P-D) = max(0,-slack)
        #   Z3: cheapest FEASIBLE bay       ->  Z3floor_i = min_{j in fit_bays} pref_cost
        self.Tfloor = np.maximum(0, -self.slack).astype(np.int64)
        self.Z3floor = np.empty(self.n, dtype=np.int64)
        for i in range(self.n):
            fb = self.fit_bays[i]
            self.Z3floor[i] = int(self.pref_cost[i, fb].min()) if len(fb) else 0
        self.Z1_floor_sum = int(self.Tfloor.sum())
        self.Z3_floor_sum = int(self.Z3floor.sum())
        # Z2 (coupled): max weighted load each bay can still reach (blocks that fit it)
        self.bay_fit_load = np.zeros(self.m, dtype=np.float64)
        for i in range(self.n):
            for j in self.fit_bays[i]:
                self.bay_fit_load[j] += self.L[i]

    def obj_floor(self):
        """Admissible GLOBAL lower bound on the exact objective (any feasible
        completion). Z2_lower = 0 globally (no load fixed). One-sided:
        obj_floor <= w1*Z1 + w2*Z2 + w3*Z3 for every feasible solution."""
        return self.w1 * self.Z1_floor_sum + self.w3 * self.Z3_floor_sum

    # ---- orientation helpers -------------------------------------------
    def orientations_for_bay(self, i, j):
        """indices of orientations of block i that fit (bbox) in bay j."""
        out = []
        for oi, og in enumerate(self.geoms[i]):
            if og.w <= self.W[j] + 1e-9 and og.h <= self.H[j] + 1e-9:
                out.append(oi)
        return out


class Solution:
    """
    A complete assignment. Arrays indexed by block id.
    bay[i], orient[i], x[i], y[i], entry[i], exit[i]  (all ints; -1 = unassigned)
    """
    __slots__ = ("prob", "bay", "orient", "x", "y", "entry", "exit",
                 "_z1", "_z2", "_z3", "_dirty")

    def __init__(self, prob):
        self.prob = prob
        n = prob.n
        self.bay = np.full(n, -1, dtype=np.int64)
        self.orient = np.zeros(n, dtype=np.int64)
        self.x = np.zeros(n, dtype=np.int64)
        self.y = np.zeros(n, dtype=np.int64)
        self.entry = np.zeros(n, dtype=np.int64)
        self.exit = np.zeros(n, dtype=np.int64)
        self._dirty = True

    def copy(self):
        s = Solution.__new__(Solution)
        s.prob = self.prob
        s.bay = self.bay.copy(); s.orient = self.orient.copy()
        s.x = self.x.copy(); s.y = self.y.copy()
        s.entry = self.entry.copy(); s.exit = self.exit.copy()
        s._dirty = True
        return s

    def og(self, i):
        return self.prob.geoms[i][self.orient[i]]

    # ---- objective ------------------------------------------------------
    def objective(self):
        """Return (scalar, Z1, Z2, Z3) using exact spec formulas (vectorized)."""
        p = self.prob
        bay = self.bay
        asg = bay >= 0
        # Z1 tardiness
        tard = np.maximum(0, self.exit - p.D)
        Z1 = int(tard[asg].sum()) if asg.any() else 0
        # Z2 workload imbalance: max pairwise |u_j*load_j - u_k*load_k|
        load = np.bincount(bay[asg], weights=p.L[asg], minlength=p.m) if asg.any() \
            else np.zeros(p.m)
        wload = p.u * load
        # The evaluator floors the maximum normalized imbalance.
        Z2 = (float(math.floor(wload.max() - wload.min()))
              if p.m > 1 else 0.0)
        # Z3 preference cost  (pref_cost[i, bay[i]] summed over assigned blocks)
        if asg.any():
            idx = np.nonzero(asg)[0]
            Z3 = int(p.pref_cost[idx, bay[idx]].sum())
        else:
            Z3 = 0
        scalar = p.w1 * Z1 + p.w2 * Z2 + p.w3 * Z3
        return scalar, Z1, Z2, Z3

    # ---- output ---------------------------------------------------------
    def to_output(self):
        """Build the grader's operations dict. EXIT ops precede ENTRY ops per day."""
        ops = {}
        p = self.prob
        for i in range(p.n):
            if self.bay[i] < 0:
                continue
            e = int(self.entry[i]); x = int(self.exit[i])
            ops.setdefault(e, {"EXIT": [], "ENTRY": []})
            ops.setdefault(x, {"EXIT": [], "ENTRY": []})
            ops[e]["ENTRY"].append({
                "type": "ENTRY", "block_id": i, "bay_id": int(self.bay[i]),
                "x": int(self.x[i]), "y": int(self.y[i]),
                "orient_idx": int(self.orient[i]),  # index into blocks[i]['shape']
            })
            ops[x]["EXIT"].append({
                "type": "EXIT", "block_id": i, "bay_id": int(self.bay[i]),
            })
        out = {}
        for day in sorted(ops.keys()):
            lst = ops[day]["EXIT"] + ops[day]["ENTRY"]  # EXIT first
            if lst:
                out[str(day)] = lst
        return {"operations": out}


def event_days(sol, blocks=None):
    """Sorted unique days on which some block enters or exits (state changes here)."""
    p = sol.prob
    idx = range(p.n) if blocks is None else blocks
    days = set()
    for i in idx:
        if sol.bay[i] < 0:
            continue
        days.add(int(sol.entry[i]))
        days.add(int(sol.exit[i]))
    return sorted(days)


def presence_on(sol, day, bay=None):
    """List of block ids present at `day` (entry<=day<exit), optionally in a bay."""
    p = sol.prob
    out = []
    for i in range(p.n):
        if sol.bay[i] < 0:
            continue
        if bay is not None and sol.bay[i] != bay:
            continue
        if sol.entry[i] <= day < sol.exit[i]:
            out.append(i)
    return out
