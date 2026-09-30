# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Solution state: committed placements, feasibility queries, insert/remove.

Feasibility model (conservative w.r.t. utils.check_feasibility):
  A block C placed in bay b at integer (x, y), orientation o, over the
  half-open day interval [t1, t2) must satisfy, for every committed block D in
  the same bay whose interval overlaps:

    boundary condition (D "present" while C's crane moves):
      - D occupies day t1           (entry moment, same-day entries of earlier
                                     commits included: ours-last ordering)
      - or D occupies day t2 - 1    (exit moment, same-day exits of earlier
                                     commits included: ours-first ordering)
      => C's layer k must avoid D's layers j >= k.

    interior condition (D's own crane moves while C sits):
      - D enters or exits strictly inside (t1, t2)
      => D's layer a must avoid C's layers >= a.

  Same-day operation ordering is realized at output time by commit sequence:
  entries ascending (later commits enter later), exits descending (later
  commits exit earlier). This makes every commit-time check consistent with
  the checker's sequential Stage-5 replay.
"""

import math
import os

import numpy as np

from _ogc.ogc_kernels import build_G, scan_best, day_scan, day_scan_bb


class State:
    def __init__(self, prep, rng=None):
        p = self.prep = prep
        n, m = p.n, p.m
        self.tcap = int(p.tcap)
        self.rng = rng or np.random.default_rng(12345)

        self.bay = np.full(n, -1, np.int64)
        self.px = np.zeros(n, np.int64)
        self.py = np.zeros(n, np.int64)
        self.oi = np.zeros(n, np.int64)
        self.t1 = np.zeros(n, np.int64)
        self.t2 = np.zeros(n, np.int64)
        self.seq = np.zeros(n, np.int64)
        self.placed = np.zeros(n, bool)
        self.seq_counter = 0

        self.bay_blocks = [[] for _ in range(m)]  # list of block ids per bay
        self.loads = np.zeros(m)
        self.sum_tard = 0.0
        self.sum_pref = 0.0
        self.occ0 = np.zeros((m, self.tcap), np.int64)

        # scratch buffers for queries
        self._dx = np.zeros(n, np.int64)
        self._dy = np.zeros(n, np.int64)
        self._doidx = np.zeros(n, np.int64)
        self._dflags = np.zeros(n, np.int64)
        self._G = [np.zeros((int(p.K.max()), int(p.bayW[b]), int(p.bayH[b])),
                            np.int16) for b in range(m)]
        self._Gin = [np.zeros((int(p.K.max()), int(p.bayW[b]),
                               int(p.bayH[b])), np.int16) for b in range(m)]
        # bitboard position test: occupancy bit-grids (bits along x, one
        # 3-word mask row per (layer, y)); requires every bay and layer bbox
        # to fit 192 columns
        self._use_bb = (os.environ.get("OGC_BITBOARD", "0") == "1"
                        and p.bb_ok
                        and all(int(p.bayW[b]) <= 192 for b in range(m)))
        self._bb_verify = os.environ.get("OGC_BB_VERIFY", "0") == "1"
        if self._use_bb:
            self._Gbit = [np.zeros((int(p.K.max()), int(p.bayH[b]), 3),
                                   np.uint64) for b in range(m)]
            self._Ginbit = [np.zeros((int(p.K.max()), int(p.bayH[b]), 3),
                                     np.uint64) for b in range(m)]
        # Gin occupancy bitboard for the run-based screen (one scratch board
        # per bay, zeroed and rebuilt inside each day_scan call). Needs every
        # bay to fit 192 columns and Prep to have produced runs.
        self._screen = (os.environ.get("OGC_SCREEN", "1") != "0"
                        and getattr(p, "runs_ok", False)
                        and all(int(p.bayW[b]) <= 192 for b in range(m)))
        self._Ginb = [np.zeros((int(p.K.max()), int(p.bayH[b]) + 2, 3),
                               np.uint64) for b in range(m)]
        # best_insertion runs 40k-140k times per 25s and slices these buffers
        # per (bay, K) on every call. None of them is ever REASSIGNED -- the
        # grids are written in place and restore() does `occ0[:] = 0` -- so the
        # views can be built once. (self.loads IS reassigned by restore, so it
        # is deliberately not cached here.)
        _Kmax = int(p.K.max())
        self._Gk = [[self._G[b][:k] for k in range(_Kmax + 1)]
                    for b in range(m)]
        self._Gink = [[self._Gin[b][:k] for k in range(_Kmax + 1)]
                      for b in range(m)]
        self._Ginbk = [[self._Ginb[b][:k] for k in range(_Kmax + 1)]
                       for b in range(m)]
        self._occ0b = [self.occ0[b] for b in range(m)]
        self._bay_cap = [int(p.bayW[b] * p.bayH[b]) for b in range(m)]
        self.exact_budget = int(os.environ.get("OGC_EXACT_BUDGET", "128"))
        # earliest-feasible-day memo: a day_scan that returns t* proves
        # "no feasible day < t* in this bay for this block" (the kernel
        # returns on the FIRST feasible day). Insertions only add
        # constraints, so the proof stays exact until a removal in that
        # bay; a per-bay removal counter is the staleness stamp.
        # CAVEAT (review 07-14): when exact_budget exhausts within a day
        # the verdict is scan-order (xdir/ydir) dependent, so the "proof"
        # is exact only for non-exhausted scans -- one reason this knob is
        # dead (hit rate 0.12%) and MUST stay default-off as-is.
        self._efd_on = os.environ.get("OGC_EFD", "0") == "1"
        self._efd_verify = os.environ.get("OGC_EFD_VERIFY", "0") == "1"
        self._efd_day = np.full((n, m), -1, np.int64)
        self._efd_stamp = np.full((n, m), -1, np.int64)
        self._rm_cnt = np.zeros(m, np.int64)
        self._efd_q = 0
        self._efd_hit = 0
        self._efd_skip = 0
        self.tau = 1e-9           # positive-overlap threshold for exact tests
        self.guide = None         # optional CP-SAT target bay per block
        self.guide_w = 0.0        # proposal bias toward the target bay
        self.lock_bay = None      # optional HARD per-block bay lock (the
                                  # locked-assignment island: every insertion
                                  # is forced to lock_bay[i], so the LNS
                                  # optimises schedule/packing only)
        # departure-aligned placement: per-bay layer-0 exit-day field
        # (cell -> occupying block's exit_day + 1; 0 = empty), maintained in
        # insert/remove and used by refine_position's attention scoring
        self._exitpack = os.environ.get("OGC_EXITPACK", "0") != "0"
        self._E = ([np.zeros((int(p.bayW[b]), int(p.bayH[b])), np.int64)
                    for b in range(m)] if self._exitpack else None)
        # space-time shadow pricing (OGC_PRICE): per-day overload price
        # prefix sums; day_scan charges pw * (priceP[t2]-priceP[t]) per
        # candidate window so placement SEES future demand. gamma=0 -> off,
        # kernel behavior bit-identical.
        self.price_prefix = np.zeros(self.tcap + 2, np.float64)
        self.price_gamma = 0.0
        # narrowest usable footprint width across all (block, orientation):
        # free runs narrower than this are dead slivers (side-view metric)
        try:
            widths = []
            heights = []
            for i2 in range(n):
                for o2 in range(int(p.n_orient[i2])):
                    r0 = int(p.layer_base[p.orient_base[i2] + o2])
                    widths.append(p.lbx1[r0] - p.lbx0[r0])
                    heights.append(p.lby1[r0] - p.lby0[r0])
            self._wmin = int(max(2, min(8, math.ceil(min(widths)))))
            self._hmin = int(max(2, min(8, math.ceil(min(heights)))))
        except Exception:
            self._wmin = 3
            self._hmin = 3
        self.tday = None          # optional day-relaxation target entry days
        self.tbay = None          # bay each day target was computed for
        self.tlate = None         # relaxation's sacrifice set (ends past due)
        # packed per-bay arrays for the day_scan kernel, kept in sync with
        # bay_blocks (same order)
        self._jid = [np.zeros(n, np.int64) for _ in range(m)]
        self._jt1 = [np.zeros(n, np.int64) for _ in range(m)]
        self._jt2 = [np.zeros(n, np.int64) for _ in range(m)]
        self._jx = [np.zeros(n, np.int64) for _ in range(m)]
        self._jy = [np.zeros(n, np.int64) for _ in range(m)]
        self._joidx = [np.zeros(n, np.int64) for _ in range(m)]
        # per-bay CONTIGUOUS copies: passing column slices of 2-D arrays into
        # numba would create a second ('A'-layout) signature of day_scan and
        # force a multi-second recompile at runtime
        fits_u8 = p.fits.astype(np.uint8)
        self._fits_b = [np.ascontiguousarray(fits_u8[:, b]) for b in range(m)]
        self._pxlo_b = [np.ascontiguousarray(p.pxlo[:, b]) for b in range(m)]
        self._pxhi_b = [np.ascontiguousarray(p.pxhi[:, b]) for b in range(m)]
        self._pylo_b = [np.ascontiguousarray(p.pylo[:, b]) for b in range(m)]
        self._pyhi_b = [np.ascontiguousarray(p.pyhi[:, b]) for b in range(m)]
        # border-reservation bands (built on demand by enable_bands), keyed by
        # side so several band VIEWS of the same bay can coexist -- the
        # lookahead arbiter scans more than one and lets the future decide
        self._bands = {}
        self._band_side = "right"   # the side `band=True` resolves to
        self._band_pxlo = None
        self._band_pxhi = None
        self._band_fits = None

    # -------------------------------------------------------------- bands --
    def enable_bands(self, frac=0.25, side="right"):
        """Precompute BORDER BANDS: a narrowed px range hugging one bay edge.

        Rationale: the free area a bay carries forward is only useful to a
        future big block if it is CONTIGUOUS. Small blocks scattered through
        the middle carve the one large region into several unusable ones, and
        because placements PERSIST for a block's whole stay, that damage is
        paid on every day of its span -- this is the static-density vs
        persistent-position gap that makes area plans over-promise.

        Confining small blocks to a band at the far edge keeps the central
        span whole. Note this is NOT a nudge off the anchor corner (measured
        harmful five times over): inside the band the ordinary bottom-left
        discipline still decides the exact cell -- only the *search range*
        moves, and the caller applies it strictly as a tie-break.
        """
        p = self.prep
        m, nO = p.m, p.pxlo.shape[0]
        lo = np.array(p.pxlo, np.int64, copy=True)
        hi = np.array(p.pxhi, np.int64, copy=True)
        for b in range(m):
            Wb = float(p.bayW[b]) * float(frac)
            for oidx in range(nO):
                if side == "left":
                    v = int(math.floor(Wb - p.fbx1[oidx]))
                    if v < hi[oidx, b]:
                        hi[oidx, b] = v
                else:
                    v = int(math.ceil(float(p.bayW[b]) - Wb - p.fbx0[oidx]))
                    if v > lo[oidx, b]:
                        lo[oidx, b] = v
        fits = ((lo <= hi) & (p.pylo <= p.pyhi)).astype(np.uint8)
        self._band_pxlo = [np.ascontiguousarray(lo[:, b]) for b in range(m)]
        self._band_pxhi = [np.ascontiguousarray(hi[:, b]) for b in range(m)]
        self._band_fits = [np.ascontiguousarray(fits[:, b]) for b in range(m)]
        self._bands[side] = (self._band_pxlo, self._band_pxhi, self._band_fits)
        self._band_side = side

    # ------------------------------------------------------------------ obj --
    def obj2(self, loads=None):
        if self.prep.m < 2:
            return 0.0
        L = self.loads if loads is None else loads
        w = self.prep.u * L
        return math.floor(w.max() - w.min())

    def objective(self):
        p = self.prep
        return p.w1 * self.sum_tard + p.w2 * self.obj2() + p.w3 * self.sum_pref

    def set_price(self, gamma, lookahead=8, eta_p=0.65):
        """Build the space-time shadow price curve (the solver's view of
        the FUTURE): per-day demand from every block's area spread over its
        on-time span, overload = max(0, demand/capacity - eta_p), smeared
        forward over `lookahead` days (the how-far-it-sees hyperparameter).
        day_scan then charges gamma * w1 * (cells/captot) * overload-days
        for each candidate window, so a big block is steered AWAY from
        days whose space many due-soon blocks will need."""
        p = self.prep
        TC = self.price_prefix.shape[0] - 1
        dem = np.zeros(TC, np.float64)
        ob = p.orient_base
        for i2 in range(p.n):
            a = float(p.ncells0[ob[i2]])
            s0 = int(p.R[i2])
            P2 = max(1, int(p.P[i2]))
            s1 = max(int(p.D[i2]), s0 + P2)
            span = max(1, s1 - s0)
            if s1 > TC:
                s1 = TC
            if s0 < s1:
                dem[s0:s1] += a * P2 / span
        captot = float((p.bayW * p.bayH).sum())
        lam = np.maximum(0.0, dem / captot - float(eta_p))
        L = max(1, int(lookahead))
        if L > 1 and lam.any():
            lam2 = lam.copy()
            for sh in range(1, L):
                lam2[:-sh] = np.maximum(lam2[:-sh], lam[sh:])
            lam = lam2
        self.price_prefix[0] = 0.0
        self.price_prefix[1:] = np.cumsum(lam) * p.w1 / captot
        self.price_gamma = float(gamma)
        if self.price_gamma > 0.0:
            self._use_bb = False  # bb path has no price support

    # -------------------------------------------------------------- queries --
    def try_day(self, i, b, t1, xw=1.0, yw=0.001, xdir=1, ydir=1):
        """Best feasible (oidx_local, px, py) for block i in bay b entering at
        day t1, or None. Scans all orientations; picks the position closest to
        the anchored corner (xdir/ydir)."""
        p = self.prep
        t2 = t1 + p.P[i]
        K = int(p.K[i])
        W = int(p.bayW[b])
        H = int(p.bayH[b])

        G = self._G[b]
        G[:K, :, :] = 0

        nd = 0
        dx, dy, doidx, dflags = self._dx, self._dy, self._doidx, self._dflags
        for j in self.bay_blocks[b]:
            e1 = self.t1[j]
            e2 = self.t2[j]
            if e1 >= t2 or e2 <= t1:
                continue
            fl = 0
            if (e1 <= t1 < e2) or (e1 < t2 <= e2):
                fl |= 1
            if (t1 < e1 < t2) or (t1 < e2 < t2):
                fl |= 2
            if fl:
                dx[nd] = self.px[j]
                dy[nd] = self.py[j]
                doidx[nd] = p.orient_base[j] + self.oi[j]
                dflags[nd] = fl
                nd += 1
        if nd:
            build_G(G[:K], nd, dx, dy, doidx, dflags,
                    p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
                    p.arena_u, p.arena_v)

        best = None
        best_s = math.inf
        o0 = p.orient_base[i]
        for o in range(int(p.n_orient[i])):
            oidx = o0 + o
            if not p.fits[oidx, b]:
                continue
            s0, s1 = p.scan_ptr[oidx], p.scan_ptr[oidx + 1]
            pxlo = int(p.pxlo[oidx, b])
            pxhi = int(p.pxhi[oidx, b])
            pylo = int(p.pylo[oidx, b])
            pyhi = int(p.pyhi[oidx, b])
            bx, by = scan_best(
                G[:K],
                p.scan_u[s0:s1], p.scan_v[s0:s1], p.scan_k[s0:s1],
                pxlo, pxhi, pylo, pyhi, xw, yw, xdir, ydir)
            if bx >= 0:
                dx = (bx - pxlo) if xdir > 0 else (pxhi - bx)
                dy = (by - pylo) if ydir > 0 else (pyhi - by)
                s = xw * dx + yw * dy
                if s < best_s:
                    best_s = s
                    best = (o, bx, by)
        return best

    def day_upper_bound(self, i, b):
        """A day at which placement is guaranteed feasible (bay empty)."""
        last = self.prep.R[i]
        for j in self.bay_blocks[b]:
            if self.t2[j] > last:
                last = self.t2[j]
        return int(last)

    def quick_reject(self, i, b, t1):
        """Necessary condition: enough free layer-0 cells on every day of the
        window for the smallest orientation of i."""
        p = self.prep
        t2 = min(t1 + int(p.P[i]), self.tcap)
        cap = int(p.bayW[b] * p.bayH[b])
        need = self._min_cells0(i, b)
        occ = self.occ0[b]
        for t in range(t1, t2):
            if cap - occ[t] < need:
                return True
        return False

    def _min_cells0(self, i, b):
        p = self.prep
        o0, o1 = p.orient_base[i], p.orient_base[i + 1]
        vals = [p.ncells0[o] for o in range(o0, o1) if p.fits[o, b]]
        return min(vals) if vals else 1 << 30

    def _next_day(self, b, t, P):
        """Next candidate entry day after t: smallest t' > t such that t' or
        t' + P coincides with a committed entry/exit event in bay b (between
        such days the feasibility of the window [t', t'+P) cannot change)."""
        best = 1 << 60
        for j in self.bay_blocks[b]:
            for e in (int(self.t1[j]), int(self.t2[j])):
                if e > t and e < best:
                    best = e
                ep = e - P + 1  # day where "e crosses the window end" flips
                if ep > t and ep < best:
                    best = ep
        return best

    def best_insertion(self, i, max_days=None, xw=1.0, yw=0.001,
                       best_cost_init=math.inf, bay_noise=0.0,
                       xdir=1, ydir=1, only_bay=None, use_guide=False,
                       use_dayguide=False, blink=0.0, band=False,
                       exact_budget=None):
        """Find min-cost insertion of block i across bays/days.

        cost = w1 * tardiness + w3 * pref_penalty + w2 * delta_obj2
        (+ tiny position tie-break, kept separate from real cost).
        `blink` skips a bay candidate with the given probability (SISR-style
        recreate diversification); 0.0 reproduces historical behavior exactly.
        `band` selects a narrowed px search range: False = full range,
        True = the last side passed to enable_bands, or a side name
        ("left"/"right"). An unbuilt band silently falls back to the full
        range, so the caller never has to check.
        Returns (cost, b, t1, o, px, py) or None.
        """
        p = self.prep
        R = int(p.R[i])
        P = int(p.P[i])
        D = int(p.D[i])

        if self.lock_bay is not None:
            lb = int(self.lock_bay[i])
            if lb >= 0:
                if only_bay is not None and only_bay != lb:
                    return None  # move targets a non-locked bay: reject
                only_bay = lb

        # static per-bay part.
        #
        # This ran `self.loads.copy()` plus TWO obj2() calls per candidate bay,
        # and one of them -- obj2() of the unchanged loads -- is the same value
        # every time. obj2 is floor(max(u*L) - min(u*L)), so on m <= 5 bays a
        # plain Python max/min beats numpy, whose per-op call overhead dominates
        # at this size. Bit-exact: w[b] keeps the original's
        # u[b] * (loads[b] + wl[i]) grouping (FP multiply does not distribute),
        # every other w[j] is the identical product, and max/min/floor over the
        # same float64 values give the same result.
        m = p.m
        w = None
        if m >= 2:
            u = p.u
            L = self.loads
            w = [float(u[j]) * float(L[j]) for j in range(m)]
            base2 = math.floor(max(w) - min(w))
            wl_i = float(p.wl[i])
        stats = []
        for b in range(m):
            if only_bay is not None and b != only_bay:
                continue
            if not p.block_fits[i, b]:
                continue
            if w is None:
                d2 = 0.0
            else:
                mx = mn = float(u[b]) * (float(L[b]) + wl_i)
                for j in range(m):
                    if j == b:
                        continue
                    v = w[j]
                    if v > mx:
                        mx = v
                    if v < mn:
                        mn = v
                d2 = math.floor(mx - mn) - base2
            stat_true = p.w3 * (p.smax[i] - p.pref[i, b]) + p.w2 * d2
            stat = stat_true
            if use_guide and self.guide is not None and b != self.guide[i]:
                # bias proposals toward the relaxation-optimal assignment;
                # the bias steers the SEARCH only -- the returned cost stays
                # true so acceptance/budget accounting are unaffected
                stat = stat + self.guide_w
            if bay_noise > 0.0:
                stat = stat + self.rng.random() * bay_noise
            stats.append((stat, stat_true, b))
        stats.sort(key=lambda s: s[0])

        # resolve the band view once: None => ordinary full-range scan
        bset = None
        if band:
            bset = self._bands.get(self._band_side if band is True else band)

        best = None
        # pruning happens in BIASED units (guide/noise steer the search); the
        # caller's budget is in TRUE units, so give the prune cap headroom for
        # the maximum bias and enforce the true budget on the result instead
        bias_max = 0.0
        if use_guide and self.guide is not None:
            bias_max += self.guide_w
        if bay_noise > 0.0:
            bias_max += bay_noise
        best_cost = best_cost_init + bias_max
        o0 = int(p.orient_base[i])
        o1 = int(p.orient_base[i + 1])
        K = int(p.K[i])
        # every one of these was re-derived per candidate bay inside the loop,
        # yet none of them depends on b: with ~60 arguments per day_scan call
        # the casts alone were a measurable share of best_insertion's Python
        # time. mc0 is the min_cells0 ROW for i, so the per-bay lookup is a
        # single 1-D index instead of a 2-D one.
        P_i = int(P)
        D_i = int(D)
        w1_f = float(p.w1)
        xw_f = float(xw)
        yw_f = float(yw)
        xdir_i = int(xdir)
        ydir_i = int(ydir)
        tmax_i = int(self.tcap - P - 1)
        # Per-call override of the exact-test budget. day_scan takes it as a
        # SCALAR argument, so raising it changes nothing about the numba
        # signature and the shipped kernel cache still matches -- which is why
        # the last-chance rescan in ogc_search is affordable at all.
        eb_i = int(self.exact_budget if exact_budget is None else exact_budget)
        tau_f = float(self.tau)
        pg_f = float(self.price_gamma)
        mc0 = p.min_cells0[i]
        Gk_i = self._Gk
        Gink_i = self._Gink
        Ginbk_i = self._Ginbk
        for stat, stat_true, b in stats:
            if stat >= best_cost:
                break
            if blink > 0.0 and self.rng.random() < blink:
                continue
            # day-relaxation target: lift the scan start to the target entry
            # day in the target bay only. The proposal is CONSTRAINED, not
            # re-priced -- the returned cost stays true, so acceptance and
            # budget accounting are unaffected (same invariant as the bay
            # guide). Passing a different R value keeps the numba signature
            # unchanged (scalar argument), so the shipped kernel cache holds.
            R_b = R
            if (use_dayguide and self.tday is not None
                    and self.tday[i] >= 0 and b == int(self.tbay[i])):
                R_b = max(R, int(self.tday[i]))
            # earliest-feasible-day memo: skip days already proven dead.
            # The memo is only written from unlifted scans (proof anchored
            # at R), so applying max() here is exact for any start >= R.
            R_scan = R_b
            # a band scan searches a SUBSET of the full px range, so reading
            # the memo stays exact (no feasible full day before h => no
            # feasible band day either) but writing it does not: the band's
            # first feasible day may sit after a day the full range can use.
            efd_write = self._efd_on and R_b == R and bset is None
            if bset is not None:
                fits_b = bset[2][b]
                pxlo_b = bset[0][b]
                pxhi_b = bset[1][b]
            else:
                fits_b = self._fits_b[b]
                pxlo_b = self._pxlo_b[b]
                pxhi_b = self._pxhi_b[b]
            if self._efd_on:
                self._efd_q += 1
                if self._efd_stamp[i, b] == self._rm_cnt[b]:
                    h = int(self._efd_day[i, b])
                    if h > R_scan:
                        self._efd_hit += 1
                        self._efd_skip += h - R_scan
                        R_scan = h
            nb = len(self.bay_blocks[b])
            if self._use_bb:
                t, o, bx, by = day_scan_bb(
                    self._G[b][:K], self._Gin[b][:K],
                    self._Gbit[b][:K], self._Ginbit[b][:K],
                    nb, self._jt1[b], self._jt2[b], self._jx[b],
                    self._jy[b], self._joidx[b],
                    p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
                    p.arena_u, p.arena_v,
                    p.icell_ptr, p.icell_cnt, p.arena_iu, p.arena_iv,
                    p.rm_ptr, p.rm_umin, p.rm_vmin, p.rm_nrows, p.rm_mask,
                    o0, o1, fits_b,
                    pxlo_b, pxhi_b,
                    self._pylo_b[b], self._pyhi_b[b],
                    int(R_scan), int(P), int(D), float(stat), float(p.w1),
                    float(best_cost),
                    self.occ0[b], int(p.bayW[b] * p.bayH[b]),
                    int(p.min_cells0[i, b]),
                    float(xw), float(yw), int(xdir), int(ydir),
                    int(self.tcap - P - 1), eb_i,
                    p.tri_ptr, p.tri_cnt, p.tri_x, p.tri_y,
                    p.lbx0, p.lby0, p.lbx1, p.lby1, p.tri_ok,
                    float(self.tau))
            if self._use_bb:
                # bitboard path has no price support: reconstruct the
                # unpriced biased cost (pricing disables bb upstream)
                cb = (stat + p.w1 * max(0, t + P - D)) if t >= 0 else 1e30
            if not self._use_bb or self._bb_verify:
                mc0_b = mc0[b]
                t_c, o_c, bx_c, by_c, cb_c = day_scan(
                    Gk_i[b][K], Gink_i[b][K],
                    nb, self._jt1[b], self._jt2[b], self._jx[b],
                    self._jy[b], self._joidx[b],
                    p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
                    p.arena_u, p.arena_v,
                    p.icell_ptr, p.icell_cnt, p.arena_iu, p.arena_iv,
                    p.scan_ptr, p.scan_u, p.scan_v, p.scan_k,
                    o0, o1, fits_b,
                    pxlo_b, pxhi_b,
                    self._pylo_b[b], self._pyhi_b[b],
                    int(R_scan), P_i, D_i, float(stat), w1_f,
                    float(best_cost),
                    self._occ0b[b], self._bay_cap[b],
                    int(mc0_b),
                    xw_f, yw_f, xdir_i, ydir_i,
                    tmax_i, eb_i,
                    p.tri_ptr, p.tri_cnt, p.tri_x, p.tri_y,
                    p.lbx0, p.lby0, p.lbx1, p.lby1, p.tri_ok,
                    tau_f,
                    self.price_prefix,
                    float(pg_f * mc0_b),
                    Ginbk_i[b][K], p.run_ptr, p.run_v, p.run_u0,
                    p.run_len, self._screen)
                if not self._use_bb:
                    t, o, bx, by, cb = t_c, o_c, bx_c, by_c, cb_c
                elif (t, o, bx, by) != (t_c, o_c, bx_c, by_c):
                    raise AssertionError(
                        f"bitboard mismatch i={i} b={b}: "
                        f"bb=({t},{o},{bx},{by}) cell=({t_c},{o_c},"
                        f"{bx_c},{by_c})")
            if (self._efd_verify and R_scan > R_b and not self._use_bb):
                # differential proof check: the memo-lifted scan must agree
                # with an unlifted scan from R_b in every component
                tv, ov, xv, yv, cbv = day_scan(
                    self._G[b][:K], self._Gin[b][:K],
                    nb, self._jt1[b], self._jt2[b], self._jx[b],
                    self._jy[b], self._joidx[b],
                    p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
                    p.arena_u, p.arena_v,
                    p.icell_ptr, p.icell_cnt, p.arena_iu, p.arena_iv,
                    p.scan_ptr, p.scan_u, p.scan_v, p.scan_k,
                    o0, o1, fits_b,
                    pxlo_b, pxhi_b,
                    self._pylo_b[b], self._pyhi_b[b],
                    int(R_b), int(P), int(D), float(stat), float(p.w1),
                    float(best_cost),
                    self.occ0[b], int(p.bayW[b] * p.bayH[b]),
                    int(p.min_cells0[i, b]),
                    float(xw), float(yw), int(xdir), int(ydir),
                    int(self.tcap - P - 1), eb_i,
                    p.tri_ptr, p.tri_cnt, p.tri_x, p.tri_y,
                    p.lbx0, p.lby0, p.lbx1, p.lby1, p.tri_ok,
                    float(self.tau),
                    self.price_prefix,
                    float(self.price_gamma * p.min_cells0[i, b]),
                    self._Ginb[b][:K], p.run_ptr, p.run_v, p.run_u0,
                    p.run_len, self._screen)
                if (tv, ov, xv, yv) != (t, o, bx, by):
                    raise AssertionError(
                        f"EFD mismatch i={i} b={b}: lifted from "
                        f"{R_scan} gave ({t},{o},{bx},{by}), full scan "
                        f"from {R_b} gave ({tv},{ov},{xv},{yv})")
            if efd_write and t >= 0:
                self._efd_day[i, b] = t
                self._efd_stamp[i, b] = self._rm_cnt[b]
            if t >= 0:
                c = cb  # kernel's biased cost: stat + w1*tard (+ price)
                if c < best_cost:
                    best_cost = c
                    c_true = stat_true + p.w1 * max(0, t + P - D)
                    best = (c_true, b, t, o, bx, by)
        # note: the winner may exceed the caller's TRUE-unit budget by up to
        # the bias headroom; callers evaluate the true objective anyway (SA),
        # and hard-rejecting here measurably increased aborted moves
        return best

    def refine_position(self, i, b, t1, o, xdir=1, ydir=1, h=5,
                        max_feas=400, px0=None, py0=None, slack=1):
        """Departure-aligned position override. The objective does not
        depend on (x, y), so among feasible positions at the chosen
        (bay, day, orientation) we are free to pick the one whose
        neighbourhood attention score is highest: sit next to blocks that
        exit when we do (score max(0, h - |their_exit - mine|) per adjacent
        occupied cell, walls count h/2). Conservative feasibility (outer
        raster): if no conservatively-free position exists the caller keeps
        the exact-refined one. Returns (px, py) or None."""
        if not self._exitpack:
            return None
        from _ogc.ogc_kernels import scan_exit_contact
        p = self.prep
        P = int(p.P[i])
        t2 = t1 + P
        K = int(p.K[i])
        oidx = int(p.orient_base[i] + o)
        if not p.fits[oidx, b]:
            return None

        G = self._G[b]
        G[:K, :, :] = 0
        nd = 0
        dx, dy, doidx, dflags = self._dx, self._dy, self._doidx, self._dflags
        for j in self.bay_blocks[b]:
            e1 = self.t1[j]
            e2 = self.t2[j]
            if e1 >= t2 or e2 <= t1:
                continue
            fl = 0
            if (e1 <= t1 < e2) or (e1 < t2 <= e2):
                fl |= 1
            if (t1 < e1 < t2) or (t1 < e2 < t2):
                fl |= 2
            if fl:
                dx[nd] = self.px[j]
                dy[nd] = self.py[j]
                doidx[nd] = p.orient_base[j] + self.oi[j]
                dflags[nd] = fl
                nd += 1
        if nd:
            build_G(G[:K], nd, dx, dy, doidx, dflags,
                    p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
                    p.arena_u, p.arena_v)

        s0, s1 = int(p.scan_ptr[oidx]), int(p.scan_ptr[oidx + 1])
        r0 = int(p.layer_base[oidx])
        c0 = int(p.cell_ptr[r0])
        cn = int(p.cell_cnt[r0])
        pxlo = int(p.pxlo[oidx, b])
        pxhi = int(p.pxhi[oidx, b])
        pylo = int(p.pylo[oidx, b])
        pyhi = int(p.pyhi[oidx, b])
        # tightness cap: never move further from the anchored corner than
        # the exact-refined original position plus a small slack -- the
        # attention only chooses among equally-dense placements
        # per-axis tightness caps: any combined metric leaks (a high-y
        # original licenses large x-drift and vice versa -- measured +8%
        # construct tardiness). The candidate box is the original position
        # plus `slack` cells on each axis, no trades.
        cap_a = 10 ** 9
        cap_b = 10 ** 9
        if px0 is not None:
            cap_a = ((px0 - pxlo) if xdir > 0 else (pxhi - px0)) + int(slack)
            cap_b = ((py0 - pylo) if ydir > 0 else (pyhi - py0)) + int(slack)
        px, py, score = scan_exit_contact(
            G[:K], self._E[b],
            p.scan_u[s0:s1], p.scan_v[s0:s1], p.scan_k[s0:s1],
            p.arena_u[c0:c0 + cn], p.arena_v[c0:c0 + cn],
            pxlo, pxhi, pylo, pyhi,
            int(t2), int(h), 1.0, 0.001, int(xdir), int(ydir),
            int(max_feas), int(cap_a), int(cap_b))
        if px < 0:
            return None
        return int(px), int(py)

    def refine_position_exact(self, i, b, t1, o, px0, py0):
        """Exact-feasibility attention refinement: probe the +-1/+-2 box
        around day_scan's chosen position with can_place_exact (the SAME
        exact test placements use, so unlike the conservative-raster
        variant this cannot lose exact-band density), score positions by
        departure-affinity attention, keep the original on ties.
        Returns (px, py) -- possibly the original."""
        from _ogc.ogc_kernels import attention_score
        p = self.prep
        P = int(p.P[i])
        t2 = t1 + P
        oidx = int(p.orient_base[i] + o)
        r0 = int(p.layer_base[oidx])
        c0 = int(p.cell_ptr[r0])
        cn = int(p.cell_cnt[r0])
        c0u = p.arena_u[c0:c0 + cn]
        c0v = p.arena_v[c0:c0 + cn]
        E = self._E[b]
        best = (px0, py0)
        best_sc = attention_score(E, c0u, c0v, int(px0), int(py0),
                                  int(t2), 5)
        pxlo = int(p.pxlo[oidx, b])
        pxhi = int(p.pxhi[oidx, b])
        pylo = int(p.pylo[oidx, b])
        pyhi = int(p.pyhi[oidx, b])
        for dxo in (-1, 0, 1):
            for dyo in (-2, -1, 0, 1, 2):
                if dxo == 0 and dyo == 0:
                    continue
                px = px0 + dxo
                py = py0 + dyo
                if not (pxlo <= px <= pxhi and pylo <= py <= pyhi):
                    continue
                sc = attention_score(E, c0u, c0v, px, py, int(t2), 5)
                if sc <= best_sc + 1e-9:
                    continue
                if self.can_place_exact(i, b, px, py, o, t1, t2):
                    best = (px, py)
                    best_sc = sc
        return int(best[0]), int(best[1])

    def refine_position_sliver(self, i, b, t1, o, px0, py0):
        """Sliver-avoiding tie-break (side-view projection metric): among
        the exact-feasible +-1/+-2 box around day_scan's position, prefer
        the one creating the fewest dead slivers -- free runs narrower than
        the narrowest remaining block width. Keeps the original on ties."""
        from _ogc.ogc_kernels import build_G, sliver_score
        p = self.prep
        P = int(p.P[i])
        t2 = t1 + P
        K = int(p.K[i])
        oidx = int(p.orient_base[i] + o)
        G = self._G[b]
        G[:K, :, :] = 0
        nd = 0
        dx, dy, doidx, dflags = (self._dx, self._dy, self._doidx,
                                 self._dflags)
        for j in self.bay_blocks[b]:
            e1 = self.t1[j]
            e2 = self.t2[j]
            if e1 >= t2 or e2 <= t1:
                continue
            fl = 0
            if (e1 <= t1 < e2) or (e1 < t2 <= e2):
                fl |= 1
            if (t1 < e1 < t2) or (t1 < e2 < t2):
                fl |= 2
            if fl:
                dx[nd] = self.px[j]
                dy[nd] = self.py[j]
                doidx[nd] = p.orient_base[j] + self.oi[j]
                dflags[nd] = fl
                nd += 1
        if nd:
            build_G(G[:K], nd, dx, dy, doidx, dflags,
                    p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
                    p.arena_u, p.arena_v)
        G0 = G[0]
        r0 = int(p.layer_base[oidx])
        c0 = int(p.cell_ptr[r0])
        cn = int(p.cell_cnt[r0])
        c0u = p.arena_u[c0:c0 + cn]
        c0v = p.arena_v[c0:c0 + cn]
        best = (int(px0), int(py0))
        best_sc = sliver_score(G0, c0u, c0v, int(px0), int(py0),
                               self._wmin, self._hmin)
        pxlo = int(p.pxlo[oidx, b])
        pxhi = int(p.pxhi[oidx, b])
        pylo = int(p.pylo[oidx, b])
        pyhi = int(p.pyhi[oidx, b])
        for dxo in (-1, 0, 1):
            for dyo in (-2, -1, 0, 1, 2):
                if dxo == 0 and dyo == 0:
                    continue
                px = px0 + dxo
                py = py0 + dyo
                if not (pxlo <= px <= pxhi and pylo <= py <= pyhi):
                    continue
                sc = sliver_score(G0, c0u, c0v, px, py, self._wmin,
                                  self._hmin)
                if sc <= best_sc + 1e-9:
                    continue
                if self.can_place_exact(i, b, px, py, o, t1, t2):
                    best = (px, py)
                    best_sc = sc
        return best

    def scan_from_day(self, i, b, t_from, xw=1.0, yw=0.001,
                      xdir=1, ydir=1):
        """Earliest exact-refined placement of block i in bay b at a day
        >= t_from: full day_scan quality (three-valued raster + budgeted
        exact triangles) anchored at an arbitrary start day. Used by the
        DP-by-parts joint repack, where the conservative try_day measurably
        cannot re-place blocks inside snug packings.
        Returns (t, o, px, py) or None."""
        p = self.prep
        P = int(p.P[i])
        D = int(p.D[i])
        K = int(p.K[i])
        o0 = int(p.orient_base[i])
        o1 = int(p.orient_base[i + 1])
        nb = len(self.bay_blocks[b])
        t, o, bx, by, _cb = day_scan(
            self._G[b][:K], self._Gin[b][:K],
            nb, self._jt1[b], self._jt2[b], self._jx[b],
            self._jy[b], self._joidx[b],
            p.layer_base, p.layer_cnt, p.cell_ptr, p.cell_cnt,
            p.arena_u, p.arena_v,
            p.icell_ptr, p.icell_cnt, p.arena_iu, p.arena_iv,
            p.scan_ptr, p.scan_u, p.scan_v, p.scan_k,
            o0, o1, self._fits_b[b],
            self._pxlo_b[b], self._pxhi_b[b],
            self._pylo_b[b], self._pyhi_b[b],
            int(t_from), int(P), int(D), 0.0, float(p.w1),
            math.inf,
            self.occ0[b], int(p.bayW[b] * p.bayH[b]),
            int(p.min_cells0[i, b]),
            float(xw), float(yw), int(xdir), int(ydir),
            int(self.tcap - P - 1), int(self.exact_budget),
            p.tri_ptr, p.tri_cnt, p.tri_x, p.tri_y,
            p.lbx0, p.lby0, p.lbx1, p.lby1, p.tri_ok,
            float(self.tau),
            self.price_prefix, 0.0,
            self._Ginb[b][:K], p.run_ptr, p.run_v, p.run_u0,
            p.run_len, self._screen)
        if t < 0:
            return None
        return int(t), int(o), int(bx), int(by)

    def can_place_exact(self, i, b, x, y, o, t1, t2):
        """Exact feasibility of a specific placement (position AND window)
        against the current committed state. Used by moves that bypass the
        normal day_scan search (e.g. exit-delay nesting)."""
        from _ogc.ogc_kernels import exact_position_ok
        p = self.prep
        oidx = int(p.orient_base[i] + o)
        if not p.fits[oidx, b]:
            return False
        if not (p.pxlo[oidx, b] <= x <= p.pxhi[oidx, b]
                and p.pylo[oidx, b] <= y <= p.pyhi[oidx, b]):
            return False
        nb = len(self.bay_blocks[b])
        return bool(exact_position_ok(
            int(x), int(y), int(p.layer_base[oidx]), int(p.layer_cnt[oidx]),
            nb, self._jt1[b], self._jt2[b], self._jx[b], self._jy[b],
            self._joidx[b], int(t1), int(t2),
            p.layer_base, p.layer_cnt,
            p.tri_ptr, p.tri_cnt, p.tri_x, p.tri_y,
            p.lbx0, p.lby0, p.lbx1, p.lby1, p.tri_ok, float(self.tau)))

    # ------------------------------------------------------- insert / remove --
    def insert(self, i, b, t1, o, px, py, seq=None, t2=None):
        p = self.prep
        if t2 is None:
            t2 = t1 + int(p.P[i])
        self.bay[i] = b
        self.px[i] = px
        self.py[i] = py
        self.oi[i] = o
        self.t1[i] = t1
        self.t2[i] = t2
        if seq is None:
            self.seq[i] = self.seq_counter
            self.seq_counter += 1
        else:
            # revert path: restore the original commit order exactly, so the
            # same-day operation ordering all other blocks were validated
            # against is preserved
            self.seq[i] = seq
        self.placed[i] = True
        q = len(self.bay_blocks[b])
        self.bay_blocks[b].append(i)
        oidx = p.orient_base[i] + o
        self._jid[b][q] = i
        self._jt1[b][q] = t1
        self._jt2[b][q] = t2
        self._jx[b][q] = px
        self._jy[b][q] = py
        self._joidx[b][q] = oidx
        self.loads[b] += p.wl[i]
        self.sum_tard += max(0, t2 - p.D[i])
        self.sum_pref += p.smax[i] - p.pref[i, b]
        self.occ0[b, t1:min(t2, self.tcap)] += p.ncells0[oidx]
        if self._exitpack:
            from _ogc.ogc_kernels import stamp_exit
            r0 = int(p.layer_base[oidx])
            c0 = int(p.cell_ptr[r0])
            cn = int(p.cell_cnt[r0])
            stamp_exit(self._E[b], p.arena_u[c0:c0 + cn],
                       p.arena_v[c0:c0 + cn], int(px), int(py),
                       int(t2) + 1)

    def remove(self, i):
        p = self.prep
        b = int(self.bay[i])
        if self._exitpack:
            from _ogc.ogc_kernels import stamp_exit
            oidx0 = int(p.orient_base[i] + self.oi[i])
            r0 = int(p.layer_base[oidx0])
            c0 = int(p.cell_ptr[r0])
            cn = int(p.cell_cnt[r0])
            stamp_exit(self._E[b], p.arena_u[c0:c0 + cn],
                       p.arena_v[c0:c0 + cn], int(self.px[i]),
                       int(self.py[i]), 0)
        self._rm_cnt[b] += 1  # removals can open earlier days: expire memos
        lst = self.bay_blocks[b]
        q = lst.index(i)
        last = len(lst) - 1
        if q != last:
            lst[q] = lst[last]
            self._jid[b][q] = self._jid[b][last]
            self._jt1[b][q] = self._jt1[b][last]
            self._jt2[b][q] = self._jt2[b][last]
            self._jx[b][q] = self._jx[b][last]
            self._jy[b][q] = self._jy[b][last]
            self._joidx[b][q] = self._joidx[b][last]
        lst.pop()
        self.loads[b] -= p.wl[i]
        self.sum_tard -= max(0, int(self.t2[i]) - p.D[i])
        self.sum_pref -= p.smax[i] - p.pref[i, b]
        oidx = p.orient_base[i] + self.oi[i]
        self.occ0[b, self.t1[i]:min(int(self.t2[i]), self.tcap)] -= p.ncells0[oidx]
        self.placed[i] = False
        self.bay[i] = -1

    def snapshot(self):
        return (self.bay.copy(), self.px.copy(), self.py.copy(), self.oi.copy(),
                self.t1.copy(), self.t2.copy(), self.seq.copy(),
                self.placed.copy(), self.seq_counter,
                self.loads.copy(), self.sum_tard, self.sum_pref)

    def restore(self, snap):
        self._rm_cnt += 1  # wholesale state swap: expire all day memos
        (bay, px, py, oi, t1, t2, seq, placed, seq_counter,
         loads, sum_tard, sum_pref) = snap
        self.bay = bay.copy()
        self.px = px.copy()
        self.py = py.copy()
        self.oi = oi.copy()
        self.t1 = t1.copy()
        self.t2 = t2.copy()
        self.seq = seq.copy()
        self.placed = placed.copy()
        self.seq_counter = seq_counter
        self.loads = loads.copy()
        self.sum_tard = sum_tard
        self.sum_pref = sum_pref
        p = self.prep
        self.bay_blocks = [[] for _ in range(p.m)]
        self.occ0[:] = 0
        for i in range(p.n):
            if self.placed[i]:
                b = int(self.bay[i])
                q = len(self.bay_blocks[b])
                self.bay_blocks[b].append(i)
                oidx = p.orient_base[i] + self.oi[i]
                self._jid[b][q] = i
                self._jt1[b][q] = self.t1[i]
                self._jt2[b][q] = self.t2[i]
                self._jx[b][q] = self.px[i]
                self._jy[b][q] = self.py[i]
                self._joidx[b][q] = oidx
                self.occ0[b, self.t1[i]:min(int(self.t2[i]), self.tcap)] += p.ncells0[oidx]
        if self._exitpack:
            from _ogc.ogc_kernels import stamp_exit
            for b in range(p.m):
                self._E[b][:, :] = 0
            for i in range(p.n):
                if self.placed[i]:
                    b = int(self.bay[i])
                    oidx = int(p.orient_base[i] + self.oi[i])
                    r0 = int(p.layer_base[oidx])
                    c0 = int(p.cell_ptr[r0])
                    cn = int(p.cell_cnt[r0])
                    stamp_exit(self._E[b], p.arena_u[c0:c0 + cn],
                               p.arena_v[c0:c0 + cn], int(self.px[i]),
                               int(self.py[i]), int(self.t2[i]) + 1)

    # ---------------------------------------------------------------- output --
    def build_operations(self):
        """Solution dict. Within a day: EXITs first (commit-seq descending),
        then ENTRYs (commit-seq ascending)."""
        days = {}
        for i in range(self.prep.n):
            if not self.placed[i]:
                continue
            days.setdefault(int(self.t2[i]), [[], []])[0].append(i)
            days.setdefault(int(self.t1[i]), [[], []])[1].append(i)
        ops = {}
        for t in sorted(days):
            exits, entries = days[t]
            exits.sort(key=lambda i: -self.seq[i])
            entries.sort(key=lambda i: self.seq[i])
            lst = []
            for i in exits:
                lst.append({"type": "EXIT", "block_id": int(i),
                            "bay_id": int(self.bay[i])})
            for i in entries:
                lst.append({"type": "ENTRY", "block_id": int(i),
                            "bay_id": int(self.bay[i]),
                            "x": int(self.px[i]), "y": int(self.py[i]),
                            "orient_idx": int(self.oi[i])})
            ops[str(t)] = lst
        return {"operations": ops}
