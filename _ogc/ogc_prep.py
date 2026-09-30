# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Instance preprocessing: parse blocks/bays, build conservative integer-grid
rasters for every (block, orientation, layer), and pack them into flat arenas
for the numba kernels.

Raster semantics: cell (u, v) is marked iff the layer polygon overlaps the
unit cell [u, u+1] x [v, v+1] expanded by a tiny pad (1e-6). Placements whose
marked cells are disjoint are guaranteed collision-free under the official
checker's exact Shapely math (positive-area overlap implies a shared marked
cell).
"""

import math

import numpy as np

from _ogc.ogc_kernels import raster_poly, raster_poly_inner

PAD = 1e-6
AREA_EPS = 1e-12


def _triangulate(xs, ys):
    """Ear-clipping triangulation of a simple polygon.

    Returns (flat_tri_xs, flat_tri_ys) with 3 vertices per triangle, or None
    if triangulation fails (caller then disables the exact path for this
    layer, which is safe/conservative)."""
    pts = list(zip(xs.tolist(), ys.tolist()))
    # drop consecutive duplicates
    ded = []
    for pt in pts:
        if not ded or (abs(pt[0] - ded[-1][0]) > 1e-12 or
                       abs(pt[1] - ded[-1][1]) > 1e-12):
            ded.append(pt)
    if len(ded) > 1 and (abs(ded[0][0] - ded[-1][0]) < 1e-12 and
                         abs(ded[0][1] - ded[-1][1]) < 1e-12):
        ded.pop()
    if len(ded) < 3:
        return None
    # signed area; ensure CCW
    sa = 0.0
    for i in range(len(ded)):
        x1, y1 = ded[i]
        x2, y2 = ded[(i + 1) % len(ded)]
        sa += x1 * y2 - x2 * y1
    poly_area = abs(sa) * 0.5
    if sa < 0:
        ded.reverse()

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1]) -
                (a[1] - o[1]) * (b[0] - o[0]))

    def in_tri(p, a, b, c):
        # inside or on boundary (with tolerance): used to REJECT ears, so
        # inclusive is the safe direction
        d1 = cross(a, b, p)
        d2 = cross(b, c, p)
        d3 = cross(c, a, p)
        eps = 1e-12
        return d1 >= -eps and d2 >= -eps and d3 >= -eps

    idx = list(range(len(ded)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        n = len(idx)
        found = False
        for ii in range(n):
            a = ded[idx[(ii - 1) % n]]
            b = ded[idx[ii]]
            c = ded[idx[(ii + 1) % n]]
            cr = cross(a, b, c)
            if cr <= 1e-12:
                continue  # reflex or degenerate corner
            ok = True
            for jj in range(n):
                if jj in ((ii - 1) % n, ii, (ii + 1) % n):
                    continue
                if in_tri(ded[idx[jj]], a, b, c):
                    ok = False
                    break
            if ok:
                tris.append((a, b, c))
                idx.pop(ii)
                found = True
                break
        if not found:
            return None
    if len(idx) == 3:
        a, b, c = (ded[idx[0]], ded[idx[1]], ded[idx[2]])
        if cross(a, b, c) > 1e-12:
            tris.append((a, b, c))
    # verify area
    ta = sum(abs(cross(a, b, c)) * 0.5 for a, b, c in tris)
    if abs(ta - poly_area) > 1e-6 * max(1.0, poly_area):
        return None
    txs = np.empty(3 * len(tris))
    tys = np.empty(3 * len(tris))
    for t, (a, b, c) in enumerate(tris):
        txs[3 * t] = a[0]
        tys[3 * t] = a[1]
        txs[3 * t + 1] = b[0]
        tys[3 * t + 1] = b[1]
        txs[3 * t + 2] = c[0]
        tys[3 * t + 2] = c[1]
    return txs, tys


class Prep:
    def __init__(self, prob):
        self.name = prob.get("name", "?")
        bays = prob["bays"]
        blocks = prob["blocks"]
        self.m = len(bays)
        self.n = len(blocks)
        self.bayW = np.array([int(b["width"]) for b in bays], np.int64)
        self.bayH = np.array([int(b["height"]) for b in bays], np.int64)
        areas = (self.bayW * self.bayH).astype(np.float64)
        self.u = areas.mean() / areas  # obj2 bay weights

        w = prob.get("weights", {}) or {}
        self.w1 = float(w.get("w1", 1.0))
        self.w2 = float(w.get("w2", 1.0))
        self.w3 = float(w.get("w3", 1.0))

        n = self.n
        self.R = np.array([int(b["release_time"]) for b in blocks], np.int64)
        self.D = np.array([int(b["due_date"]) for b in blocks], np.int64)
        self.P = np.array([max(1, int(b["processing_time"])) for b in blocks], np.int64)
        # Dense day-indexed acceleration arrays need to cover not just the
        # release horizon but the guaranteed empty-bay fallback schedule. If
        # every block were serialized in one bay, it completes no later than
        # max(R) + sum(P). The old hard-coded 8192 truncated legal later dates.
        max_r = int(self.R.max()) if n else 0
        sum_p = int(self.P.sum()) if n else 0
        max_p = int(self.P.max()) if n else 1
        self.tcap = max(8192, max_r + sum_p + max_p + 16)
        self.wl = np.array([float(b["workload"]) for b in blocks], np.float64)
        self.pref = np.array([b["bay_preferences"] for b in blocks], np.float64)
        self.smax = self.pref.max(axis=1)

        # ---- shapes -> rasters ------------------------------------------------
        # global orientation indexing: oidx = orient_base[i] + o
        self.n_orient = np.array([len(b["shape"]) for b in blocks], np.int64)
        self.orient_base = np.zeros(n + 1, np.int64)
        np.cumsum(self.n_orient, out=self.orient_base[1:])
        nO = int(self.orient_base[-1])

        self.K = np.zeros(n, np.int64)  # layer count per block
        for i, b in enumerate(blocks):
            # empty layers are dropped, matching the checker's _resolve_layers
            self.K[i] = max(sum(1 for v in s["layers"] if v)
                            for s in b["shape"])

        # per global orient: float bbox over all layers (for containment bounds)
        self.fbx0 = np.zeros(nO)
        self.fby0 = np.zeros(nO)
        self.fbx1 = np.zeros(nO)
        self.fby1 = np.zeros(nO)

        # layer table: row = layer_base[oidx] + l
        self.layer_base = np.zeros(nO, np.int64)
        self.layer_cnt = np.zeros(nO, np.int64)

        cell_ptr = []   # per layer row: start into arena
        cell_cnt = []
        au = []         # arena u coords (per cell)
        av = []
        icell_ptr = []  # inner-cell arena (cells fully covered by the layer)
        icell_cnt = []
        aiu = []
        aiv = []
        # per layer row: float bbox and triangulation
        lb = []         # (x0, y0, x1, y1)
        tri_list = []   # per row: (n_tri, flat xs, flat ys) or None
        # scan arrays: per oidx concatenated cells of all layers with layer ids
        scan_ptr = np.zeros(nO + 1, np.int64)
        scu = []
        scv = []
        sck = []
        self.ncells0 = np.zeros(nO, np.int64)  # layer-0 cell count per orient
        # row masks per layer row for the bitboard position test: bits along
        # u (relative to rm_umin), one 3-word mask per v-row
        rm_ptr_l = []
        rm_umin_l = []
        rm_vmin_l = []
        rm_nrows_l = []
        rm_rows = []
        rm_pos = 0
        bb_ok = True

        shapely_poly = None  # lazy import for the invalid-polygon fallback

        row = 0
        arena_pos = 0
        for i, b in enumerate(blocks):
            for o, s in enumerate(b["shape"]):
                oidx = self.orient_base[i] + o
                self.layer_base[oidx] = row
                layers = [v for v in s["layers"] if v]  # _resolve_layers semantics
                self.layer_cnt[oidx] = len(layers)
                minx = miny = math.inf
                maxx = maxy = -math.inf
                orient_cells = 0
                for l, verts in enumerate(layers):
                    xs = np.array([v[0] for v in verts], np.float64)
                    ys = np.array([v[1] for v in verts], np.float64)
                    minx = min(minx, xs.min())
                    maxx = max(maxx, xs.max())
                    miny = min(miny, ys.min())
                    maxy = max(maxy, ys.max())
                    u0 = math.floor(xs.min())
                    v0 = math.floor(ys.min())
                    nu = max(math.ceil(xs.max()) - u0, 1)
                    nv = max(math.ceil(ys.max()) - v0, 1)
                    # validity guard: fall back to shapely for weird polygons
                    simple = _is_probably_simple(xs, ys)
                    if simple:
                        grid = raster_poly(xs, ys, u0, v0, nu, nv, PAD, AREA_EPS)
                        igrid = raster_poly_inner(xs, ys, u0, v0, nu, nv)
                    else:
                        if shapely_poly is None:
                            import shapely
                            shapely_poly = shapely
                        grid = _raster_shapely(shapely_poly, verts, u0, v0, nu, nv)
                        igrid = np.zeros_like(grid)
                    uu, vv = np.nonzero(grid)
                    uu = (uu + u0).astype(np.int64)
                    vv = (vv + v0).astype(np.int64)
                    cell_ptr.append(arena_pos)
                    cell_cnt.append(len(uu))
                    arena_pos += len(uu)
                    orient_cells += len(uu)
                    au.append(uu)
                    av.append(vv)
                    iu, iv = np.nonzero(igrid)
                    icell_ptr.append(sum(map(len, aiu)) if False else 0)
                    aiu.append((iu + u0).astype(np.int64))
                    aiv.append((iv + v0).astype(np.int64))
                    icell_cnt.append(len(iu))
                    lb.append((xs.min(), ys.min(), xs.max(), ys.max()))
                    tri_list.append(_triangulate(xs, ys) if simple else None)
                    scu.append(uu)
                    scv.append(vv)
                    sck.append(np.full(len(uu), l, np.int64))
                    # row masks for this layer row
                    if len(uu):
                        um = int(uu.min())
                        vm = int(vv.min())
                        nr = int(vv.max()) - vm + 1
                        bits = uu - um
                        if int(bits.max()) > 191:
                            bb_ok = False  # layer wider than 3 words
                            nr = 0
                        else:
                            mrows = np.zeros((nr, 3), np.uint64)
                            np.bitwise_or.at(
                                mrows, (vv - vm, bits >> 6),
                                np.uint64(1) << (bits & 63).astype(np.uint64))
                            rm_rows.append(mrows)
                    else:
                        um = 0
                        vm = 0
                        nr = 0
                    rm_ptr_l.append(rm_pos)
                    rm_umin_l.append(um)
                    rm_vmin_l.append(vm)
                    rm_nrows_l.append(nr)
                    rm_pos += nr
                    if l == 0:
                        self.ncells0[oidx] = len(uu)
                    row += 1
                self.fbx0[oidx] = minx
                self.fby0[oidx] = miny
                self.fbx1[oidx] = maxx
                self.fby1[oidx] = maxy
                scan_ptr[oidx + 1] = scan_ptr[oidx] + orient_cells

        self.cell_ptr = np.array(cell_ptr, np.int64)
        self.cell_cnt = np.array(cell_cnt, np.int64)
        self.arena_u = np.concatenate(au) if au else np.zeros(0, np.int64)
        self.arena_v = np.concatenate(av) if av else np.zeros(0, np.int64)
        # inner arena (fix the running pointers)
        pos = 0
        for r in range(len(icell_cnt)):
            icell_ptr[r] = pos
            pos += icell_cnt[r]
        self.icell_ptr = np.array(icell_ptr, np.int64)
        self.icell_cnt = np.array(icell_cnt, np.int64)
        self.arena_iu = np.concatenate(aiu) if aiu else np.zeros(0, np.int64)
        self.arena_iv = np.concatenate(aiv) if aiv else np.zeros(0, np.int64)
        # per-layer-row float bboxes
        n_rows = len(lb)
        self.lbx0 = np.array([b[0] for b in lb])
        self.lby0 = np.array([b[1] for b in lb])
        self.lbx1 = np.array([b[2] for b in lb])
        self.lby1 = np.array([b[3] for b in lb])
        # triangle arena
        self.tri_ptr = np.zeros(n_rows, np.int64)
        self.tri_cnt = np.zeros(n_rows, np.int64)
        self.tri_ok = np.zeros(n_rows, np.uint8)
        tx = []
        ty = []
        tpos = 0
        for r, tri in enumerate(tri_list):
            self.tri_ptr[r] = tpos
            if tri is not None:
                txs, tys = tri
                ntri = len(txs) // 3
                self.tri_cnt[r] = ntri
                self.tri_ok[r] = 1
                tx.append(txs)
                ty.append(tys)
                tpos += ntri
        self.tri_x = np.concatenate(tx) if tx else np.zeros(0)
        self.tri_y = np.concatenate(ty) if ty else np.zeros(0)
        # ---- horizontal RUN decomposition of every layer row ---------------
        # A layer row is a set of unit cells; along each grid row v they form a
        # handful of contiguous u-runs (measured mean 29.3 runs per orientation
        # against 303 cells). The run form is what lets the placement scan test
        # ALL px for a given py at once: a run of length L at offset u0 blocks
        # exactly the px in (dilate(occ_row, L) >> u0), and dilate is O(log L)
        # 192-bit shift-ORs by doubling. See day_scan_runs.
        run_row_ptr = np.zeros(n_rows + 1, np.int64)
        r_v = []
        r_u0 = []
        r_len = []
        for row in range(n_rows):
            c0 = int(cell_ptr[row])
            cn = int(cell_cnt[row])
            if cn:
                uu = self.arena_u[c0:c0 + cn]
                vv = self.arena_v[c0:c0 + cn]
                order = np.lexsort((uu, vv))     # by v, then u
                us = uu[order]
                vs = vv[order]
                k = 0
                while k < cn:
                    j = k + 1
                    while (j < cn and vs[j] == vs[k]
                           and us[j] == us[j - 1] + 1):
                        j += 1
                    r_v.append(int(vs[k]))
                    r_u0.append(int(us[k]))
                    r_len.append(j - k)
                    k = j
            run_row_ptr[row + 1] = len(r_v)
        self.run_ptr = run_row_ptr
        self.run_v = np.array(r_v, np.int64) if r_v else np.zeros(0, np.int64)
        self.run_u0 = np.array(r_u0, np.int64) if r_u0 else np.zeros(0, np.int64)
        self.run_len = (np.array(r_len, np.int64) if r_len
                        else np.zeros(0, np.int64))
        # runs are only usable when every layer fits the 192-bit board
        self.runs_ok = bool(len(r_u0) == 0
                            or (int(self.run_u0.min()) >= -192
                                and int((self.run_u0 + self.run_len).max())
                                <= 192))

        self.scan_ptr = scan_ptr
        self.scan_u = np.concatenate(scu) if scu else np.zeros(0, np.int64)
        self.scan_v = np.concatenate(scv) if scv else np.zeros(0, np.int64)
        self.scan_k = np.concatenate(sck) if sck else np.zeros(0, np.int64)
        self.rm_ptr = np.array(rm_ptr_l, np.int64)
        self.rm_umin = np.array(rm_umin_l, np.int64)
        self.rm_vmin = np.array(rm_vmin_l, np.int64)
        self.rm_nrows = np.array(rm_nrows_l, np.int64)
        self.rm_mask = (np.ascontiguousarray(np.concatenate(rm_rows))
                        if rm_rows else np.zeros((0, 3), np.uint64))
        self.bb_ok = bb_ok

        # position bounds per (oidx, bay): px in [pxlo, pxhi], py in [pylo, pyhi]
        self.pxlo = np.zeros((nO, self.m), np.int64)
        self.pxhi = np.zeros((nO, self.m), np.int64)
        self.pylo = np.zeros((nO, self.m), np.int64)
        self.pyhi = np.zeros((nO, self.m), np.int64)
        for oidx in range(nO):
            for b in range(self.m):
                self.pxlo[oidx, b] = math.ceil(-self.fbx0[oidx])
                self.pxhi[oidx, b] = math.floor(self.bayW[b] - self.fbx1[oidx])
                self.pylo[oidx, b] = math.ceil(-self.fby0[oidx])
                self.pyhi[oidx, b] = math.floor(self.bayH[b] - self.fby1[oidx])
        self.fits = (self.pxlo <= self.pxhi) & (self.pylo <= self.pyhi)  # (nO, m)

        # block fits bay at all (any orientation)
        self.block_fits = np.zeros((n, self.m), bool)
        self.min_cells0 = np.full((n, self.m), 1 << 30, np.int64)
        for i in range(n):
            o0, o1 = self.orient_base[i], self.orient_base[i + 1]
            self.block_fits[i] = self.fits[o0:o1].any(axis=0)
            for b in range(self.m):
                vals = [self.ncells0[o] for o in range(o0, o1)
                        if self.fits[o, b]]
                if vals:
                    self.min_cells0[i, b] = min(vals)

        # structural lower bound on any insertion's tard+pref cost: a block
        # released too late to make its due date pays w1*(R+P-D) no matter
        # where it lands, plus the cheapest reachable preference penalty
        minpref = np.zeros(n)
        for i in range(n):
            pens = [self.smax[i] - self.pref[i, b] for b in range(self.m)
                    if self.block_fits[i, b]]
            minpref[i] = min(pens) if pens else 0.0
        self.lbc = (self.w1 * np.maximum(0, self.R + self.P - self.D)
                    + self.w3 * minpref)

        self.nO = nO


def _is_probably_simple(xs, ys):
    """Cheap self-intersection screen: O(n^2) segment crossing test."""
    n = len(xs)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            x3, y3 = xs[j], ys[j]
            x4, y4 = xs[(j + 1) % n], ys[(j + 1) % n]
            d1 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d2 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
            d3 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            d4 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return False
    return True


def _raster_shapely(shapely, verts, u0, v0, nu, nv):
    """Fallback exact-conservative raster via shapely (handles invalid polys
    the same way utils.py does: buffer(0) repair)."""
    poly = shapely.Polygon(verts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    grid = np.zeros((nu, nv), np.uint8)
    for a in range(nu):
        for b in range(nv):
            cell = shapely.box(u0 + a - PAD, v0 + b - PAD,
                               u0 + a + 1 + PAD, v0 + b + 1 + PAD)
            inter = poly.intersection(cell)
            if not inter.is_empty and inter.area > AREA_EPS:
                grid[a, b] = 1
    return grid
