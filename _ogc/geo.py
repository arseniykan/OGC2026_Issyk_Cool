# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""
geo.py — exact, integer-scaled geometry core for OGC 2026.

Design principles (see ultimate_solver_guide.md sections 4 & 12A):
  * All polygon coordinates are scaled by SCALE=10000 and stored as int64.
    Verified on the data: every vertex has <=4 decimals, so scaling is EXACT
    (no floating-point error anywhere in the hot path).
  * Placement (x, y) are integers in ORIGINAL bay units; translation by a
    placement is (x*SCALE, y*SCALE) — still exact integers.
  * Non-convex layers are decomposed ONCE into triangles (ear clipping).
    Two polygons' interiors intersect  <=>  some triangle of one strictly
    overlaps some triangle of the other (Separating Axis Theorem).
  * "Interiors intersect" = positive-area overlap. Touching edges / shared
    boundaries are NOT collisions (spec allows them). SAT encodes this by
    requiring STRICT overlap on every candidate axis.

The three predicates the rest of the solver needs:
  * contained_in_bay  — every layer inside [0,W]x[0,H] (rectangle => vertex test, exact)
  * same_level_collision — common-level interior intersection between two blocks
  * crane_blocked     — can a block be lifted straight up given neighbours (l1<=l2)

This module is the LOCAL ground truth; it must match the grader's shapely-based
check exactly on the touching-edge convention.
"""

import numpy as np

SCALE = 10000

# --------------------------------------------------------------------------
# Optional numba acceleration. Falls back to pure Python if unavailable.
# --------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        # no-op decorator supporting both @njit and @njit(...)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(f):
            return f
        return wrap


# ==========================================================================
# Polygon preprocessing (runs once at load time, NOT on the hot path)
# ==========================================================================

def scale_poly(verts):
    """List[[x,y]] float -> (k,2) int64 array, scaled and dedup of consecutive."""
    pts = []
    for (x, y) in verts:
        ix = int(round(x * SCALE))
        iy = int(round(y * SCALE))
        if pts and pts[-1] == (ix, iy):
            continue
        pts.append((ix, iy))
    # drop closing duplicate if present
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts.pop()
    return np.array(pts, dtype=np.int64) if pts else np.zeros((0, 2), dtype=np.int64)


def signed_area2(poly):
    """Twice the signed area (int). Positive => CCW."""
    n = len(poly)
    a = 0
    for i in range(n):
        x1, y1 = int(poly[i, 0]), int(poly[i, 1])
        x2, y2 = int(poly[(i + 1) % n, 0]), int(poly[(i + 1) % n, 1])
        a += x1 * y2 - x2 * y1
    return a


def poly_bbox(poly):
    """(minx,miny,maxx,maxy) int of a (k,2) array."""
    if len(poly) == 0:
        return (0, 0, 0, 0)
    return (int(poly[:, 0].min()), int(poly[:, 1].min()),
            int(poly[:, 0].max()), int(poly[:, 1].max()))


def _cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _point_in_tri_strict(px, py, ax, ay, bx, by, cx, cy):
    """True if (px,py) is strictly inside triangle abc (assumed CCW, positive area)."""
    d1 = _cross(ax, ay, bx, by, px, py)
    d2 = _cross(bx, by, cx, cy, px, py)
    d3 = _cross(cx, cy, ax, ay, px, py)
    return d1 > 0 and d2 > 0 and d3 > 0


def _point_in_tri_incl(px, py, ax, ay, bx, by, cx, cy):
    """True if inside or on boundary of triangle abc (CCW)."""
    d1 = _cross(ax, ay, bx, by, px, py)
    d2 = _cross(bx, by, cx, cy, px, py)
    d3 = _cross(cx, cy, ax, ay, px, py)
    return d1 >= 0 and d2 >= 0 and d3 >= 0


def triangulate(poly):
    """
    Ear-clipping triangulation of a simple polygon (int coords).
    Returns list of (3,2) int64 arrays, all CCW, zero-area triangles dropped.
    Robust to CW/CCW input; falls back to a fan if ear clipping stalls.
    """
    poly = np.asarray(poly, dtype=np.int64)
    n = len(poly)
    if n < 3:
        return []
    # normalize to CCW
    if signed_area2(poly) < 0:
        poly = poly[::-1].copy()
        n = len(poly)

    verts = [(int(poly[i, 0]), int(poly[i, 1])) for i in range(n)]
    # remove collinear/duplicate handled implicitly by ear test
    idx = list(range(n))
    tris = []
    guard = 0
    max_guard = 3 * n + 10
    while len(idx) > 3 and guard < max_guard:
        guard += 1
        ear_found = False
        m = len(idx)
        for k in range(m):
            i0 = idx[(k - 1) % m]
            i1 = idx[k]
            i2 = idx[(k + 1) % m]
            ax, ay = verts[i0]
            bx, by = verts[i1]
            cx, cy = verts[i2]
            cr = _cross(ax, ay, bx, by, cx, cy)
            if cr <= 0:
                continue  # reflex or collinear -> not an ear tip
            # no other vertex strictly inside this triangle
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                px, py = verts[j]
                if _point_in_tri_incl(px, py, ax, ay, bx, by, cx, cy):
                    ok = False
                    break
            if not ok:
                continue
            tris.append(np.array([[ax, ay], [bx, by], [cx, cy]], dtype=np.int64))
            del idx[k]
            ear_found = True
            break
        if not ear_found:
            break  # stall (degenerate) -> fan fallback below

    if len(idx) == 3:
        i0, i1, i2 = idx
        tris.append(np.array([verts[i0], verts[i1], verts[i2]], dtype=np.int64))
    elif len(idx) > 3:
        # fan fallback for the remaining polygon (approx; used only on degenerate input)
        for k in range(1, len(idx) - 1):
            tris.append(np.array([verts[idx[0]], verts[idx[k]], verts[idx[k + 1]]],
                                 dtype=np.int64))

    # drop zero-area triangles, ensure CCW
    out = []
    for t in tris:
        a2 = signed_area2(t)
        if a2 == 0:
            continue
        if a2 < 0:
            t = t[::-1].copy()
        out.append(t)
    return out


def stack_triangles(tri_list):
    """List of (3,2) arrays -> single (T,3,2) int64 array (empty -> (0,3,2))."""
    if not tri_list:
        return np.zeros((0, 3, 2), dtype=np.int64)
    return np.stack(tri_list).astype(np.int64)


# ==========================================================================
# Hot-path kernels (numba-jitted when available)
# ==========================================================================

RASTER_CELL = SCALE   # 1 bay unit per cell -> a placement offset (bay units) is an
                      # EXACT integer cell shift (no rounding).


@njit(cache=True)
def _pt_in_tri_j(px, py, ax, ay, bx, by, cx, cy):
    """True if (px,py) is inside-or-on triangle (a,b,c) (int64)."""
    d1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    d2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
    d3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


@njit(cache=True)
def _tri_hits_aabb(t, x0, y0, x1, y1):
    """True iff triangle t (3,2 int64) overlaps the axis-aligned box [x0,x1]x[y0,y1]
    with POSITIVE area. Full triangle-vs-AABB SAT: separating axes = the box's two
    axes (bbox reject) + the triangle's three edge normals. Strict '<=' ⇒ touching
    counts as NO overlap (matches the strict interior overlap of _tri_overlap)."""
    txmin = t[0, 0]; txmax = t[0, 0]; tymin = t[0, 1]; tymax = t[0, 1]
    for k in range(1, 3):
        if t[k, 0] < txmin: txmin = t[k, 0]
        if t[k, 0] > txmax: txmax = t[k, 0]
        if t[k, 1] < tymin: tymin = t[k, 1]
        if t[k, 1] > tymax: tymax = t[k, 1]
    if txmax <= x0 or x1 <= txmin:
        return False
    if tymax <= y0 or y1 <= tymin:
        return False
    for e in range(3):
        a0 = t[e, 0]; a1 = t[e, 1]
        f = (e + 1) % 3
        nx = -(t[f, 1] - a1); ny = (t[f, 0] - a0)      # edge normal
        p0 = nx * t[0, 0] + ny * t[0, 1]
        p1 = nx * t[1, 0] + ny * t[1, 1]
        p2 = nx * t[2, 0] + ny * t[2, 1]
        tmin = p0; tmax = p0
        if p1 < tmin: tmin = p1
        if p1 > tmax: tmax = p1
        if p2 < tmin: tmin = p2
        if p2 > tmax: tmax = p2
        q0 = nx * x0 + ny * y0; q1 = nx * x1 + ny * y0
        q2 = nx * x0 + ny * y1; q3 = nx * x1 + ny * y1
        cmin = q0; cmax = q0
        if q1 < cmin: cmin = q1
        if q1 > cmax: cmax = q1
        if q2 < cmin: cmin = q2
        if q2 > cmax: cmax = q2
        if q3 < cmin: cmin = q3
        if q3 > cmax: cmax = q3
        if tmax <= cmin or cmax <= tmin:
            return False
    return True


@njit(cache=True)
def _rasterize_level(tris, g, r0, r1, c0, c1):
    """Jitted cell loop for ONE level. Returns (outer, inner) uint8 masks over the
    cell box [r0,r1]x[c0,c1]. OUTER = positive-area triangle overlap (over-cover);
    INNER = all 4 cell corners inside-or-on a SINGLE triangle (under-cover)."""
    nr = r1 - r0 + 1; nc = c1 - c0 + 1
    outer = np.zeros((nr, nc), np.uint8)
    inner = np.zeros((nr, nc), np.uint8)
    T = tris.shape[0]
    for r in range(r0, r1 + 1):
        y0 = r * g; y1 = y0 + g
        for c in range(c0, c1 + 1):
            x0 = c * g; x1 = x0 + g
            hit = False
            for ti in range(T):
                if _tri_hits_aabb(tris[ti], x0, y0, x1, y1):
                    hit = True; break
            if not hit:
                continue
            outer[r - r0, c - c0] = 1
            for ti in range(T):
                t = tris[ti]
                if (_pt_in_tri_j(x0, y0, t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1])
                        and _pt_in_tri_j(x1, y0, t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1])
                        and _pt_in_tri_j(x0, y1, t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1])
                        and _pt_in_tri_j(x1, y1, t[0, 0], t[0, 1], t[1, 0], t[1, 1], t[2, 0], t[2, 1])):
                    inner[r - r0, c - c0] = 1
                    break
    return outer, inner


def rasterize_poly(tris, g=RASTER_CELL):
    """Rasterize ONE level's triangles (T,3,2 int64 scaled) into two cell sets
    (INNER/OUTER raster prefilter, GEOMETRY_AND_ISLAND_SPEEDUP_PLAN §A1):
      OUTER = cells with positive-area polygon overlap (over-covers the polygon);
      INNER = cells the polygon covers COMPLETELY (under-covers the interior).
    Returns (r0, c0, outer, inner): cell-grid origin (r0,c0 in cell units) and lists
    of (dr,dc) relative cells. Both one-sided: OUTER∩OUTER=∅ ⇒ certainly free;
    OUTER∩INNER≠∅ ⇒ certainly blocked; else defer to exact SAT. Exact-safe.
    The heavy cell loop is jitted (_rasterize_level); this shell only extracts the
    nonzero cells and stays O(#cells) in Python."""
    if tris.shape[0] == 0:
        return 0, 0, [], []
    xs = tris[:, :, 0]; ys = tris[:, :, 1]
    xmin = int(xs.min()); xmax = int(xs.max()); ymin = int(ys.min()); ymax = int(ys.max())
    c0 = xmin // g; c1 = (xmax - 1) // g if xmax > xmin else xmin // g
    r0 = ymin // g; r1 = (ymax - 1) // g if ymax > ymin else ymin // g
    outer_m, inner_m = _rasterize_level(np.ascontiguousarray(tris), g, r0, r1, c0, c1)
    oi = np.nonzero(outer_m); ii = np.nonzero(inner_m)
    outer = list(zip(oi[0].tolist(), oi[1].tolist()))
    inner = list(zip(ii[0].tolist(), ii[1].tolist()))
    return r0, c0, outer, inner


def rasterize_poly_bits(tris, g=RASTER_CELL):
    """Bit-packed variant of rasterize_poly for ONE level. Returns
    (r0, c0, nrows, nwords, ow, iw): cell-grid origin, row/word dims, and the
    OUTER/INNER row-word bitsets ([nrows*nwords] uint64, LSB-first). Empty ⇒
    (0,0,0,0, empty, empty)."""
    empty = np.zeros(0, np.uint64)
    if tris.shape[0] == 0:
        return 0, 0, 0, 0, empty, empty
    xs = tris[:, :, 0]; ys = tris[:, :, 1]
    xmin = int(xs.min()); xmax = int(xs.max()); ymin = int(ys.min()); ymax = int(ys.max())
    c0 = xmin // g; c1 = (xmax - 1) // g if xmax > xmin else xmin // g
    r0 = ymin // g; r1 = (ymax - 1) // g if ymax > ymin else ymin // g
    outer_m, inner_m = _rasterize_level(np.ascontiguousarray(tris), g, r0, r1, c0, c1)
    nrows = r1 - r0 + 1
    ncols = c1 - c0 + 1
    nwords = (ncols + 63) // 64
    ow = pack_mask(outer_m, nwords)
    iw = pack_mask(inner_m, nwords)
    return r0, c0, nrows, nwords, ow, iw


@njit(cache=True)
def _tri_overlap(a, b, adx, ady, bdx, bdy):
    """
    SAT interior-overlap of two triangles a,b (each (3,2) int64), translated by
    (adx,ady) and (bdx,bdy). Returns True iff interiors have positive-area overlap
    (strict). Touching / edge-sharing returns False.
    """
    # candidate axes = edge normals of a then of b
    for src in range(2):
        if src == 0:
            poly = a
            ox = adx
            oy = ady
        else:
            poly = b
            ox = bdx
            oy = bdy
        for e in range(3):
            x1 = poly[e, 0]
            y1 = poly[e, 1]
            x2 = poly[(e + 1) % 3, 0]
            y2 = poly[(e + 1) % 3, 1]
            # edge vector (ex,ey); normal = (-ey, ex)
            nx = -(y2 - y1)
            ny = (x2 - x1)
            if nx == 0 and ny == 0:
                continue
            # project a
            amin = 0
            amax = 0
            first = True
            for i in range(3):
                p = (a[i, 0] + adx) * nx + (a[i, 1] + ady) * ny
                if first:
                    amin = p
                    amax = p
                    first = False
                else:
                    if p < amin:
                        amin = p
                    if p > amax:
                        amax = p
            bmin = 0
            bmax = 0
            first = True
            for i in range(3):
                p = (b[i, 0] + bdx) * nx + (b[i, 1] + bdy) * ny
                if first:
                    bmin = p
                    bmax = p
                    first = False
                else:
                    if p < bmin:
                        bmin = p
                    if p > bmax:
                        bmax = p
            # separated or merely touching on this axis => no interior overlap
            if amax <= bmin or bmax <= amin:
                return False
    return True


@njit(cache=True)
def batch_first_overlap(A, Abb, adx, ady, B, Bbb):
    """
    First interior-overlap between new-block triangles A (relative, (Ta,3,2)) with
    per-triangle relative bboxes Abb ((Ta,4)) translated by (adx,ady), and a batch
    of co-present triangles B (ABSOLUTE, (Tb,3,2)) with absolute bboxes Bbb ((Tb,4)).
    Per-triangle AABB reject inside the JIT makes this the fast collision kernel.
    """
    for ia in range(A.shape[0]):
        axmin = Abb[ia, 0] + adx
        aymin = Abb[ia, 1] + ady
        axmax = Abb[ia, 2] + adx
        aymax = Abb[ia, 3] + ady
        for ib in range(B.shape[0]):
            if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                continue
            if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                continue
            if _tri_overlap(A[ia], B[ib], adx, ady, 0, 0):
                return True
    return False


@njit(cache=True)
def first_feasible_offset(A, Abb, offX, offY, B, Bbb):
    """
    Scan candidate offsets (offX,offY are parallel scaled-int arrays, already in
    bottom-left order) and return the index of the FIRST offset at which moving
    triangles A (relative, with relative per-tri bboxes Abb) do NOT overlap any
    co-present triangle B (absolute, bboxes Bbb). Returns -1 if none.

    The whole candidate x neighbour x axis loop runs inside one jitted call —
    this eliminates the ~217k Python-level dispatches the profiler flagged.
    """
    C = offX.shape[0]
    Ta = A.shape[0]
    Tb = B.shape[0]
    for c in range(C):
        dx = offX[c]
        dy = offY[c]
        free = True
        for ia in range(Ta):
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Tb):
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], B[ib], dx, dy, 0, 0):
                    free = False
                    break
            if not free:
                break
        if free:
            return c
    return -1


@njit(cache=True)
def first_feasible_offset_lvl(A, Alvl, Abb, offX, offY, B, Blvl, Bbb):
    """
    Level-aware, EARLY-EXIT candidate scan for any number of layers. A/B hold ALL
    layers' triangles concatenated, with per-triangle level ids Alvl/Blvl and
    per-triangle bboxes Abb/Bbb (A relative, B absolute). A moving triangle only
    collides with a co-present triangle at the SAME level (same-level collision).
    Returns the first bottom-left candidate index that is collision-free, or -1.

    Uniform for K=1..4 and it early-exits at the first feasible candidate — this
    fixes the multi-layer slowdown of the mask-based path.
    """
    C = offX.shape[0]
    Ta = A.shape[0]
    Tb = B.shape[0]
    for c in range(C):
        dx = offX[c]
        dy = offY[c]
        free = True
        for ia in range(Ta):
            la = Alvl[ia]
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Tb):
                if Blvl[ib] != la:
                    continue
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], B[ib], dx, dy, 0, 0):
                    free = False
                    break
            if not free:
                break
        if free:
            return c
    return -1


@njit(cache=True)
def best_envelope_offset(A, Alvl, Abb, offX, offY, Bs, Bbb, Bstart,
                        mx0, my0, mx1, my1, ox0, oy0, ox1, oy1):
    """HPS placement (Fang et al. 2024, "knowledge reuse + improved heuristic"):
    among the FEASIBLE candidate offsets, return the one that grows the bay's
    occupied bounding box the least (minimum envelope increment) — i.e. the tightest
    nestle against what's already placed. Feasibility is the SAME exact level-aware
    triangle scan (the authority); only the SELECTION among feasible spots changes
    from bottom-left first-fit to min-envelope. Falls back to first-feasible if the
    occupied box is degenerate."""
    C = offX.shape[0]
    Ta = A.shape[0]
    nlv = Bstart.shape[0] - 1
    best_c = -1
    best_area = -1.0
    for c in range(C):
        dx = offX[c]
        dy = offY[c]
        free = True
        for ia in range(Ta):
            la = Alvl[ia]
            if la >= nlv:
                continue
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Bstart[la], Bstart[la + 1]):
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], Bs[ib], dx, dy, 0, 0):
                    free = False
                    break
            if not free:
                break
        if not free:
            continue
        # envelope area if placed here (union of occupied box and moving box)
        nx0 = mx0 + dx if mx0 + dx < ox0 else ox0
        ny0 = my0 + dy if my0 + dy < oy0 else oy0
        nx1 = mx1 + dx if mx1 + dx > ox1 else ox1
        ny1 = my1 + dy if my1 + dy > oy1 else oy1
        area = float(nx1 - nx0) * float(ny1 - ny0)
        if best_c < 0 or area < best_area:
            best_area = area
            best_c = c
    return best_c


@njit(cache=True)
def first_feasible_offset_cde(A, Alvl, Abb, mcx, mcy, mr,
                             offX, offY, Bs, Bbb, Bstart, Bcx, Bcy, Bcr):
    """
    CDE-accelerated variant of first_feasible_offset_slots (jagua-rs fail-fast
    surrogate). Per candidate offset: if the moving block's bounding circle is
    STRICTLY disjoint from every co-present block's bounding circle, the candidate
    is provably collision-free -> return it WITHOUT touching a single triangle
    (the CLEAR fast-path). Otherwise fall back to the exact level-aware triangle
    scan, which stays the sole authority (so touching is handled correctly).
    """
    C = offX.shape[0]
    Ta = A.shape[0]
    nlv = Bstart.shape[0] - 1
    nb = Bcx.shape[0]
    for c in range(C):
        dx = offX[c]
        dy = offY[c]
        # ---- broad phase: bounding-circle CLEAR test ----
        acx = mcx + dx
        acy = mcy + dy
        clear = True
        for b in range(nb):
            ddx = acx - Bcx[b]
            ddy = acy - Bcy[b]
            rs = mr + Bcr[b]
            if ddx * ddx + ddy * ddy <= rs * rs:
                clear = False
                break
        if clear:
            return c                            # provably collision-free
        # ---- narrow phase: exact level-aware triangle scan (authority) ----
        free = True
        for ia in range(Ta):
            la = Alvl[ia]
            if la >= nlv:
                continue
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Bstart[la], Bstart[la + 1]):
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], Bs[ib], dx, dy, 0, 0):
                    free = False
                    break
            if not free:
                break
        if free:
            return c
    return -1


@njit(cache=True)
def first_feasible_offset_slots(A, Alvl, Abb, offX, offY, Bs, Bbb, Bstart):
    """
    Fast level-aware early-exit scan: co-present triangles Bs are grouped by level
    with Bstart[l]..Bstart[l+1] the slice for level l, so a moving triangle at
    level `la` only tests the matching level's triangles (no wasted iteration),
    while still early-exiting at the first feasible bottom-left candidate.
    """
    C = offX.shape[0]
    Ta = A.shape[0]
    nlv = Bstart.shape[0] - 1
    for c in range(C):
        dx = offX[c]
        dy = offY[c]
        free = True
        for ia in range(Ta):
            la = Alvl[ia]
            if la >= nlv:
                continue                       # no co-present at this level
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Bstart[la], Bstart[la + 1]):
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], Bs[ib], dx, dy, 0, 0):
                    free = False
                    break
            if not free:
                break
        if free:
            return c
    return -1


# ============================ bit-packed raster core ========================
# The INNER/OUTER cell sets of every polygon are stored as BIT-PACKED ROWS: for
# each grid row, a run of uint64 words whose bit c (LSB-first within a word) marks
# cell column c. A collision query is then row-wise shift+AND+"any-nonzero" over a
# few machine words (RASTER_GEOMETRY_EXPLAINED §7) instead of per-cell array
# indexing — 64 cells per word-op, no per-cell branch. Exact SAT still decides the
# uncertain band.

@njit(cache=True)
def pack_mask(mask, nwords):
    """Pack a 2D uint8 cell mask [nrows,ncols] into row-major uint64 words
    [nrows*nwords], LSB-first (bit c%64 of word c//64)."""
    nrows = mask.shape[0]; ncols = mask.shape[1]
    out = np.zeros(nrows * nwords, np.uint64)
    one = np.uint64(1)
    for r in range(nrows):
        base = r * nwords
        for c in range(ncols):
            if mask[r, c] != 0:
                out[base + (c >> 6)] |= (one << np.uint64(c & 63))
    return out


@njit(cache=True, inline='always')
def _row_shift_hit(mw, mstart, nbw, occ, l, orow, GW, colbase):
    """True iff the block row (words mw[mstart:mstart+nbw], block-column j -> occ
    column colbase+j) intersects occupancy row occ[l, orow, :]. colbase may be
    negative; block bits mapping outside [0,64*GW) simply miss (no co-present)."""
    for wb in range(nbw):
        v = mw[mstart + wb]
        if v == 0:
            continue
        base = colbase + (wb << 6)          # occ column of this word's bit 0
        wo = base >> 6 if base >= 0 else -((-base + 63) >> 6)   # floor(base/64)
        sh = base - (wo << 6)               # in [0,64)
        if 0 <= wo < GW:
            if ((v << np.uint64(sh)) & occ[l, orow, wo]) != 0:
                return True
        if sh != 0:
            wo1 = wo + 1
            if 0 <= wo1 < GW:
                if ((v >> np.uint64(64 - sh)) & occ[l, orow, wo1]) != 0:
                    return True
    return False


@njit(cache=True)
def or_block_into(occ, l, mw, mstart, nrows, nwords, rowbase, colbase, GR, GW):
    """OR one block mask (packed rows mw[mstart:...]) into occupancy layer l at
    row offset rowbase / column offset colbase. Used to build the bay occupancy."""
    for dr in range(nrows):
        orow = rowbase + dr
        if orow < 0 or orow >= GR:
            continue
        rb = mstart + dr * nwords
        for wb in range(nwords):
            v = mw[rb + wb]
            if v == 0:
                continue
            base = colbase + (wb << 6)
            wo = base >> 6 if base >= 0 else -((-base + 63) >> 6)
            sh = base - (wo << 6)
            if 0 <= wo < GW:
                occ[l, orow, wo] |= (v << np.uint64(sh))
            if sh != 0:
                wo1 = wo + 1
                if 0 <= wo1 < GW:
                    occ[l, orow, wo1] |= (v >> np.uint64(64 - sh))


@njit(cache=True)
def build_occupancy(occ_o, occ_i, all_ow, all_iw, meta, GR, GW):
    """Build bay occupancy bitsets in ONE jitted pass (no per-block Python
    dispatch). meta rows = (layer, wstart, nrows, nwords, rowbase, colbase) index
    into the concatenated co-present word arrays all_ow/all_iw."""
    for i in range(meta.shape[0]):
        l = meta[i, 0]; ws = meta[i, 1]; nr = meta[i, 2]
        nw = meta[i, 3]; rb = meta[i, 4]; cb = meta[i, 5]
        or_block_into(occ_o, l, all_ow, ws, nr, nw, rb, cb, GR, GW)
        or_block_into(occ_i, l, all_iw, ws, nr, nw, rb, cb, GR, GW)


@njit(cache=True)
def first_feasible_offset_raster(A, Alvl, Abb, offX, offY, Bs, Bbb, Bstart,
                                 cxa, cya,
                                 r_r0, r_c0, r_nrows, r_nwords, r_wstart,
                                 mow, miw, occ_o, occ_i, gr0, gc0):
    """Bit-packed INNER/OUTER raster gate (RASTER_GEOMETRY_EXPLAINED §7). Per
    candidate (bottom-left order): OUTER∩INNER or INNER∩OUTER ⇒ certainly BLOCKED
    (skip); OUTER∩OUTER=∅ ⇒ certainly FREE (return); else UNCERTAIN ⇒ exact
    triangle-SAT for that one offset. Returns first feasible candidate index or -1.
    Exact-safe: FREE⊆SAT-feasible, BLOCKED⊆SAT-infeasible, band ⇒ _tri_overlap.
    occ_o/occ_i are [nlev,GR,GW] uint64 bay occupancy bitsets; (gr0,gc0) their
    cell origin. cxa/cya are candidate offsets in CELL units (=bay units)."""
    C = cxa.shape[0]
    Ta = A.shape[0]
    nlv = Bstart.shape[0] - 1
    occ_nlev = occ_o.shape[0]
    GR = occ_o.shape[1]
    GW = occ_o.shape[2]
    bK = r_r0.shape[0]
    for c in range(C):
        cx = cxa[c]
        cy = cya[c]
        blocked = False
        shared = False
        for l in range(bK):
            if l >= occ_nlev:
                continue
            nrows = r_nrows[l]
            if nrows == 0:
                continue
            nwords = r_nwords[l]
            wstart = r_wstart[l]
            rowbase = r_r0[l] + cy - gr0
            colbase = r_c0[l] + cx - gc0
            for dr in range(nrows):
                orow = rowbase + dr
                if orow < 0 or orow >= GR:
                    continue
                mstart = wstart + dr * nwords
                if _row_shift_hit(mow, mstart, nwords, occ_i, l, orow, GW, colbase):
                    blocked = True
                    break                              # OUTER∩INNER
                if not shared:
                    if _row_shift_hit(mow, mstart, nwords, occ_o, l, orow, GW, colbase):
                        shared = True                  # OUTER∩OUTER
            if blocked:
                break
            for dr in range(nrows):
                orow = rowbase + dr
                if orow < 0 or orow >= GR:
                    continue
                mstart = wstart + dr * nwords
                if _row_shift_hit(miw, mstart, nwords, occ_o, l, orow, GW, colbase):
                    blocked = True
                    break                              # INNER∩OUTER
            if blocked:
                break
        if blocked:
            continue
        if not shared:
            return c                                   # certainly free
        # uncertain band -> exact SAT for this single offset (the sole authority)
        dx = offX[c]
        dy = offY[c]
        free = True
        for ia in range(Ta):
            la = Alvl[ia]
            if la >= nlv:
                continue
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Bstart[la], Bstart[la + 1]):
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], Bs[ib], dx, dy, 0, 0):
                    free = False
                    break
            if not free:
                break
        if free:
            return c
    return -1


@njit(cache=True)
def collision_free_mask(A, Abb, offX, offY, B, Bbb, out):
    """Multi-level helper: AND into `out` (uint8, 1=still feasible) whether each
    candidate offset keeps moving level A clear of co-present level B. Skips
    candidates already excluded by another level."""
    C = offX.shape[0]
    Ta = A.shape[0]
    Tb = B.shape[0]
    for c in range(C):
        if out[c] == 0:
            continue
        dx = offX[c]
        dy = offY[c]
        free = 1
        for ia in range(Ta):
            axmin = Abb[ia, 0] + dx
            aymin = Abb[ia, 1] + dy
            axmax = Abb[ia, 2] + dx
            aymax = Abb[ia, 3] + dy
            for ib in range(Tb):
                if axmax <= Bbb[ib, 0] or Bbb[ib, 2] <= axmin:
                    continue
                if aymax <= Bbb[ib, 1] or Bbb[ib, 3] <= aymin:
                    continue
                if _tri_overlap(A[ia], B[ib], dx, dy, 0, 0):
                    free = 0
                    break
            if free == 0:
                break
        out[c] = free


@njit(cache=True)
def _tris_overlap(A, B, adx, ady, bdx, bdy,
                  a_minx, a_miny, a_maxx, a_maxy,
                  b_minx, b_miny, b_maxx, b_maxy):
    """
    Any triangle of A overlaps any triangle of B (interiors), given translations
    and the (already translated) bounding boxes for a fast reject.
    """
    # broad-phase AABB reject (translated bboxes passed in)
    if a_maxx <= b_minx or b_maxx <= a_minx:
        return False
    if a_maxy <= b_miny or b_maxy <= a_miny:
        return False
    na = A.shape[0]
    nb = B.shape[0]
    for i in range(na):
        for j in range(nb):
            if _tri_overlap(A[i], B[j], adx, ady, bdx, bdy):
                return True
    return False


# ==========================================================================
# Block-level predicates (operate on precomputed OrientGeom-like structures)
# ==========================================================================
# An "orientation geometry" (built in model.py) is a dict-ish with:
#   K            : int number of layers
#   tris         : list length K of (T,3,2) int64 arrays (relative, scaled)
#   lbbox        : (K,4) int64 array of per-layer bbox (relative, scaled)
#   ubbox        : (4,) int64 union bbox (relative, scaled)
# Placement offset for a block placed at integer (x,y): (x*SCALE, y*SCALE).


def contained_in_bay(og, x, y, W, H):
    """
    Every layer of orientation `og` placed at integer (x,y) lies within the
    rectangle [0,W]x[0,H]. Exact: rectangle is convex => all vertices in range.
    Uses the union bbox (relative) which bounds all layers' vertices.
    """
    ox = x * SCALE
    oy = y * SCALE
    minx, miny, maxx, maxy = og.ubbox
    if minx + ox < 0 or miny + oy < 0:
        return False
    if maxx + ox > W * SCALE or maxy + oy > H * SCALE:
        return False
    return True


def same_level_collision(a, xa, ya, b, xb, yb):
    """
    True if blocks a,b (orientation geoms) at integer placements collide on ANY
    common level l = 0..min(Ka,Kb)-1 (interior intersection).
    """
    adx = xa * SCALE
    ady = ya * SCALE
    bdx = xb * SCALE
    bdy = yb * SCALE
    # quick union-bbox reject
    au = a.ubbox
    bu = b.ubbox
    if au[2] + adx <= bu[0] + bdx or bu[2] + bdx <= au[0] + adx:
        return False
    if au[3] + ady <= bu[1] + bdy or bu[3] + bdy <= au[1] + ady:
        return False
    K = a.K if a.K < b.K else b.K
    for l in range(K):
        A = a.tris[l]
        B = b.tris[l]
        if A.shape[0] == 0 or B.shape[0] == 0:
            continue
        la = a.lbbox[l]
        lb = b.lbbox[l]
        if _tris_overlap(A, B, adx, ady, bdx, bdy,
                         la[0] + adx, la[1] + ady, la[2] + adx, la[3] + ady,
                         lb[0] + bdx, lb[1] + bdy, lb[2] + bdx, lb[3] + bdy):
            return True
    return False


def crane_blocked_by(mover, xm, ym, other, xo, yo):
    """
    True if `other` (at xo,yo) blocks the straight-up crane lift of `mover`
    (at xm,ym). Blocked iff for some level l1 of mover, some level l2 >= l1 of
    other has interior-intersecting polygon with mover's level l1.
    """
    adx = xm * SCALE
    ady = ym * SCALE
    bdx = xo * SCALE
    bdy = yo * SCALE
    au = mover.ubbox
    bu = other.ubbox
    if au[2] + adx <= bu[0] + bdx or bu[2] + bdx <= au[0] + adx:
        return False
    if au[3] + ady <= bu[1] + bdy or bu[3] + bdy <= au[1] + ady:
        return False
    for l1 in range(mover.K):
        A = mover.tris[l1]
        if A.shape[0] == 0:
            continue
        la = mover.lbbox[l1]
        for l2 in range(l1, other.K):
            B = other.tris[l2]
            if B.shape[0] == 0:
                continue
            lb = other.lbbox[l2]
            if _tris_overlap(A, B, adx, ady, bdx, bdy,
                             la[0] + adx, la[1] + ady, la[2] + adx, la[3] + ady,
                             lb[0] + bdx, lb[1] + bdy, lb[2] + bdx, lb[3] + bdy):
                return True
    return False


def warmup():
    """Trigger numba compilation once with a tiny dummy so the first real call is fast."""
    t = np.array([[[0, 0], [10000, 0], [0, 10000]]], dtype=np.int64)
    bb = np.array([[0, 0, 10000, 10000]], dtype=np.int64)
    offX = np.array([0, 50000], dtype=np.int64)
    offY = np.array([0, 0], dtype=np.int64)
    out = np.ones(2, dtype=np.uint8)
    lvl = np.zeros(1, dtype=np.int64)
    bstart = np.array([0, 1], dtype=np.int64)
    try:
        # only the LIVE hot-path kernels are compiled here (unused variants stay
        # lazily uncompiled -> shorter warm-up).  _tris_overlap covers the
        # same-level-collision / crane predicates used by the checker & crane.
        _tris_overlap(t, t, 0, 0, 5000, 0, 0, 0, 10000, 10000, 5000, 0, 15000, 10000)
        _tri_overlap(t[0], t[0], 0, 0, 0, 0)
        first_feasible_offset_slots(t, lvl, bb, offX, offY, t, bb, bstart)
        _cf = np.zeros(1, dtype=np.float64)
        first_feasible_offset_cde(t, lvl, bb, 0.0, 0.0, 7000.0, offX, offY,
                                  t, bb, bstart, _cf, _cf, _cf + 1000.0)
        best_envelope_offset(t, lvl, bb, offX, offY, t, bb, bstart,
                             0, 0, 10000, 10000, 0, 0, 10000, 10000)
    except Exception:
        pass
