# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Numba kernels for the OGC2026 solver: polygon rasterization, forbidden-grid
construction, and placement scans."""

import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# Polygon-cell clipping (Sutherland-Hodgman) for exact conservative rasters
# -----------------------------------------------------------------------------

@njit(cache=True)
def _clip_area(xs, ys, x0, y0, x1, y1):
    """Area of polygon (xs, ys) clipped to axis-aligned box [x0,x1]x[y0,y1].

    Sutherland-Hodgman with a convex clip window. For a simple (possibly
    non-convex) subject polygon the shoelace area of the S-H output equals the
    true clipped area (degenerate zero-width bridges contribute zero area).
    """
    n = xs.shape[0]
    cap = 16 * n + 16
    px = np.empty(cap)
    py = np.empty(cap)
    qx = np.empty(cap)
    qy = np.empty(cap)
    for i in range(n):
        px[i] = xs[i]
        py[i] = ys[i]
    m = n
    for edge in range(4):
        if m == 0:
            return 0.0
        k = 0
        for i in range(m):
            cx = px[i]
            cy = py[i]
            j = i + 1
            if j == m:
                j = 0
            jx = px[j]
            jy = py[j]
            if edge == 0:
                ci = cx >= x0
                ji = jx >= x0
            elif edge == 1:
                ci = cx <= x1
                ji = jx <= x1
            elif edge == 2:
                ci = cy >= y0
                ji = jy >= y0
            else:
                ci = cy <= y1
                ji = jy <= y1
            if ci:
                qx[k] = cx
                qy[k] = cy
                k += 1
            if ci != ji:
                if edge == 0:
                    t = (x0 - cx) / (jx - cx)
                    ix = x0
                    iy = cy + t * (jy - cy)
                elif edge == 1:
                    t = (x1 - cx) / (jx - cx)
                    ix = x1
                    iy = cy + t * (jy - cy)
                elif edge == 2:
                    t = (y0 - cy) / (jy - cy)
                    iy = y0
                    ix = cx + t * (jx - cx)
                else:
                    t = (y1 - cy) / (jy - cy)
                    iy = y1
                    ix = cx + t * (jx - cx)
                qx[k] = ix
                qy[k] = iy
                k += 1
        for i in range(k):
            px[i] = qx[i]
            py[i] = qy[i]
        m = k
    if m < 3:
        return 0.0
    s = 0.0
    for i in range(m):
        j = i + 1
        if j == m:
            j = 0
        s += px[i] * py[j] - px[j] * py[i]
    return abs(s) * 0.5


@njit(cache=True)
def raster_poly(xs, ys, u0, v0, nu, nv, pad, thresh):
    """Conservative raster: out[a,b] = 1 iff polygon overlaps unit cell
    (u0+a, v0+b) expanded by `pad` with clipped area > thresh."""
    out = np.zeros((nu, nv), np.uint8)
    for a in range(nu):
        for b in range(nv):
            x0 = u0 + a
            y0 = v0 + b
            if _clip_area(xs, ys, x0 - pad, y0 - pad, x0 + 1.0 + pad, y0 + 1.0 + pad) > thresh:
                out[a, b] = 1
    return out


@njit(cache=True)
def raster_poly_inner(xs, ys, u0, v0, nu, nv):
    """Inner raster: out[a,b] = 1 iff unit cell (u0+a, v0+b) is (essentially)
    fully covered by the polygon. Used only for certain-collision rejection,
    so both false negatives and (measure-1e-9) false positives are safe."""
    out = np.zeros((nu, nv), np.uint8)
    for a in range(nu):
        for b in range(nv):
            x0 = u0 + a
            y0 = v0 + b
            if _clip_area(xs, ys, x0, y0, x0 + 1.0, y0 + 1.0) >= 1.0 - 1e-9:
                out[a, b] = 1
    return out


# -----------------------------------------------------------------------------
# Forbidden-grid construction
# -----------------------------------------------------------------------------
# G has shape (K, W, H) where K = number of layers of the block being placed.
# For each already-placed block D overlapping the candidate time window:
#   flag bit 0 (boundary): D present at the entry or exit moment. The moving
#       block's layer k must avoid D's layers j >= k, so D's layer l stamps
#       G[0 .. min(l, K-1)].
#   flag bit 1 (interior): D itself enters or exits strictly inside the
#       window; D's layer a must avoid the placed block's layers >= a, so D's
#       layer l stamps G[l .. K-1] (nothing if l >= K).

@njit(cache=True)
def build_G(G, nd, dx, dy, doidx, dflags, layer_base, layer_cnt,
            cell_ptr, cell_cnt, arena_u, arena_v):
    K = G.shape[0]
    for t in range(nd):
        base = layer_base[doidx[t]]
        nl = layer_cnt[doidx[t]]
        fl = dflags[t]
        ox = dx[t]
        oy = dy[t]
        for l in range(nl):
            row = base + l
            p0 = cell_ptr[row]
            p1 = p0 + cell_cnt[row]
            if fl & 1:
                khi = l
                if khi > K - 1:
                    khi = K - 1
                for k in range(0, khi + 1):
                    for c in range(p0, p1):
                        G[k, arena_u[c] + ox, arena_v[c] + oy] = 1
            if fl & 2:
                if l <= K - 1:
                    for k in range(l, K):
                        for c in range(p0, p1):
                            G[k, arena_u[c] + ox, arena_v[c] + oy] = 1
    return G


# -----------------------------------------------------------------------------
# Placement scans
# -----------------------------------------------------------------------------

@njit(cache=True)
def scan_first(G, cu, cv, ck, pxlo, pxhi, pylo, pyhi):
    """First feasible reference position scanning x asc then y asc.
    Returns (-1, -1) if none."""
    nc = cu.shape[0]
    for px in range(pxlo, pxhi + 1):
        for py in range(pylo, pyhi + 1):
            ok = True
            for c in range(nc):
                if G[ck[c], cu[c] + px, cv[c] + py] != 0:
                    ok = False
                    break
            if ok:
                return px, py
    return -1, -1


@njit(cache=True)
def scan_best(G, cu, cv, ck, pxlo, pxhi, pylo, pyhi, xw, yw, xdir, ydir):
    """Best feasible position minimizing xw*dx + yw*dy where dx/dy are the
    distances from the anchored corner (xdir/ydir = +1: left/bottom,
    -1: right/top). Scans from the anchor outward with monotone pruning."""
    nc = cu.shape[0]
    best_px = -1
    best_py = -1
    best_s = 1e30
    nx = pxhi - pxlo + 1
    ny = pyhi - pylo + 1
    for a in range(nx):
        px = pxlo + a if xdir > 0 else pxhi - a
        base = xw * a
        if base + 0.0 >= best_s:
            break
        for bcnt in range(ny):
            py = pylo + bcnt if ydir > 0 else pyhi - bcnt
            s = base + yw * bcnt
            if s >= best_s:
                break
            ok = True
            for c in range(nc):
                if G[ck[c], cu[c] + px, cv[c] + py] != 0:
                    ok = False
                    break
            if ok:
                best_s = s
                best_px = px
                best_py = py
                break
    return best_px, best_py


@njit(cache=True)
def attention_score(E, c0u, c0v, px, py, my_exit, h):
    """Departure-affinity attention score of one position: each occupied
    4-neighbour cell contributes 1 + max(0, h-|their_exit-mine|)/h, each
    wall neighbour 1."""
    W = E.shape[0]
    H = E.shape[1]
    sc = 0.0
    for c in range(c0u.shape[0]):
        u = c0u[c] + px
        v = c0v[c] + py
        for duv in range(4):
            if duv == 0:
                uu = u - 1
                vv = v
            elif duv == 1:
                uu = u + 1
                vv = v
            elif duv == 2:
                uu = u
                vv = v - 1
            else:
                uu = u
                vv = v + 1
            if uu < 0 or uu >= W or vv < 0 or vv >= H:
                sc += 1.0
            else:
                e = E[uu, vv]
                if e > 0:
                    d = e - 1 - my_exit
                    if d < 0:
                        d = -d
                    if d < h:
                        sc += 1.0 + (h - d) / h
                    else:
                        sc += 1.0
    return sc


@njit(cache=True)
def sliver_score(G0, c0u, c0v, px, py, wmin, hmin):
    """Fragmentation penalty of a position (side-view projection idea):
    for every free cell horizontally adjacent to the placed footprint,
    measure the free RUN it belongs to along x; runs narrower than wmin
    (the narrowest width any remaining block can use) are dead slivers.
    Returns the negative count of sliver cells created next to this
    placement (higher = better). G0 is the day's layer-0 occupancy grid;
    the candidate itself is not yet stamped."""
    W = G0.shape[0]
    H = G0.shape[1]
    pen = 0
    for c in range(c0u.shape[0]):
        u = c0u[c] + px
        v = c0v[c] + py
        # horizontal neighbours: runs narrower than wmin are dead
        for side in range(2):
            uu = u - 1 if side == 0 else u + 1
            if uu < 0 or uu >= W:
                continue
            if G0[uu, v] != 0:
                continue
            run = 0
            step = -1 if side == 0 else 1
            w = uu
            while 0 <= w < W and G0[w, v] == 0 and run <= wmin:
                run += 1
                w += step
            if run < wmin:
                pen += 1
        # vertical neighbours: runs shorter than hmin are dead channels
        # (a horizontal-only metric rewards floating placements)
        for side in range(2):
            vv = v - 1 if side == 0 else v + 1
            if vv < 0 or vv >= H:
                continue
            if G0[u, vv] != 0:
                continue
            run = 0
            step = -1 if side == 0 else 1
            w = vv
            while 0 <= w < H and G0[u, w] == 0 and run <= hmin:
                run += 1
                w += step
            if run < hmin:
                pen += 1
    return -float(pen)


@njit(cache=True)
def stamp_exit(E, cu, cv, px, py, val):
    """Write `val` into the layer-0 exit-day field at the block's cells.
    val = exit_day + 1 on insert (0 means empty), 0 on remove."""
    W = E.shape[0]
    H = E.shape[1]
    for c in range(cu.shape[0]):
        u = cu[c] + px
        v = cv[c] + py
        if 0 <= u < W and 0 <= v < H:
            E[u, v] = val


@njit(cache=True)
def scan_exit_contact(G, E, cu, cv, ck, c0u, c0v,
                      pxlo, pxhi, pylo, pyhi, my_exit, h,
                      xw, yw, xdir, ydir, max_feas, cap_a, cap_b):
    """Departure-aligned contact placement: among (conservatively) feasible
    positions, maximize the attention score over the neighbourhood --
    each occupied 4-neighbour cell contributes max(0, h - |their_exit -
    my_exit|) (blocks that leave together should sit together), each
    bay-wall neighbour contributes h/2 (walls never fragment). Corner
    distance breaks ties. Scans from the anchored corner; stops after
    max_feas feasible candidates. Returns (px, py, score) or (-1,-1,-1)."""
    nc = cu.shape[0]
    n0 = c0u.shape[0]
    W = E.shape[0]
    H = E.shape[1]
    best_px = -1
    best_py = -1
    best_score = -1.0
    best_s = 1e30
    half = 0.5 * h
    nx = pxhi - pxlo + 1
    ny = pyhi - pylo + 1
    nfeas = 0
    for a in range(nx):
        px = pxlo + a if xdir > 0 else pxhi - a
        if a > cap_a:
            break  # per-axis tightness caps: stay in a tight box around
        for bcnt in range(ny):  # the exact-refined original position
            py = pylo + bcnt if ydir > 0 else pyhi - bcnt
            if bcnt > cap_b:
                break
            ok = True
            for c in range(nc):
                if G[ck[c], cu[c] + px, cv[c] + py] != 0:
                    ok = False
                    break
            if not ok:
                continue
            # attention score over the layer-0 footprint neighbourhood
            sc = 0.0
            for c in range(n0):
                u = c0u[c] + px
                v = c0v[c] + py
                # 4-neighbourhood
                for duv in range(4):
                    if duv == 0:
                        uu = u - 1
                        vv = v
                    elif duv == 1:
                        uu = u + 1
                        vv = v
                    elif duv == 2:
                        uu = u
                        vv = v - 1
                    else:
                        uu = u
                        vv = v + 1
                    if uu < 0 or uu >= W or vv < 0 or vv >= H:
                        sc += 1.0
                    else:
                        e = E[uu, vv]
                        if e > 0:
                            # base contact dominates (dense packing);
                            # departure affinity modulates on top
                            d = e - 1 - my_exit
                            if d < 0:
                                d = -d
                            if d < h:
                                sc += 1.0 + (h - d) / h
                            else:
                                sc += 1.0
            s = xw * a + yw * bcnt
            if (sc > best_score + 1e-9
                    or (sc > best_score - 1e-9 and s < best_s)):
                best_score = sc
                best_s = s
                best_px = px
                best_py = py
            nfeas += 1
            if nfeas >= max_feas:
                return best_px, best_py, best_score
    return best_px, best_py, best_score


@njit(cache=True)
def any_feasible(G, cu, cv, ck, pxlo, pxhi, pylo, pyhi):
    nc = cu.shape[0]
    for px in range(pxlo, pxhi + 1):
        for py in range(pylo, pyhi + 1):
            ok = True
            for c in range(nc):
                if G[ck[c], cu[c] + px, cv[c] + py] != 0:
                    ok = False
                    break
            if ok:
                return True
    return False


# -----------------------------------------------------------------------------
# Exact geometry: triangle-triangle clipped area for sub-cell precision
# -----------------------------------------------------------------------------

@njit(cache=True)
def _tri_clip_area(ax0, ay0, ax1, ay1, ax2, ay2,
                   bx0, by0, bx1, by1, bx2, by2):
    """Area of intersection of two triangles (Sutherland-Hodgman: clip A by
    the halfplanes of CCW triangle B)."""
    # working polygon
    px = np.empty(16)
    py = np.empty(16)
    qx = np.empty(16)
    qy = np.empty(16)
    px[0] = ax0
    py[0] = ay0
    px[1] = ax1
    py[1] = ay1
    px[2] = ax2
    py[2] = ay2
    m = 3
    # ensure B is CCW
    if (bx1 - bx0) * (by2 - by0) - (by1 - by0) * (bx2 - bx0) < 0.0:
        t = bx1
        bx1 = bx2
        bx2 = t
        t = by1
        by1 = by2
        by2 = t
    for e in range(3):
        if e == 0:
            ex0, ey0, ex1, ey1 = bx0, by0, bx1, by1
        elif e == 1:
            ex0, ey0, ex1, ey1 = bx1, by1, bx2, by2
        else:
            ex0, ey0, ex1, ey1 = bx2, by2, bx0, by0
        dx = ex1 - ex0
        dy = ey1 - ey0
        if m == 0:
            return 0.0
        k = 0
        for i in range(m):
            cx = px[i]
            cy = py[i]
            j = i + 1
            if j == m:
                j = 0
            jx = px[j]
            jy = py[j]
            side_c = dx * (cy - ey0) - dy * (cx - ex0)
            side_j = dx * (jy - ey0) - dy * (jx - ex0)
            if side_c >= 0.0:
                qx[k] = cx
                qy[k] = cy
                k += 1
            if (side_c < 0.0) != (side_j < 0.0):
                denom = side_c - side_j
                if denom != 0.0:
                    t = side_c / denom
                    qx[k] = cx + t * (jx - cx)
                    qy[k] = cy + t * (jy - cy)
                    k += 1
        for i in range(k):
            px[i] = qx[i]
            py[i] = qy[i]
        m = k
    if m < 3:
        return 0.0
    s = 0.0
    for i in range(m):
        j = i + 1
        if j == m:
            j = 0
        s += px[i] * py[j] - px[j] * py[i]
    return abs(s) * 0.5


@njit(cache=True)
def _tri_sep_margin(ax0, ay0, ax1, ay1, ax2, ay2,
                    bx0, by0, bx1, by1, bx2, by2, eps):
    """True if some edge axis of CCW triangle A or B separates the triangles
    with margin >= eps (guaranteed zero-area intersection even under small
    float perturbation)."""
    for which in range(2):
        for e in range(3):
            if which == 0:
                if e == 0:
                    px0, py0, px1, py1 = ax0, ay0, ax1, ay1
                elif e == 1:
                    px0, py0, px1, py1 = ax1, ay1, ax2, ay2
                else:
                    px0, py0, px1, py1 = ax2, ay2, ax0, ay0
                q0x, q0y, q1x, q1y, q2x, q2y = bx0, by0, bx1, by1, bx2, by2
            else:
                if e == 0:
                    px0, py0, px1, py1 = bx0, by0, bx1, by1
                elif e == 1:
                    px0, py0, px1, py1 = bx1, by1, bx2, by2
                else:
                    px0, py0, px1, py1 = bx2, by2, bx0, by0
                q0x, q0y, q1x, q1y, q2x, q2y = ax0, ay0, ax1, ay1, ax2, ay2
            ex = px1 - px0
            ey = py1 - py0
            elen = (ex * ex + ey * ey) ** 0.5
            if elen < 1e-12:
                continue
            # CCW interior is left of the edge; other triangle fully on the
            # right side with margin => separated
            lim = -eps * elen
            c0 = ex * (q0y - py0) - ey * (q0x - px0)
            c1 = ex * (q1y - py0) - ey * (q1x - px0)
            c2 = ex * (q2y - py0) - ey * (q2x - px0)
            if c0 <= lim and c1 <= lim and c2 <= lim:
                return True
    return False


@njit(cache=True)
def _layers_overlap_exact(rowA, oxA, oyA, rowB, oxB, oyB,
                          tri_ptr, tri_cnt, tri_x, tri_y,
                          lbx0, lby0, lbx1, lby1, tau):
    """Tri-state exact overlap of layer rowA translated by (oxA, oyA) vs
    layer rowB translated by (oxB, oyB):
      0 = certainly zero-area (free), 1 = certainly positive-area overlap,
      2 = ambiguous (near-degenerate; caller must treat as blocked).
    AABB-disjoint (gap >= 0, exact float compare) implies area 0 exactly."""
    # AABB filter (touching boxes => intersection has zero area: safe free)
    if (lbx0[rowA] + oxA >= lbx1[rowB] + oxB or
            lbx0[rowB] + oxB >= lbx1[rowA] + oxA or
            lby0[rowA] + oyA >= lby1[rowB] + oyB or
            lby0[rowB] + oyB >= lby1[rowA] + oyA):
        return 0
    eps = 1e-7
    a0 = tri_ptr[rowA]
    a1 = a0 + tri_cnt[rowA]
    b0 = tri_ptr[rowB]
    b1 = b0 + tri_cnt[rowB]
    ambiguous = False
    for ta in range(a0, a1):
        ax0 = tri_x[3 * ta] + oxA
        ay0 = tri_y[3 * ta] + oyA
        ax1 = tri_x[3 * ta + 1] + oxA
        ay1 = tri_y[3 * ta + 1] + oyA
        ax2 = tri_x[3 * ta + 2] + oxA
        ay2 = tri_y[3 * ta + 2] + oyA
        amnx = min(ax0, min(ax1, ax2))
        amxx = max(ax0, max(ax1, ax2))
        amny = min(ay0, min(ay1, ay2))
        amxy = max(ay0, max(ay1, ay2))
        for tb in range(b0, b1):
            bx0 = tri_x[3 * tb] + oxB
            by0 = tri_y[3 * tb] + oyB
            bx1 = tri_x[3 * tb + 1] + oxB
            by1 = tri_y[3 * tb + 1] + oyB
            bx2 = tri_x[3 * tb + 2] + oxB
            by2 = tri_y[3 * tb + 2] + oyB
            if (amnx >= max(bx0, max(bx1, bx2)) or
                    min(bx0, min(bx1, bx2)) >= amxx or
                    amny >= max(by0, max(by1, by2)) or
                    min(by0, min(by1, by2)) >= amxy):
                continue
            area = _tri_clip_area(ax0, ay0, ax1, ay1, ax2, ay2,
                                  bx0, by0, bx1, by1, bx2, by2)
            if area > tau:
                return 1
            if not _tri_sep_margin(ax0, ay0, ax1, ay1, ax2, ay2,
                                   bx0, by0, bx1, by1, bx2, by2, eps):
                ambiguous = True
    if ambiguous:
        return 2
    return 0


@njit(cache=True)
def exact_position_ok(px, py, o_layer_base, K,
                      nd, jt1, jt2, jx, jy, joidx, t, t2,
                      layer_base, layer_cnt,
                      tri_ptr, tri_cnt, tri_x, tri_y,
                      lbx0, lby0, lbx1, lby1, tri_ok, tau):
    """Exact feasibility of placing the query block (layer rows
    o_layer_base .. o_layer_base+K-1, where K MUST be the layer count of the
    query orientation, i.e. layer_cnt[oidx]) at (px, py) for window [t, t2):
    replays the class rules with exact polygon overlap tests."""
    for q in range(nd):
        e1 = jt1[q]
        e2 = jt2[q]
        if e1 >= t2 or e2 <= t:
            continue
        fl = 0
        if (e1 <= t and t < e2) or (e1 < t2 and t2 <= e2):
            fl |= 1
        if (t < e1 and e1 < t2) or (t < e2 and e2 < t2):
            fl |= 2
        if fl == 0:
            continue
        dbase = layer_base[joidx[q]]
        nl = layer_cnt[joidx[q]]
        ox = jx[q]
        oy = jy[q]
        for l in range(nl):
            rowD = dbase + l
            if fl & 1:
                khi = l
                if khi > K - 1:
                    khi = K - 1
                for k in range(0, khi + 1):
                    rowI = o_layer_base + k
                    if tri_ok[rowI] == 0 or tri_ok[rowD] == 0:
                        return False  # no exact data: stay conservative
                    if _layers_overlap_exact(rowI, px, py, rowD, ox, oy,
                                             tri_ptr, tri_cnt, tri_x, tri_y,
                                             lbx0, lby0, lbx1, lby1,
                                             tau) != 0:
                        return False
            if fl & 2:
                if l <= K - 1:
                    for k in range(l, K):
                        rowI = o_layer_base + k
                        if tri_ok[rowI] == 0 or tri_ok[rowD] == 0:
                            return False
                        if _layers_overlap_exact(rowI, px, py, rowD, ox, oy,
                                                 tri_ptr, tri_cnt,
                                                 tri_x, tri_y,
                                                 lbx0, lby0, lbx1, lby1,
                                                 tau) != 0:
                            return False
    return True


# -----------------------------------------------------------------------------
# Full day-scan kernel: for one (block, bay) pair, find the earliest candidate
# entry day with a feasible position, scanning orientations, at machine speed.
# -----------------------------------------------------------------------------

@njit(cache=True)
def _stamp_out(Gf, WH, Hg, sgn, fl, row_base, nl, ox, oy, K,
               cell_ptr, cell_cnt, arena_u, arena_v):
    """Add (sgn=+1) or remove (sgn=-1) block D's OUTER cells for class bits fl
    in the count grid G.

    Gf is the FLAT (K*W*H,) view of the (K, W, H) grid, the same view
    day_scan's probe loop uses. For any contiguous (K, W, H),

        G[k, u + ox, v + oy] == Gf[k*WH + (u*Hg + v) + ox*Hg + oy]

    exactly. Hoisting ox*Hg + oy out of the loop turns the two multiplies of
    3-index addressing into one per cell. The cell sweep stays innermost so
    each k streams one grid plane; hoisting the cell loop outside the k walk
    measured 2x SLOWER on the stamp-heaviest instance (one extra cache miss
    per cell op).
    """
    obase = ox * Hg + oy
    for l in range(nl):
        row = row_base + l
        p0 = cell_ptr[row]
        p1 = p0 + cell_cnt[row]
        if fl & 1:
            khi = l if l < K - 1 else K - 1
            for k in range(0, khi + 1):
                kb = k * WH + obase
                for c in range(p0, p1):
                    Gf[kb + arena_u[c] * Hg + arena_v[c]] += sgn
        if fl & 2 and l <= K - 1:
            for k in range(l, K):
                kb = k * WH + obase
                for c in range(p0, p1):
                    Gf[kb + arena_u[c] * Hg + arena_v[c]] += sgn


@njit(cache=True)
def _stamp_in_s(Ginf, Ginb, WH, Hg, sgn, fl, row_base, nl, ox, oy, K,
                icell_ptr, icell_cnt, arena_iu, arena_iv, freecnt):
    """Inner cells: the Gin count grid, the occupancy BITBOARD the run-based
    screen reads (bit (k, y, x) is set exactly while Gin[k, x, y] != 0), and
    freecnt[k], the number of cells with Gin[k] == 0.

    Flat-addressed as _stamp_out. Splitting inner from outer costs one extra
    row loop when both are wanted, but lets day_scan skip the outer grid
    entirely on the 97-99% of orientation-days the screen rejects.
    """
    one = np.uint64(1)
    obase = ox * Hg + oy
    for l in range(nl):
        row = row_base + l
        i0 = icell_ptr[row]
        i1 = i0 + icell_cnt[row]
        if fl & 1:
            khi = l if l < K - 1 else K - 1
            for k in range(0, khi + 1):
                kb = k * WH + obase
                for c in range(i0, i1):
                    idx = kb + arena_iu[c] * Hg + arena_iv[c]
                    cnt = Ginf[idx] + sgn
                    Ginf[idx] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            freecnt[k] -= 1
                            uu = arena_iu[c] + ox
                            Ginb[k, arena_iv[c] + oy, uu >> 6] |= \
                                one << np.uint64(uu & 63)
                    elif cnt == 0:
                        freecnt[k] += 1
                        uu = arena_iu[c] + ox
                        Ginb[k, arena_iv[c] + oy, uu >> 6] &= \
                            ~(one << np.uint64(uu & 63))
        if fl & 2 and l <= K - 1:
            for k in range(l, K):
                kb = k * WH + obase
                for c in range(i0, i1):
                    idx = kb + arena_iu[c] * Hg + arena_iv[c]
                    cnt = Ginf[idx] + sgn
                    Ginf[idx] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            freecnt[k] -= 1
                            uu = arena_iu[c] + ox
                            Ginb[k, arena_iv[c] + oy, uu >> 6] |= \
                                one << np.uint64(uu & 63)
                    elif cnt == 0:
                        freecnt[k] += 1
                        uu = arena_iu[c] + ox
                        Ginb[k, arena_iv[c] + oy, uu >> 6] &= \
                            ~(one << np.uint64(uu & 63))


@njit(cache=True)
def _stamp_in(Ginf, WH, Hg, sgn, fl, row_base, nl, ox, oy, K,
              icell_ptr, icell_cnt, arena_iu, arena_iv, freecnt):
    """_stamp_in_s without the bitboard, for the use_screen=0 path."""
    obase = ox * Hg + oy
    for l in range(nl):
        row = row_base + l
        i0 = icell_ptr[row]
        i1 = i0 + icell_cnt[row]
        if fl & 1:
            khi = l if l < K - 1 else K - 1
            for k in range(0, khi + 1):
                kb = k * WH + obase
                for c in range(i0, i1):
                    idx = kb + arena_iu[c] * Hg + arena_iv[c]
                    cnt = Ginf[idx] + sgn
                    Ginf[idx] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            freecnt[k] -= 1
                    elif cnt == 0:
                        freecnt[k] += 1
        if fl & 2 and l <= K - 1:
            for k in range(l, K):
                kb = k * WH + obase
                for c in range(i0, i1):
                    idx = kb + arena_iu[c] * Hg + arena_iv[c]
                    cnt = Ginf[idx] + sgn
                    Ginf[idx] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            freecnt[k] -= 1
                    elif cnt == 0:
                        freecnt[k] += 1


@njit(cache=True)
def _stamp_bb(G, Gin, Gbit, Ginbit, sgn, fl, row_base, nl, ox, oy, K,
              cell_ptr, cell_cnt, arena_u, arena_v,
              icell_ptr, icell_cnt, arena_iu, arena_iv):
    """_stamp plus maintenance of the occupancy bit-grids: bit (k, y, x) is
    set exactly while the count grid at (k, x, y) is nonzero."""
    one = np.uint64(1)
    for l in range(nl):
        row = row_base + l
        p0 = cell_ptr[row]
        p1 = p0 + cell_cnt[row]
        i0 = icell_ptr[row]
        i1 = i0 + icell_cnt[row]
        if fl & 1:
            khi = l if l < K - 1 else K - 1
            for k in range(0, khi + 1):
                for c in range(p0, p1):
                    uu = arena_u[c] + ox
                    vv = arena_v[c] + oy
                    cnt = G[k, uu, vv] + sgn
                    G[k, uu, vv] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            Gbit[k, vv, uu >> 6] |= one << np.uint64(uu & 63)
                    elif cnt == 0:
                        Gbit[k, vv, uu >> 6] &= ~(one << np.uint64(uu & 63))
                for c in range(i0, i1):
                    uu = arena_iu[c] + ox
                    vv = arena_iv[c] + oy
                    cnt = Gin[k, uu, vv] + sgn
                    Gin[k, uu, vv] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            Ginbit[k, vv, uu >> 6] |= one << np.uint64(uu & 63)
                    elif cnt == 0:
                        Ginbit[k, vv, uu >> 6] &= ~(one << np.uint64(uu & 63))
        if fl & 2 and l <= K - 1:
            for k in range(l, K):
                for c in range(p0, p1):
                    uu = arena_u[c] + ox
                    vv = arena_v[c] + oy
                    cnt = G[k, uu, vv] + sgn
                    G[k, uu, vv] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            Gbit[k, vv, uu >> 6] |= one << np.uint64(uu & 63)
                    elif cnt == 0:
                        Gbit[k, vv, uu >> 6] &= ~(one << np.uint64(uu & 63))
                for c in range(i0, i1):
                    uu = arena_iu[c] + ox
                    vv = arena_iv[c] + oy
                    cnt = Gin[k, uu, vv] + sgn
                    Gin[k, uu, vv] = cnt
                    if sgn > 0:
                        if cnt == 1:
                            Ginbit[k, vv, uu >> 6] |= one << np.uint64(uu & 63)
                    elif cnt == 0:
                        Ginbit[k, vv, uu >> 6] &= ~(one << np.uint64(uu & 63))


# -----------------------------------------------------------------------------
# 192-bit primitives for the run-based feasibility screen
# -----------------------------------------------------------------------------
# A bay row is at most 192 columns, held as three little-endian uint64 words.
# `dilate(x, L) = OR_{d<L} (x >> d)` marks every column from which a run of
# length L would touch an occupied cell; it costs O(log L) shift-ORs by
# doubling rather than L of them.

@njit(cache=True, inline="always")
def _shr3(w0, w1, w2, n):
    """Shift a 192-bit little-endian value right by n in [0, 192)."""
    if n <= 0:
        return w0, w1, w2
    ws = n >> 6
    bs = n & 63
    if ws == 0:
        a0, a1, a2 = w0, w1, w2
    elif ws == 1:
        a0, a1, a2 = w1, w2, np.uint64(0)
    elif ws == 2:
        a0, a1, a2 = w2, np.uint64(0), np.uint64(0)
    else:
        return np.uint64(0), np.uint64(0), np.uint64(0)
    if bs == 0:
        return a0, a1, a2
    b = np.uint64(bs)
    ib = np.uint64(64 - bs)
    return (a0 >> b) | (a1 << ib), (a1 >> b) | (a2 << ib), a2 >> b


@njit(cache=True, inline="always")
def _shl3(w0, w1, w2, n):
    """Shift a 192-bit little-endian value left by n in [0, 192)."""
    if n <= 0:
        return w0, w1, w2
    ws = n >> 6
    bs = n & 63
    if ws == 0:
        a0, a1, a2 = w0, w1, w2
    elif ws == 1:
        a0, a1, a2 = np.uint64(0), w0, w1
    elif ws == 2:
        a0, a1, a2 = np.uint64(0), np.uint64(0), w0
    else:
        return np.uint64(0), np.uint64(0), np.uint64(0)
    if bs == 0:
        return a0, a1, a2
    b = np.uint64(bs)
    ib = np.uint64(64 - bs)
    return a0 << b, (a1 << b) | (a0 >> ib), (a2 << b) | (a1 >> ib)


@njit(cache=True)
def _any_free_pos(Ginbit, run_ptr, run_v, run_u0, run_len,
                  layer_base, layer_cnt, oidx,
                  pxlo, pxhi, pylo, pyhi, Hb):
    """Does ANY (px, py) in the box place every cell of orientation `oidx` on a
    Gin-free cell?

    This is exactly the position sweep's precondition (the sweep rejects a
    position the moment one cell hits Gin), so a False here can never discard a
    position the sweep would have accepted -- the screen is EXACT, and the
    scan's output stays bit-identical. 94-99% of scanned orientation-days have
    no placement at all, and each of those currently pays a full nx*ny sweep to
    discover it.
    """
    zero = np.uint64(0)
    one = np.uint64(1)
    ones = np.uint64(0xFFFFFFFFFFFFFFFF)
    nl = layer_cnt[oidx]
    olb = layer_base[oidx]
    # window mask over px in [pxlo, pxhi]
    m0, m1, m2 = _shl3(ones, ones, ones, pxlo)
    hi = pxhi + 1
    if hi < 192:
        k0, k1, k2 = _shl3(ones, ones, ones, hi)
        m0 &= ~k0
        m1 &= ~k1
        m2 &= ~k2
    if (m0 | m1 | m2) == zero:
        return True                      # empty window: let the sweep decide
    for py in range(pylo, pyhi + 1):
        a0, a1, a2 = m0, m1, m2
        dead = False
        for l in range(nl):
            row = olb + l
            for j in range(run_ptr[row], run_ptr[row + 1]):
                y = py + run_v[j]
                if y < 0 or y >= Hb:
                    dead = True
                    break
                o0w = Ginbit[l, y, 0]
                o1w = Ginbit[l, y, 1]
                o2w = Ginbit[l, y, 2]
                if (o0w | o1w | o2w) == zero:
                    continue             # row entirely free: blocks nothing
                # dilate by the run length, O(log L) shift-ORs
                L = run_len[j]
                sft = 1
                while sft < L:
                    t = sft
                    if L - sft < t:
                        t = L - sft
                    e0, e1, e2 = _shr3(o0w, o1w, o2w, t)
                    o0w |= e0
                    o1w |= e1
                    o2w |= e2
                    sft += t
                # positions blocked by this run: dilation shifted by -u0
                u0 = run_u0[j]
                if u0 >= 0:
                    b0, b1, b2 = _shr3(o0w, o1w, o2w, u0)
                else:
                    b0, b1, b2 = _shl3(o0w, o1w, o2w, -u0)
                a0 &= ~b0
                a1 &= ~b1
                a2 &= ~b2
                if (a0 | a1 | a2) == zero:
                    dead = True
                    break
            if dead:
                break
        if not dead and (a0 | a1 | a2) != zero:
            return True
    return False


@njit(cache=True)
def _class_of(e1, e2, t, t2):
    fl = 0
    if (e1 <= t and t < e2) or (e1 < t2 and t2 <= e2):
        fl |= 1
    if (t < e1 and e1 < t2) or (t < e2 and e2 < t2):
        fl |= 2
    return fl


@njit(cache=True)
def day_scan(G, Gin,                # (K, W, H) int16 scratch: outer / inner
             nb, jt1, jt2, jx, jy, joidx,   # committed blocks in bay
             layer_base, layer_cnt, cell_ptr, cell_cnt, arena_u, arena_v,
             icell_ptr, icell_cnt, arena_iu, arena_iv,
             scan_ptr, scan_u, scan_v, scan_k,
             o0, o1,                # global orient index range of block i
             fits_o,                # (nO,) uint8 for this bay
             pxlo_o, pxhi_o, pylo_o, pyhi_o,  # (nO,) position bounds, this bay
             R, P, D, stat, w1, best_cost,
             occ0, cap, need,       # quick reject: occ0[t] per day, capacity
             xw, yw, xdir, ydir, tmax,
             exact_budget,          # max exact checks per day (0 = raster only)
             tri_ptr, tri_cnt, tri_x, tri_y,
             lbx0, lby0, lbx1, lby1, tri_ok, tau,
             priceP, pw,            # space-time shadow prices (pw=0: off)
             Ginb, run_ptr, run_v, run_u0, run_len, use_screen):
    """Returns (t, o_local, px, py, cb) of the best-priced feasible day whose cost
    stat + w1*max(0, t+P-D) < best_cost, else t = -1.

    Position feasibility: certain-free if no outer-grid hit; certain-blocked
    if any inner-grid hit; otherwise uncertain and resolved with an exact
    triangulated overlap replay (budgeted per candidate day).

    The forbidden count-grids are maintained incrementally across candidate
    days: each committed block changes class O(1) times as the window slides,
    so total stamping work is O(nb * cells) for the whole scan."""
    K = G.shape[0]
    # ---- flat addressing -------------------------------------------------
    # G / Gin are C-contiguous (K, W, H). The inner probe used to evaluate
    # G[ck[c], cu[c] + px, cv[c] + py]: three loads plus two multiplies and
    # two adds per cell, ~300k positions per insertion. Precomputing
    # off[c] = ck*W*H + cu*H + cv once per CALL makes each probe one add and
    # one load, because
    #     G[k, u+px, v+py] == Gf[off + px*H + py]
    # exactly, for any contiguous (K, W, H). Addressing only: identical
    # loads, identical order, identical results (verified: 0 mismatches over
    # 3000 differential probes, byte-identical construction objective).
    Wg = G.shape[1]
    Hg = G.shape[2]
    Gf = G.reshape(K * Wg * Hg)
    Ginf = Gin.reshape(K * Wg * Hg)
    c_lo = scan_ptr[o0]
    c_hi = scan_ptr[o1]
    offs = np.empty(c_hi - c_lo, np.int64)
    for c in range(c_lo, c_hi):
        offs[c - c_lo] = (scan_k[c] * Wg + scan_u[c]) * Hg + scan_v[c]
    # upper-bound day: bay empty from t_ub on
    t_ub = R
    for q in range(nb):
        if jt2[q] > t_ub:
            t_ub = jt2[q]
    # Zero grids and stamp the initial day's classes.
    #
    # G (outer counts) is read ONLY by the position probe's `uncertain` test,
    # and the screen rejects 97-99% of orientation-days before any probe
    # happens -- so maintaining G across every candidate day is mostly work
    # that is never read. Only the INNER grids (Gin, its bitboard, freecnt),
    # which the screen itself needs, are kept up to date eagerly. G is
    # materialised from fl_cur the first time a probe actually needs it, and
    # kept live from then on, so a scan pays for it at most once and scans that
    # never reach a probe never build it at all.
    WH = Wg * Hg
    g_live = False
    Ginf[:] = 0
    if use_screen:
        for k in range(K):
            for yy in range(Ginb.shape[1]):
                Ginb[k, yy, 0] = np.uint64(0)
                Ginb[k, yy, 1] = np.uint64(0)
                Ginb[k, yy, 2] = np.uint64(0)
    freecnt = np.empty(K, np.int64)
    ncell_bay = G.shape[1] * G.shape[2]
    for k in range(K):
        freecnt[k] = ncell_bay
    fl_cur = np.zeros(nb, np.int64)
    t = R
    t2 = t + P
    for q in range(nb):
        fl = _class_of(jt1[q], jt2[q], t, t2)
        fl_cur[q] = fl
        if fl:
            if use_screen:
                _stamp_in_s(Ginf, Ginb, WH, Hg, 1, fl, layer_base[joidx[q]],
                            layer_cnt[joidx[q]], jx[q], jy[q], K,
                            icell_ptr, icell_cnt, arena_iu, arena_iv, freecnt)
            else:
                _stamp_in(Ginf, WH, Hg, 1, fl, layer_base[joidx[q]],
                          layer_cnt[joidx[q]], jx[q], jy[q], K,
                          icell_ptr, icell_cnt, arena_iu, arena_iv, freecnt)
    found_t = -1
    found_o = -1
    found_px = -1
    found_py = -1
    found_cb = best_cost
    # MONOTONE SCREEN CARRY-OVER. Both day-level screens are monotone in
    # occupancy: "fewer free cells than this layer needs" and "no position puts
    # every cell on a Gin-free cell" can only stay true when occupancy GROWS.
    # So when a day transition is purely additive (no class bit removed
    # anywhere), every orientation rejected on the previous day is still
    # rejected and its screen can be skipped entirely -- and the screen is
    # 19-29% of this kernel. One bit per local orientation in an int64, so no
    # allocation; any removal clears the mask and re-screens from scratch.
    rej_mask = 0
    use_rej = (o1 - o0) <= 63
    while True:
        tard = t + P - D
        if tard < 0:
            tard = 0
        cfree = stat + w1 * tard
        if cfree >= found_cb:
            return found_t, found_o, found_px, found_py, found_cb
        if t >= tmax:
            return found_t, found_o, found_px, found_py, found_cb
        t2 = t + P
        # quick reject on layer-0 free area
        reject = False
        if t < t_ub:
            for d in range(t, t + P):
                if d >= occ0.shape[0]:
                    break
                if cap - occ0[d] < need:
                    reject = True
                    break
        if not reject:
            # scan orientations
            best_o = -1
            best_px = -1
            best_py = -1
            best_s = 1e30
            budget = exact_budget
            for oidx in range(o0, o1):
                if fits_o[oidx] == 0:
                    continue
                o_bit = 1 << (oidx - o0)
                if use_rej and (rej_mask & o_bit) != 0:
                    continue        # rejected earlier, occupancy only grew
                # impossibility screen: every outer cell of layer l must land
                # on a Gin-free cell, so fewer free cells than the layer's
                # cell count means no feasible position exists this day
                poss = True
                for l in range(layer_cnt[oidx]):
                    if freecnt[l] < cell_cnt[layer_base[oidx] + l]:
                        poss = False
                        break
                if not poss:
                    rej_mask |= o_bit
                    continue
                if use_screen and not _any_free_pos(
                        Ginb, run_ptr, run_v, run_u0, run_len,
                        layer_base, layer_cnt, oidx,
                        pxlo_o[oidx], pxhi_o[oidx],
                        pylo_o[oidx], pyhi_o[oidx], G.shape[2]):
                    rej_mask |= o_bit
                    continue
                if not g_live:
                    # first probe of this scan: build G for the CURRENT window
                    # from fl_cur, which already holds every committed block's
                    # class. Identical to the incrementally maintained grid --
                    # a count grid is a pure function of {(block, class)}, and
                    # the per-cell adds commute.
                    Gf[:] = 0
                    for q2 in range(nb):
                        f2 = fl_cur[q2]
                        if f2:
                            _stamp_out(Gf, WH, Hg, 1, f2,
                                       layer_base[joidx[q2]],
                                       layer_cnt[joidx[q2]],
                                       jx[q2], jy[q2], K,
                                       cell_ptr, cell_cnt, arena_u, arena_v)
                    g_live = True
                s0 = scan_ptr[oidx]
                s1 = scan_ptr[oidx + 1]
                ob = s0 - c_lo          # base into the precomputed offsets
                pxlo = pxlo_o[oidx]
                pxhi = pxhi_o[oidx]
                pylo = pylo_o[oidx]
                pyhi = pyhi_o[oidx]
                nx = pxhi - pxlo + 1
                ny = pyhi - pylo + 1
                nc = s1 - s0
                olb = layer_base[oidx]
                for a in range(nx):
                    px = pxlo + a if xdir > 0 else pxhi - a
                    base_s = xw * a
                    if base_s >= best_s:
                        break
                    pxH = px * Hg
                    for bcnt in range(ny):
                        py = pylo + bcnt if ydir > 0 else pyhi - bcnt
                        s = base_s + yw * bcnt
                        if s >= best_s:
                            break
                        ok = True
                        uncertain = False
                        base = pxH + py
                        for c in range(nc):
                            idx = offs[ob + c] + base
                            if Ginf[idx] != 0:
                                ok = False
                                break
                            if Gf[idx] != 0:
                                uncertain = True
                        if ok and uncertain:
                            if budget > 0:
                                budget -= 1
                                # pass THIS orientation's true layer count so
                                # layer-row indexing never crosses into other
                                # orientations' rows
                                ok = exact_position_ok(
                                    px, py, olb, layer_cnt[oidx],
                                    nb, jt1, jt2, jx, jy, joidx, t, t2,
                                    layer_base, layer_cnt,
                                    tri_ptr, tri_cnt, tri_x, tri_y,
                                    lbx0, lby0, lbx1, lby1, tri_ok, tau)
                            else:
                                ok = False
                        if ok:
                            best_s = s
                            best_o = oidx - o0
                            best_px = px
                            best_py = py
                            break
            if best_o >= 0:
                cb = cfree
                if pw > 0.0:
                    cb = cb + pw * (priceP[t2] - priceP[t])
                if cb < found_cb:
                    found_cb = cb
                    found_t = t
                    found_o = best_o
                    found_px = best_px
                    found_py = best_py
                if pw == 0.0:
                    # unpriced: the first accepted day is final (identical
                    # to the historical earliest-day semantics)
                    return found_t, found_o, found_px, found_py, found_cb
        if t >= t_ub:
            return found_t, found_o, found_px, found_py, found_cb
        # next candidate day: event day or event-P+1 crossing
        tn = t_ub
        for q in range(nb):
            e = jt1[q]
            if e > t and e < tn:
                tn = e
            e = e - P + 1
            if e > t and e < tn:
                tn = e
            e = jt2[q]
            if e > t and e < tn:
                tn = e
            e = e - P + 1
            if e > t and e < tn:
                tn = e
        t = tn
        # incremental class update for the new window [t, t+P)
        t2n = t + P
        any_rem = False
        for q in range(nb):
            fl = _class_of(jt1[q], jt2[q], t, t2n)
            if fl != fl_cur[q]:
                rem = fl_cur[q] & ~fl
                add = fl & ~fl_cur[q]
                if rem:
                    any_rem = True
                if g_live:
                    # only worth maintaining once something has read it
                    if rem:
                        _stamp_out(Gf, WH, Hg, -1, rem,
                                   layer_base[joidx[q]], layer_cnt[joidx[q]],
                                   jx[q], jy[q], K,
                                   cell_ptr, cell_cnt, arena_u, arena_v)
                    if add:
                        _stamp_out(Gf, WH, Hg, 1, add,
                                   layer_base[joidx[q]], layer_cnt[joidx[q]],
                                   jx[q], jy[q], K,
                                   cell_ptr, cell_cnt, arena_u, arena_v)
                if use_screen:
                    if rem:
                        _stamp_in_s(Ginf, Ginb, WH, Hg, -1, rem,
                                    layer_base[joidx[q]], layer_cnt[joidx[q]],
                                    jx[q], jy[q], K,
                                    icell_ptr, icell_cnt, arena_iu, arena_iv,
                                    freecnt)
                    if add:
                        _stamp_in_s(Ginf, Ginb, WH, Hg, 1, add,
                                    layer_base[joidx[q]], layer_cnt[joidx[q]],
                                    jx[q], jy[q], K,
                                    icell_ptr, icell_cnt, arena_iu, arena_iv,
                                    freecnt)
                else:
                    if rem:
                        _stamp_in(Ginf, WH, Hg, -1, rem,
                                  layer_base[joidx[q]], layer_cnt[joidx[q]],
                                  jx[q], jy[q], K,
                                  icell_ptr, icell_cnt, arena_iu, arena_iv,
                                  freecnt)
                    if add:
                        _stamp_in(Ginf, WH, Hg, 1, add,
                                  layer_base[joidx[q]], layer_cnt[joidx[q]],
                                  jx[q], jy[q], K,
                                  icell_ptr, icell_cnt, arena_iu, arena_iv,
                                  freecnt)
                fl_cur[q] = fl
        if any_rem:
            rej_mask = 0        # occupancy shrank: earlier rejections expire


@njit(cache=True)
def day_scan_bb(G, Gin, Gbit, Ginbit,   # count grids + occupancy bit-grids
                nb, jt1, jt2, jx, jy, joidx,
                layer_base, layer_cnt, cell_ptr, cell_cnt, arena_u, arena_v,
                icell_ptr, icell_cnt, arena_iu, arena_iv,
                rm_ptr, rm_umin, rm_vmin, rm_nrows, rm_mask,
                o0, o1, fits_o,
                pxlo_o, pxhi_o, pylo_o, pyhi_o,
                R, P, D, stat, w1, best_cost,
                occ0, cap, need,
                xw, yw, xdir, ydir, tmax,
                exact_budget,
                tri_ptr, tri_cnt, tri_x, tri_y,
                lbx0, lby0, lbx1, lby1, tri_ok, tau):
    """day_scan with the per-cell position probe replaced by word-parallel
    row-mask tests. Verdicts are identical by construction: any inner-grid
    bit under the block's outer mask = certainly blocked; any outer-grid bit
    = uncertain (exact check); neither = certainly free. The masks are
    shifted once per (orientation, px) column and reused for every py.
    Requires bay width <= 192 and per-layer bbox width <= 192."""
    K = G.shape[0]
    t_ub = R
    for q in range(nb):
        if jt2[q] > t_ub:
            t_ub = jt2[q]
    zero = np.uint64(0)
    # scratch for the per-column shifted masks + per-layer row metadata
    mxrows = 1
    mxnl = 1
    for oidx in range(o0, o1):
        nlo = layer_cnt[oidx]
        if nlo > mxnl:
            mxnl = nlo
        tot = 0
        for l in range(nlo):
            tot += rm_nrows[layer_base[oidx] + l]
        if tot > mxrows:
            mxrows = tot
    shmask = np.empty((mxrows, 3), np.uint64)
    lay_off = np.empty(mxnl, np.int64)
    lay_nr = np.empty(mxnl, np.int64)
    lay_v0 = np.empty(mxnl, np.int64)
    for k in range(K):
        for a in range(G.shape[1]):
            for bb in range(G.shape[2]):
                G[k, a, bb] = 0
                Gin[k, a, bb] = 0
    for k in range(K):
        for a in range(Gbit.shape[1]):
            for bb in range(3):
                Gbit[k, a, bb] = zero
                Ginbit[k, a, bb] = zero
    fl_cur = np.zeros(nb, np.int64)
    t = R
    t2 = t + P
    for q in range(nb):
        fl = _class_of(jt1[q], jt2[q], t, t2)
        fl_cur[q] = fl
        if fl:
            _stamp_bb(G, Gin, Gbit, Ginbit, 1, fl,
                      layer_base[joidx[q]], layer_cnt[joidx[q]],
                      jx[q], jy[q], K,
                      cell_ptr, cell_cnt, arena_u, arena_v,
                      icell_ptr, icell_cnt, arena_iu, arena_iv)
    while True:
        tard = t + P - D
        if tard < 0:
            tard = 0
        if stat + w1 * tard >= best_cost:
            return -1, -1, -1, -1
        if t >= tmax:
            return -1, -1, -1, -1
        t2 = t + P
        reject = False
        if t < t_ub:
            for d in range(t, t + P):
                if d >= occ0.shape[0]:
                    break
                if cap - occ0[d] < need:
                    reject = True
                    break
        if not reject:
            best_o = -1
            best_px = -1
            best_py = -1
            best_s = 1e30
            budget = exact_budget
            for oidx in range(o0, o1):
                if fits_o[oidx] == 0:
                    continue
                pxlo = pxlo_o[oidx]
                pxhi = pxhi_o[oidx]
                pylo = pylo_o[oidx]
                pyhi = pyhi_o[oidx]
                nx = pxhi - pxlo + 1
                ny = pyhi - pylo + 1
                olb = layer_base[oidx]
                nl = layer_cnt[oidx]
                for a in range(nx):
                    px = pxlo + a if xdir > 0 else pxhi - a
                    base_s = xw * a
                    if base_s >= best_s:
                        break
                    # shift this orientation's mask rows to column px once;
                    # the py loop below only ANDs
                    pos = 0
                    for l in range(nl):
                        r = olb + l
                        nr = rm_nrows[r]
                        lay_off[l] = pos
                        lay_nr[l] = nr
                        lay_v0[l] = rm_vmin[r]
                        if nr <= 0:
                            continue
                        mb = rm_ptr[r]
                        sft = px + rm_umin[r]
                        woff = sft >> 6
                        rr = sft & 63
                        rr_u = np.uint64(rr)
                        nrr_u = np.uint64(64 - rr)
                        for dy in range(nr):
                            for w in range(3):
                                src = w - woff
                                mm = zero
                                if 0 <= src < 3:
                                    mm = rm_mask[mb + dy, src] << rr_u
                                if rr != 0 and 0 <= src - 1 < 3:
                                    mm |= rm_mask[mb + dy, src - 1] >> nrr_u
                                shmask[pos, w] = mm
                            pos += 1
                    for bcnt in range(ny):
                        py = pylo + bcnt if ydir > 0 else pyhi - bcnt
                        s = base_s + yw * bcnt
                        if s >= best_s:
                            break
                        ok = True
                        uncertain = False
                        for l in range(nl):
                            nr = lay_nr[l]
                            if nr <= 0:
                                continue
                            off = lay_off[l]
                            yb = py + lay_v0[l]
                            for dy in range(nr):
                                yy = yb + dy
                                for w in range(3):
                                    mm = shmask[off + dy, w]
                                    if mm == zero:
                                        continue
                                    if (Ginbit[l, yy, w] & mm) != zero:
                                        ok = False
                                        break
                                    if (Gbit[l, yy, w] & mm) != zero:
                                        uncertain = True
                                if not ok:
                                    break
                            if not ok:
                                break
                        if ok and uncertain:
                            if budget > 0:
                                budget -= 1
                                ok = exact_position_ok(
                                    px, py, olb, nl,
                                    nb, jt1, jt2, jx, jy, joidx, t, t2,
                                    layer_base, layer_cnt,
                                    tri_ptr, tri_cnt, tri_x, tri_y,
                                    lbx0, lby0, lbx1, lby1, tri_ok, tau)
                            else:
                                ok = False
                        if ok:
                            best_s = s
                            best_o = oidx - o0
                            best_px = px
                            best_py = py
                            break
            if best_o >= 0:
                return t, best_o, best_px, best_py
        if t >= t_ub:
            return -1, -1, -1, -1
        tn = t_ub
        for q in range(nb):
            e = jt1[q]
            if e > t and e < tn:
                tn = e
            e = e - P + 1
            if e > t and e < tn:
                tn = e
            e = jt2[q]
            if e > t and e < tn:
                tn = e
            e = e - P + 1
            if e > t and e < tn:
                tn = e
        t = tn
        t2n = t + P
        for q in range(nb):
            fl = _class_of(jt1[q], jt2[q], t, t2n)
            if fl != fl_cur[q]:
                rem = fl_cur[q] & ~fl
                add = fl & ~fl_cur[q]
                if rem:
                    _stamp_bb(G, Gin, Gbit, Ginbit, -1, rem,
                              layer_base[joidx[q]], layer_cnt[joidx[q]],
                              jx[q], jy[q], K,
                              cell_ptr, cell_cnt, arena_u, arena_v,
                              icell_ptr, icell_cnt, arena_iu, arena_iv)
                if add:
                    _stamp_bb(G, Gin, Gbit, Ginbit, 1, add,
                              layer_base[joidx[q]], layer_cnt[joidx[q]],
                              jx[q], jy[q], K,
                              cell_ptr, cell_cnt, arena_u, arena_v,
                              icell_ptr, icell_cnt, arena_iu, arena_iv)
                fl_cur[q] = fl


def warmup():
    """Trigger JIT compilation with tiny inputs."""
    xs = np.array([0.0, 2.0, 2.0, 0.0])
    ys = np.array([0.0, 0.0, 2.0, 2.0])
    raster_poly(xs, ys, 0, 0, 2, 2, 1e-6, 1e-12)
    G = np.zeros((2, 8, 8), np.int16)
    dx = np.zeros(1, np.int64)
    dy = np.zeros(1, np.int64)
    doidx = np.zeros(1, np.int64)
    dflags = np.ones(1, np.int64)
    layer_base = np.zeros(1, np.int64)
    layer_cnt = np.ones(1, np.int64)
    cell_ptr = np.zeros(1, np.int64)
    cell_cnt = np.ones(1, np.int64)
    arena_u = np.zeros(1, np.int64)
    arena_v = np.zeros(1, np.int64)
    build_G(G, 1, dx, dy, doidx, dflags, layer_base, layer_cnt,
            cell_ptr, cell_cnt, arena_u, arena_v)
    cu = np.zeros(1, np.int64)
    cv = np.zeros(1, np.int64)
    ck = np.zeros(1, np.int64)
    scan_first(G, cu, cv, ck, 0, 3, 0, 3)
    scan_best(G, cu, cv, ck, 0, 3, 0, 3, 1.0, 0.1, 1, 1)
    any_feasible(G, cu, cv, ck, 0, 3, 0, 3)
    scan_ptr = np.array([0, 1], np.int64)
    fits_o = np.ones(1, np.uint8)
    pb = np.zeros(1, np.int64)
    pe = np.full(1, 3, np.int64)
    occ0 = np.zeros(64, np.int64)
    Gin = np.zeros_like(G)
    tri_ptr = np.zeros(1, np.int64)
    tri_cnt = np.ones(1, np.int64)
    tri_x = np.array([0.0, 1.0, 0.0])
    tri_y = np.array([0.0, 0.0, 1.0])
    lb0 = np.zeros(1)
    lb1 = np.ones(1)
    tri_ok = np.ones(1, np.uint8)
    day_scan(G, Gin, 1, np.zeros(1, np.int64), np.ones(1, np.int64),
             np.zeros(1, np.int64), np.zeros(1, np.int64),
             np.zeros(1, np.int64),
             layer_base, layer_cnt, cell_ptr, cell_cnt, arena_u, arena_v,
             cell_ptr, cell_cnt, arena_u, arena_v,
             scan_ptr, cu, cv, ck,
             0, 1, fits_o, pb, pe, pb, pe,
             0, 2, 5, 0.0, 1.0, 1e18,
             occ0, 64, 1,
             1.0, 0.001, 1, 1, 60,
             8, tri_ptr, tri_cnt, tri_x, tri_y,
             lb0, lb0, lb1, lb1, tri_ok, 1e-9,
             np.zeros(70, np.float64), 0.0,
             np.zeros((2, 10, 3), np.uint64),
             np.zeros(2, np.int64), np.zeros(1, np.int64),
             np.zeros(1, np.int64), np.ones(1, np.int64), True)
    raster_poly_inner(xs, ys, 0, 0, 2, 2)
    E8 = np.zeros((8, 8), np.int64)
    stamp_exit(E8, cu, cv, 0, 0, 3)
    scan_exit_contact(G, E8, cu, cv, ck, cu, cv,
                      int(pb[0]), int(pe[0]), int(pb[0]), int(pe[0]),
                      2, 5, 1.0, 0.001, 1, 1, 8, 99, 99)
    exact_position_ok(0, 0, 0, 1,
                      1, np.zeros(1, np.int64), np.ones(1, np.int64),
                      np.zeros(1, np.int64), np.zeros(1, np.int64),
                      np.zeros(1, np.int64), 0, 2,
                      layer_base, layer_cnt,
                      tri_ptr, tri_cnt, tri_x, tri_y,
                      lb0, lb0, lb1, lb1, tri_ok, 1e-9)
    import os
    if os.environ.get("OGC_BITBOARD", "0") == "1":
        Gbit = np.zeros((2, 8, 3), np.uint64)
        Ginbit = np.zeros((2, 8, 3), np.uint64)
        rm_mask = np.ones((1, 3), np.uint64)
        day_scan_bb(G, Gin, Gbit, Ginbit,
                    1, np.zeros(1, np.int64), np.ones(1, np.int64),
                    np.zeros(1, np.int64), np.zeros(1, np.int64),
                    np.zeros(1, np.int64),
                    layer_base, layer_cnt, cell_ptr, cell_cnt,
                    arena_u, arena_v,
                    cell_ptr, cell_cnt, arena_u, arena_v,
                    np.zeros(1, np.int64), np.zeros(1, np.int64),
                    np.zeros(1, np.int64), np.ones(1, np.int64), rm_mask,
                    0, 1, fits_o, pb, pe, pb, pe,
                    0, 2, 5, 0.0, 1.0, 1e18,
                    occ0, 64, 1,
                    1.0, 0.001, 1, 1, 60,
                    8, tri_ptr, tri_cnt, tri_x, tri_y,
                    lb0, lb0, lb1, lb1, tri_ok, 1e-9)
