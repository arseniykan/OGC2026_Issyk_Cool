# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""
crane.py — crane constraint as a relocation / accessibility partial order.

The crane rule (spec 1.3): an ENTRY or EXIT of block i on day t in bay j is
feasible only if, at the instant of the operation, no other present block
obstructs i's straight vertical lift (some level l2>=l1 of the other block
overlaps i's level-l1 footprint).

Because ENTRY/EXIT change the present set, feasibility is about the ORDER of
operations within a day, exactly like container pre-marshalling / block
relocation.  We model an obstruction digraph and peel it (Kahn's algorithm,
~40 lines, no networkx — see guide 12A.5).

Key simplification (proved from the data): for single-layer (K=1) blocks the
crane can never be blocked once same-level collision-freeness holds, because
crane_blocked_by only fires on a level pair l1<=l2 and the l2==l1 case is
exactly the same-level test.  So crane only constrains multi-layer overhangs.
"""
from . import geo


def obstructs(prob, sol, mover, other):
    """Does `other` block the straight vertical lift of `mover` (both placed)?"""
    if prob.K[mover] == 1 and prob.K[other] == 1:
        return False  # single-layer: collision-freeness already guarantees liftability
    ogm = sol.og(mover)
    ogo = sol.og(other)
    return geo.crane_blocked_by(ogm, int(sol.x[mover]), int(sol.y[mover]),
                                ogo, int(sol.x[other]), int(sol.y[other]))


def _build_obstruction(prob, sol, movers, against):
    """
    obstruction[m] = set of blocks in `against` that block mover m's lift.
    movers: iterable of block ids we want to move (enter/exit).
    against: iterable of blocks that could obstruct (present blocks).
    """
    against = list(against)
    obs = {}
    for m in movers:
        s = set()
        for o in against:
            if o == m:
                continue
            if obstructs(prob, sol, m, o):
                s.add(o)
        obs[m] = s
    return obs


def order_exits(prob, sol, prev_present, exiting):
    """
    Feasible EXIT order for `exiting` (subset of prev_present) or None.
    A block can exit when no still-present block obstructs it. Non-exiting
    blocks in prev_present stay for the whole process.
    """
    exiting = list(exiting)
    if not exiting:
        return []
    present = set(prev_present)
    # obstruction of each exiting block by ANY currently-present block
    order = []
    remaining = set(exiting)
    # precompute static obstruction sets against the full prev_present universe
    obs = _build_obstruction(prob, sol, exiting, prev_present)
    guard = 0
    while remaining and guard <= len(exiting) + 1:
        guard += 1
        progressed = False
        for b in list(remaining):
            # b is liftable if none of its obstructors are still present
            if not (obs[b] & present):
                order.append(b)
                remaining.discard(b)
                present.discard(b)
                progressed = True
        if not progressed:
            return None  # obstruction cycle -> spatial infeasibility
    return order if not remaining else None


def order_entries(prob, sol, base_present, entering):
    """
    Feasible ENTRY order for `entering` given `base_present` already in the bay
    (after this day's exits), or None.

    An entry (lowering a block into place) is the time-reverse of an exit
    (lifting it out): a feasible insertion order is exactly the reverse of a
    feasible peeling (exit) order of the entering set out of the full present
    state.  This peeling also correctly detects the infeasible case where a
    permanently-present base block obstructs an entering block (it can never be
    peeled), so a naive "pick any insertable" greedy — which is NOT complete —
    is avoided.
    """
    entering = list(entering)
    if not entering:
        return []
    full_present = list(base_present) + entering
    peel = order_exits(prob, sol, full_present, entering)
    if peel is None:
        return None
    return list(reversed(peel))


def can_add(prob, sol, i, cop_ids):
    """
    Fast O(k) incremental crane check: assuming the bay was crane-feasible before,
    can block i (tentatively placed in sol) be added while keeping every ENTRY/EXIT
    orderable against the co-present blocks `cop_ids`?

    Covers all pairwise (i,c) obstruction/timing cases and same-day mutual cycles.
    Rare higher-order cycles through i are not detected here — the full
    bay_crane_ok in the final feasibility check is the guarantee; construction
    uses this filter for speed and repairs any residue.
    """
    if prob.K[i] == 1 and all(prob.K[c] == 1 for c in cop_ids):
        return True
    ei = int(sol.entry[i]); xi = int(sol.exit[i])
    for c in cop_ids:
        ec = int(sol.entry[c]); xc = int(sol.exit[c])
        i_above_c = obstructs(prob, sol, c, i)   # i blocks c's lift (i above c)
        c_above_i = obstructs(prob, sol, i, c)   # c blocks i's lift (c above i)
        if not (i_above_c or c_above_i):
            continue
        # --- i's ENTRY (day ei): c firmly present above i ---
        if c_above_i and ec < ei < xc:
            return False
        # --- i's EXIT (day xi): c present above i (c stays past xi) ---
        if c_above_i and ec <= xi < xc:
            return False
        # --- c's EXIT (day xc) while i firmly present above c ---
        if i_above_c and ei < xc < xi:
            return False
        # --- c's ENTRY (day ec) while i firmly present above c ---
        if i_above_c and ei < ec < xi:
            return False
        # --- same-day entry: mutual obstruction is unorderable ---
        if ei == ec and i_above_c and c_above_i:
            return False
        # --- same-day exit: mutual obstruction is unorderable ---
        if xi == xc and i_above_c and c_above_i:
            return False
    return True


def bay_crane_ok(prob, sol, j, members, lo=None, hi=None):
    """
    True iff every ENTRY/EXIT among `members` (block ids in bay j) has a feasible
    crane order on every event day (optionally restricted to days in [lo,hi]).
    Cheap early-out when all members are single-layer.
    """
    if not members:
        return True
    if all(prob.K[i] == 1 for i in members):
        return True  # single-layer: collision-freeness => always liftable
    ev = set()
    for i in members:
        ev.add(int(sol.entry[i]))
        ev.add(int(sol.exit[i]))
    if lo is not None:
        ev = {t for t in ev if lo <= t <= hi}
    for t in sorted(ev):
        prev = [i for i in members if sol.entry[i] <= t - 1 < sol.exit[i]]
        exiting = [i for i in members if sol.exit[i] == t]
        entering = [i for i in members if sol.entry[i] == t]
        if exiting and order_exits(prob, sol, prev, exiting) is None:
            return False
        base = [i for i in prev if i not in set(exiting)]
        if entering and order_entries(prob, sol, base, entering) is None:
            return False
    return True


def crane_certificate(prob, sol, movers, against, is_exit):
    """
    When ordering is impossible, return a minimal set of blocks whose mutual
    obstruction causes the failure (for LBBD no-good cuts / targeted repair).
    """
    if is_exit:
        obs = _build_obstruction(prob, sol, movers, against)
    else:
        obs = _build_obstruction(prob, sol, movers, set(against) | set(movers))
    # collect blocks that participate in an unresolved obstruction
    cert = set()
    for m, s in obs.items():
        if s:
            cert.add(m)
            cert |= s
    return frozenset(cert)
