# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""BRKGA over construction order (Goncalves & Resende 2011).

Construction quality decides overloaded instances (LNS only improves ~4%
post-construction), so search the construction ORDER itself: a random-key
vector over blocks decodes to an insertion order via argsort, and the decoder
is the existing greedy `construct()` + `day_scan` placement -- the closest
published precedent is the BRKGA family for 2D/3D packing where the decoder
is exactly such a placement heuristic.

The initial population is seeded with the four historical EDD construction
variants, and those four are always evaluated even if the time budget is
gone, so the BRKGA result can never be worse than the best historical
construction for the island.
"""

import time

import numpy as np

from _ogc.ogc_search import construct


def _variant_order(p, variant):
    """Mirror ogc_search.construct's built-in orderings."""
    if variant == 1:
        return sorted(range(p.n), key=lambda i: (p.R[i], p.D[i]))
    if variant == 2:
        o0 = p.orient_base[:-1]
        return sorted(range(p.n),
                      key=lambda i: (p.D[i],
                                     int(p.P[i]) * int(p.ncells0[o0[i]])))
    if variant == 3:
        o0 = p.orient_base[:-1]
        return sorted(range(p.n),
                      key=lambda i: (p.D[i], -int(p.ncells0[o0[i]])))
    return sorted(range(p.n), key=lambda i: (p.D[i], p.R[i], -p.wl[i]))


def brkga_construct(state, rng, t_deadline, log=None,
                    pop_size=24, pe=0.25, pm=0.15, rho=0.7):
    """Evolve construction orders until t_deadline; leave `state` holding the
    best construction found. Returns its objective."""
    p = state.prep
    n = p.n
    empty = state.snapshot()

    def keys_of(order):
        k = np.empty(n)
        k[np.asarray(order)] = np.linspace(0.0, 1.0, n)
        return k

    def decode(keys):
        state.restore(empty)
        construct(state, order=list(np.argsort(keys, kind="stable")),
                  deadline=t_deadline, look=False)
        return state.objective()

    # initial population: the 4 EDD variants, gaussian jitter around them,
    # and uniform-random immigrants
    pop = [keys_of(_variant_order(p, v)) for v in range(4)]
    while len(pop) < pop_size:
        if rng.random() < 0.5:
            base = pop[int(rng.integers(0, 4))]
            pop.append(np.clip(base + rng.normal(0.0, 0.08, n), 0.0, 1.0))
        else:
            pop.append(rng.random(n))

    best_f, best_snap = float("inf"), None
    fit = []
    evals = 0
    for q, keys in enumerate(pop):
        if time.time() >= t_deadline and q >= 4:
            fit.append(float("inf"))
            continue
        f = decode(keys)
        evals += 1
        fit.append(f)
        if f < best_f:
            best_f = f
            best_snap = state.snapshot()

    ne = max(2, int(pe * pop_size))
    nm = max(1, int(pm * pop_size))
    gen = 0
    while time.time() < t_deadline:
        gen += 1
        idx = np.argsort(fit)
        elites = [pop[int(j)] for j in idx[:ne]]
        newpop = list(elites)
        newfit = [fit[int(j)] for j in idx[:ne]]
        for _ in range(nm):
            newpop.append(rng.random(n))
            newfit.append(None)
        while len(newpop) < pop_size:
            e = elites[int(rng.integers(0, ne))]
            o = pop[int(rng.integers(0, len(pop)))]
            child = np.where(rng.random(n) < rho, e, o)
            newpop.append(child)
            newfit.append(None)
        pop = newpop
        fit = []
        for q, keys in enumerate(pop):
            if newfit[q] is not None:
                fit.append(newfit[q])
                continue
            if time.time() >= t_deadline:
                fit.append(float("inf"))
                continue
            f = decode(keys)
            evals += 1
            fit.append(f)
            if f < best_f:
                best_f = f
                best_snap = state.snapshot()

    # Every decode already produced a complete state and the best one's
    # snapshot was retained above. v9 rebuilt the winner once (twice with
    # lookahead) using fresh 30-second deadlines *after* t_deadline, allowing
    # the bounded BRKGA phase to overrun by as much as a minute.
    if best_snap is not None:
        state.restore(best_snap)
    else:
        state.restore(empty)
    if log:
        log(f"brkga gens={gen} evals={evals} obj={state.objective():.0f}")
    return state.objective()
