# Architecture notes

Supplementary detail to the overview in the [README](../README.md). This document covers
the reasoning behind the design choices, the parts that are easy to misread from the
source, and the things we tried that did not work.

## Why the geometry is the architecture

We profiled the solver early and the result set the direction for everything after:
whole-solution feasibility checking is **negligible**, and the per-iteration cost is
almost entirely the **placement geometry** — finding where a block can legally go. The
large-neighborhood search is greedy-bound, not acceptance-bound.

That means the productive question is not "what is a smarter metaheuristic" but "how many
legal-placement queries per second can we afford, and how exact are they". Our first
formulations answered a handful per second. The shipped kernel answers enough to run
48–126 LNS iterations per second per island — roughly five orders of magnitude more
search on the same hardware.

## The three-valued raster in detail

A block is a stack of polygon layers. The crane descends vertically, so placing block `b`
at `(x, y)` on day `t` is legal only if, for every layer `k` of `b` and every layer `j ≥ k`
of every block already resident in the bay during `b`'s stay, the two layers do not
intersect. Checking that directly with polygon algebra is correct and far too slow.

Instead each orientation of each block is rasterized **twice** onto an integer grid:

- the **outer** raster covers every cell the polygon touches at all;
- the **inner** raster covers only cells entirely contained in the polygon.

Both are produced by Sutherland–Hodgman clipping of the polygon against each grid cell, so
each raster is *exact* about what it claims. Consequently, for a candidate position:

| test | conclusion |
|---|---|
| resident outer grids show no hit against our inner cells | **certainly free** |
| resident inner grids hit our inner cells | **certainly blocked** |
| neither | **uncertain** |

Only the uncertain minority goes to the exact path: a triangulated overlap replay
(`_layers_overlap_exact`, `exact_position_ok`) that resolves the question in exact integer
arithmetic. That replay has a per-day budget (`exact_budget`), so a pathological instance
degrades into scanning fewer candidate days rather than into a timeout.

The important property is that **the raster never produces a wrong answer** — it abstains.
An approximate grid would silently admit infeasible placements, and with a checker this
strict that means a zero score.

### Making the scan incremental

A naive implementation rebuilds the occupancy grid per candidate day. Instead the grids are
**stamped**: when a block is committed its cells are added once, and as the day-window
slides each resident block changes membership O(1) times. A full day-window scan costs
`O(nb · cells)` rather than `O(days · nb · cells)`.

Two further prunings matter:

- **Earliest-feasible-day memo.** If a scan for block `i` returns day `t*`, it has
  *proved* no day before `t*` admits `i`. Later scans skip the proven-dead prefix. This is
  only sound if the memo is written from unlifted scans — a scan run with a relaxed
  constraint proves nothing — so the write is gated on that.
- **Day-capacity quick reject.** Compare the day's committed area against bay capacity
  before scanning anything.

`day_scan` also uses a flat-addressed inner probe: the three-dimensional index
`Gin[k, u, v]` becomes a single precomputed offset per cell plus one add, with the `px`
term hoisted out of the inner loop. That change alone moved the board on six of eight
finals instances, purely by raising the iteration count at identical search behavior.

## Classification, and why it sets knobs

`driver.classify()` computes:

- **peak space-time utilization** — summed layer-0 area of blocks live on each day,
  against total bay area, maximized over the horizon;
- **median processing time**, **block count**, **`w1/w3` ratio**;
- refined congestion bounds: a layer-aware earliest-start profile, a **Hall-type subset
  bound**, and an **energetic (compulsory-part)** bound.

The refined bounds are computed and logged but, by default, do **not** move the router:
the thresholds were tuned against the legacy `peak_util` statistic, so swapping the
statistic underneath them would re-route instances wholesale on no evidence.
`OGC_CLASSIFY=v2` switches to `max(legacy, refined)` for whoever wants to earn that
change with a benchmark run.

The regimes are `soft`, `overload-ext`, `overload-mod`, `pref`, `congested`,
`light-small`, `light-large`. Each maps to a knob preset — the SISR dose and whether
guides are on — not to a different engine.

An earlier generation genuinely shipped two engines behind a router. Collapsing them into
one engine plus a classifier was the single best structural decision in the project: it
removed divergence bugs between the engines, made every A/B comparable, and let the
"pocket" behavior that one fork had discovered be expressed as `sisr=all, guides=off` on
the shared code base.

`classify()` also records the **margin** — how close the instance sits to flipping regime.
The thresholds are hard edges: 0.949 and 0.951 buy different portfolios for the same
problem. Logging the margin does not remove the discontinuity, but it makes a hidden
instance sitting on an edge identifiable instead of mysteriously high-variance.

## Islands, and why collapsing them is correct

Four islands, one per available core. They differ in construction arm, initial
temperature (`T0 ∈ {0.5, 1.0, 0.25, 2.0} · w1`), acceptance criterion, and whether they
take guide targets.

Every ~20 seconds each island publishes its best to a shared slot and adopts the global
best if that is better. We audited this because it looked like a bug: the islands do in
fact **collapse onto one trajectory** — on 2 of 3 rolls all four end up identical. The
audit conclusion was the opposite of the suspicion. Cooperating islands beat independent
islands by about 18%. The collapse is the mechanism, not a failure of it: four cores
refining one good solution beats four cores refining four mediocre ones, at these budgets.

What the low basin-hit rate *does* reveal is a real limit on search diversity, and that
limit is where the remaining gap lives (see [RESULTS.md](RESULTS.md)). The answer is not to
stop cooperating; it is to give the islands genuinely different *starting basins*.

Island 3 stays guide-free deliberately. A relaxation can be confidently wrong, and if
every island took its targets a bad relaxation would cost the whole solve. The guide-free
arm means cross-pollination arbitrates between the guided and unguided basins per instance
instead of the guide being trusted a priori.

On loose instances one island takes a different role: it hard-locks to a bay assignment
that the relaxation has **certified optimal** and searches only schedule and packing.
Because the area-based relaxation overestimates what the multi-layer crane rules actually
allow, it first probes each candidate capacity arm with a short forced construction and
locks the most *realizable* one — measurement rather than a guess.

## Exact methods as guides, not as solvers

Solving the joint problem exactly is hopeless at 150–300 blocks. What worked was using
exact solvers to produce **targets** and **bounds**:

- `ogc_guide.py` — CP-SAT bay guide. Target bay per block.
- `ogc_dayguide.py` — CP-SAT day guide. Target entry day per block, applied only inside
  the bay the bay guide chose, so an island with no bay guide has no day targets either.
  It also surfaces the relaxation's chosen **sacrifice set** — the blocks it decided to
  finish late — which becomes role information for a sacrifice-swap move.
- `ogc_globalguide.py` — Gurobi MILP relaxation over the horizon. Optional.
- `ogc_assign.py` — the geometry-free bay-assignment problem. On spatially loose
  instances (`peak_util < 0.45`) tardiness is easy, the objective collapses to
  `w2·Z2 + w3·Z3`, and this assignment's optimum is *certifiable*.
- `ogc_retime.py` — CP-SAT retiming with geometry frozen. Once positions and orientations
  are fixed, feasibility couples blocks within a bay only pairwise, which is small enough
  to solve exactly as a tail pass.

Guides enter the search as a soft penalty (`guide_w = 10·w3`) on deviating from target,
not as a constraint. Tail passes are adopted only if strictly better, with wholesale
restore on disagreement — so no exact component can make the answer worse than the
heuristic found on its own.

A caution from our own logs: the guide **weight** is not a free parameter. Raising it won
8 of 8 rolls at TL=600 and then reversed sign at TL=300. It is not shipped.

## The search layer

Twelve removal operators, under adaptive weights that track reward:

`_rm_random` · `_rm_tardy` · `_rm_window` · `_rm_pref` · `_rm_bay_load` · `_rm_single` ·
`_rm_rect` · `_rm_offguide` · `_rm_offday` · `_rm_hole` · `_rm_crit` · `_rm_string`

`_rm_crit` uses a **criticality index**: how often each block has been the one a repair
aborted on, decayed so it tracks the currently binding constraints rather than the whole
run's history. `_rm_string` is SISR-style string removal, which is the strongest operator
on overloaded instances and starts above par in the weight vector.

Ruin size `k` is adaptive with a cap of 12–13, with per-`k` weights learned on measured
reward from a flat prior over the tuned range. Recreate is blink (stochastic greedy)
reinsertion. **Ejection chains** let a tardy block force its way into a target bay and
window by ejecting the occupants and rehoming them across all bays — the move that made
congested instances tractable (6/6 rolls better, −1.4%).

Acceptance is Metropolis SA by default. Island 2 runs a **Self-Tuning Lam** controller
(Cicirello, *Applied Sciences* 11(21):9828, 2021) instead. The motivation was a direct
measurement: against Lam's target uphill-acceptance trajectory, our SA schedule was
10× to 48× too cold from 15% to 65% of the budget — a hill climber from a fifth of the
way in. The controller measures realized acceptance and inverts for the temperature that
delivers the target.

Two adaptations were necessary. Lam's published 0.44 mid-run plateau was derived for
single-variable moves; one iteration here ruins up to 13 blocks and rebuilds, so holding
44% uphill acceptance through mid-run is too disruptive — `OGC_LAM_PLATEAU=0.20` keeps the
trajectory's shape at half its level. And `OGC_LAM_OBS_MOVES=1` makes the controller
observe the special moves' accept/reject decisions too, raising its observation supply
10–17× (13,473 observations vs 811 on one instance), which is what closes the feedback
loop on starved instances instead of leaving it open-loop on the schedule.

Honest status: at the shipped tuning this is `geomean −1.34%`, better on 7 of 10, paired
`t = −2.14`, i.e. `p ≈ 0.06`. Suggestive, not significant.

## Robustness engineering

Competition scoring punishes a crash far more than it rewards a good answer, so a
disproportionate share of the code is defensive:

- **Nothing ships unverified.** The organizers' shapely checker runs when importable,
  our independent exact integer checker otherwise, then repair, then a
  guaranteed-feasible fallback construction. 17 submissions, zero infeasible boards.
- **Every knob parse fails into its default.** A malformed or `NaN` environment value
  must not kill an island worker, and `OGC_GUIDE_W=nan` must not poison pruning.
- **Knobs are read per run, not at import.** The driver sets some knobs by regime *after*
  the search module is cached, so a module-level read would pin the first instance's
  value for the whole process.
- **Per-island scoping instead of environment mutation.** Writing `os.environ` from an
  island is safe under `fork` and catastrophic when the pool is bypassed: it leaks into
  every later instance in the same process. Found by reasoning about the non-fork path.
- **No shipped JIT cache.** A numba cache is keyed on CPU features, and making one
  portable requires pinning `NUMBA_CPU_NAME` — which makes LLVM *emit* AVX2/FMA/BMI2. On
  a host without them that is a SIGILL on every instance. Refusing the cache costs ~15 s
  of compilation; guessing wrong costs the entire submission. The probe never imports
  numba, because importing it would lock codegen in before the decision is made.
- **Optional-component failures are surfaced, not swallowed.** An earlier version erased
  them unless logging was on, which made a dependency regression indistinguishable from a
  normally-unavailable guide.

## Things that did not work

Recorded because the negative results cost as much to obtain as the positive ones.

| tried | outcome |
|---|---|
| Deep reinsertion reach (50× candidate scan) | Null on overload at 0.7% resolution, no dose-response |
| Raising the guide weight | Won 8/8 at TL=600, **reversed sign** at TL=300 |
| Depth-weighted overlap penalty | Null |
| Due-date homotopy | Negative |
| Joint bounding-box CP-SAT | Refuted |
| Exit-time-aware bbox proxy | Refuted — a bounding box cannot distinguish interlocking from a crane trap |
| Z-decomposition of the assignment | A weak-baseline artifact; worse than the real baseline on both components |
| Disabling island sync | 18% worse |
| Whole-bay NFP re-pack | Real packing lever, but instance-dependent and never beat the champion |
| Plan-and-Realize at final-round densities | Disabled in the scored build: unrealizable, and it burned 25% of an island |

The one cross-cutting lesson: **A/B against the strongest baseline you have, paired, with
the noise floor measured first.** Several of the rows above looked like wins against a
weaker reference or on a single roll.

## Where we would go next

The robustness re-run says the deficit is quality per iteration, not throughput. The
avenues we would pursue, in order:

1. **Cross-bay freedom.** A 10-block exact window allowed to move blocks *between* bays
   found −3.1% where per-bay exact search found ≤1%. This is the largest verified
   untapped lever we know of. An earlier attempt to productize it regressed ~30% of
   instances — the lever is proven, the architecture around it is not.
2. **Genuine construction diversity.** Per-island *different* seeds, so cooperation
   arbitrates between real alternatives rather than near-copies.
3. **Soft-overlap separation.** Allow temporary overlap and drive it out with a
   separation force (in the style of `sparrow`/ROMA/LP compaction), rather than only ever
   admitting legal placements.
4. **P8.** The worst slot on the official board, orthogonal to every other instance in the
   set. It needs its own diagnosis, not a transfer from the slots we could already move.
