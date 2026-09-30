# OGC 2026 — The Grand Shipyard Puzzle (Team Issyk Cool)

Our solver for the **Optimization Grand Challenge 2026**, "The Grand Shipyard Puzzle:
Pack the Block, Beat the Clock".

**Final result: 3rd of 958 teams — 2nd Prize.** 1,448 participants from 55 countries; 40
teams advanced to the final round, where this solver scored 284.

---

## The problem in one page

A shipyard builds each structural block of a ship inside a **bay** — a rectangular
workspace `W × H` served by a single overhead crane. You are given:

- **Blocks** `i ∈ N`: each is a stack of `K_i` polygon **layers** (irregular, often
  non-convex), with several precomputed **orientations**, a release date `R_i`, a due
  date `D_i`, a processing time `P_i`, a workload `L_i`, and a per-bay **preference
  score** `S_ij` (summing to 100 across bays).
- **Bays** `j ∈ M`: sizes `W_j × H_j`.

For every block you must decide **which bay**, **where and how** (`x`, `y`,
orientation), and **when** (`ENTRY_i`, `EXIT_i`). Minimize

```
Z  =  w1·Z1  +  w2·Z2  +  w3·Z3
      tardiness   imbalance   preference shortfall
```

with `w1 >> w3 > w2` in every instance — so **tardiness dominates**, typically 90%+ of
the objective. Subject to: blocks in the same bay must not overlap in space while they
overlap in time, and every placement must be crane-reachable.

**The crane is the hard part.** It moves vertically only: to place a block it lowers it
straight down, so every layer `k` of the descending block sweeps through all heights
above its resting level. Formally a candidate position is blocked if layer `k` of the
new block intersects layer `j` of a resident block for any `j >= k`. A position that
looks free in 2-D footprint can be crane-infeasible, and a bay can be packed into a
state where a block is trapped. Full spec in [docs/problem_statement.md](docs/problem_statement.md).

Two consequences shaped everything we built:

1. **Disjoint footprints are always crane-feasible**, so crane *sequencing* is not an
   independent lever — the problem reduces to cumulative-area scheduling at the
   congestion peaks plus packing efficiency.
2. On the hard instances the tardiness lower bound is **0** — the tardiness we pay is
   entirely *packing-induced*. Better geometry, not better scheduling, is the prize.

---

## Quick start

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate ogc2026

python run.py data/finals/fin_4.json
```

`run.py` solves one instance, verifies it with the organizers' own checker, and prints
the objective and its `Z1/Z2/Z3` breakdown. The official time limits are baked in, so
`--tl` is only needed to override them.

```bash
python run.py --suite finals        # all 8 finals instances at official TLs
python run.py --suite preliminary   # all 6 preliminary instances
```

Requires Python 3.12, `numpy`, `numba`, `shapely`. `ortools` and `gurobipy` are
**optional** — they power the CP-SAT and MILP guides, and every call site degrades
gracefully when they are absent (you lose guidance quality, not feasibility).

First run pays ~15 s of numba JIT compilation. No compiled cache is shipped: a cache is
keyed on CPU features, and pinning those to make one portable would emit AVX2/FMA on
machines that may not have them — a crash on every instance to save 15 seconds.

> ### Run this on Linux if you want the scored numbers
>
> The island pool requires `fork`. On Windows and macOS spawn-only setups,
> [`_ogc/ogc_solve.py`](_ogc/ogc_solve.py) deliberately falls back to a **single
> in-process island** rather than risk a `spawn` pool re-importing the harness — correct,
> but roughly a quarter of the search. Measured on `fin_4`: 4,127,504 single-island on
> Windows against the **3,288,730** the scored 4-island Linux run produced, a 25% gap from
> platform alone. Light instances are unaffected (`prelim_1` reproduces its scored 11,280
> exactly, since one island already closes it). The evaluation server was 4-core Linux.

---

## How the solver works

The whole design follows from one measurement: **placement geometry is the bottleneck.**
Whole-solution feasibility checking is negligible; the cost per LNS iteration is almost
entirely the geometric candidate scan. So the architecture spends everything on making
that scan fast and exact, then runs a large-neighborhood search on top of it.

### 1. A three-valued conservative raster (the foundation)

Placement feasibility is decided on **two rasterized count-grids**, not on polygons:

| | meaning |
|---|---|
| **outer grid**, no hit | position is *certainly free* |
| **inner grid**, any hit | position is *certainly blocked* |
| otherwise | *uncertain* → exact triangulated overlap replay, budgeted per candidate day |

The grids are built by Sutherland–Hodgman polygon-cell clipping, which makes them
**exact conservative** rather than approximate — the raster never lies, it only
abstains. So the hot path is integer grid arithmetic, and exact geometry is paid for
only on the ambiguous minority.

Everything hot lives in `numba` `njit` kernels ([`_ogc/ogc_kernels.py`](_ogc/ogc_kernels.py)),
with a flat-addressed inner probe in `day_scan`. On top of that:

- **incremental stamping** — each committed block changes class O(1) times as the
  day-window slides, so a whole scan costs `O(nb · cells)` instead of re-rasterizing
  per day;
- **earliest-feasible-day memo** — a scan returning day `t*` *proves* no earlier day is
  feasible, so later scans skip days already proven dead (written only from unlifted
  scans, so the proof stays sound);
- **bitboard occupancy** along `x` for fast position rejection;
- **bottom-left corner discipline** — placements anchor to corners. Kept deliberately:
  our own A/Bs showed this is the anti-fragmentation mechanism.

This is what buys ~48–126 LNS iterations/second/island on instances where the naive
formulation manages a handful.

### 2. A classifier that sets knobs, not a portfolio of engines

[`_ogc/driver.py`](_ogc/driver.py) profiles each instance — peak space-time utilization,
median processing time, the `w1/w3` ratio, block count, plus Hall-type and energetic
congestion bounds — and routes it to one of seven regimes: `soft`, `overload-ext`,
`overload-mod`, `pref`, `congested`, `light-small`, `light-large`.

The regime does **not** select a different solver. It sets knobs on the *same* engine:
the SISR string-removal dose, and whether the CP-SAT guides are on. An earlier
generation shipped two engines behind a router; collapsing that into one engine plus a
knob-setting classifier removed a whole class of divergence bug and made every A/B
comparable. The classifier also logs how close an instance sits to each threshold, so a
hidden instance landing on a hard edge is visible rather than silently high-variance.

### 3. Four cooperating islands

The evaluation server allows 4 cores, so [`_ogc/ogc_solve.py`](_ogc/ogc_solve.py) runs
four island searches — three worker processes plus one in-process — each a
deliberately different arm:

| island | construction | acceptance | guidance |
|---|---|---|---|
| 0 | BRKGA over insertion order (on overload) | SA, `T0 = 0.5·w1` | guided |
| 1 | BRKGA over insertion order | SA, `T0 = 1.0·w1` | guided |
| 2 | relaxation-seeded (day targets) | **Self-Tuning Lam** controller | guided |
| 3 | greedy variant | SA, `T0 = 2.0·w1`, LAHC arm | **guide-free hedge** + SISR |

Islands publish their best to a shared slot every ~20 s and adopt the global best if it
is better. This **collapses the islands onto one trajectory** — and that is why it
works: we tested it, and cooperating islands beat independent ones by ~18%. Island 3
stays guide-free on purpose, so a misdirected relaxation cannot poison every arm at
once. On loose instances one island instead hard-locks to a *certified-optimal* bay
assignment and optimizes only schedule and packing, chasing the assignment bound.

### 4. Guides: exact methods where they pay

Rather than solve the whole thing exactly (hopeless at this size), we use exact solvers
to produce **targets** that bias the heuristic:

- **bay guide** and **day guide** (CP-SAT, `ortools`) — target bay and target entry day
  per block, applied as a soft penalty `guide_w = 10·w3` inside the placement scan;
- **global guide** (Gurobi, optional) — a MILP relaxation over the whole horizon;
- **assignment relaxation** — a geometry-free assignment bound, which on spatially loose
  instances (`peak_util < 0.45`) is tight enough that its optimum is *certifiable*;
- **CP-SAT retiming** at frozen geometry ([`_ogc/ogc_retime.py`](_ogc/ogc_retime.py)) —
  with positions frozen, feasibility couples blocks of one bay only pairwise, so entry
  days can be re-optimized exactly as a tail pass.

Each of these is adopted only if it strictly improves, with wholesale restore on
disagreement, so a guide can never make the answer worse than the search found.

### 5. The search itself

Ruin-and-recreate LNS ([`_ogc/ogc_search.py`](_ogc/ogc_search.py)) with twelve removal
operators — random, tardy, time-window, preference, bay-load, single, rectangle,
off-guide, off-day, hole-ruin, criticality, and SISR string removal — under **adaptive
operator weights** and an **adaptive ruin size** `k` (cap 12). Recreate is a blink
(stochastic greedy) reinsertion with ejection chains: a tardy block can force its way
into a target bay/window by ejecting occupants and rehoming them across all bays.

Acceptance is Metropolis SA with `T0 ∝ w1`, LAHC on some arms, and on island 2 a
**Self-Tuning Lam** controller (Cicirello 2021) that measures the realized uphill
acceptance rate and sets the temperature which produces Lam's target trajectory,
instead of following a predetermined decay curve.

### 6. Safety tail

Nothing ships unverified. The driver runs the organizers' own shapely checker when it is
importable, falls back to our independent exact integer checker otherwise, attempts
repair on failure, and keeps a guaranteed-feasible fallback construction in reserve.
Across 17 competition submissions we never returned an infeasible board.

---

## What we measured, including the negatives

Every mechanism here was A/B'd in paired experiments against a measured noise floor. The
negative results are as much of the finding as the positive ones:

**Validated.** Cooperating island sync (+18% vs independent). The conservative raster
and compiled kernels (~10^5× the iteration rate of our first formulation). Bottom-left
anchoring as anti-fragmentation. Cycle cadence on congested instances at TL=300
(−3.18% pooled, p=0.012) — though it reaches only a narrow band of instances.

**Refuted, after looking like wins.** A guide-weight increase won 8/8 at TL=600 and then
*reversed sign* at TL=300 — caught before it shipped. Deep reinsertion reach (50×): null
on overload, no dose-response. Depth-weighted overlap penalties, due-date homotopy, joint
bounding-box CP-SAT, exit-time-aware bbox proxies: all null or negative.

**The measurement lesson.** Re-sending a *byte-identical* submission moved our mean gap
by 1.97 percentage points — larger than the spread across eight genuinely different
builds. Below roughly 2pp, a single submission resolves nothing. Same-build CV is 1–2%
on 600 s overload instances but 9–16% on 300 s preference instances, so a 5% effect
needs `n≈1–5` runs in one place and `n≈71–155` in the other. Most "improvements" in this
problem class are indistinguishable from roll variance.

**Where the remaining gap is.** After the round closed, every team's final submission was
re-run at 0.70× and 1.5× the official time limits. We placed **3rd at 0.70×** and **4th
at 1.5×** — our standing rises as time gets scarcer and falls as it loosens. More time
helps our competitors more than it helps us, which says the deficit is **quality per
iteration, not iterations per second**. The kernel is not the bottleneck any more; the
neighborhood and the guidance are.

Per-instance results and the official board are in [docs/RESULTS.md](docs/RESULTS.md).

---

## Repository layout

```
myalgorithm.py            competition entry point: sets two knobs, imports the driver
_ogc/
  driver.py               instance classifier, orchestration, safety tail
  ogc_kernels.py          numba geometry kernels (day_scan, rasterization, exact replay)
  ogc_search.py           LNS: operators, acceptance, Lam controller, construction
  ogc_state.py            solution state, incremental placement, move primitives
  ogc_solve.py            island parallelism and cross-island sync
  ogc_prep.py             instance preprocessing, orientation rasterization
  geo.py, crane.py        exact polygon and crane-clearance geometry
  feas.py                 independent exact integer feasibility checker
  model.py                problem/solution data model
  ogc_guide.py            CP-SAT bay guide
  ogc_dayguide.py         CP-SAT day guide
  ogc_globalguide.py      Gurobi global relaxation (optional)
  ogc_assign.py           exact bay-assignment endgame for loose instances
  ogc_retime.py           CP-SAT retiming at frozen geometry
  ogc_brkga.py            BRKGA over construction order
  relaxarea.py            area-relaxation helper
  ogc_fallback.py         guaranteed-feasible fallback construction
  official_checker.py     the organizers' checker, unmodified (see NOTICE)
run.py                    solve + verify one instance or a whole suite
data/finals/              the 8 finals instances (published by the organizers)
data/preliminary/         the 6 preliminary instances
docs/ALGORITHM.md         deeper architecture notes
docs/RESULTS.md           official scores, per-instance
docs/problem_statement.md the problem specification
```

### Tuning knobs

Nearly every mechanism is gated by an `OGC_*` environment variable, so any arm can be
A/B'd from the same build without recompiling. `myalgorithm.py` sets only the two groups
that differ from defaults in the scored configuration:

- `OGC_PLAN=0` — Plan-and-Realize disabled. The capacity-plan gate fires on 13/40 of our
  training instances at final-round densities, and *every* loss to our reference solver
  was a gated instance: the plan is unrealizable that dense and costs 25% of an island's
  budget.
- `OGC_LAM=2`, `OGC_LAM_PLATEAU=0.20`, `OGC_LAM_OBS_MOVES=1` — the Self-Tuning Lam
  configuration on island 2. Lam's published 0.44 plateau was derived for
  single-variable moves; one iteration here ruins up to 13 blocks and rebuilds, so 0.20
  keeps the trajectory's shape at half its level.

Explicit environment variables still override both, so every experiment stays runnable
from this one build.

---

## Credits

**Team Issyk Cool** — Arseniy Kan and Alina Akhmetbek.

## License

MIT for our own code — see [LICENSE](LICENSE). The organizers' checker and the
competition instances are redistributed unmodified under their own terms and are **not**
covered by it; see [NOTICE](NOTICE).
