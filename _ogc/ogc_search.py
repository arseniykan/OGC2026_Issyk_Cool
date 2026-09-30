# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================
"""Greedy construction + LNS improvement for OGC2026."""

import math
import os
import time

import numpy as np


def _fenv(name, default):
    """Non-negative float knob, hardened: a malformed or NaN value falls back
    to the default instead of killing the island worker at import. Defined
    FIRST because module-level knobs below call it -- the same ordering trap
    that made a corrupt cache manifest a NameError in driver.py."""
    try:
        v = float(os.environ.get(name, default))
        return v if math.isfinite(v) and v >= 0.0 else float(default)
    except (TypeError, ValueError):
        return float(default)


# Reinsertion caps are read by each LNS instance, not here at module import:
# the evaluator may call algorithm() for several hidden instances in one
# interpreter, and the classifier legitimately selects different caps.
# DEEP-REPAIR dose (v4.23 graft): the reinsertion budget
# old_cost + cap_t*T + cap_w1*w1 collapses to ~old_cost + w1 once the
# annealing temperature decays, so late repairs can only ever consider
# placements within about one tardy day of what they removed -- expensive
# but ENABLING placements are never generated. With probability OGC_DEEP_P
# a repair runs with a greatly relaxed budget instead. p=0 (default) takes
# no rng draw, so the default stream is bit-identical.
_DEEP_P = float(os.environ.get("OGC_DEEP_P", "0") or 0)
_DEEP_MULT = _fenv("OGC_DEEP_MULT", 50)
# lower-bound budget tightening: subtract the structural minimum cost of the
# still-uninserted blocks from each scan's cap, and abort before scanning
# when even a block's own lower bound exceeds what is left
_LBCAP = os.environ.get("OGC_LBCAP", "0") == "1"
# ---------------------------------------------------------------------------
# ADMISSIBLE WORKLOAD ACCOUNTING IN THE REPAIR BUDGET (OGC_BALHEAD)
#
# The repair budget was built from the removed set's tardiness and preference
# cost only, while best_insertion's returned cost carries a w2 (workload
# imbalance) term as well. Three consequences, all one-directional:
#
#   * the removal was never credited for imbalance it relieved, though every
#     insertion was charged for imbalance it created;
#   * an insertion with NEGATIVE cost -- one that improved the objective by
#     moving load toward the lightest bay -- was clamped to zero by
#     `spent += max(0.0, c)`, so it bought the rest of the repair nothing;
#   * the cap is consumed front-to-back, so a repair whose payoff is a
#     rebalance completing on the last insertions aborts on the first
#     expensive one and is never evaluated.
#
# Together those bias the search toward moves whose benefit appears early and
# suppress coordinated reassignments -- the balance term is precisely the one
# that needs several blocks to move at once. This knob fixes all three, using
# w2 * obj2 as the extra headroom: obj2 is a floor()ed non-negative range, so
# that product is a true bound on the remaining w2 gain, which makes the prune
# admissible for the workload term rather than merely plausible.
#
# SPLIT BY CONFIDENCE, and defaulted OFF. Crediting the removal with the w2
# imbalance it relieved is a pure consistency repair -- it makes old_cost use
# the same cost function best_insertion already returns -- so that part is
# unconditional. Widening the cap and letting `spent` go negative are POLICY:
# they buy admissibility with iteration throughput, exactly the trade the
# caps exist to make, and a first 30s probe on prob_22 came out worse (3998
# against 1964). One sample on a low-tardiness instance is not a verdict, but
# it is not a licence either, so the default stays where the evidence is until
# a paired A/B says otherwise. OGC_BALHEAD=1 enables.
_BALHEAD = os.environ.get("OGC_BALHEAD", "0") == "1"
# Fraction of a locked island's budget the hard bay lock may hold without
# producing a new incumbent before it is dropped. See the comment in run().
#
# DEFAULT 0 (escape disabled). It shipped at 0.25 for one revision on the
# argument that an area-based lock cannot see geometry or crane order, which is
# true but does not make THIS trigger the right response:
#
#   * The failure it claims to rescue is already screened. ogc_solve probes
#     every capacity arm with forced_construct and locks the most REALIZABLE
#     one -- "measurement beats guessing", in its own words. An unrealizable
#     lock has already been filtered before the island starts.
#   * The trigger measures convergence, not unrealizability. Every island stops
#     finding incumbents late in a run, so on a healthy 300s solve the escape
#     fires anyway, for reasons unrelated to whether the lock was good.
#   * Releasing it destroys the arm. Exactly one island of four is locked, and
#     it is set up with guide=None -- so with the lock gone it has neither lock
#     nor guide and duplicates the unguided islands. The portfolio trades a
#     distinct basin for a redundant one, in a design whose whole rationale is
#     that no single hidden distribution should be able to kill a basin.
#   * It leaves the state inconsistent. tbay is the LOCKED assignment, so after
#     release the day guide still lifts scans toward the locked bays while
#     blocks are free to sit elsewhere -- contradictory pressure, at low
#     temperature, over a suddenly much larger neighbourhood.
#
# The concern is real; the trigger is wrong. A defensible version would use the
# COMPARATIVE signal the island already receives through sync_cb -- this
# island's best sitting far behind the shared cross-island best is evidence the
# lock is losing, where stagnation alone is not. That is a different change and
# it has to be measured before it ships.
_LOCK_ESC = _fenv("OGC_LOCK_ESC", 0.0)
# Multiplier on exact_budget for the one retry a block gets before the repair
# is abandoned. 1 disables the rescan (the scan's first verdict is final).
#
# MEASURED AND DEFAULTED OFF. Instrumented on prob_5 (n=150, m=2, the densest
# 2-bay instance) at 8x budget: 8706 rescans, 3 saves -- 0.03%. Exact-test
# rationing is simply not why repairs abort. day_scan already enumerates EVERY
# integer position in the legal range for every orientation, so what fails is
# almost never "the position was not generated"; it is that no generated
# position priced under cap_i, which is the repair BUDGET (see OGC_BALHEAD),
# not candidate generation. Worth keeping as a knob because the failure mode
# is real and instance-dependent -- a bay full of near-touching concave
# polygons makes ambiguous cells common -- but not worth ~8700 extra scans per
# 25s to buy three placements.
_LASTCHANCE = int(_fenv("OGC_LASTCHANCE", 1))
# 25% hardest-first reinsertion order (structurally tardy / large first)
_ORD2 = os.environ.get("OGC_ORD2", "0") == "1"
# restart from best + large perturbation after this many stalled seconds
_STALL_S = _fenv("OGC_STALL_S", 0)
# ejection-chain move: force a tardy block into a target bay/window by
# ejecting the occupants and rehoming them across all bays (paired A/B:
# 6/6 rolls better, -1.4% on the congested set)
_CHAIN = os.environ.get("OGC_CHAIN", "1") == "1"
# exit-delay nesting move gate (the guide-free fork has no such move; the
# pocket preset may disable it for full fork emulation)
_NEST = os.environ.get("OGC_NEST", "1") == "1"
# DP-by-parts joint batch repack (bounded DFS placement with backtracking)
_DP = os.environ.get("OGC_DP", "0") == "1"
# departure-aligned placement: corner zoning by exit time (early-leavers
# anchor left, late-leavers right -- cohorts cluster, holes open contiguous)
_EXITCORNER = os.environ.get("OGC_EXITCORNER", "0") == "1"
# ...and full attention-scored position refinement (exit-affinity contact)
_EXITPACK = os.environ.get("OGC_EXITPACK", "0")  # 0|1(conservative)|2(exact box)
# size-gradient anchoring: big blocks anchor LEFT, small anchor RIGHT --
# size-homogeneous regions tile with fewer awkward gaps (user hypothesis;
# distinct from the failed exit-zoning: size sorts by tiling compatibility)
_SIZECORNER = os.environ.get("OGC_SIZECORNER", "0") == "1"
# hole-targeted ruin: destroy the blocks WALLING IN the largest free region
# of a congested day so repair can refill it coherently. Placement-side
# hole-awareness fails (fights the corner discipline); ruin-side does not.
_HOLERUIN = os.environ.get("OGC_HOLERUIN", "0") == "1"
# BORDER RESERVATION: confine SMALL blocks (area below the _BAND_Q quantile)
# to a band hugging one bay edge, so the central span stays contiguous for
# blocks that will need it later. Applied strictly as a TIE-BREAK -- the full
# range is scanned first and the band placement is taken only when it costs
# no more -- which is what separates it from the five measured losses that
# nudged placements off the anchor corner at a real objective cost.
_BAND = os.environ.get("OGC_BAND", "0") == "1"
_BAND_Q = _fenv("OGC_BAND_Q", 0.25)
_BAND_W = _fenv("OGC_BAND_W", 0.25)
_BAND_SIDE = os.environ.get("OGC_BAND_SIDE", "right")

# ---------------------------------------------------------------------------
# SELF-TUNING LAM ANNEALING  (OGC_LAM)
#
# Cicirello, "Self-Tuning Lam Annealing: Learning Hyperparameters While Problem
# Solving", Applied Sciences 11(21):9828, 2021.  Algorithms 3-7 of that paper.
#
# WHAT THE SHIPPED SCHEDULE DOES WRONG.  run() cools on a curve fixed before
# the first iteration:  T(q) = T0 * 0.02**q  with  q = elapsed/budget  and
# T0 = t0_scale * w1.  MEASURED uphill-acceptance rate -- the fraction of
# worsening candidates the island actually took -- median over prob_2/5/12/36
# at TL=180, by decile of the anneal budget and by island:
#
#   decile  q(mid)  Lam tgt |  t0=0.25   t0=0.5   t0=1.0   t0=2.0
#        0    0.05     0.51 |    0.140    0.167    0.284    0.362
#        1    0.15     0.44 |    0.044    0.118    0.220    0.283
#        2    0.25     0.44 |    0.011    0.071    0.172    0.346
#        3    0.35     0.44 |    0.009    0.057    0.019    0.235
#        4    0.45     0.44 |    0.011    0.028    0.056    0.142
#        5    0.55     0.44 |    0.026    0.017    0.000    0.121
#        6    0.65     0.44 |    0.017    0.021    0.038    0.052
#        7    0.75     0.08 |    0.010    0.013    0.000    0.020
#        8    0.85     0.01 |    0.000    0.008    0.009    0.015
#        9    0.95     0.00 |    0.000    0.011    0.016    0.026
#
# The FREEZE is fine: from q=0.65 on, every island is within a factor of two of
# the target. The defect is the PLATEAU. Lam asks for 0.44 from 15% to 65% of
# the budget; island 2 delivers 0.009-0.044, i.e. 10x to 48x too cold, for half
# the run. It is a hill climber from a fifth of the way in, whether or not it
# has anything left to climb -- and that cannot be retuned away, because
# t0_scale and Tmin_frac set the ENDPOINTS of a curve whose middle is wrong.
#
# Note what is NOT wrong, since the obvious hypothesis is that T0 scaling off
# w1 is the error (the objective is w1*tard + w2*imbalance + w3*pref, so why
# w1?). The measured median uphill delta across all 16 island-runs is
# 0.65..1.00 * w1, median 0.88: one tardy day on one block is both the quantum
# and the typical uphill move, so w1 is a good normaliser and the tuning phase
# keeps landing on dC ~= 1.00*w1. Scale matters in a different way -- the
# spread WITHIN one island is 300..8000, and no multiplier expresses a
# distribution (see the OGC_LAM_MODEL block).
#
# WHAT LAM DOES INSTEAD.  Lam & Delosme observed that near-optimal SA runs all
# follow one acceptance-rate trajectory: ~1.0 falling to 0.44 over the first
# 15% of the run, flat 0.44 until 65%, then a fast freeze. The Modified Lam
# (Swartz/Boyan) tracks that trajectory with a feedback controller -- EMA of
# the realised acceptance rate against the target, temperature moved UP or DOWN
# to close the gap. Cicirello's Self-Tuning Lam adds a short tuning phase that
# derives T0 and the controller gain from sampled cost differences, so the
# controller starts on target instead of converging to it.
#
# THREE ADAPTATIONS, all forced by this solver rather than chosen:
#
# 1. WALL-CLOCK PROGRESS.  The paper indexes LamRate by i/N for a known
#    iteration budget N. Here the budget is time (an island's t_end), one
#    iteration costs anywhere from 5 to 40 ms by design (k_max adapts to hit
#    that band) and the special moves cost more, so N is neither known nor
#    uniform. So q = elapsed/(t_end - t_start), and every per-iteration
#    constant of the paper is re-derived in the time domain:
#
#      paper                     here
#      M = 0.001N iterations     the tuning phase's own measured duration
#      AcceptRate(0) = lam_0.001 LamRate(f), f = tuning duration / budget
#      R = lam_0.002             LamRate(2f)
#      alpha = 2/(0.01N+1)       per-observation weight of a 0.01*budget EMA
#      T0 * beta**M = T1         T0 * exp(-s * f*budget) = T1
#
#    Table 1 of the paper is exactly this at f = 0.001 and f = 0.01 -- the
#    zeta constants are what _lam_temp() returns under its gamma clamp -- so
#    the generalisation reproduces the paper's numbers at the paper's f and
#    stays defined at ours.
#
# 2. THE CONTROLLED POPULATION IS THE WORSENING CANDIDATES (OGC_LAM_POP).
#    This is the change without which the schedule is strictly worse than what
#    it replaces, and it took two measured failures to arrive at.
#
#    The paper controls the acceptance rate over ALL candidates, modelled as
#    P(T) = gamma + (1-gamma)*exp(-DeltaC/T) with gamma the share accepted
#    regardless of temperature. That is fine when gamma is small. Here it is
#    not, twice over:
#
#      * a MEDIAN OF 96.5% of valid candidates have delta EXACTLY 0 (range 73%
#        to 99.8% over 160 island-deciles of prob_2/5/12/36) -- the repair puts
#        the ruined blocks back at the same cost. Ties are accepted
#        unconditionally, so the raw rate has median 0.978 and never leaves
#        0.87-1.00 all run; the controller reads "far above target" from
#        q=0.15 onward and cools to the floor forever.
#      * Excluding ties is not enough. Of the candidates that DO change the
#        objective, 69-97% are improvements during the tuning window, and on
#        the large instances they stay dominant for most of the run: measured
#        on island 2, prob_13 sat at A=0.88 at q=0.47 and prob_36 at A=0.86 at
#        q=0.28. Both exceed 0.44 on free improvements alone, so again the
#        controller could only cool, and it drove T to ~0 by q=0.4-0.6 -- about
#        30x COLDER than the schedule it replaced.
#
#    Both failures are the same failure: the temperature does not control the
#    part of the rate that dominates it. Temperature controls exactly one
#    thing -- whether a WORSENING candidate is taken. So control that:
#
#        observations = candidates with delta > 0
#        target       = LamRate(q)
#        model        = P(T) = exp(-DeltaC/T),  i.e. gamma == 0
#
#    which is the practical content of Lam's plateau ("accept about 44% of the
#    uphill moves you consider") and reduces to the paper exactly in the regime
#    the paper was derived from, where gamma -> 0 and every neighbour changes
#    the cost. It is also always well posed: w in (0,1) is reachable by some T
#    for any instance, so the Eq (18) gamma clamp -- which is what collapsed
#    T0/T1 and with them the gain -- can never be reached. gamma survives as a
#    reported diagnostic and as the population knob: OGC_LAM_POP=nontie gives
#    delta != 0 with gamma the improving share, and OGC_LAM_POP=all gives the
#    paper's own population with ties counted as deterministic accepts. Those
#    two are the measured failures above; they are kept because they are the
#    A/B that justifies the default.
#
# 3. THE GAIN IS CAPPED BY THE MEASUREMENT.  Bang-bang control against a lagged
#    estimate limit-cycles with log-amplitude ~ gain * lag. The paper's ratio
#    of gain to EMA weight is ~3.4 (classic Modified Lam: 0.5), which is fine
#    at its sample rates. Here the non-tie rate is as low as 0.7 per second
#    (prob_5) against 30 per second (prob_12) -- a 40x spread WITHIN one
#    solver -- so a fixed gain is either blind or wild. Two guards:
#      * the per-observation log-step never exceeds OGC_LAM_KAPPA * alpha, so
#        the temperature cannot move further between two useful measurements
#        than one EMA horizon can resolve;
#      * the step is scaled by tanh(err / OGC_LAM_BAND) instead of sign(err),
#        which is bang-bang outside the band and proportional inside it.
#      OGC_LAM_BAND=0 restores sign(err), OGC_LAM_KAPPA=0 removes the cap.
#
# WHAT IT DOES NOT TOUCH.  Temperature has a SECOND job in this solver: the
# reinsertion budget  old_cost + cap_t*T + cap_w1*w1  (and the same term in
# try_chain). Letting the controller drive that closes a loop -- acceptance
# falls, T rises, repairs get a wider funnel, iterations slow, the candidate
# distribution moves, acceptance changes again -- and it changes which
# candidates EXIST, not just which are taken. So the budget keeps the shipped
# monotone curve (T below) and only the Metropolis test reads the controller
# (Ta). Repair and geometry aborts produce no candidate objective and therefore
# no Metropolis decision, so they update nothing either.
#
# DEFAULT: island 2 ONLY.  That is the 0.25*w1 island, which freezes earliest
# and is the least specialised of the four (island 0 runs the day re-solve,
# island 1 can take the bay lock, island 3 carries the SISR arm). Three
# measured baseline islands are preserved, cross-island sync propagates
# whatever the Lam island finds, and a loss is capped at one arm of four.
# OGC_LAM=all|0|"0,2" selects otherwise.
_LAM_ISL = os.environ.get("OGC_LAM", "2").strip().lower()
# tuning phase: stop at OGC_LAM_CAL_N non-tie samples but never before
# OGC_LAM_CAL_MIN and never after OGC_LAM_CAL_S (both fractions of budget).
# The floor matters twice: LamRate(f) -> 1 as f -> 0, which sends T0 -> inf,
# and a longer phase starts the controller nearer its operating point (at
# f=0.02, gamma=0.1 the derived T0 is 2.3*DeltaC against 19*DeltaC at
# f=0.002, and the capped gain needs ~20 observations to close that).
_LAM_CAL_N = int(_fenv("OGC_LAM_CAL_N", 24))
# ...and never before OGC_LAM_CAL_W of those are WORSENING. gamma and DeltaC
# are both statistics of the worsening branch; a phase that ends with one
# worsening sample has measured nothing, and the failure is silent (see below).
_LAM_CAL_W = int(_fenv("OGC_LAM_CAL_W", 8))
_LAM_CAL_MIN = _fenv("OGC_LAM_CAL_MIN", 0.010)
_LAM_CAL_S = _fenv("OGC_LAM_CAL_S", 0.030)
# ---------------------------------------------------------------------------
# WHY THE TUNING PHASE DOES NOT ACCEPT EVERYTHING (OGC_LAM_CAL_ACC=0 default)
#
# Paper Alg 6 line 12 accepts every neighbour during tuning, so that DeltaC is
# a statistic of the neighbourhood rather than of the incumbent's basin. In a
# bit-flip landscape that walk is unbiased. Here it is not, and the bias is
# severe enough to disable the controller. MEASURED, prob_13 @ TL=70, the
# accept-all phase on island 2 reported:
#
#     f=0.0132 k=24 ties=85 dC=51151 (7.67*w1) gam=0.960 T0=1.40*w1
#     T1=8622  s=0.104/s   -> end of run: A=0.596 against tgt=0.001
#
# i.e. the island never froze. Both inputs were wrong, for the same reason:
#
#   * gamma = 0.96. Repair is GREEDY (best_insertion), so from a solution the
#     walk has just degraded, almost every repair is an improvement. 23 of the
#     24 non-tie samples went downhill. gamma >= AcceptRate(0) then trips the
#     clamp of Eq (18), which collapses T0/T1 to the constant zeta ratio and
#     with it the controller gain s -- so the controller cannot follow the
#     freeze, and the failure looks like a tuned schedule rather than an error.
#   * DeltaC = 7.67*w1 off ONE worsening sample, against a true typical
#     worsening of 0.9-2.9*w1 (probe over prob_2/5/12/36).
#
# Observing instead of accepting costs the paper's unbiasedness argument and
# buys statistics taken where the search will actually run. gamma and DeltaC
# describe the CANDIDATE distribution, which depends on the incumbent and not
# on the acceptance rule, so sampling at the incumbent is the relevant
# population. OGC_LAM_CAL_ACC=1 restores paper fidelity.
_LAM_CAL_ACC = os.environ.get("OGC_LAM_CAL_ACC", "0") == "1"
# EMA horizon as a fraction of budget (paper: 0.01N), and the floor on its
# effective sample count -- alpha is capped at 2/(N_MIN+1), which on the
# sparse instances is what actually sets it.
_LAM_EMA = _fenv("OGC_LAM_EMA", 0.010)
_LAM_EMA_N = _fenv("OGC_LAM_EMA_N", 25)
_LAM_KAPPA = _fenv("OGC_LAM_KAPPA", 1.0)
_LAM_BAND = _fenv("OGC_LAM_BAND", 0.05)
# gain floor: the share of the budget within which the feedback correction must
# be able to sweep the schedule's whole log-span. See the authority-floor
# comment in maybe_finish().
_LAM_FRZ = max(0.02, _fenv("OGC_LAM_FRZ", 0.25))
# DeltaC estimator: "wor" | "all" | "mean" | "meanall".
#
# Two departures from the paper's mean-of-|delta|-over-non-ties, both measured:
#
#   * WORSENING BRANCH ONLY. Eq (15) uses DeltaC as "an estimate of the cost
#     difference" inside exp(-DeltaC/T0) -- the term that prices worsening
#     candidates. Pooling improvements into it is only harmless if the two
#     branches have similar scale, and here they do not: prob_36 decile 0
#     measures 31600 over all non-ties against 5778 over worsenings alone.
#   * MEDIAN, NOT MEAN. The worsening distribution is heavy-tailed (a compound
#     ruin-recreate can move several blocks a day late at once), and the mean
#     of a heavy-tailed positive variable is not its typical value -- which is
#     what a Boltzmann factor responds to. One outlier in a 24-sample phase
#     put DeltaC at 7.67*w1 on prob_13. The median is O(1) work here because
#     the phase is bounded, and it is what makes the estimate survive a phase
#     that only got a handful of samples.
_LAM_DC = os.environ.get("OGC_LAM_DC", "wor").strip().lower()
# controlled population: "wor" (delta>0, gamma==0) | "nontie" | "all".
# See item 2 of the block above -- this is the knob that decides whether the
# controller can do anything at all on this solver.
_LAM_POP = os.environ.get("OGC_LAM_POP", "wor").strip().lower()
# ---------------------------------------------------------------------------
# ACCEPTANCE MODEL: "emp" (default) | "point"
#
# Eq (15) prices the worsening branch as exp(-DeltaC/T): one point estimate
# through the Boltzmann factor. But Metropolis averages that factor over the
# whole delta distribution, and here the distribution is wide -- a compound
# ruin-recreate can make one block a day late or five, so the sampled uphill
# deltas span more than an order of magnitude within a single island. A point
# estimate does not merely shift such a schedule, it COMPRESSES it:
#
#   * a high target rate has to be paid for on the EXPENSIVE deltas, so the
#     true temperature at the hot end is well above the point model's;
#   * a low target rate is carried by the CHEAP deltas alone, so the true
#     temperature at the frozen end is well below it.
#
# Measured over one island's sampled deltas (300..8000, median 1100), against
# the median point model:
#
#     rate    0.90   0.807   0.60   0.44   0.10   0.01   0.002
#     ratio   1.91    1.78   1.45   1.20   0.71   0.47    0.40
#
# -- a log-span of 5.78 against the point model's 4.18, so the point model asks
# for a range 4.8x too narrow at both ends at once. No amount of feedback gain
# fixes that: a scalar correction can slide the schedule, not stretch it.
#
# "emp" replaces the point mass with the measured distribution and inverts the
# same equation,
#
#     Phat(T) = gamma + (1-gamma) * mean_j exp(-d_j / T) = LamRate(q),
#
# which is one fewer approximation rather than one more assumption. Phat is
# monotone in T so bisection is safe, and because the schedule depends on q
# only through LamRate(q) it is tabulated once over rate at tuning time -- the
# per-iteration cost is an index and a lerp.
_LAM_MODEL = os.environ.get("OGC_LAM_MODEL", "emp").strip().lower()
# ---------------------------------------------------------------------------
# LATE RECALIBRATION.
#
# The tuning phase can end with too few worsening samples to build a
# distribution from -- prob_13 finished its 3% window with k=37 non-ties of
# which ZERO worsened, because that early the search is still in free descent.
# There is nothing to wait for (a window with no uphill candidates is a window
# where temperature does nothing), so the phase ends on its cap and the point
# fallback is used. But the samples do arrive later, and the first
# OGC_LAM_CAL_W of them are enough to replace a guessed scale with a measured
# one. One
# rebuild, once, and only when the tuning phase came up short; the correction
# resets with it because what it had accumulated was mostly that scale error.
_LAM_RECAL = os.environ.get("OGC_LAM_RECAL", "1") == "1"
# ceiling on T_accept as a multiple of the tuned T0. The controller cannot
# raise the rate above gamma + (1-gamma) once saturated, so without a ceiling
# an unreachable target heats forever.
_LAM_HI = _fenv("OGC_LAM_HI", 1000.0)
_LAM_TRACE = os.environ.get("OGC_LAM_TRACE", "0") == "1"
# ---------------------------------------------------------------------------
# WHICH OF TEMPERATURE'S TWO JOBS THE CONTROLLER DRIVES (OGC_LAM_REPAIR)
#
#   "0"      acceptance only. v10.5's default, and the strategy note's
#            recommendation, on the grounds that driving the reinsertion budget
#            from a feedback loop closes a dangerous cycle: acceptance falls ->
#            T rises -> the funnel widens -> iterations slow -> the candidate
#            distribution moves -> acceptance changes again.
#   "sched"  acceptance from the closed loop, funnel from the OPEN-LOOP
#            calibrated schedule. The cycle cannot form, because the schedule is
#            a fixed function of q and DeltaC settled at tuning time and never
#            reads the acceptance rate.
#   "full"   both from the corrected temperature. This is the coupling the note
#            warns about; kept only to measure what the warning is worth.
#
# WHY THIS IS THE BIGGER HALF, against the note's assumption that it is the
# lesser. MEASURED tie fraction -- the share of valid candidates whose objective
# delta is EXACTLY 0, which are accepted whatever T is, so the Metropolis test
# never sees them -- averaged over prob_2/5/12/36:
#
#     island t0    mean tie fraction    so T governs
#       0.25x           0.963            3.7% of decisions
#       0.5x            0.957            4.3%
#       1.0x            0.949            5.1%
#       2.0x            0.945            5.5%
#
# Acceptance is a lever on ~4% of this search's decisions. That is a ceiling on
# any acceptance-only schedule change, however well tuned. The funnel is a lever
# on which candidates EXIST, and by the end of the shipped curve it has closed:
#
#     q                  0.00   0.15   0.40   0.65   1.00
#     cap_t*T/w1, isl2   0.500  0.278  0.105  0.039  0.010
#
# At q=0.65 the budget is old_cost + w1 to three decimal places, so a repair can
# only consider placements within one tardy day of what it removed -- "expensive
# but ENABLING placements are never generated", which is exactly the diagnosis
# the OGC_DEEP_P comment already records at the top of this file. Lam's plateau
# holds the same term near 2.0*w1, some 50x wider.
#
# And the tie fraction itself falls monotonically with island temperature in the
# table above, which is the direct evidence: T acting through the funnel changes
# the candidate distribution, which is the thing acceptance cannot reach.
_LAM_REPAIR = os.environ.get("OGC_LAM_REPAIR", "0").strip().lower()
if _LAM_REPAIR not in ("0", "off", "sched", "full"):
    _LAM_REPAIR = "0"
# ---------------------------------------------------------------------------
# TARGET PLATEAU LEVEL (OGC_LAM_PLATEAU, default 0.44 = Lam & Delosme's)
#
# 0.44 is an empirical constant of Lam & Delosme's study of near-optimal SA runs
# over neighbourhoods of SINGLE-VARIABLE moves. One iteration here is a ruin of
# k blocks (k_max adapts up to 13+) followed by a greedy rebuild -- a vastly
# larger jump, whose acceptance at the same rate is a correspondingly larger
# perturbation. The SHAPE of the trajectory and the self-calibration of its
# scale are the paper's contributions and are independent of the level, so the
# level is exposed rather than assumed. Scales the whole curve, so the ratios
# between its three phases (1.0 -> plateau -> freeze) are preserved.
_LAM_PLAT = min(0.95, max(0.005, _fenv("OGC_LAM_PLATEAU", 0.44)))
# Observe the special moves' Metropolis decisions too (try_nest, try_chain,
# try_roleswap, try_separate, try_dp_repack). They decide at the same Ta on the
# same objective, so they are the same population -- they were left out of v10.5
# only because each needs its own abort/reject split. On the instances where the
# loop is starved (34 uphill decisions in a whole run on prob_5) this is the
# cheapest available increase in observation supply.
_LAM_OBS_MOVES = os.environ.get("OGC_LAM_OBS_MOVES", "0") == "1"
# diagnostic only: count valid candidates / ties / uphill (OGC_TIEPROBE)
_TIEPROBE = os.environ.get("OGC_TIEPROBE", "0") == "1"
# Normalise the string knobs at import, the same way _fenv hardens the numeric
# ones: an island worker must not die on a typo in an experiment's environment,
# and silently running an unintended population would be worse than either.
if _LAM_POP not in ("wor", "nontie", "all"):
    _LAM_POP = "wor"
if _LAM_DC not in ("wor", "all", "mean", "meanall"):
    _LAM_DC = "wor"
if _LAM_MODEL not in ("emp", "point"):
    _LAM_MODEL = "emp"


def _median(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    s = sorted(xs)
    h = n // 2
    return s[h] if n % 2 else 0.5 * (s[h - 1] + s[h])


def _lam_rate(q, plat=None):
    """Lam & Delosme's idealised target acceptance rate (paper Eq (1)),
    indexed by wall-clock progress q in [0,1] rather than iteration i/N.

    `plat` replaces the 0.44 plateau (see OGC_LAM_PLATEAU); the two 0.56s are
    1-0.44, i.e. the height of the initial descent above the plateau, so they
    move with it and the trajectory keeps its shape."""
    p = _LAM_PLAT if plat is None else plat
    if q <= 0.0:
        return 1.0
    if q <= 0.15:
        return p + (1.0 - p) * 560.0 ** (-q / 0.15)
    if q <= 0.65:
        return p
    if q >= 1.0:
        return p / 440.0
    return p * 440.0 ** (-(q - 0.65) / 0.35)


def _lam_temp(rate, gam, dC):
    """Invert the paper's acceptance model  P(T) = gam + (1-gam)*exp(-dC/T)
    for T (Eqs (15)-(18)): the temperature at which a neighbourhood whose
    typical worsening costs dC, and whose improving share is gam, is accepted
    at `rate`. gam >= rate has no solution -- the deterministic branch alone
    already exceeds the target -- so gam is clamped just below it, which is
    exactly the substitution behind the zeta constants of the paper's Table 1.
    """
    r = min(max(rate, 1e-9), 1.0 - 1e-9)
    # paper Eq (18): gamma >= rate has no solution, so shift gamma just below
    # it. Never below zero -- at gamma == 0 the model needs no clamp at all and
    # T = -dC/ln(rate) is exact, which is the default population's case.
    g = max(0.0, min(max(gam, 0.0), r - 1e-3))
    return -dC / math.log((r - g) / (1.0 - g))


def _lam_accept(T, ds, gam):
    """Empirical acceptance probability of the controlled population at T:
    gamma for the deterministic members, plus the mean Boltzmann factor over
    the sampled worsening deltas. Monotone increasing in T."""
    tot = 0.0
    for d in ds:
        x = -d / T
        tot += math.exp(x) if x > -700.0 else 0.0
    return gam + (1.0 - gam) * (tot / len(ds))


def _lam_invert(rate, ds, gam, t_hint):
    """Solve _lam_accept(T) == rate for T by bisection in log T. t_hint sets
    the bracket's scale (the point-model answer, which is within an order of
    magnitude by construction)."""
    r = min(max(rate, 1e-9), 1.0 - 1e-9)
    if r <= gam:
        # already met by the deterministic members alone; the paper's Eq (18)
        # clamp in empirical form
        r = gam + (1.0 - gam) * 1e-3
    lo, hi = math.log(t_hint) - 12.0, math.log(t_hint) + 12.0
    if _lam_accept(math.exp(hi), ds, gam) < r:
        return math.exp(hi)
    if _lam_accept(math.exp(lo), ds, gam) > r:
        return math.exp(lo)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _lam_accept(math.exp(mid), ds, gam) < r:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


# SCHEDULE TABLE COORDINATE.  The table is uniform in
#
#     u = ln(-ln(rate)),
#
# not in rate or ln(rate), because the point model T = -dC/ln(rate) is
# ln T = ln(dC) - u -- exactly LINEAR in u, slope -1. So u is the coordinate in
# which the thing being tabulated is straight, the empirical schedule is a
# gentle curve on top of it, and linear interpolation is accurate everywhere.
# In ln(rate) the hot end has a local slope of ~7 and a 48-point grid was off
# by 0.39 in log T; in u the same budget lands within 1e-3.
_LAM_R_LO = 0.44 / 440.0                     # LamRate(1.0)
_LAM_R_HI = 0.98
_LAM_TAB_N = 64


def _lam_islands(mode):
    """Parse OGC_LAM into a membership test over island indices."""
    if mode in ("", "0", "off", "no", "none"):
        return frozenset()
    if mode in ("1", "all", "yes"):
        return None                      # None == every island
    out = set()
    for tok in mode.replace(" ", "").split(","):
        if tok:
            try:
                out.add(int(tok))
            except ValueError:
                pass
    return frozenset(out)


_LAM_SET = _lam_islands(_LAM_ISL)


class _LamAnneal:
    """Self-Tuning Lam controller for the Metropolis acceptance temperature.

    The tuning phase measures the neighbourhood's uphill cost distribution;
    that distribution plus Lam's trajectory gives a complete CALIBRATED
    SCHEDULE, defined implicitly by

        mean_j exp(-d_j / T_sched(q))  =  LamRate(q)

    i.e. the temperature at which the sampled worsening candidates would be
    accepted at exactly the rate Lam's trajectory asks for at progress q.
    (OGC_LAM_MODEL=point replaces the sample mean with the paper's single
    DeltaC, which is Eq (15) verbatim.) The feedback loop then rides on top as
    a multiplicative correction, so measurement only has to fix the MODEL ERROR
    -- the distribution came from a few dozen samples and it drifts as the
    search converges -- rather than discover the schedule from nothing.

    That split is what makes the controller safe on this solver. Uphill
    decisions are scarce and their number depends on the temperature itself --
    a hot T pushes cur_obj above its basin floor, and from a degraded solution
    a greedy repair nearly always improves, so worsening candidates dry up.
    MEASURED on island 2 at TL=90: 2034 uphill decisions on prob_12 against 39
    on prob_13, 34 on prob_5, 91 on prob_36. A pure feedback controller is
    open-loop on three of those four, and a pure feedback controller with
    nothing underneath it is then just a random walk in temperature.

    With the schedule underneath, the starved case degrades to "Lam's shape at
    the instance's own measured cost scale" -- which is still strictly more
    than the shipped curve knows, since t0_scale*w1 has no idea what a
    candidate's delta actually costs -- and the well-sampled case gets the full
    closed loop. The scarcity is also self-limiting in the right direction:
    few uphill decisions means few decisions the temperature governs, so the
    instances where the loop is blind are the instances where it matters least.

    Life cycle:  sample() while tuning -> maybe_finish() derives DeltaC, the
    schedule and the gain -> temp() every iteration, observe() per uphill
    Metropolis decision.
    """

    def __init__(self, total, w1, T_seed, log=None):
        self.total = max(1e-9, float(total))
        self.w1 = max(1.0, float(w1))
        # until the phase ends there is no calibrated scale, so the special
        # moves read the shipped T0
        self.T = max(1e-9, float(T_seed))
        self.T0 = self.T
        self.T_hi = math.inf
        self.T_lo = 1e-9
        self.tuning = True
        self.s = 0.0
        self.s_tuned = 0.0
        self.dC = 0.0
        self.gam = 0.0
        self.f = 0.0
        self.corr = 0.0                  # log-space feedback correction
        self.corr_lim = math.log(max(1.0 + 1e-9, _LAM_HI))
        self.A = 1.0
        self.tgt = 1.0
        self.t_prev = 0.0
        # tuning accumulators. The samples are kept, not just summed, because
        # the estimator is a median -- the phase is bounded by cal_max, so the
        # lists cannot grow without limit.
        self.k = 0
        self.n_imp = 0
        self.abs_s = []
        self.wor_s = []
        self.n_tie = 0
        self.cal_n = max(2, _LAM_CAL_N)
        self.cal_w = max(1, _LAM_CAL_W)
        self.cal_min = max(0.20, _LAM_CAL_MIN * self.total)
        self.cal_max = max(self.cal_min, _LAM_CAL_S * self.total)
        self.tau = max(1e-9, 0.5 * _LAM_EMA * self.total)
        self.a_cap = min(0.5, 2.0 / (max(2.0, _LAM_EMA_N) + 1.0))
        # population membership (see OGC_LAM_POP): "wor" keeps only worsening
        # candidates, which is what makes gamma vanish from the model
        self.pop_wor = (_LAM_POP == "wor")
        self.pop_all = (_LAM_POP == "all")
        self.n_obs = 0
        self.n_cool = 0
        self.n_heat = 0
        self.emp = False
        self.recal = None
        self.T_rep = None
        self.log = log
        self.trace = [] if _LAM_TRACE else None

    # -- tuning phase ------------------------------------------------------
    def sample(self, d):
        """Record one valid candidate's objective delta (paper Alg 6)."""
        if abs(d) <= 1e-9:
            self.n_tie += 1
            return
        self.k += 1
        self.abs_s.append(abs(d))
        if d > 0.0:
            self.wor_s.append(d)
        else:
            self.n_imp += 1

    def _member(self, d):
        """Is this candidate in the controlled population?"""
        if self.pop_all:
            return True
        if self.pop_wor:
            return d > 1e-9
        return abs(d) > 1e-9

    def maybe_finish(self, now, t_start):
        """End the tuning phase once it has enough samples, or has run long
        enough that waiting for them costs more than the estimate is worth."""
        if not self.tuning:
            return
        el = now - t_start
        if el < self.cal_min:
            return
        if el < self.cal_max and (self.k < self.cal_n
                                  or len(self.wor_s) < self.cal_w):
            return
        el = max(1e-6, el)
        k = self.k
        n_wor = len(self.wor_s)
        # gamma is the share of the CONTROLLED POPULATION accepted regardless
        # of temperature. Under the default population every member worsens, so
        # that share is zero by construction and the model reduces to
        # P(T) = exp(-DeltaC/T) -- no clamp, nothing to collapse.
        if self.pop_wor:
            m, n_det = n_wor, 0
        elif self.pop_all:
            m, n_det = k + self.n_tie, self.n_imp + self.n_tie
        else:
            m, n_det = k, self.n_imp
        if _LAM_DC in ("all", "wor"):
            src = self.wor_s if (_LAM_DC == "wor" and n_wor) else self.abs_s
            dC = _median(src)
        else:
            src = self.wor_s if (_LAM_DC == "mean" and n_wor) else self.abs_s
            dC = (sum(src) / len(src)) if src else 0.0
        if not (dC > 0.0) or not math.isfinite(dC):
            # the tuning walk never left a plateau. The paper uses DeltaC = 1
            # (its costs are unit integers); here the smallest objective step a
            # repair can actually produce is one tardy day on one block, w1.
            dC = self.w1
        # paper Eq (14), with the d == M adjustment that keeps gamma < 1
        gam = (n_det / m) if (m > 0 and n_det < m) else (
            n_det / (1.0 + m) if m > 0 else 0.0)
        f = min(0.4, max(1e-4, el / self.total))
        A0 = _lam_rate(f)                       # Eq (9), generalised
        R = _lam_rate(2.0 * f)                  # Eq (20), generalised
        self.dC, self.gam, self.f = dC, gam, f
        self._build_sched()
        T0 = self.sched(f)                      # Eq (19)
        T1 = self.sched(2.0 * f)                # Eq (22)
        # Eq (23) T0*beta**M = T1, in the time domain: the controller must be
        # able to move by that log-drop over one more tuning-phase duration.
        self.s = max(0.0, math.log(max(T0, 1e-300) / max(T1, 1e-300))) / el
        if not math.isfinite(self.s):
            self.s = 0.0
        # ------------------------------------------------------------------
        # AUTHORITY FLOOR on the correction's slew rate. Eq (24) sizes the gain
        # off one T0/T1 step measured in the first few percent of the run, and
        # that step can be arbitrarily small -- it collapsed to 0.018/s on
        # three of four probe instances. A correction that cannot move inside
        # the run is not feedback. So floor it on the log-span the schedule
        # itself sweeps, from where the run starts (rate A0) to frozen (rate at
        # q=1), and require that span to be traversable within OGC_LAM_FRZ of
        # the budget. This can only ever RAISE s, and the per-observation
        # kappa*alpha cap still bounds what any single step may do.
        span = abs(math.log(max(T0, 1e-300) / max(self.sched(1.0), 1e-300)))
        self.s_tuned = self.s
        self.s = max(self.s, span / max(1e-9, _LAM_FRZ * self.total))
        self.T0 = T0
        self.T_hi = max(T0, self.sched(f)) * max(1.0, _LAM_HI)
        self.A = A0
        self.tgt = A0
        self.tuning = False
        self.t_prev = now
        if _LAM_RECAL and n_wor < self.cal_w:
            self.recal = []
        self.T = self.temp(now, now - el)
        if self.log:
            self.log(f"lam tuned: pop={_LAM_POP} f={f:.4f} k={k} "
                     f"wor={n_wor} ties={self.n_tie} "
                     f"dC={dC:.0f} ({dC / self.w1:.2f}*w1) gam={gam:.3f} "
                     f"A0={A0:.3f} R={R:.3f} "
                     f"Tsched: {T0 / self.w1:.2f} -> "
                     f"{self.sched(0.4) / self.w1:.2f} -> "
                     f"{self.sched(1.0) / self.w1:.3f} (*w1)  "
                     f"s={self.s:.3f}/s (tuned {self.s_tuned:.3f}) "
                     f"acap={self.a_cap:.4f}")

    # -- schedule + control ------------------------------------------------
    def _build_sched(self):
        """Tabulate the open-loop schedule over rate. The schedule depends on q
        only through LamRate(q), so one table serves every q and the
        per-iteration cost is an index plus a lerp. See _LAM_TAB_N for why the
        grid is uniform in u = ln(-ln(rate))."""
        self._u0 = u0 = math.log(-math.log(_LAM_R_HI))
        self._du = (math.log(-math.log(_LAM_R_LO)) - u0) / (_LAM_TAB_N - 1)
        ds = self.wor_s if (_LAM_MODEL == "emp" and self.wor_s) else None
        tab = []
        for i in range(_LAM_TAB_N):
            r = math.exp(-math.exp(u0 + i * self._du))
            t = _lam_temp(r, self.gam, self.dC)
            if ds is not None:
                t = _lam_invert(r, ds, self.gam, t)
            tab.append(math.log(max(t, 1e-300)))
        self._tab = tab
        self.emp = ds is not None

    def sched(self, q):
        """The calibrated open-loop schedule: the temperature at which the
        controlled population is accepted at exactly LamRate(q)."""
        r = min(_LAM_R_HI, max(_LAM_R_LO, _lam_rate(q)))
        x = (math.log(-math.log(r)) - self._u0) / self._du
        if x <= 0.0:
            return math.exp(self._tab[0])
        if x >= _LAM_TAB_N - 1:
            return math.exp(self._tab[-1])
        i = int(x)
        w = x - i
        return math.exp(self._tab[i] + w * (self._tab[i + 1] - self._tab[i]))

    def temp(self, now, t_start):
        """Acceptance temperature at wall-clock `now`: the calibrated schedule
        times the feedback correction. Also latches T_rep, the temperature the
        reinsertion budget should use under OGC_LAM_REPAIR -- "sched" takes the
        open-loop schedule, which cannot form a feedback cycle with acceptance
        because it never reads it."""
        if self.tuning:
            self.T_rep = None
            return self.T
        q = (now - t_start) / self.total
        base = self.sched(q)
        T = base * math.exp(self.corr)
        if not math.isfinite(T):
            T = self.T0
        self.T = min(self.T_hi, max(self.T_lo, T))
        if _LAM_REPAIR == "sched":
            self.T_rep = base
        elif _LAM_REPAIR == "full":
            self.T_rep = self.T
        else:
            self.T_rep = None
        return self.T

    def observe(self, accepted, d, now, q):
        """One Metropolis decision on a member of the controlled population:
        update the EMA of the realised rate and move the correction toward
        whatever closes the gap to LamRate(q)."""
        if not self._member(d):
            # a plateau move carries no temperature information, and neither
            # does an improvement: both are taken whatever T is
            return
        dt = now - self.t_prev
        if dt < 0.0:
            dt = 0.0
        self.t_prev = now
        if self.recal is not None and d > 0.0:
            self.recal.append(d)
            if len(self.recal) >= self.cal_w:
                self._recalibrate()
        # paper Eq (6)/(10) in the time domain: alpha is the weight of a
        # 0.01*budget EMA at this observation's spacing, floored in SAMPLE
        # count so a sparse island does not control off one Bernoulli draw
        a = min(self.a_cap, -math.expm1(-dt / self.tau))
        self.A += a * ((1.0 if accepted else 0.0) - self.A)
        self.tgt = tgt = _lam_rate(q)
        err = self.A - tgt
        step = self.s * dt
        if _LAM_KAPPA > 0.0:
            step = min(step, _LAM_KAPPA * a)
        if _LAM_BAND > 0.0:
            step *= math.tanh(err / _LAM_BAND)
        elif err <= 0.0:
            step = -step
        c = self.corr - step
        self.corr = min(self.corr_lim, max(-self.corr_lim, c))
        self.n_obs += 1
        if step > 0.0:
            self.n_cool += 1
        elif step < 0.0:
            self.n_heat += 1
        if self.trace is not None and self.n_obs % 32 == 0:
            self.trace.append((round(q, 4), round(self.A, 4), round(tgt, 4),
                               round(self.T, 2)))

    def _recalibrate(self):
        """Rebuild the schedule from worsening deltas that arrived after a
        tuning phase which had too few of them to model. Once only."""
        ds = self.recal
        self.recal = None
        self.wor_s = list(ds)
        dC_new = _median(ds)
        if not (dC_new > 0.0) or not math.isfinite(dC_new):
            return
        dC_old, self.dC = self.dC, dC_new
        self._build_sched()
        # the correction so far was mostly this scale error; the new schedule
        # supersedes it, and keeping it would double-count
        self.corr = 0.0
        if self.log:
            self.log(f"lam recal: dC {dC_old:.0f} -> {dC_new:.0f} "
                     f"({dC_new / self.w1:.2f}*w1) from {len(ds)} uphill "
                     f"samples, Tsched {self.sched(0.4) / self.w1:.2f}*w1 "
                     f"at the plateau")

    def report(self):
        if self.tuning:
            return (f"lam=UNTUNED k={self.k} wor={len(self.wor_s)} "
                    f"ties={self.n_tie}")
        return (f"lam pop={_LAM_POP} plat={_LAM_PLAT:.3f} "
                f"rep={_LAM_REPAIR} "
                f"model={'emp' if self.emp else 'point'} "
                f"T={self.T:.3g} ({self.T / self.w1:.3f}"
                f"*w1) corr={math.exp(self.corr):.2f}x A={self.A:.3f} "
                f"tgt={self.tgt:.3f} dC={self.dC:.0f} ({self.dC / self.w1:.2f}"
                f"*w1) gam={self.gam:.3f} s={self.s:.3f} obs={self.n_obs} "
                f"cool/heat={self.n_cool}/{self.n_heat}")


# ---------------------------------------------------------------------------
# LOOKAHEAD-ARBITRATED PLACEMENT (OGC_LOOK)
#
# Every spatial-discipline experiment so far (exit zoning, exit-aligned
# refinement, sliver tie-breaks, size-gradient anchoring, border reservation)
# picked its placement from a fixed rule and hoped the rule was right. All six
# lost. The rules are not obviously wrong -- what is wrong is deciding blind:
# the immediate cost of a placement cannot see the fragmentation it leaves
# behind, and a placement PERSISTS for the block's whole stay, so the damage
# is paid on every day of its span.
#
# So do not pick a rule. Generate the alternatives each rule would propose,
# then let the next few blocks vote. This is the paper's planning framework
# (Zhao et al., "Deliberate Planning of 3D-BPP on Packing Configuration
# Trees") reduced to what applies here:
#
#   * model-based planning (Sec 3.4): score a placement by rolling the next
#     few arrivals forward, execute only the FIRST node, replan every step;
#   * spatial ensemble (Sec 3.3): the roll-outs are not commensurable, so
#     compare candidates by ascending RANK within each view and take the
#     candidate with the best WORST rank, rather than summing magnitudes.
#
# Two invariants keep this out of the failure mode above:
#   1. a view is admissible only when its TRUE cost does not exceed the
#      incumbent's -- the objective is never traded for a spatial preference;
#   2. ties resolve to candidate 0, the ordinary full-range bottom-left
#      placement, so an uninformative lookahead reproduces the baseline
#      trajectory exactly.
_LOOK = os.environ.get("OGC_LOOK", "0") == "1"
# which alternative views to propose alongside the incumbent
_LOOK_MODES = [s for s in os.environ.get("OGC_LOOK_MODES",
                                         "flip,left,right").split(",") if s]
_LOOK_L = int(os.environ.get("OGC_LOOK_L", "6"))      # blocks rolled forward
_LOOK_AGG = os.environ.get("OGC_LOOK_AGG", "rank")    # rank | sum
_LOOK_Q = _fenv("OGC_LOOK_Q", 1.0)   # band views: size quantile
_LOOK_W = _fenv("OGC_LOOK_W", 0.25)  # band width
# CONGESTION GATE: arbitrate only where space actually binds. In a bay-day
# that is half empty no arrangement of this block denies the next one
# anything, so the roll-out returns the same key for every candidate and the
# probes are pure cost. Cost matters here: construction budget is spent
# against a deadline, and measured at EQUAL WALL CLOCK an ungated arbiter
# (~8x a plain construction) loses to plain multi-start despite being -6.7%
# better per construction. 0 disables the gate.
_LOOK_OCC = _fenv("OGC_LOOK_OCC", 0.55)
_EPS = 1e-9
_BIG = 1e15
LOOK_STATS = {"steps": 0, "gated": 0, "multi": 0, "override": 0, "probes": 0,
              "paced_off": 0}


def _look_tight(state, p, i, c):
    """Is the incumbent placement going into contested space? Peak layer-0
    occupancy of the target bay over the placement's own day span, including
    the block itself, as a fraction of bay area."""
    b, t = int(c[1]), int(c[2])
    t2 = min(t + int(p.P[i]), state.occ0.shape[1])
    if t2 <= t:
        return True
    area = float(p.bayW[b]) * float(p.bayH[b])
    if area <= 0.0:
        return True
    peak = float(state.occ0[b, t:t2].max())
    cells = float(p.ncells0[int(p.orient_base[i]) + int(c[3])])
    return (peak + cells) / area >= _LOOK_OCC


def _look_views(state, i, c0, xd, lift, wide, modes=None, use_guide=True,
                ydir=1):
    """Alternative placements of block i that cost no more than the incumbent.

    Returns [c0, ...] -- the incumbent is always first, so a caller that
    cannot separate them keeps today's behavior. Views are generated without
    bay noise or blink: they exist to be COMPARED against the incumbent, and
    the admissibility test below is on true cost, so a noise-free view can
    never smuggle in a worse placement.
    """
    cands = [c0]
    seen = {(c0[1], c0[2], c0[3], c0[4], c0[5])}
    cap = c0[0] + _EPS
    for mode in (_LOOK_MODES if modes is None else modes):
        if mode == "flip":
            # mirrored anchor: same discipline, opposite corner
            c = state.best_insertion(i, use_guide=use_guide,
                                     use_dayguide=lift, xdir=-xd, ydir=ydir,
                                     best_cost_init=cap)
        elif mode in ("left", "right"):
            if not wide:
                continue
            c = state.best_insertion(i, use_guide=use_guide,
                                     use_dayguide=lift, xdir=xd, ydir=ydir,
                                     band=mode, best_cost_init=cap)
        else:
            continue
        # best_cost_init is a BIASED cap (guide bias has headroom), so the
        # returned winner can exceed the true budget -- re-check invariant 1.
        if c is None or c[0] > cap:
            continue
        key = (c[1], c[2], c[3], c[4], c[5])
        if key in seen:
            continue
        seen.add(key)
        cands.append(c)
    return cands


def _look_future(p, i, order, k, cands, L):
    """The next unplaced blocks that could actually collide with block i:
    they must fit one of the candidate bays and arrive before the placement
    ends. The forward scan is windowed so the step stays O(L), not O(n)."""
    bays = {c[1] for c in cands}
    tmax = max(int(c[2]) for c in cands) + int(p.P[i])
    out = []
    for j in order[k + 1:k + 1 + 12 * L]:
        j = int(j)
        if int(p.R[j]) >= tmax:
            continue
        if not any(p.block_fits[j, b] for b in bays):
            continue
        out.append(j)
        if len(out) >= L:
            break
    return out


def _look_probe(state, i, cands, future):
    """Roll each candidate forward: place it, then ask what the next blocks
    would cost, then take it back. Returns keys[cand][view].

    When every candidate sits in the same bay -- the common case, since
    cost-tied views rarely change bay -- probing is restricted to that bay.
    That is exact (an insertion into bay b changes no other bay) and m times
    cheaper, and it measures the MARGINAL damage rather than drowning it in
    the unchanged alternatives. When the candidates disagree about the bay,
    the restriction would compare different bays' costs, so probe globally.

    Probes run under an INCUMBENT-BEST CAP: at each view we already know the
    best key any earlier candidate achieved, and a candidate that cannot beat
    it changes no ranking decision, so the scan is capped there and allowed to
    give up early. Candidates that fail the cap all collapse to one "no better
    here" bucket -- which is exactly the resolution the rank argmin needs,
    since distinctions among candidates that are all worse than the leader can
    never change the winner. Only the first candidate pays an uncapped scan.
    The `sum` aggregation adds magnitudes, so it opts out of the cap.
    """
    only = cands[0][1] if len({c[1] for c in cands}) == 1 else None
    guide = only is None   # with a single bay to order, the guide only costs
    capped = _LOOK_AGG != "sum"
    nv = len(future)
    keys = [[None] * nv for _ in cands]
    bestv = [None] * nv
    seq0 = state.seq_counter
    for ci, c in enumerate(cands):
        _, b, t, o, px, py = c
        state.insert(i, b, t, o, px, py)
        for v, j in enumerate(future):
            cap = _BIG if (bestv[v] is None or not capped) else (
                bestv[v][0] + _EPS)
            r = state.best_insertion(j, only_bay=only, use_guide=guide,
                                     best_cost_init=cap)
            if r is None or r[0] > cap:
                continue                      # no better than the leader
            k = (r[0], float(r[2]))
            keys[ci][v] = k
            if bestv[v] is None or k < bestv[v]:
                bestv[v] = k
        state.remove(i)
    for row in keys:
        for v in range(nv):
            if row[v] is None:
                row[v] = (_BIG, _BIG)
    # insert/remove is exactly self-inverse for a block appended last, but it
    # does burn commit sequence numbers; give them back so a construction with
    # the arbiter on is comparable to one without.
    state.seq_counter = seq0
    LOOK_STATS["probes"] += len(cands) * len(future)
    return keys


def _look_pick(cands, keys, rng=None):
    """Spatial ensemble: best worst-rank across views (paper Sec 3.3).

    `blind`/`rand` are CONTROLS, not modes to ship: they take an alternative
    view without consulting the roll-out, and exist to prove the gain comes
    from the arbitration rather than from merely having a second candidate.
    """
    nc = len(cands)
    if _LOOK_AGG == "blind":
        return nc - 1
    if _LOOK_AGG == "rand":
        return int(rng.integers(nc)) if rng is not None else nc - 1
    nv = len(keys[0]) if keys else 0
    if nv == 0:
        return 0
    if _LOOK_AGG == "sum":
        return min(range(nc),
                   key=lambda ci: (sum(k[0] for k in keys[ci]),
                                   sum(k[1] for k in keys[ci]), ci))
    worst = [0] * nc
    tot = [0] * nc
    for v in range(nv):
        col = sorted(range(nc), key=lambda ci: keys[ci][v])
        # competition ranking: equal keys share a rank, so a view that cannot
        # tell the candidates apart contributes no separation at all
        rank = [0] * nc
        r, prev = 0, None
        for pos, ci in enumerate(col):
            if prev is None or keys[ci][v] != prev:
                r, prev = pos, keys[ci][v]
            rank[ci] = r
        for ci in range(nc):
            if rank[ci] > worst[ci]:
                worst[ci] = rank[ci]
            tot[ci] += rank[ci]
    # invariant 2: `ci` last => ties keep the incumbent
    return min(range(nc), key=lambda ci: (worst[ci], tot[ci], ci))


def _size_xdir(p, i, med_area):
    return 1 if float(p.ncells0[p.orient_base[i]]) >= med_area else -1


def _exit_xdir(p, i, med):
    """Corner zone from a block's earliest-exit estimate."""
    return 1 if float(p.R[i] + p.P[i]) <= med else -1
# ---------------------------------------------------------------------------
# FAILURE-DIRECTED DESTRUCTION (OGC_CRIT)
#
# The destroy portfolio samples structurally: at random, by tardiness, by
# window, by bay load, by guide deviation. None of it uses what the search has
# already LEARNED about which blocks are hard. On the long-TL overload
# instances our islands plateau, and a plateau is precisely the regime where
# the same few blocks keep failing to reinsert, iteration after iteration,
# while the destroy portfolio keeps sampling elsewhere.
#
# Nascimento et al. (EJOR 2026), "Improving the Efficiency of Logic-Based
# Benders Decomposition for p-Batch Scheduling Problems with 2D Packing",
# find that the single most effective accelerator (-77%..-98% run time, far
# ahead of every subproblem-side method) is a CRITICALITY INDEX: score each
# part by how often it took part in a subproblem that turned out infeasible,
# and steer the master toward solutions that avoid the critical combinations.
# Their conclusion is explicit -- subproblem-side acceleration alone is not
# enough, because the master admits many equal-objective solutions and
# progress requires steering it toward the ones that will actually pack.
#
# The LNS analogue: a repair attempt that aborts is our infeasible subproblem,
# and the block it aborted on is the binding constraint. Accumulate that, then
# destroy AROUND those blocks -- eject the incumbents holding the window the
# critical block keeps failing to reach. The pattern (seed + make room) is the
# one _rm_offguide/_rm_offday already use, but seeded from measured failure
# instead of from a relaxation target, so it keeps working after the guides
# have stopped saying anything new.
_CRIT = os.environ.get("OGC_CRIT", "0") == "1"
_CRIT_DECAY = _fenv("OGC_CRIT_DECAY", 0.995)
_CRIT_W0 = _fenv("OGC_CRIT_W0", 1.0)  # initial op weight

# ---------------------------------------------------------------------------
# LOOKAHEAD-ARBITRATED REPAIR (OGC_RLOOK)
#
# The construct-side arbiter above wins -6.7% per construction but has to GUESS
# which blocks come next. In LNS repair there is nothing to guess: the blocks
# that follow are exactly `order[idx+1:]`, already chosen. So the roll-out is
# over the true continuation, and the horizon is bounded by the ruin size
# rather than by an arbitrary window.
#
# It is still ~5x the cost of a plain repair, and in LNS throughput is the
# whole game -- so it is not a mode, it is a MOVE, taking an adaptive share of
# iterations driven by its own acceptance rate exactly like try_nest/try_chain.
# If it earns its cost the share grows; if it does not it decays to ~2% and
# costs almost nothing. That is the same self-limiting contract every other
# expensive move in this file already honours.
_RLOOK = os.environ.get("OGC_RLOOK", "0") == "1"
_RLOOK_L = int(os.environ.get("OGC_RLOOK_L", "4"))
_RLOOK_MODES = [s for s in os.environ.get("OGC_RLOOK_MODES",
                                          "flip").split(",") if s]

# ---------------------------------------------------------------------------
# FAIL-FAST REPAIR ORDER (OGC_RORD)
#
# The same LBBD paper's subproblem accelerator (4.2.1) is a search-order rule:
# place the parts with the FEWEST feasible positions first, tie-broken by the
# ones that block the most space. It cuts their infeasibility-proof time by
# ~3.3x, because a repair that is going to fail should fail on its first
# insertion, not its last -- everything placed before the failure is wasted.
#
# Our repair-order portfolio sorts by due date, area-time, day target or
# structural cost. None of it is failure-aware. This branch orders by measured
# criticality first, then by the cheap standing analogues of |IFP_j|: how few
# bays the block fits, and how little slack its window has. Costs nothing --
# it is a sort key, not a scan.
#
# Measured: 66% of repairs abort, and they abort at index ~11 of ~16 -- every
# insertion before that point was built and thrown away (48k wasted insertions
# per 45s of search). Ordering failure-first cuts the abort index to ~6.7 and
# the waste by 29%; single-island A/B (8 seeds x 4 dense instances) -5.26%.
# Applied to EVERY repair (P=1.0) rather than a share: at P=0.30 the effect is
# -0.38%, i.e. noise. This does cost repair-order diversity, which the other
# branches exist to provide -- a real trade-off, and the reason the share is
# left tunable.
#
# ADOPTED (default ON, 2026-07-31). Official-TL A/B vs the shipping engine,
# 3 rolls per arm: prob_5@600 -13.8%, prob_4@480 -16.8%, prob_3@240 -3.3%,
# 9/9 paired rolls, and on 4 and 5 the two arms' roll ranges do not even
# overlap. Set OGC_RORD=0 to restore the previous repair-order portfolio.
_RORD = os.environ.get("OGC_RORD", "1") == "1"
_RORD_P = _fenv("OGC_RORD_P", 1.0)

# ---------------------------------------------------------------------------
# ADAPTIVE RUIN SIZE (OGC_KADAPT)
#
# PCT's recursive packing (Sec 4.5, Fig. 9 / Table 8) finds that solution
# quality peaks sharply when the sub-problem size matches what the base policy
# is actually good at -- they measure that scale (95th percentile of what the
# policy packs, tau=30) rather than guessing it, and both larger and smaller
# decompositions lose. Our ruin size is the same quantity: `_ksize` samples
# uniformly from [3, k_max] with a fixed k_max=12, chosen once and never
# revisited. Learn it instead, from the acceptance rate each size actually
# earns, with the same multiplicative-weights scheme the operator portfolio
# already uses.
#
# Measured: the learned distribution drops the abort RATE from 0.69 to 0.55 --
# it finds the ruin sizes the repair can actually satisfy. Single-island A/B
# -4.81%; combined with OGC_RORD, -9.38% on 32/32 paired seeds. The two are
# complementary: RORD makes failures cheap, KADAPT makes them rarer, and
# together they cut wasted insertions 34.9k -> 19.5k per 45s.
#
# VERDICT (2026-08-11): the two defects below are fixed, and the LEARNING IS
# THEN TURNED OFF, because with the defects fixed it stops paying.
#
# Paired single-island A/B, 300s, 12 jobs over prob_5/12/13/16/20/26, arms:
# the shipping engine, the corrected range with uniform sampling, and the
# corrected range with the rewritten learner (bench/ab_kadapt.log).
#   uniform over [3,13]  -0.74%  (better on 7/12)
#   learned over [3,13]  +1.16%  (better on 6/12; +3.0% prob_13, +8.3% prob_16)
# The learner takes 18% MORE iterations and still loses: it tilts toward small
# ruins (mean 7.0 against uniform's 8.0), which buys throughput at the cost of
# move quality, and on congested instances that is the wrong trade. So the
# default is OGC_KADAPT=0 -- which now genuinely means "uniform over the tuned
# range", not "uniform over a range nobody chose". The learner is kept, correct
# and measured, behind OGC_KADAPT=1.
#
# REWRITTEN 2026-08-11 -- the first implementation did not do what this comment
# says, for two independent reasons, both measured on a live island:
#
#  1. `k_w` was sized `k_max + 2` = 14 AT CONSTRUCTION, but `k_max` is a live
#     variable that the iteration-time controller at the bottom of run() walks
#     up to 30. numpy slicing truncates silently, so `k_w[lo:k_max+1]` returned
#     11 entries no matter how large k_max grew and the drawn size was capped
#     at 13. Instrumented on prob_12/prob_5: k_max hits its 30 ceiling within
#     ~17 iterations and stays there for >99.9% of the run, yet no ruin larger
#     than 13 was ever drawn. The adopted -4.81% A/B was therefore not
#     measuring a learned distribution at all -- it was measuring "cap the ruin
#     at 13" against "sample uniformly up to 30", with the cap arriving as an
#     off-by-one in an array length.
#
#  2. The multiplicative update could not discriminate. Reward was x1.05 on an
#     accepted move and x0.999 on a rejected one, so the break-even acceptance
#     rate is ln(1/0.999)/ln(1.05*1/0.999) = 2.0%. Every ruin size beats that
#     by an order of magnitude, so every arm walked to the 5.0 cap and stayed
#     pinned: measured shares 8.0%-9.8% across all 11 live sizes -- uniform --
#     while the acceptance rates they were supposed to be learning from ranged
#     76% (k=3) to 33% (k=13) on prob_12 and 64% to 9% on prob_5. A 7x signal,
#     entirely discarded.
#
# The two defects pulled in opposite directions and very nearly cancelled: a
# learner that could not learn, sampling a range that was accidentally right.
# The fix therefore keeps the range (see _K_CEIL -- widening it to the value
# the code intended LOSES, and that is measured, not assumed) and replaces the
# learner.
#
# What replaces it is the scheme the operator portfolio in this same file
# already uses and that demonstrably works: accumulate a reward per arm, then
# periodically re-derive the weights from the arm's MEAN reward with an
# exploration floor. Three things are specific to ruin size:
#
#  * Reward is per SECOND. A ruin of 30 costs several times what a ruin of 3
#    costs, so crediting a bare acceptance would systematically overpay large
#    sizes for the extra work they were given. Objective gain per unit wall
#    clock is the currency LNS actually spends.
#  * Neighbouring sizes pool their statistics (3-tap smoothing of the reward
#    SUMS and COUNTS separately, so the pooling is a weighted mean and not a
#    mean of means). This is the same unimodality PCT's Fig. 9 reports: reward
#    is smooth in the sub-problem scale, so with up to 28 arms and 512 samples
#    per window an unpooled per-arm estimate is mostly noise.
#  * The floor is 0.15 of the top arm rather than the operator portfolio's
#    0.05: every size in [3, 13] is part of the tuned policy and worth keeping
#    in play, so the learner tilts the distribution instead of collapsing it.
_KADAPT = os.environ.get("OGC_KADAPT", "0") == "1"
# Hard ceiling on the ruin size: the length of the weight vector AND the bound
# the iteration-time controller in run() clamps k_max to. These two must agree
# -- when they did not, the smaller one silently won, and 13 is the value that
# accident produced. It is kept, deliberately, because it MEASURES BEST.
#
# The controller's own ceiling was 30 and it reaches it in ~17 iterations, so
# with the array bug fixed the natural reading is "sample [3, 30] and let the
# learner sort it out". That was tested: 4 exploration-floor settings against
# the shipping engine, 8 paired 300s single-island jobs each (prob_5/12/13),
# bench/ab_floor_range30.log. Every one of them LOST -- +0.50%, +1.42%,
# +0.36%, +1.17% -- with prob_13 (n=300, m=2, the most congested instance in
# the set) driving +1.2% to +3.6% of it. Ruins above ~13 do not pay for their
# wall clock on congested instances no matter how the distribution is learned,
# and the accidental cap was doing real work.
#
# So the cap stays, but as a stated constant applied to BOTH sampling paths
# instead of an off-by-one that only bound one of them. That also repairs what
# the OGC_KADAPT knob MEANS: it used to switch the range (uniform [3,30]) and
# the learning together, so measuring "KADAPT on vs off" measured mostly the
# range. Now both paths sample [3, 13] and the knob isolates the learning.
_K_CEIL = 13
_K_LO = 3            # smallest ruin any caller asks for (_ksize default)
_K_WIN = 512         # credits between weight re-derivations


# Exploration floor, as a fraction of the best arm; it bounds the achievable
# weight ratio at (1 + f) / f, so at 0.15 the most-favoured ruin size can be
# drawn at most 7.7x as often as the least. Deliberately conservative: over
# [3, 13] every arm is already known to be worth sampling (that whole range is
# the tuned policy), so the learner's job is to tilt the distribution, not to
# collapse it onto one size. Compare the operator portfolio's 0.05 -- looser
# there because a bad OPERATOR really is worth starving.
_K_FLOOR = _fenv("OGC_K_FLOOR", 0.15)
_K_ACC = _fenv("OGC_K_ACC", 0.1)    # accepted-but-not-improving, in w1 units
_K_BEST = _fenv("OGC_K_BEST", 1.0)  # new-incumbent bonus, in w1 units

# ---------------------------------------------------------------------------
# INSERT-THEN-SEPARATE WITH VICTIM RETIMING (OGC_SEPMOVE)
#
# Probe #2 (experiments/sparrow_probe_t.py, 2026-08-04) is the SPEC: force a
# tardy seed into an earlier window at the min-overlap position (overlap
# allowed transiently), then legalize by moving the colliding victims --
# spatially, or LATER IN TIME -- under a net-positive w1-day budget (seed
# tardy-days freed minus victims' tardy-days added must stay > 0). Measured
# 7/8 legalizations, net +3..+204 w1-days per success on congested cells;
# ZERO-SUM at capacity-exceeded density, which is why the driver band-gates
# the knob to peak_util [0.70, 2.0) on overload/congested groups only.
# This is an LNS move, not the probe: engine state/kernels only, cell-level
# separation, exact commit through the same test every other move uses.
# Read PER-RUN in LNS.__init__ (the OGC_DEEP_P pattern) because the driver
# writes the knob after this module is import-cached. Default OFF: the run-
# loop guard short-circuits before any rng draw, stream bit-identical.

# depth-2 chains: a stuck ejected block may evict its preferred bay's window
_CHAIN2 = os.environ.get("OGC_CHAIN2", "0") == "1"
# sacrifice-role swap: exchange the tardy/on-time roles of two blocks when
# the day relaxation disagrees with the incumbent about WHO should be late
_ROLESWAP = os.environ.get("OGC_ROLESWAP", "0") == "1"

from _ogc.ogc_prep import Prep
from _ogc.ogc_state import State


def construct(state, order=None, variant=0, deadline=None, day_seed=False,
              look=None):
    p = state.prep
    if order is None:
        if day_seed and state.tday is not None:
            # relaxation-seeded construction: insert in the global/day
            # relaxation's start order -- an 'OptiFrame-shaped' start for
            # congested instances, run as one hedged island of the
            # portfolio. NOTE: the adopted -8.96%/-11.5% A/Bs measured the
            # ORDERING alone; OGC_SEED_LIFT additionally lifts each scan
            # into its target window and is unmeasured, default off.
            td = state.tday
            order = sorted(range(p.n),
                           key=lambda i: (int(td[i]) if td[i] >= 0
                                          else int(p.R[i]),
                                          p.D[i], p.R[i]))
        elif variant == 1:
            order = sorted(range(p.n), key=lambda i: (p.R[i], p.D[i]))
        elif variant == 2:
            # EDD with shortest area-time first within due cohorts
            o0 = p.orient_base[:-1]
            order = sorted(range(p.n),
                           key=lambda i: (p.D[i],
                                          int(p.P[i]) * int(p.ncells0[o0[i]])))
        elif variant == 3:
            o0 = p.orient_base[:-1]
            order = sorted(range(p.n),
                           key=lambda i: (p.D[i], -int(p.ncells0[o0[i]])))
        else:
            order = sorted(range(p.n), key=lambda i: (p.D[i], p.R[i], -p.wl[i]))
    lift = day_seed and os.environ.get("OGC_SEED_LIFT", "0") == "1"
    med = float(np.median(p.R + p.P)) if _EXITCORNER else 0.0
    med_a = (float(np.median(p.ncells0[p.orient_base[:-1]]))
             if _SIZECORNER else 0.0)
    # the arbiter costs ~8x a plain construction, which is nothing for a
    # one-shot island seed and fatal inside a generation loop -- so the call
    # site, not just the env, decides
    look = (_LOOK if look is None else look) and _LOOK_L > 0
    look_wide = None
    if look:
        if "left" in _LOOK_MODES:
            state.enable_bands(frac=_LOOK_W, side="left")
        if "right" in _LOOK_MODES:
            state.enable_bands(frac=_LOOK_W, side="right")
        if _LOOK_Q >= 1.0:
            look_wide = np.ones(p.n, bool)
        else:
            a = p.ncells0[p.orient_base[:-1]].astype(np.float64)
            look_wide = a <= np.quantile(a, _LOOK_Q)
    small = None
    if _BAND:
        areas = p.ncells0[p.orient_base[:-1]].astype(np.float64)
        small = areas <= np.quantile(areas, _BAND_Q)
        state.enable_bands(frac=_BAND_W, side=_BAND_SIDE)
    t_look0 = time.time()
    n_ord = len(order)
    for k_pos, i in enumerate(order):
        if look:
            LOOK_STATS["gated"] += 1
            # PACE GUARD. Falling off `deadline` mid-order is far worse than
            # any arbitration is worth: the remaining blocks get dumped at
            # their empty-bay upper-bound day. So keep arbitrating only while
            # the construction is on pace to finish inside half its budget,
            # and drop to the plain decoder the moment it is not.
            if deadline is not None and (k_pos & 31) == 31:
                el = time.time() - t_look0
                if el * n_ord > 0.5 * (deadline - t_look0) * (k_pos + 1):
                    look = False
                    LOOK_STATS["paced_off"] += 1
        cand = None
        if _SIZECORNER:
            xd = _size_xdir(p, i, med_a)
        elif _EXITCORNER:
            xd = _exit_xdir(p, i, med)
        else:
            xd = 1
        if deadline is None or time.time() < deadline:
            cand = state.best_insertion(i, use_guide=True,
                                        use_dayguide=lift, xdir=xd)
            if cand is not None and look and _look_tight(state, p, i, cand):
                LOOK_STATS["steps"] += 1
                cands = _look_views(state, i, cand, xd, lift,
                                    bool(look_wide[i]))
                if len(cands) > 1:
                    LOOK_STATS["multi"] += 1
                    pick = 0
                    if _LOOK_AGG in ("blind", "rand"):
                        pick = _look_pick(cands, None, state.rng)
                    else:
                        fut = _look_future(p, i, order, k_pos, cands, _LOOK_L)
                        if fut:
                            pick = _look_pick(
                                cands, _look_probe(state, i, cands, fut))
                    if pick:
                        LOOK_STATS["override"] += 1
                        cand = cands[pick]
            if cand is not None and small is not None and small[i]:
                # retry inside the reserved border band; adopt only when the
                # true cost does not rise, so the objective is never traded
                # away for the spatial discipline
                cb = state.best_insertion(i, use_guide=True,
                                          use_dayguide=lift, xdir=xd,
                                          band=True,
                                          best_cost_init=cand[0] + 1e-9)
                if cb is not None and cb[0] <= cand[0] + 1e-9:
                    cand = cb
        if cand is None:
            # past deadline or (should not happen) no candidate found: place
            # at the guaranteed-feasible empty-bay day of a fitting bay
            for b in range(p.m):
                if p.block_fits[i, b]:
                    t = state.day_upper_bound(i, b)
                    r = state.try_day(i, b, t)
                    if r is not None:
                        o, px, py = r
                        state.insert(i, b, t, o, px, py)
                        break
            else:
                raise RuntimeError(f"block {i} fits no bay")
            continue
        c, b, t, o, px, py = cand
        if _EXITPACK == "1":
            ref = state.refine_position(i, b, t, o, xdir=xd,
                                        px0=px, py0=py)
            if ref is not None:
                px, py = ref
        elif _EXITPACK == "2":
            px, py = state.refine_position_exact(i, b, t, o, px, py)
        elif _EXITPACK == "3":
            px, py = state.refine_position_sliver(i, b, t, o, px, py)
        state.insert(i, b, t, o, px, py)
    return state


def polish(state, t_deadline, max_rounds=6):
    """Strict-improvement relocation descent: for every block carrying cost
    (tardiness or preference penalty, or sitting in the max-loaded bay), try
    re-inserting it; keep only strict total-objective improvements."""
    p = state.prep
    improved_any = True
    rounds = 0
    while improved_any and rounds < max_rounds and time.time() < t_deadline:
        rounds += 1
        improved_any = False
        tard = np.maximum(0, state.t2 - p.D) * state.placed
        prefpen = ((p.smax - p.pref[np.arange(p.n),
                                    np.maximum(state.bay, 0)])
                   * state.placed)
        gain = p.w1 * tard + p.w3 * prefpen
        # blocks in the most weighted-loaded bay can also help obj2
        wload = p.u * state.loads
        bmax = int(np.argmax(wload))
        order = np.argsort(-gain)
        for i in order:
            i = int(i)
            if time.time() > t_deadline:
                break
            if not state.placed[i]:
                continue
            if gain[i] <= 0 and int(state.bay[i]) != bmax:
                break
            old_obj = state.objective()
            old = (int(state.bay[i]), int(state.t1[i]), int(state.oi[i]),
                   int(state.px[i]), int(state.py[i]), int(state.seq[i]),
                   int(state.t2[i]))
            # cap the search at the block's current cost: polish only keeps
            # strict improvements, so anything at or above it is useless --
            # this also bounds the scan time
            cap = (p.w1 * max(0, int(state.t2[i]) - int(p.D[i]))
                   + p.w3 * (p.smax[i] - p.pref[i, old[0]])
                   + p.w2 * (state.obj2() + 1.0))
            state.remove(i)
            cand = state.best_insertion(i, best_cost_init=cap + 1e-6)
            if cand is None:
                state.insert(i, old[0], old[1], old[2], old[3], old[4],
                             seq=old[5], t2=old[6])
                continue
            c, b, t, o, px, py = cand
            state.insert(i, b, t, o, px, py)
            if state.objective() < old_obj - 1e-9:
                improved_any = True
            else:
                state.remove(i)
                state.insert(i, old[0], old[1], old[2], old[3], old[4],
                             seq=old[5], t2=old[6])
    return state


class LNS:
    def __init__(self, state, rng, t_end, log=None, sync_cb=None,
                 sync_every=None, day_resolve=False, sisr=False, lahc=0):
        self.state = state
        self.rng = rng
        self.t_end = t_end
        self.log = log
        self.sync_cb = sync_cb      # cross-island best sharing
        self.sync_every = (float(os.environ.get("OGC_SYNC_EVERY", "20"))
                           if sync_every is None else sync_every)
        p = state.prep
        try:
            self.cap_t = float(os.environ.get("OGC_CAP_T", "2.0"))
            self.cap_w1 = float(os.environ.get("OGC_CAP_W1", "1.0"))
            if not math.isfinite(self.cap_t) or self.cap_t < 0.0:
                self.cap_t = 2.0
            if not math.isfinite(self.cap_w1) or self.cap_w1 < 0.0:
                self.cap_w1 = 1.0
        except (TypeError, ValueError):
            self.cap_t, self.cap_w1 = 2.0, 1.0
        self.n = p.n
        # operator weights (adaptive)
        self.ops = [self._rm_random, self._rm_tardy, self._rm_window,
                    self._rm_pref, self._rm_bay_load, self._rm_single,
                    self._rm_rect, self._rm_offguide]
        if _HOLERUIN:
            self.ops.append(self._rm_hole)
        # criticality index: how often each block has been the one a repair
        # aborted on. Decayed so it tracks the CURRENT binding constraints,
        # not the whole run's history -- same reasoning as the adaptive
        # operator weights.
        self.crit = np.zeros(self.n)
        self._crit_n = 0
        if _CRIT:
            self.ops.append(self._rm_crit)
        # day-target machinery only exists when the day guide supplied
        # targets; without them the operator list, rng stream and weights
        # are exactly the tuned defaults
        if getattr(state, "tday", None) is not None:
            self.ops.append(self._rm_offday)
        # SISR string-removal arm (ported from the engine_v47 fork, where it
        # produced the hidden P5 records); appended LAST and only when the
        # island opts in, so every non-SISR island keeps the tuned operator
        # list and rng stream bit-for-bit
        self._sisr = bool(sisr)
        if self._sisr:
            self.ops.append(self._rm_string)
        try:
            self._day_p = float(os.environ.get("OGC_DAY_P", "0.5"))
            if not (0.0 <= self._day_p <= 1.0):  # NaN fails into default
                self._day_p = 0.5
        except (TypeError, ValueError):
            self._day_p = 0.5
        self._day_p0 = self._day_p  # recovery ceiling for the valve
        self._day_reg = 0           # consecutive-regression counter
        self._day_resolve = bool(day_resolve)  # one-shot mid-run re-solve
        self._day_resolved = False
        self.op_w = np.ones(len(self.ops))
        self.op_w[5] = 0.35  # single-block moves are weak; let adaptation grow them
        if _CRIT:
            self.op_w[self.ops.index(self._rm_crit)] = _CRIT_W0
        if self._sisr:
            # strings are the strongest overload operator; start above par
            self.op_w[self.ops.index(self._rm_string)] = 1.5
        # knob parsing mirrors the OGC_DAY_P hardening: malformed values
        # fall back to defaults instead of killing the island worker
        self.blink = 0.0
        self.lahc_len = 0
        if self._sisr:
            try:
                bl = float(os.environ.get("OGC_BLINK", "0.01"))
                # NaN and out-of-range (>=1 would blink away every bay)
                # both fail into the default
                self.blink = bl if 0.0 <= bl < 1.0 else 0.01
            except (TypeError, ValueError):
                self.blink = 0.01
            # Acceptance portfolio: late-acceptance hill climbing instead of
            # Metropolis SA when a length is given. The caller passes `lahc`
            # for per-island selection (ogc_solve's full-dose mode); OGC_LAHC
            # remains an explicit user override and wins when set, which is
            # also why the caller's value is only consulted if it is not.
            try:
                self.lahc_len = max(
                    0, int(float(os.environ.get("OGC_LAHC", lahc or 0))))
            except (TypeError, ValueError):
                self.lahc_len = max(0, int(lahc or 0))
        self._nest_acc = 0
        self._nest_try = 0
        self._dp_acc = 0
        self._dp_try = 0
        # insert-then-separate move (OGC_SEPMOVE): read per-run, never at
        # import -- the driver band-gates the knob after this module is
        # cached (same reasoning as OGC_DEEP_P below)
        self._sep_on = os.environ.get("OGC_SEPMOVE", "0") == "1"
        self._sep_acc = 0
        self._sep_try = 0
        self._sep_commit = 0
        self._sep_dbg = {}
        self._exit_med = float(np.median(p.R + p.P))
        self._size_med = float(np.median(p.ncells0[p.orient_base[:-1]]))
        self._chain_acc = 0
        self._chain_try = 0
        self._rs_acc = 0
        self._rs_try = 0
        self._rl_acc = 0
        self._rl_best = 0
        self._rl_try = 0
        self._rs_dbg = {}
        self.op_score = np.zeros(len(self.ops))
        self.op_cnt = np.zeros(len(self.ops))
        self._deep_used = 0
        # read per-run, not at import: the driver sets OGC_DEEP_P by group
        # AFTER this module is (load-once) cached, so a module-level read
        # would pin the first dispatch's value for the whole process
        try:
            self._deep_p = float(os.environ.get("OGC_DEEP_P", _DEEP_P))
        except (TypeError, ValueError):
            self._deep_p = _DEEP_P
        self.k_max = 12  # adaptive removal-size cap (walked by the dt controller)
        # per-ruin-size weights (KADAPT), indexed by k DIRECTLY and sized to the
        # controller's ceiling, not to k_max's initial value -- see _K_CEIL.
        # Flat prior: [3, _K_CEIL] IS the tuned range, so the island starts at
        # exactly the distribution the engine shipped with and the learner only
        # ever moves it on measured reward.
        self.k_w = np.ones(_K_CEIL + 1)
        self.k_score = np.zeros(_K_CEIL + 1)
        self.k_cnt = np.zeros(_K_CEIL + 1)
        self._k_cdf = {}     # (lo, hi) -> cumulative weights; cleared on reweight
        self._last_k = None
        self._k_n = 0
        self._k_drawn = np.zeros(_K_CEIL + 1)  # lifetime draws, for the log line
        # RORD's tiebreak keys are STATIC (they read only Prep), so build them
        # once instead of re-deriving them from numpy inside every sort key.
        # Measured 19.6us -> 2.4us per repair sort on prob_12, orders identical.
        _fits = [int(v) for v in p.block_fits.sum(1)]
        _slack = [int(a) - int(b) - int(c) for a, b, c in zip(p.D, p.R, p.P)]
        _size = [-int(v) for v in p.ncells0[p.orient_base[:-1]]]
        self._rord_key = list(zip(_fits, _slack, _size))
        # abort accounting: ~3 in 4 repairs abort, and every insertion made
        # before the abort was wasted work. Tracking WHERE in the order the
        # abort lands is what says whether a fail-fast order can pay.
        self._lc_try = 0        # last-chance rescans attempted
        # set for real by run(); defined here so a move can never reach an
        # unset attribute however the class is driven
        self._lam = None
        self._lam_t0 = 0.0
        self._lc_save = 0       # ... that turned an abort into a placement
        self._abort_pos = 0     # sum of abort indices
        self._abort_len = 0     # sum of order lengths at abort
        self._ins_wasted = 0    # insertions thrown away by aborts

    # ---------------------------------------------------------- removal ops --
    def _ksize(self, lo=_K_LO):
        hi = min(_K_CEIL, max(lo + 1, self.k_max))
        if _KADAPT:
            # inverse-CDF draw off a cached cumulative vector. rng.choice(p=)
            # costs 11.4us per call regardless of length and is called at least
            # once per iteration; searchsorted on a cached cumsum is 1.9us.
            key = (lo, hi)
            cdf = self._k_cdf.get(key)
            if cdf is None:
                cdf = np.cumsum(self.k_w[lo:hi + 1])
                self._k_cdf[key] = cdf
            k = lo + int(np.searchsorted(cdf, self.rng.random() * cdf[-1]))
            if k > hi:  # float-rounding guard on the last cell
                k = hi
            self._last_k = k
            return k
        return int(self.rng.integers(lo, hi + 1))

    def _k_credit(self, accepted, gain=0.0, dt=0.0, best=False):
        """Bandit credit for the ruin size this iteration drew.

        The arm is the size that was SAMPLED (not the number of blocks the
        operator ended up removing) -- that is the decision the distribution
        actually made, so it is the one the reward has to be attributed to.

        Reward is objective gain PER SECOND: a ruin of 30 costs several times
        what a ruin of 3 costs, and a scheme that credits bare acceptances pays
        large sizes for the extra work they were handed rather than for what
        they returned. Non-improving acceptances and new incumbents get fixed
        bonuses in w1 (tardy-day) units so every term shares one scale.
        """
        k = self._last_k
        if k is None:
            return
        self._last_k = None
        w1 = self.state.prep.w1
        r = gain
        if accepted:
            r += _K_ACC * w1
        if best:
            r += _K_BEST * w1
        self.k_score[k] += r / max(dt, 1e-4)
        self.k_cnt[k] += 1.0
        self._k_drawn[k] += 1.0
        self._k_n += 1
        if self._k_n % _K_WIN == 0:
            self._k_reweight()

    def _k_reweight(self):
        """Re-derive the ruin-size weights from the mean reward each size
        earned in the last window, mirroring the operator-portfolio update at
        the bottom of run().

        Sizes pool with their neighbours before the mean is taken: reward is a
        smooth, roughly unimodal function of the sub-problem scale (PCT Fig. 9),
        so a size that drew 15 samples this window can borrow from the sizes
        either side of it instead of reporting noise. Sums and counts are
        smoothed separately so the result is a weighted mean over the three
        sizes, not an unweighted mean of three per-size means.
        """
        s = self.k_score[_K_LO:]
        c = self.k_cnt[_K_LO:]
        ps = s.copy()
        pc = c.copy()
        ps[1:-1] = 0.25 * s[:-2] + 0.5 * s[1:-1] + 0.25 * s[2:]
        pc[1:-1] = 0.25 * c[:-2] + 0.5 * c[1:-1] + 0.25 * c[2:]
        ps[0] = 0.75 * s[0] + 0.25 * s[1]
        pc[0] = 0.75 * c[0] + 0.25 * c[1]
        ps[-1] = 0.75 * s[-1] + 0.25 * s[-2]
        pc[-1] = 0.75 * c[-1] + 0.25 * c[-2]
        sc = ps / np.maximum(pc, 1e-9)
        top = float(sc.max())
        if top > 0.0:
            self.k_w[_K_LO:] = (0.7 * self.k_w[_K_LO:]
                                + 0.3 * (_K_FLOOR + sc / top))
        self.k_score[:] = 0.0
        self.k_cnt[:] = 0.0
        self._k_cdf.clear()

    def _k_mean(self):
        """Mean ruin size actually drawn over the run -- the one number that
        says whether the distribution learned anything or stayed uniform."""
        tot = float(self._k_drawn.sum())
        if tot <= 0.0:
            return 0.0
        return float((self._k_drawn * np.arange(len(self._k_drawn))).sum() / tot)

    def _rm_random(self):
        k = self._ksize()
        ids = np.nonzero(self.state.placed)[0]
        if len(ids) == 0:
            return []
        k = min(k, len(ids))
        return list(self.rng.choice(ids, size=k, replace=False))

    def _rm_tardy(self):
        st = self.state
        p = st.prep
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = np.nonzero(tard > 0)[0]
        if len(cand) == 0:
            return self._rm_random()
        i = int(self.rng.choice(cand))
        # remove i plus blocks in its bay overlapping its ideal window
        b = int(st.bay[i])
        lo = int(p.R[i])
        hi = int(st.t2[i])
        others = [j for j in st.bay_blocks[b]
                  if j != i and st.t1[j] < hi and st.t2[j] > lo]
        self.rng.shuffle(others)
        return [i] + others[:self._ksize() - 1]

    def _rm_window(self):
        st = self.state
        p = st.prep
        b = int(self.rng.integers(0, p.m))
        if not st.bay_blocks[b]:
            return self._rm_random()
        j = st.bay_blocks[b][int(self.rng.integers(0, len(st.bay_blocks[b])))]
        t = int(st.t1[j])
        span = int(2 * p.P.mean())
        sel = [i for i in st.bay_blocks[b]
               if st.t1[i] < t + span and st.t2[i] > t]
        self.rng.shuffle(sel)
        return sel[:self._ksize(4)]

    def _rm_pref(self):
        st = self.state
        p = st.prep
        pen = (p.smax - p.pref[np.arange(self.n), np.maximum(st.bay, 0)]) * st.placed
        cand = np.nonzero(pen > 0)[0]
        if len(cand) == 0:
            return self._rm_random()
        k = min(int(self.rng.integers(2, 8)), len(cand))
        sel = list(self.rng.choice(cand, size=k, replace=False))
        # plus a few random for freedom. This operator sizes itself, so the
        # ruin size _rm_random draws does not describe the move -- drop it
        # rather than hand the KADAPT bandit an action it did not take.
        keep_k = self._last_k
        extra = self._rm_random()[:3]
        self._last_k = keep_k
        return list(set(sel + extra))

    def _rm_single(self):
        # one block, sampled with probability proportional to its cost
        st = self.state
        p = st.prep
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        prefpen = ((p.smax - p.pref[np.arange(self.n),
                                    np.maximum(st.bay, 0)]) * st.placed)
        w = p.w1 * tard + p.w3 * prefpen + 1e-9
        w = w * st.placed
        tot = w.sum()
        if tot <= 0:
            return self._rm_random()[:1]
        i = int(self.rng.choice(self.n, p=w / tot))
        return [i]

    def _rm_bay_load(self):
        # rebalance: remove from the most (weighted-)loaded bay
        st = self.state
        p = st.prep
        w = p.u * st.loads
        b = int(np.argmax(w))
        if not st.bay_blocks[b]:
            return self._rm_random()
        blocks = list(st.bay_blocks[b])
        self.rng.shuffle(blocks)
        return blocks[:self._ksize()]

    def _rm_string(self):
        """SISR-style ruin (Christiaens & Vanden Berghe 2020): remove 1-3
        temporally CONTIGUOUS strings of blocks from bays around a cost-biased
        seed, occasionally preserving a middle substring ("split string").
        Contiguous removal frees a coherent space-time hole the recreate can
        genuinely repack, unlike scattered removal. Ported verbatim from the
        engine_v47 fork (hidden P5/P4 record holder)."""
        st = self.state
        p = st.prep
        rng = self.rng
        ids = np.nonzero(st.placed)[0]
        if len(ids) == 0:
            return []
        # seed biased toward tardy blocks (the cost carriers on overload)
        tard = (np.maximum(0, st.t2 - p.D) * st.placed)[ids].astype(np.float64)
        if tard.sum() > 0 and rng.random() < 0.6:
            seed = int(rng.choice(ids, p=(tard + 1e-9) / (tard + 1e-9).sum()))
        else:
            seed = int(rng.choice(ids))
        t_seed = int(st.t1[seed])
        k_target = self._ksize(4)
        n_strings = 1 + int(rng.integers(0, 3))
        per = max(2, k_target // n_strings)
        removed = []
        seen_bays = set()
        b = int(st.bay[seed])
        for s in range(n_strings):
            if s > 0:
                # neighbor string: another bay active in the seed's window
                cands = [int(j) for j in ids
                         if int(st.bay[j]) not in seen_bays
                         and st.t1[j] < t_seed + p.P[seed]
                         and st.t2[j] > t_seed]
                if not cands:
                    break
                b = int(st.bay[int(rng.choice(cands))])
            seen_bays.add(b)
            run = sorted(st.bay_blocks[b],
                         key=lambda j: (int(st.t1[j]), int(st.seq[j])))
            if not run:
                continue
            pos = min(range(len(run)),
                      key=lambda q: abs(int(st.t1[run[q]]) - t_seed))
            L = min(len(run), per)
            start = max(0, min(pos - int(rng.integers(0, L)), len(run) - L))
            string = run[start:start + L]
            if len(string) > 3 and rng.random() < 0.35:
                # split string: keep a middle chunk in place
                mlen = 1 + int(rng.integers(0, len(string) - 3))
                a = 1 + int(rng.integers(0, len(string) - mlen - 1))
                string = string[:a] + string[a + mlen:]
            removed.extend(int(j) for j in string)
        return list(dict.fromkeys(removed))

    def _rm_crit(self):
        """Failure-directed ruin: eject whoever is holding the space the
        blocks that keep failing to reinsert are trying to reach.

        The seed is drawn proportional to the criticality index, so it is the
        block the search has actually been unable to place recently -- not a
        block that merely looks hard structurally. Around it we remove the
        incumbents whose stay overlaps the seed's own feasible window in the
        bays it can use, which is the set that must move for the seed to get
        an on-time placement. Without any recorded failure yet there is
        nothing to direct us, so fall back to the tuned default operator.
        """
        st = self.state
        p = st.prep
        w = self.crit * (self.crit > 0)
        tot = float(w.sum())
        if tot <= 0.0:
            return self._rm_tardy()
        i = int(self.rng.choice(self.n, p=w / tot))
        sel = [i] if st.placed[i] else []
        # the window the seed needs to land in to be on time, widened by its
        # own duration so we also clear what sits just before it
        lo = int(p.R[i])
        hi = max(lo + int(p.P[i]), int(p.D[i]))
        bays = [b for b in range(p.m) if p.block_fits[i, b]]
        if st.guide is not None and p.block_fits[i, int(st.guide[i])]:
            bays = [int(st.guide[i])] + [b for b in bays
                                         if b != int(st.guide[i])]
        room = []
        for b in bays[:2]:
            occ = [j for j in st.bay_blocks[b]
                   if j != i and st.t1[j] < hi and st.t2[j] > lo]
            if occ:
                self.rng.shuffle(occ)
                room.extend(occ[:max(2, self._ksize() // 2)])
        return list(dict.fromkeys(int(x) for x in sel + room))

    def _crit_hit(self, i):
        """Record that a repair aborted on block i (our 'infeasible
        subproblem'). Recorded unconditionally -- it is one array increment
        and no rng draw, so it changes nothing observable while the consumers
        (_rm_crit, the fail-fast repair order) are switched off. Decay runs on
        a schedule rather than every hit to keep the array write off the hot
        path."""
        self.crit[i] += 1.0
        self._crit_n += 1
        if self._crit_n % 256 == 0:
            self.crit *= _CRIT_DECAY ** 256

    def _rm_offguide(self):
        # blocks sitting outside their relaxation-optimal bay, plus a few
        # blocks occupying their landing window in the target bay (make room)
        st = self.state
        p = st.prep
        if st.guide is None:
            return self._rm_random()
        off = [i for i in range(self.n)
               if st.placed[i] and int(st.bay[i]) != st.guide[i]]
        if not off:
            return self._rm_random()
        self.rng.shuffle(off)
        sel = off[:max(1, self._ksize() // 2)]
        room = []
        for i in sel[:3]:
            tb = st.guide[i]
            lo = int(p.R[i])
            hi = lo + int(p.P[i])
            occ = [j for j in st.bay_blocks[tb]
                   if st.t1[j] < hi and st.t2[j] > lo and j not in sel]
            if occ:
                self.rng.shuffle(occ)
                room.extend(occ[:2])
        # room lists from different blocks may share occupants: dedupe
        return list(dict.fromkeys(int(i) for i in sel + room))

    def _rm_offday(self):
        # day-level twin of _rm_offguide: blocks placed far from their
        # relaxation-optimal entry day (tardy ones first), plus the occupants
        # of their target window in the target bay (make room). Only ever
        # registered when day targets exist.
        st = self.state
        p = st.prep
        if st.tday is None:
            return self._rm_random()
        dev = []
        for i in range(self.n):
            if not st.placed[i] or st.tday[i] < 0:
                continue
            d = abs(int(st.t1[i]) - int(st.tday[i]))
            if d > 0:
                dev.append((0 if st.t2[i] > p.D[i] else 1, -d, i))
        if not dev:
            return self._rm_random()
        dev.sort()
        sel = [i for (_tardy, _nd, i) in dev[:max(1, self._ksize() // 2)]]
        room = []
        for i in sel[:3]:
            tb = int(st.tbay[i])
            lo = int(st.tday[i])
            hi = lo + int(p.P[i])
            occ = [j for j in st.bay_blocks[tb]
                   if st.t1[j] < hi and st.t2[j] > lo and j not in sel]
            if occ:
                self.rng.shuffle(occ)
                room.extend(occ[:2])
        return list(dict.fromkeys(int(i) for i in sel + room))

    def _resolve_days(self):
        """One-shot day-target refresh from the live incumbent: membership
        from the CURRENT bays (not the guide assignment), per-bay capacity
        calibrated from the realized layer-0 occupancy (p90, clipped to
        [0.55, 0.90] of bay area), incumbent entry days as stability-anchored
        hints. Blocks the search ~2.5 s once; long-TL main island only."""
        st = self.state
        p = st.prep
        try:
            from _ogc.ogc_dayguide import recompute_day_targets
            bay_tasks = {}
            caps = {}
            for b in range(p.m):
                ids = list(st.bay_blocks[b])
                if not ids:
                    continue
                area = float(p.bayW[b] * p.bayH[b])
                occ = st.occ0[b]
                nz = occ[occ > 0]
                if len(nz) == 0:
                    continue
                eta_b = min(0.90, max(0.55,
                                      float(np.percentile(nz, 90)) / area))
                cap = max(1, int(eta_b * area))
                rows = []
                for i in ids:
                    a = max(1, min(int(p.min_cells0[i, b]), cap))
                    rows.append((int(i), int(p.R[i]), int(p.D[i]),
                                 int(p.P[i]), a, int(st.t1[i])))
                bay_tasks[int(b)] = rows
                caps[int(b)] = cap
            out = recompute_day_targets(p.n, bay_tasks, caps,
                                        budget_s=min(2.5, 0.5 * p.m))
            if out is None:
                return
            tday, tbay = out
            st.tday = np.asarray(tday, np.int64)
            st.tbay = np.asarray(tbay, np.int64)
            # roles must come from the SAME relaxation as the windows
            st.tlate = (st.tday >= 0) & (st.tday + p.P > p.D)
            # fresh targets: let the valve re-learn from full pressure
            self._day_p = self._day_p0
            self._day_reg = 0
            self._day_prev = None
            if self.log:
                self.log(f"day re-solve: {sum(1 for t in tday if t >= 0)} "
                         f"targets, caps={sorted(caps.values())}")
        except Exception as exc:
            # Mid-run guide refresh is optional, but genuine errors must stay
            # observable even when the normal verbose log is disabled.
            from _ogc.driver import _optional_failure
            _optional_failure("day re-solve", exc, self.log)

    def _lam_move_obs(self, accepted, delta):
        """A special move's Metropolis decision (OGC_LAM_OBS_MOVES). Called at
        the move's own accept line, so an abort -- which returns before ever
        pricing a complete neighbour -- cannot reach it."""
        lam = self._lam
        if lam is not None and not lam.tuning and _LAM_OBS_MOVES:
            now = time.time()
            lam.observe(accepted, delta, now,
                        (now - self._lam_t0) / lam.total)

    def try_roleswap(self, T):
        """Sacrifice-role swap. On mid/heavy-congestion instances the real
        decision is WHICH blocks end late; the day relaxation computed an
        optimal sacrifice set (tlate). When the incumbent disagrees -- a
        block the relaxation wants on time is currently tardy -- evict one
        on-time relaxation-sacrificed block from its target window and
        reinsert with the roles exchanged. SA on the true objective decides;
        full revert otherwise."""
        st = self.state
        p = st.prep
        rng = self.rng
        dbg = self._rs_dbg
        if st.tday is None or st.tlate is None:
            return False
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = [i for i in np.nonzero(tard > 0)[0]
                if st.tday[i] >= 0 and not st.tlate[i]]
        if not cand:
            dbg["nocand"] = dbg.get("nocand", 0) + 1
            return False
        d = int(cand[int(rng.integers(0, len(cand)))])
        b = int(st.tbay[d])
        P_d = int(p.P[d])
        win_lo = max(int(p.R[d]), int(st.tday[d]))
        win_hi = win_lo + P_d
        # ALL sacrifice-role occupants of d's target window: evicting one is
        # measured to be insufficient (97% of single-evict swaps left d late)
        js = [j for j in st.bay_blocks[b]
              if j != d and st.tlate[j]
              and st.t1[j] < win_hi and st.t2[j] > win_lo]
        if not js:
            dbg["nojs"] = dbg.get("nojs", 0) + 1
            return False
        if len(js) > 6:
            js.sort(key=lambda x: -int(p.D[x]))
            js = js[:6]

        moved = [d] + js
        old = [(i, int(st.bay[i]), int(st.t1[i]), int(st.oi[i]),
                int(st.px[i]), int(st.py[i]), int(st.seq[i]),
                int(st.t2[i]))
               for i in moved]
        cur_obj = st.objective()
        for i in moved:
            st.remove(i)

        def revert():
            for i in moved:
                if st.placed[i]:
                    st.remove(i)
            for (i, bb, tt1, oo, xx, yy, sq, tt2) in old:
                st.insert(i, bb, tt1, oo, xx, yy, seq=sq, t2=tt2)

        # d claims its relaxation slot (scan lifted to tday via use_dayguide;
        # only_bay == tbay[d] so the lift applies) and must actually land ON
        # TIME -- otherwise the window is geometrically unreachable even
        # with its sacrifices evicted, and we bail
        cand_d = st.best_insertion(d, only_bay=b, use_dayguide=True)
        if cand_d is None:
            dbg["noins_d"] = dbg.get("noins_d", 0) + 1
            revert()
            return False
        _c, _b, t_d, o_d, px_d, py_d = cand_d
        # require strict tardiness PROGRESS for d (full on-time is measured
        # to be geometrically unreachable in ~90% of windows even with the
        # sacrifices evicted); SA prices the rest of the move truthfully
        if t_d + P_d >= int(old[0][7]):
            dbg["d_noprog"] = dbg.get("d_noprog", 0) + 1
            revert()
            return False
        st.insert(d, b, t_d, o_d, px_d, py_d)

        # rehome the sacrifices wherever is cheapest (EDD order, tardiness
        # allowed -- they are the relaxation's designated late set)
        for j in sorted(js, key=lambda x: (p.D[x], p.R[x])):
            cand_j = st.best_insertion(j)
            if cand_j is None:
                dbg["noins_j"] = dbg.get("noins_j", 0) + 1
                revert()
                return False
            _c, b_j, t_j, o_j, px_j, py_j = cand_j
            st.insert(j, b_j, t_j, o_j, px_j, py_j)

        delta = st.objective() - cur_obj
        _ok_sa = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        self._lam_move_obs(_ok_sa, delta)
        if _ok_sa:
            if delta < -1e-9:
                self._rs_acc += 1
            else:
                dbg["sa_acc"] = dbg.get("sa_acc", 0) + 1
            return True
        dbg["sa_rej"] = dbg.get("sa_rej", 0) + 1
        revert()
        return False

    def _rm_rect(self):
        # coherent spatio-temporal hole: blocks intersecting a random
        # x-interval of a random bay during a random time window
        st = self.state
        p = st.prep
        b = int(self.rng.integers(0, p.m))
        if not st.bay_blocks[b]:
            return self._rm_random()
        j = st.bay_blocks[b][int(self.rng.integers(0, len(st.bay_blocks[b])))]
        t = int(st.t1[j])
        span = int(self.rng.integers(1, max(2, 2 * int(p.P.mean()))))
        W = int(p.bayW[b])
        rw = int(self.rng.integers(max(1, W // 4), max(2, W // 2 + 1)))
        rx = int(self.rng.integers(0, max(1, W - rw + 1)))
        sel = []
        for i in st.bay_blocks[b]:
            if st.t1[i] >= t + span or st.t2[i] <= t:
                continue
            oidx = p.orient_base[i] + st.oi[i]
            x0 = p.fbx0[oidx] + st.px[i]
            x1 = p.fbx1[oidx] + st.px[i]
            if x1 > rx and x0 < rx + rw:
                sel.append(i)
        if not sel:
            return self._rm_random()
        self.rng.shuffle(sel)
        return sel[:self._ksize(4)]

    # ------------------------------------------------ balance move -----------
    def try_balance(self, T):
        """Targeted obj2 reduction: move one block from the most weighted-
        loaded bay into the least loaded one; accept by SA on the total."""
        st = self.state
        p = st.prep
        rng = self.rng
        if p.m < 2:
            return False
        w = p.u * st.loads
        bmax = int(np.argmax(w))
        bmin = int(np.argmin(w))
        if bmax == bmin or not st.bay_blocks[bmax]:
            return False
        # the workload that would even out the pair
        target = (w[bmax] - w[bmin]) / (p.u[bmax] + p.u[bmin])
        cand = [j for j in st.bay_blocks[bmax] if p.block_fits[j, bmin]]
        if not cand:
            return False
        # prefer blocks whose workload best evens out the pair
        cand.sort(key=lambda j: abs(p.wl[j] - target))
        pick = cand[:5]
        i = pick[int(rng.integers(0, len(pick)))]
        cur_obj = st.objective()
        orig = (i, int(st.bay[i]), int(st.t1[i]), int(st.oi[i]),
                int(st.px[i]), int(st.py[i]), int(st.seq[i]),
                int(st.t2[i]))
        st.remove(i)
        cand2 = st.best_insertion(i, only_bay=bmin)
        if cand2 is None:
            st.insert(i, orig[1], orig[2], orig[3], orig[4], orig[5],
                      seq=orig[6], t2=orig[7])
            return False
        c, b, t, o, px, py = cand2
        st.insert(i, b, t, o, px, py)
        delta = st.objective() - cur_obj
        _ok_sa = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        self._lam_move_obs(_ok_sa, delta)
        if _ok_sa:
            return True
        st.remove(i)
        st.insert(i, orig[1], orig[2], orig[3], orig[4], orig[5],
                  seq=orig[6], t2=orig[7])
        return False

    # ----------------------------------------------------- chain move --------
    def try_chain(self, T, T_rep=None):
        """Ejection chain: force a tardy block into a chosen bay/window by
        removing the occupants, then rehome the ejected blocks anywhere
        (cross-bay displacement); accept by SA on the total objective.

        T is the Metropolis temperature; T_rep is the one that prices the
        rehoming budget. They are the same number unless the Lam controller is
        driving acceptance, in which case the budget must stay on the shipped
        monotone curve -- see the OGC_LAM block."""
        st = self.state
        p = st.prep
        rng = self.rng
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = np.nonzero(tard > 0)[0]
        if len(cand) == 0:
            return False
        d = int(rng.choice(cand))
        P_d = int(p.P[d])

        # target bay: the relaxation-optimal bay, the most preferred, or the
        # current one (each a different escape direction)
        roll = rng.random()
        if st.guide is not None and roll < 0.4:
            b = int(st.guide[d])
        elif roll < 0.7:
            b = int(np.argmax(p.pref[d]))
        else:
            b = int(st.bay[d])
        if not p.block_fits[d, b]:
            return False
        # target window: the day-relaxation target, as early as possible, or
        # partway back (one rng draw either way -- the default stream is
        # unchanged when no day targets exist)
        R_d = int(p.R[d])
        roll_t = rng.random()
        if (st.tday is not None and st.tday[d] >= 0
                and b == int(st.tbay[d]) and roll_t < 0.3
                and int(st.tday[d]) < int(st.t1[d])):
            # only when the target is EARLIER than the current entry --
            # sacrifice targets at/after t1 would trip the no-gain guard
            # (same-bay) or evict a window d never lands in (cross-bay),
            # burning the chain move's adaptive share on dead rolls
            tt = max(R_d, int(st.tday[d]))
        else:
            tt = (R_d if roll_t < 0.6
                  else max(R_d, (R_d + int(st.t1[d])) // 2))
        t2t = tt + P_d
        if b == int(st.bay[d]) and t2t >= int(st.t2[d]):
            return False  # no gain possible in place

        # eject the occupants of the target window; blocks with the latest
        # due dates have the most rescheduling freedom, so evict those first
        occ = [j for j in st.bay_blocks[b]
               if j != d and st.t1[j] < t2t and st.t2[j] > tt]
        if len(occ) > 6:
            occ.sort(key=lambda j: -int(p.D[j]))
            occ = occ[:6]

        moved = [d] + occ
        old = [(i, int(st.bay[i]), int(st.t1[i]), int(st.oi[i]),
                int(st.px[i]), int(st.py[i]), int(st.seq[i]),
                int(st.t2[i]))
               for i in moved]
        d_old_cost = (p.w1 * max(0, old[0][7] - int(p.D[d]))
                      + p.w3 * (p.smax[d] - p.pref[d, old[0][1]]))
        ej_old_cost = sum(p.w1 * max(0, t2o - p.D[i])
                          + p.w3 * (p.smax[i] - p.pref[i, bb])
                          for (i, bb, _t1, _o, _x, _y, _s, t2o) in old[1:])
        cur_obj = st.objective()
        for i in moved:
            st.remove(i)

        def revert():
            for i in moved:
                if st.placed[i]:
                    st.remove(i)
            for (i, bb, tt1, oo, xx, yy, sq, tt2) in old:
                st.insert(i, bb, tt1, oo, xx, yy, seq=sq, t2=tt2)

        # d goes first, forced into the target bay, and must strictly improve.
        # Price d's OLD slot from the same post-ejection baseline
        # best_insertion prices candidates from, so the signed w2 term is
        # credited symmetrically (else transient imbalance credit lets no-op
        # chains pass, and genuine max-bay escapes get vetoed).
        nl = st.loads.copy()
        nl[old[0][1]] += p.wl[d]
        d_old_sym = d_old_cost + p.w2 * (st.obj2(nl) - st.obj2())
        cand_d = st.best_insertion(d, only_bay=b,
                                   best_cost_init=d_old_sym)
        if cand_d is None:
            revert()
            return False
        c_d, _b2, t_d, o_d, px_d, py_d = cand_d
        st.insert(d, b, t_d, o_d, px_d, py_d)

        # rehome the ejected blocks anywhere, under a pooled budget: their
        # old cost plus what d's improvement bought, plus SA slack
        budget = (ej_old_cost + max(0.0, d_old_sym - c_d)
                  + self.cap_t * (T if T_rep is None else T_rep))
        spent = 0.0
        queue = sorted(occ, key=lambda i: (p.D[i], p.R[i], -p.wl[i]))
        ok = True
        d2_left = 1 if _CHAIN2 else 0
        qi = 0
        while qi < len(queue):
            i = queue[qi]
            qi += 1
            cnd = st.best_insertion(i, best_cost_init=budget - spent,
                                    use_guide=st.guide is not None)
            if cnd is None and d2_left > 0 and len(moved) < 12:
                # depth-2: the stuck block may evict occupants of its
                # preferred bay's ideal window (their old cost joins the
                # pooled budget; they join the rehoming queue)
                d2_left -= 1
                b2 = int(np.argmax(p.pref[i]))
                if p.block_fits[i, b2]:
                    lo2 = int(p.R[i])
                    hi2 = lo2 + int(p.P[i])
                    occ2 = [j for j in st.bay_blocks[b2]
                            if j not in moved and st.t1[j] < hi2
                            and st.t2[j] > lo2]
                    occ2.sort(key=lambda j: -int(p.D[j]))
                    occ2 = occ2[:3]
                    for j in occ2:
                        old.append((j, int(st.bay[j]), int(st.t1[j]),
                                    int(st.oi[j]), int(st.px[j]),
                                    int(st.py[j]), int(st.seq[j]),
                                    int(st.t2[j])))
                        budget += (p.w1 * max(0, int(st.t2[j])
                                              - int(p.D[j]))
                                   + p.w3 * (p.smax[j]
                                             - p.pref[j, int(st.bay[j])]))
                        moved.append(j)
                        st.remove(j)
                        queue.append(j)
                    if occ2:
                        cnd = st.best_insertion(
                            i, only_bay=b2, best_cost_init=budget - spent)
            if cnd is None:
                ok = False
                break
            c, bb, t, o, px, py = cnd
            spent += max(0.0, c)
            st.insert(i, bb, t, o, px, py)
        if not ok:
            revert()
            return False

        new_obj = st.objective()
        delta = new_obj - cur_obj
        _ok_sa = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        self._lam_move_obs(_ok_sa, delta)
        if _ok_sa:
            # only genuine improvements feed the adaptive share -- no-op
            # acceptances must not inflate the move's probability
            if delta < -1e-9:
                self._chain_acc += 1
            return True
        revert()
        return False

    # ------------------------------------------------------ nest move --------
    def try_nest(self, T):
        """Exit-delay nesting: pick a tardy block D; extend the exits of
        blocks already sitting in a bay so that D can enter earlier with its
        upper layers overhanging their low profiles; accept by SA."""
        st = self.state
        p = st.prep
        rng = self.rng
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = np.nonzero(tard > 0)[0]
        if len(cand) == 0:
            return False
        d = int(rng.choice(cand))
        P_d = int(p.P[d])
        # target bay: current bay usually; sometimes the most-preferred one
        b = int(st.bay[d]) if rng.random() < 0.6 else int(np.argmax(p.pref[d]))
        if not p.block_fits[d, b]:
            return False
        # target window: usually as early as possible, sometimes partway back
        # from the current (late) entry, or the latest on-time day
        R_d = int(p.R[d])
        roll = rng.random()
        if roll < 0.6:
            t1 = R_d
        elif roll < 0.85:
            t1 = max(R_d, (R_d + int(st.t1[d])) // 2)
        else:
            t1 = max(R_d, min(int(st.t1[d]) - 1, int(p.D[d]) - P_d))
        t2 = t1 + P_d
        if t2 >= int(st.t2[d]):
            return False  # no gain possible

        # extension candidates: blocks that entered at or before t1 and exit
        # strictly inside the window -- extending them to t2 turns them into
        # pure "boundary" blocks whose low layers D may overhang
        ext = [j for j in st.bay_blocks[b]
               if j != d and st.t1[j] <= t1 and t1 < st.t2[j] < t2]
        if not ext:
            return False
        # cheapest extensions first (least added tardiness)
        ext.sort(key=lambda j: (max(0, t2 - int(p.D[j]))
                                - max(0, int(st.t2[j]) - int(p.D[j]))))
        ext = ext[:int(rng.integers(1, 5))]

        cur_obj = st.objective()
        touched = []  # original tuples of every block we modified

        def revert():
            for (i, bb, tt1, oo, xx, yy, sq, tt2) in reversed(touched):
                if st.placed[i]:
                    st.remove(i)
                st.insert(i, bb, tt1, oo, xx, yy, seq=sq, t2=tt2)

        n_ext = 0
        for j in ext:
            orig = (j, int(st.bay[j]), int(st.t1[j]), int(st.oi[j]),
                    int(st.px[j]), int(st.py[j]), int(st.seq[j]),
                    int(st.t2[j]))
            st.remove(j)
            if st.can_place_exact(j, orig[1], orig[4], orig[5], orig[3],
                                  orig[2], t2):
                st.insert(j, orig[1], orig[2], orig[3], orig[4], orig[5],
                          t2=t2)
                touched.append(orig)
                n_ext += 1
            else:
                st.insert(j, orig[1], orig[2], orig[3], orig[4], orig[5],
                          seq=orig[6], t2=orig[7])
        if n_ext == 0:
            return False

        # re-place D on the extended landscape; a nest only pays if D's own
        # cost strictly improves, so cap the search there (fast rejection)
        d_orig = (d, int(st.bay[d]), int(st.t1[d]), int(st.oi[d]),
                  int(st.px[d]), int(st.py[d]), int(st.seq[d]),
                  int(st.t2[d]))
        d_old_cost = (p.w1 * max(0, d_orig[7] - int(p.D[d]))
                      + p.w3 * (p.smax[d] - p.pref[d, d_orig[1]]))
        st.remove(d)
        cand2 = st.best_insertion(d, best_cost_init=d_old_cost)
        if cand2 is None:
            st.insert(d, d_orig[1], d_orig[2], d_orig[3], d_orig[4],
                      d_orig[5], seq=d_orig[6], t2=d_orig[7])
            revert()
            return False
        touched.append(d_orig)
        c, bb, tt, oo, px, py = cand2
        st.insert(d, bb, tt, oo, px, py)

        new_obj = st.objective()
        delta = new_obj - cur_obj
        _ok_sa = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        self._lam_move_obs(_ok_sa, delta)
        if _ok_sa:
            self._nest_acc += 1
            return True
        revert()
        return False

    # ---------------- shared separation internals (single-seed + batch) ------
    # try_separate is the shipped, A/B-validated template; batch_separate
    # generalizes it to a SET of forced seeds. Everything both need lives
    # here so the two cannot drift: the wholesale revert, the conservative
    # layer-0 cell view, the min-overlap force position, and the bounded
    # victim resolution (exact relocate / 1..3-day retime inside a shared
    # net-positive w1-day budget). Pure refactor: no rng draws, identical
    # arithmetic in identical order, so try_separate's behavior (and its
    # A/B evidence) is unchanged.

    def _sep_revert(self, touched):
        """Wholesale revert: walk the touched originals in reverse commit
        order, remove whatever is placed and restore the pre-call tuple
        verbatim (seq included, so same-day entry order survives). A tuple
        with bay < 0 marks a block that was UNPLACED pre-call: it is
        removed, not re-inserted."""
        st = self.state
        for (i, bb, tt1, oo, xx, yy, sq, tt2) in reversed(touched):
            if st.placed[i]:
                st.remove(i)
            if bb >= 0:
                st.insert(i, bb, tt1, oo, xx, yy, seq=sq, t2=tt2)

    def _l0_cells(self, j, W, H):
        """In-bay layer-0 raster cells of placed block j. Outer raster
        cells over-cover the true shape, so any test built on these is
        conservative: zero cell overlap implies zero true overlap."""
        st = self.state
        p = st.prep
        oj = int(p.orient_base[j] + st.oi[j])
        rj = int(p.layer_base[oj])
        cj = int(p.cell_ptr[rj])
        nj = int(p.cell_cnt[rj])
        uu = p.arena_u[cj:cj + nj] + int(st.px[j])
        vv = p.arena_v[cj:cj + nj] + int(st.py[j])
        m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        return uu[m], vv[m]

    def _min_ov_pos(self, b, oidx, t_new, t2_new, W, H, near=None,
                    radius=None):
        """Min layer-0-overlap force position over the target window:
        cell-count occupancy of the co-present committed blocks correlated
        with the block's own layer-0 cells (numpy slice accumulation; valid
        anchors keep every slice in-bay). Cell level only -- the exact
        commit test at the end of the move is the arbiter.
        `near=None` keeps the historical first-index argmin (try_separate
        bit-identical); near=(px0, py0) breaks ties among MINIMAL-overlap
        anchors toward that point -- a batch of seeds must not all pile
        onto the same first-index minimum (measured: 196/200 batch aborts
        via v_scan_late without this), and it keeps a pull=0 seed on its
        own old spot so a band can reconstruct in place. `radius` (with
        near) restricts the anchor search to a box around near: least-
        motion relocation at a fraction of the slice-accumulation cost
        (the batch loop timed out 394/400 on global re-forces).
        Returns (sx, sy, cu, cv) or None when the orientation cannot
        anchor in the bay."""
        st = self.state
        p = st.prep
        if not p.fits[oidx, b]:
            return None
        pxlo = int(p.pxlo[oidx, b])
        pxhi = int(p.pxhi[oidx, b])
        pylo = int(p.pylo[oidx, b])
        pyhi = int(p.pyhi[oidx, b])
        if near is not None and radius is not None:
            pxlo = max(pxlo, int(near[0]) - radius)
            pxhi = min(pxhi, int(near[0]) + radius)
            pylo = max(pylo, int(near[1]) - radius)
            pyhi = min(pyhi, int(near[1]) + radius)
        if pxlo > pxhi or pylo > pyhi:
            return None
        r0 = int(p.layer_base[oidx])
        c0 = int(p.cell_ptr[r0])
        cn = int(p.cell_cnt[r0])
        cu = p.arena_u[c0:c0 + cn]
        cv = p.arena_v[c0:c0 + cn]
        occ = np.zeros((W, H), np.int64)
        for j in st.bay_blocks[b]:
            if st.t1[j] >= t2_new or st.t2[j] <= t_new:
                continue
            uu, vv = self._l0_cells(j, W, H)
            occ[uu, vv] += 1
        nx = pxhi - pxlo + 1
        ny = pyhi - pylo + 1
        ov = np.zeros((nx, ny), np.int64)
        for q in range(cn):
            x0 = pxlo + int(cu[q])
            y0 = pylo + int(cv[q])
            ov += occ[x0:x0 + nx, y0:y0 + ny]
        if near is None:
            q0 = int(np.argmin(ov))
            return pxlo + q0 // ny, pylo + q0 % ny, cu, cv
        xs, ys = np.nonzero(ov == ov.min())
        dd = (np.abs(xs + pxlo - int(near[0]))
              + np.abs(ys + pylo - int(near[1])))
        q0 = int(np.argmin(dd))
        return pxlo + int(xs[q0]), pylo + int(ys[q0]), cu, cv

    def _sep_victim(self, v, b, budget, spent):
        """Resolve one colliding victim against the live state (which may
        contain transiently overlapping forced seeds): (a) exact
        relocation / shift-and-relocate anchored at its own entry day,
        (b) pure 1..3-day window shift keeping (x, y) -- all inside the
        caller's shared net-positive w1-day budget. The victim never
        changes bay, so w2/w3 are untouched.
        Returns (status, why, v_orig, spent):
          'cleared' -- the incumbent placement was proven exact-clear
                       (raster false positive) and restored verbatim;
          'moved'   -- relocated/retimed through an exact path, spent
                       updated; v_orig is the caller's touched entry;
          'stuck'   -- restored verbatim, `why` is the failure key."""
        st = self.state
        p = st.prep
        v_orig = (v, b, int(st.t1[v]), int(st.oi[v]), int(st.px[v]),
                  int(st.py[v]), int(st.seq[v]), int(st.t2[v]))
        e_v, x_v = v_orig[2], v_orig[7]
        D_v = int(p.D[v])
        st.remove(v)
        stuck_why = "v_scan_none"
        # (a) relocate at the fixed window / shift-and-relocate: the scan
        # is anchored at the victim's own entry day and is an EXACT
        # placement against the live state including every forced seed, so
        # success means the victim is genuinely clear of them all
        r = st.scan_from_day(v, b, e_v)
        if r is not None:
            t_v, o_v, px_v, py_v = r
            if (t_v, o_v, px_v, py_v) == (e_v, v_orig[3], v_orig[4],
                                          v_orig[5]):
                # the scan proves the INCUMBENT placement is already
                # exact-clear of the seeds (the raster overlap was a
                # false positive): restore it verbatim, stop colliding
                st.insert(v, b, e_v, v_orig[3], v_orig[4], v_orig[5],
                          seq=v_orig[6], t2=v_orig[7])
                return ("cleared", None, v_orig, spent)
            if 0 <= t_v - e_v <= 3:
                t2_v = t_v + int(p.P[v])
                tadd = max(0, t2_v - D_v) - max(0, x_v - D_v)
                if spent + tadd <= budget:
                    st.insert(v, b, t_v, o_v, px_v, py_v)
                    return ("moved", None, v_orig, spent + tadd)
                stuck_why = "v_budget"
            else:
                stuck_why = "v_scan_late"
        # (b) pure window shift keeping (x, y): rarely clears an in-window
        # seed but free when the layer flags line up
        for s in (1, 2, 3):
            tadd = max(0, x_v + s - D_v) - max(0, x_v - D_v)
            if spent + tadd > budget:
                break
            if st.can_place_exact(v, b, v_orig[4], v_orig[5],
                                  v_orig[3], e_v + s, x_v + s):
                st.insert(v, b, e_v + s, v_orig[3], v_orig[4],
                          v_orig[5], t2=x_v + s)
                return ("moved", None, v_orig, spent + tadd)
        # victim stuck: restore it, let the caller abandon the whole move
        st.insert(v, v_orig[1], v_orig[2], v_orig[3], v_orig[4],
                  v_orig[5], seq=v_orig[6], t2=v_orig[7])
        return ("stuck", stuck_why, v_orig, spent)

    # -------------------------------------------- insert-then-separate move --
    def try_separate(self, T):
        """Insert-then-separate with victim retiming (probe #2 graft; the
        probe is the spec, the engine primitives are the implementation).

        (1) tardy seed, tardiness-proportional; (2) target entry pulling its
        exit toward the due date, pull depth sampled in [1, 21] days per
        move (probe: modest pulls legalize, huge ones do not); (3) FORCE-
        place the seed at
        the min layer-0-overlap position in its own bay at the target window
        -- state.insert does no feasibility check, so the overlap is carried
        transiently and every subsequent scan simply sees the seed as an
        obstacle (conservative); (4) bounded separation: each colliding
        victim either relocates via the state's insertion scan anchored at
        its own entry day (t == entry: pure relocation; entry < t <= +3:
        shift-and-relocate) or shifts 1..3 days keeping (x, y), all within
        the net-positive w1-day budget; (5) zero collisions -> the seed's
        forced placement is re-validated through can_place_exact -- the SAME
        exact test every committed placement passes -- and the move commits;
        anything else reverts wholesale. Victims never change bay, so w2/w3
        are untouched and delta = -w1 * net_days exactly, as in the probe."""
        st = self.state
        p = st.prep
        rng = self.rng
        dbg = self._sep_dbg

        def _fail(key):
            dbg[key] = dbg.get(key, 0) + 1
            return False

        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = np.nonzero(tard > 0)[0]
        if len(cand) == 0:
            return _fail("no_tardy")
        w = tard[cand].astype(np.float64)
        d = int(rng.choice(cand, p=w / w.sum()))
        b = int(st.bay[d])
        P_d = int(p.P[d])
        o_d = int(st.oi[d])
        oidx = int(p.orient_base[d] + o_d)
        cur_t1 = int(st.t1[d])
        cur_t2 = int(st.t2[d])
        # target entry: pull the exit toward the due date. Depth SAMPLED in
        # [1, 21] rather than always the 21-day cap: a deep pull into a
        # packed window demands every victim clear within the 3-day retime
        # bound at once (measured all-v_scan_late at fixed depth 21), while
        # a shallow pull is a micro-compaction the bounded separation can
        # actually finish -- probe #2's own evidence is that modest pulls
        # legalize and huge ones do not.
        pull = int(rng.integers(1, 22))
        t_new = max(int(p.R[d]), int(p.D[d]) - P_d, cur_t1 - pull)
        if t_new >= cur_t1:
            return _fail("no_pull")
        t2_new = t_new + P_d
        gain = max(0, cur_t2 - int(p.D[d])) - max(0, t2_new - int(p.D[d]))
        if gain <= 0:
            return _fail("no_gain")

        W = int(p.bayW[b])
        H = int(p.bayH[b])
        if not p.fits[oidx, b]:
            return _fail("no_fit")
        if (int(p.pxlo[oidx, b]) > int(p.pxhi[oidx, b])
                or int(p.pylo[oidx, b]) > int(p.pyhi[oidx, b])):
            return _fail("no_fit")

        cur_obj = st.objective()
        touched = []  # original tuples of every block modified, commit order

        def revert():
            self._sep_revert(touched)

        d_orig = (d, b, cur_t1, o_d, int(st.px[d]), int(st.py[d]),
                  int(st.seq[d]), cur_t2)
        st.remove(d)
        touched.append(d_orig)

        # min layer-0-overlap force position over the target window (shared
        # helper _min_ov_pos; cell level only -- the exact commit test at
        # the end is the arbiter)
        fp = self._min_ov_pos(b, oidx, t_new, t2_new, W, H)
        if fp is None:  # unreachable: the no_fit guards above already ran
            revert()
            return _fail("no_fit")
        sx, sy, cu, cv = fp

        # force-place the seed (overlap allowed transiently)
        st.insert(d, b, t_new, o_d, sx, sy)
        seedmask = np.zeros((W, H), bool)
        seedmask[cu + sx, cv + sy] = True

        budget = gain - 1  # victims' added tardy-days: net must stay > 0
        spent = 0
        t_stop = time.time() + 0.03
        iters = 0
        ok = True
        # victims whose placement an EXACT scan already validated against
        # the live state including the forced seed. The collision test below
        # is conservative (outer-raster cells over-cover the true shape), so
        # without this set an exact-clear-but-raster-overlapping victim
        # would be re-detected forever and cycle the loop into its cap.
        cleared = set()
        while iters < 200:
            iters += 1
            # worst colliding victim vs the LIVE state, recomputed each pass
            vic, vic_ov = -1, 0
            for j in st.bay_blocks[b]:
                if (j == d or j in cleared
                        or st.t1[j] >= t2_new or st.t2[j] <= t_new):
                    continue
                uu, vv = self._l0_cells(j, W, H)
                c = int(np.count_nonzero(seedmask[uu, vv]))
                if c > vic_ov:
                    vic_ov = c
                    vic = int(j)
            if vic < 0:
                break  # zero unproven cell-level overlap: try to commit
            if time.time() > t_stop:
                ok = _fail("timeout")
                break
            # bounded victim resolution: shared helper _sep_victim (exact
            # relocate / retime inside the shared net-positive budget)
            status, why, v_orig, spent = self._sep_victim(vic, b, budget,
                                                          spent)
            if status == "cleared":
                cleared.add(vic)
                continue
            if status == "moved":
                touched.append(v_orig)
                cleared.add(vic)
                continue
            # victim stuck (already restored verbatim): abandon the move
            ok = _fail(why)
            break
        else:
            ok = _fail("iter_cap")

        if ok:
            # the seed's transient placement must pass the SAME exact test
            # every other committed placement passed: re-place through it
            st.remove(d)
            if st.can_place_exact(d, b, sx, sy, o_d, t_new, t2_new):
                st.insert(d, b, t_new, o_d, sx, sy)
                delta = st.objective() - cur_obj
                # budget guarantees delta <= -w1 < 0; SA is belt & braces
                _ok_sa = delta <= 0 or rng.random() < math.exp(
                    -delta / max(T, 1e-9))
                self._lam_move_obs(_ok_sa, delta)
                if _ok_sa:
                    if delta < -1e-9:
                        self._sep_acc += 1
                    self._sep_commit += 1
                    return True
                _fail("sa_rej")
            else:
                _fail("exact_fail")
        revert()
        return False

    def _rm_hole(self):
        """Hole-targeted ruin: on a tardiness-weighted bay at its most
        congested day, find the largest connected FREE region of the
        layer-0 forbidden grid and remove the blocks bordering it (they
        wall the hole in). Repair then refills the region coherently
        instead of leaving it as dead fragmented area."""
        st = self.state
        p = st.prep
        rng = self.rng
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = np.nonzero(tard > 0)[0]
        if len(cand) == 0:
            return self._rm_random()
        k_target = self.k_max
        w = tard[cand].astype(np.float64)
        d = int(rng.choice(cand, p=w / w.sum()))
        b = int(st.bay[d])
        # most congested day around d's wait
        t0 = int(p.R[d])
        t1 = max(t0 + 1, int(st.t1[d]))
        seg = st.occ0[b, t0:t1]
        t = t0 + int(np.argmax(seg)) if len(seg) else t0
        members = [j for j in st.bay_blocks[b]
                   if st.t1[j] <= t < st.t2[j]]
        if len(members) < 3:
            return self._rm_random()
        # layer-0 occupancy + owner map at day t
        W = int(p.bayW[b])
        H = int(p.bayH[b])
        occ = np.zeros((W, H), np.int32)
        owner = np.full((W, H), -1, np.int32)
        for j in members:
            oidx = int(p.orient_base[j] + st.oi[j])
            r0 = int(p.layer_base[oidx])
            c0 = int(p.cell_ptr[r0])
            cn = int(p.cell_cnt[r0])
            uu = p.arena_u[c0:c0 + cn] + int(st.px[j])
            vv = p.arena_v[c0:c0 + cn] + int(st.py[j])
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            occ[uu[m], vv[m]] = 1
            owner[uu[m], vv[m]] = j
        # largest connected free region (4-neighbour BFS)
        seen = occ.astype(bool).copy()
        best_region = None
        for su in range(0, W, 3):
            for sv in range(0, H, 3):
                if seen[su, sv]:
                    continue
                stack = [(su, sv)]
                seen[su, sv] = True
                region = []
                while stack:
                    u, v = stack.pop()
                    region.append((u, v))
                    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nu, nv = u + du, v + dv
                        if 0 <= nu < W and 0 <= nv < H and not seen[nu, nv]:
                            seen[nu, nv] = True
                            stack.append((nu, nv))
                if best_region is None or len(region) > len(best_region):
                    best_region = region
        if not best_region:
            return self._rm_random()
        # blocks bordering the hole
        border = set()
        for (u, v) in best_region:
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nu, nv = u + du, v + dv
                if 0 <= nu < W and 0 <= nv < H and owner[nu, nv] >= 0:
                    border.add(int(owner[nu, nv]))
        border.discard(d)
        rest = list(border)
        if st.lock_bay is not None:
            rest = [i for i in rest
                    if int(st.lock_bay[i]) in (-1, int(st.bay[i]))]
        rng.shuffle(rest)
        out = [d] + rest
        return out[:max(3, k_target)]

    def try_dp_repack(self, T):
        """DP-by-parts joint repack. Take a congested bay-window -- the
        present members plus queued tardy blocks that waited past it -- and
        re-place the whole batch JOINTLY by bounded depth-first search:
        candidates are corner-anchored best positions (try_day), the state
        itself is the search state (insert on descend, remove on backtrack),
        largest blocks are decided first. This is the lookahead that
        one-at-a-time greedy insertion structurally lacks: a big block gets
        first pick of the space it needs, queued blocks enter at the window
        start instead of after it. Priced exactly, SA accept, wholesale
        revert."""
        st = self.state
        p = st.prep
        rng = self.rng
        tard = np.maximum(0, st.t2 - p.D) * st.placed
        cand = np.nonzero(tard > 0)[0]
        if len(cand) == 0:
            return False
        w = tard[cand].astype(np.float64)
        d = int(rng.choice(cand, p=w / w.sum()))
        b = int(st.bay[d])
        if st.lock_bay is not None and int(st.lock_bay[d]) not in (-1, b):
            return False
        R_d = int(p.R[d])
        cur_t1 = int(st.t1[d])
        t1 = R_d if cur_t1 <= R_d else int(rng.integers(R_d, cur_t1))
        Ps = [int(p.P[j]) for j in st.bay_blocks[b]]
        span = max(2, int(sum(Ps) / max(1, len(Ps))))
        t2 = t1 + span

        # re-place only blocks ENTERING inside the window; blocks already
        # present before t1 keep their positions and act as fixed obstacles
        # (removing every present block made congested windows exceed any
        # sane batch size: 292/300 aborts measured on prob_27)
        o0 = p.orient_base
        members = [int(j) for j in st.bay_blocks[b]
                   if t1 <= st.t1[j] < t2]
        if len(members) > 12:
            members.sort(key=lambda j: -int(p.ncells0[o0[j] + st.oi[j]]))
            members = members[:12]
        queued = [int(j) for j in cand
                  if int(st.bay[j]) == b and int(st.t1[j]) >= t2
                  and int(p.R[j]) <= t1 and j not in members]
        queued.sort(key=lambda j: -tard[j])
        queued = queued[:4]
        if not queued:
            return False

        area = {j: int(p.ncells0[o0[j] + st.oi[j]]) for j in
                members + queued}
        entry = {j: int(st.t1[j]) for j in members}
        for j in queued:
            entry[j] = max(int(p.R[j]), t1)
        # queued blocks FIRST: they are the point of the move -- placing
        # members first lets them re-take the space and reproduces exactly
        # the greedy outcome the move exists to escape
        batch = (sorted(queued, key=lambda j: -area[j])
                 + sorted(members, key=lambda j: -area[j]))

        cur_obj = st.objective()
        origs = {}
        for j in batch:
            origs[j] = (int(st.bay[j]), int(st.t1[j]), int(st.oi[j]),
                        int(st.px[j]), int(st.py[j]), int(st.seq[j]),
                        int(st.t2[j]))
            st.remove(j)

        def revert():
            for j in reversed(batch):
                if st.placed[j]:
                    st.remove(j)
                bb, tt1, oo, xx, yy, sq, tt2 = origs[j]
                st.insert(j, bb, tt1, oo, xx, yy, seq=sq, t2=tt2)

        corners = ((1, 1), (-1, 1), (1, -1), (-1, -1))
        budget = [200]
        is_queued = set(queued)

        def dfs(k):
            if k == len(batch):
                return True
            if budget[0] <= 0:
                return False
            j = batch[k]
            tj = entry[j]
            seen = set()
            tried = 0
            for (xd, yd) in corners:
                budget[0] -= 1
                r = st.scan_from_day(j, b, tj, xdir=xd, ydir=yd)
                if r is None:
                    break
                t, o, px, py = r
                if j in is_queued:
                    # any day strictly earlier than the original entry is
                    # progress; later is pointless
                    if t >= origs[j][1]:
                        break
                elif t != tj:
                    break  # a member must keep its entry day
                if (t, o, px, py) in seen:
                    continue
                seen.add((t, o, px, py))
                st.insert(j, b, t, o, px, py)
                if dfs(k + 1):
                    return True
                st.remove(j)
                tried += 1
                if tried >= 3:
                    break
            return False

        if not dfs(0):
            revert()
            return False

        new_obj = st.objective()
        delta = new_obj - cur_obj
        _ok_sa = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        self._lam_move_obs(_ok_sa, delta)
        if _ok_sa:
            self._dp_acc += 1
            return True
        revert()
        return False

    # ------------------------------------------------------------- main loop --
    def run(self, T0=None, Tmin_frac=None, cycle_s=None, island=None):
        if Tmin_frac is None:
            Tmin_frac = float(os.environ.get("OGC_TMIN_FRAC", "0.02"))
        # Four islands, cross-island adoption, and the independent stall-kick
        # mechanism already supply restarts.  Reheating on a wall-clock grid
        # was both redundant and unstable near int(total/cycle_s) boundaries,
        # so the production default is one uninterrupted cooling trajectory.
        # Retain OGC_CYCLE_S / the argument solely as an explicit experiment.
        cycle_override = (cycle_s is not None
                          or "OGC_CYCLE_S" in os.environ)
        if cycle_s is None:
            cycle_s = os.environ.get("OGC_CYCLE_S", "inf")
        try:
            cycle_s = float(cycle_s)
        except (TypeError, ValueError):
            cycle_s = math.inf
        st = self.state
        p = st.prep
        rng = self.rng
        cur_obj = st.objective()
        best_obj = cur_obj
        best_snap = st.snapshot()
        # LAHC history (Burke & Bykov 2017); None keeps Metropolis SA
        lahc = [cur_obj] * self.lahc_len if self.lahc_len > 0 else None

        T0 = T0 or max(1.0, 0.5 * p.w1)
        Tmin = max(1e-6, T0 * Tmin_frac)
        t_start = time.time()
        total = max(1e-9, self.t_end - t_start)
        if (not cycle_override or not math.isfinite(cycle_s)
                or cycle_s <= 0.0):
            n_cycles = 1
        else:
            n_cycles = max(1, min(8, int(total / cycle_s)))
        cyc_len = total / n_cycles
        it = 0
        # T is the SCHEDULE temperature -- the shipped monotone curve, which
        # keeps its second job of pricing the reinsertion budget. Ta is the
        # ACCEPTANCE temperature; without the Lam controller it is the same
        # number, and every path below reads Ta for its Metropolis test so the
        # two cases share one code path. See the OGC_LAM block.
        T = Ta = T0
        lam = None
        if (lahc is None
                and (_LAM_SET is None
                     or (island is not None and island in _LAM_SET))):
            lam = _LamAnneal(total, p.w1, T0, log=self.log)
        self._lam = lam
        self._lam_t0 = t_start
        self._tp = [0, 0, 0]
        self._aborts = 0
        self._accepts = 0
        next_sync = t_start + self.sync_every
        # LOCKED-ASSIGNMENT ESCAPE.
        #
        # The locked island has state.lock_bay set, and best_insertion rejects
        # outright any candidate in a different bay -- so the lock does not
        # bias the search, it deletes every solution that disagrees with it
        # from this island's space. The assignment it enforces comes from an
        # area-based relaxation that cannot see polygon shape, layer stacking
        # or crane order, and ogc_solve's own comment concedes the point
        # ("area-based eta overestimates what multi-layer crane rules allow").
        # When the locked assignment has no good geometric realisation there is
        # no recovery: the island cannot reach one, and it spends its entire
        # budget failing inside a space that does not contain the answer.
        #
        # So the lock is a HYPOTHESIS with a deadline. If it has produced no
        # new incumbent for a quarter of the island's budget, it has had its
        # chance; drop it and let the island search the full assignment space
        # for the remainder, keeping everything it found. Strictly widening --
        # the incumbent is untouched, only the constraint is lifted.
        lock_patience = (_LOCK_ESC * total
                         if st.lock_bay is not None and _LOCK_ESC > 0.0
                         else math.inf)
        last_best_t = t_start
        last_best_obj = best_obj
        while True:
            now = time.time()
            if now >= self.t_end:
                break
            if best_obj < last_best_obj - 1e-9:
                last_best_obj = best_obj
                last_best_t = now
            elif (st.lock_bay is not None
                  and now - last_best_t > lock_patience):
                st.lock_bay = None
                lock_patience = math.inf
                if self.log:
                    self.log(f"lock released: no new best for "
                             f"{now - last_best_t:.0f}s, searching all bays")
            if it % 8 == 0:
                el = now - t_start
                cyc_frac = (el % cyc_len) / cyc_len
                # later cycles start cooler and restart from the best solution
                cyc_idx = int(el / cyc_len)
                T0c = T0 * (0.5 ** cyc_idx)
                T = max(Tmin, T0c * (Tmin / T0c) ** cyc_frac)
                if lam is not None and lam.tuning:
                    # also checked here so a tuning phase starved of main-loop
                    # candidates (the special moves take up to ~30% of
                    # iterations, and repair aborts yield nothing) still ends
                    # on its wall-clock cap instead of running to t_end
                    lam.maybe_finish(now, t_start)
                if cyc_idx != getattr(self, "_cyc", 0):
                    self._cyc = cyc_idx
                    st.restore(best_snap)
                    cur_obj = best_obj
                if self.sync_cb is not None and now >= next_sync:
                    next_sync = now + self.sync_every
                    try:
                        adopted = self.sync_cb(best_obj, best_snap)
                        if adopted is not None:
                            g_obj, g_snap = adopted
                            if g_obj < best_obj - 1e-9:
                                best_obj = g_obj
                                best_snap = g_snap
                                st.restore(g_snap)
                                cur_obj = g_obj
                    except Exception:
                        self.sync_cb = None
                if _STALL_S > 0:
                    if best_obj < getattr(self, "_stall_best",
                                          math.inf) - 1e-9:
                        self._stall_best = best_obj
                        self._stall_t = now
                    elif (now - getattr(self, "_stall_t", t_start) > _STALL_S
                          and self.t_end - now > 8.0):
                        # stalled: restart from the best solution kicked by a
                        # large greedy relocation (fresh basin, keeps quality)
                        self._stall_t = now
                        st.restore(best_snap)
                        ids = np.nonzero(st.placed)[0]
                        kk = int(min(len(ids), 40, max(12, 3 * self.k_max)))
                        pert = [int(x) for x in
                                rng.choice(ids, size=kk, replace=False)]
                        olds = [(j, int(st.bay[j]), int(st.t1[j]),
                                 int(st.oi[j]), int(st.px[j]), int(st.py[j]),
                                 int(st.seq[j]), int(st.t2[j]))
                                for j in pert]
                        for j in pert:
                            st.remove(j)
                        olds.sort(key=lambda z: (p.D[z[0]], p.R[z[0]]))
                        kick_dl = self.t_end - 2.0
                        done = True
                        for (j, bb, tt1, oo, xx, yy, sq, tt2) in olds:
                            cand = (st.best_insertion(j, use_guide=True)
                                    if time.time() < kick_dl else None)
                            if cand is None:
                                done = False
                                break
                            _, nb2, nt, no2, npx, npy = cand
                            st.insert(j, nb2, nt, no2, npx, npy)
                        if not done:
                            # out of time (or no candidate) mid-kick: a
                            # partial mix of moved and original placements
                            # is not self-consistent -- revert the whole kick
                            for (j, *_rest) in olds:
                                if st.placed[j]:
                                    st.remove(j)
                            for (j, bb, tt1, oo, xx, yy, sq, tt2) in olds:
                                st.insert(j, bb, tt1, oo, xx, yy,
                                          seq=sq, t2=tt2)
                        cur_obj = st.objective()
                        if cur_obj < best_obj - 1e-9:
                            best_obj = cur_obj
                            best_snap = st.snapshot()
            it += 1

            it_t0 = time.time()
            # the acceptance temperature for THIS iteration: the controller's
            # calibrated schedule times its correction, or the shipped curve
            # when it is not running. it_t0 is already this iteration's clock,
            # so following the schedule costs no extra time.time().
            #
            # Tr is the temperature that prices the reinsertion budget. It stays
            # on the shipped monotone curve unless OGC_LAM_REPAIR says otherwise
            # -- see that knob for why this is the lever that matters more.
            if lam is None:
                Ta = T
            else:
                Ta = lam.temp(it_t0, t_start)
            Tr = T if (lam is None or lam.T_rep is None) else lam.T_rep
            # exit-delay nesting move (own accept/revert); its share of
            # iterations adapts to its acceptance rate
            nest_p = 0.02 + min(0.10, 0.6 * self._nest_acc
                                / max(30, self._nest_try))
            if _NEST and self.state.sum_tard > 0 and rng.random() < nest_p:
                self._nest_try += 1
                if self.try_nest(Ta):
                    cur_obj = st.objective()
                    if cur_obj < best_obj - 1e-9:
                        best_obj = cur_obj
                        best_snap = st.snapshot()
                continue
            # insert-then-separate move (OGC_SEPMOVE, band-gated by the
            # driver): own accept/revert, adaptive share. The knob guard
            # short-circuits before any rng draw, so with the move off the
            # default stream is bit-identical.
            if self._sep_on and self.state.sum_tard > 0:
                sep_p = 0.02 + min(0.10, 0.6 * self._sep_acc
                                   / max(30, self._sep_try))
                if rng.random() < sep_p:
                    self._sep_try += 1
                    if self.try_separate(Ta):
                        cur_obj = st.objective()
                        if cur_obj < best_obj - 1e-9:
                            best_obj = cur_obj
                            best_snap = st.snapshot()
                    continue
            # DP-by-parts joint repack (own accept/revert, adaptive share)
            if _DP and self.state.sum_tard > 0:
                dp_p = 0.02 + min(0.10, 0.6 * self._dp_acc
                                  / max(30, self._dp_try))
                if rng.random() < dp_p:
                    self._dp_try += 1
                    if self.try_dp_repack(Ta):
                        cur_obj = st.objective()
                        if cur_obj < best_obj - 1e-9:
                            best_obj = cur_obj
                            best_snap = st.snapshot()
                    continue
            # ejection-chain move; its share adapts to its acceptance rate
            if _CHAIN and self.state.sum_tard > 0:
                chain_p = 0.04 + min(0.10, 0.6 * self._chain_acc
                                     / max(30, self._chain_try))
                if rng.random() < chain_p:
                    self._chain_try += 1
                    if self.try_chain(Ta, T_rep=Tr):
                        cur_obj = st.objective()
                        if cur_obj < best_obj - 1e-9:
                            best_obj = cur_obj
                            best_snap = st.snapshot()
                    continue
            # sacrifice-role swap (needs day targets; adaptive share). The
            # env-knob guard short-circuits before any rng draw, keeping the
            # default stream identical
            if (_ROLESWAP and st.tday is not None
                    and self.state.sum_tard > 0):
                rs_p = 0.03 + min(0.08, 0.5 * self._rs_acc
                                  / max(30, self._rs_try))
                if rng.random() < rs_p:
                    self._rs_try += 1
                    if self.try_roleswap(Ta):
                        cur_obj = st.objective()
                        if cur_obj < best_obj - 1e-9:
                            best_obj = cur_obj
                            best_snap = st.snapshot()
                    continue
            # targeted balance move (cheap, single block)
            if rng.random() < 0.05:
                if self.try_balance(Ta):
                    cur_obj = st.objective()
                    if cur_obj < best_obj - 1e-9:
                        best_obj = cur_obj
                        best_snap = st.snapshot()
                continue

            oi = int(rng.choice(len(self.ops), p=self.op_w / self.op_w.sum()))
            removed = list(dict.fromkeys(
                int(i) for i in self.ops[oi]() if self.state.placed[i]))
            if not removed:
                # the operator drew a ruin size but produced no move; there is
                # no outcome to attribute, and leaving _last_k set would credit
                # this draw with the NEXT iteration's result
                self._last_k = None
                continue
            old = [(i, int(st.bay[i]), int(st.t1[i]), int(st.oi[i]),
                    int(st.px[i]), int(st.py[i]), int(st.seq[i]),
                    int(st.t2[i]))
                   for i in removed]
            # Cost of the removed set in the current solution: what the repair
            # is allowed to spend re-achieving it. This counted the tard and
            # pref parts ONLY, which left the accounting asymmetric, because
            # best_insertion's returned cost DOES carry a w2 term (stat_true =
            # w3*pref + w2*d2, ogc_state.py). So every insertion was charged
            # for the imbalance it created while the removal was never
            # credited for the imbalance it relieved -- repairs that shuffle
            # load between bays looked more expensive than they are, in one
            # direction only. try_chain already had this right (see d_old_sym);
            # the main loop now matches it.
            old_cost = 0.0
            for (i, b, t1, o, px, py, sq, t2o) in old:
                old_cost += (p.w1 * max(0, t2o - p.D[i])
                             + p.w3 * (p.smax[i] - p.pref[i, b]))
            obj2_pre = st.obj2()
            for i in removed:
                st.remove(i)
            old_cost += p.w2 * (obj2_pre - st.obj2())

            r_ord = rng.random()
            if _RORD and r_ord < _RORD_P:
                # fail-fast: the block the search has most recently been
                # unable to place goes first, then the ones with the fewest
                # bays open to them and the least window slack. If this repair
                # is doomed, it dies on insertion 1 instead of insertion k.
                # _rord_key[i] == (bays the block fits, window slack, -area);
                # static, so it is built once in __init__ instead of being
                # re-derived from numpy inside every comparison
                rk = self._rord_key
                order = sorted(removed,
                               key=lambda i: (-self.crit[i], rk[i]))
            elif _ORD2 and r_ord < 0.25:
                # hardest-first: structurally tardy / large blocks claim
                # space while the freed region is still empty
                ob = p.orient_base
                order = sorted(removed,
                               key=lambda i: (-p.lbc[i],
                                              -int(p.P[i])
                                              * int(p.ncells0[ob[i]]),
                                              p.D[i]))
            elif r_ord < (0.55 if _ORD2 else 0.30):
                order = list(removed)
                rng.shuffle(order)
            elif (self._sisr
                    and r_ord < (0.55 if _ORD2 else 0.30) + 0.20):
                # SISR recreate variety (the fork pairs this with string
                # removal): largest area-time footprint first is packing-
                # friendly when a contiguous hole was just opened. Consumes
                # no rng draw, so non-SISR islands are bit-identical; in
                # "all" mode it carves its 20% share from the day-order slot.
                ob = p.orient_base
                order = sorted(removed,
                               key=lambda i: -int(p.P[i])
                               * int(p.ncells0[ob[i]]))
            elif (st.tday is not None
                    and r_ord < (0.55 if _ORD2 else 0.30) + 0.35):
                # day-relaxation order: insert in target-entry-day order so
                # earlier targets claim the freed space first (carved from
                # the plain-EDD share; inactive without day targets)
                td = st.tday
                order = sorted(removed,
                               key=lambda i: (int(td[i]) if td[i] >= 0
                                              else int(p.R[i]),
                                              p.D[i], -p.wl[i]))
            else:
                order = sorted(removed,
                               key=lambda i: (p.D[i], p.R[i], -p.wl[i]))
            # lookahead-arbitrated repair takes an adaptive share of
            # iterations, grown by its own acceptance rate. `_RLOOK and ...`
            # short-circuits before the draw, so islands with it off keep the
            # tuned rng stream bit-for-bit.
            use_rlook = False
            if _RLOOK and rng.random() < (
                    0.02 + min(0.08, 0.6 * self._rl_best
                               / max(20, self._rl_try))):
                use_rlook = True
                self._rl_try += 1
            xdir = 1 if rng.random() < 0.7 else -1
            ydir = 1 if rng.random() < 0.7 else -1
            noise = 0.0 if rng.random() < 0.5 else 0.3 * p.w3
            guide_on = st.guide is not None and rng.random() < 0.6
            # short-circuits before the rng draw when no day targets exist,
            # keeping the default rng stream identical
            dayguide_on = (guide_on and st.tday is not None
                           and rng.random() < self._day_p)
            # deep repair (v4.23 graft; short-circuits before any rng draw
            # when off, so the default stream is unchanged): let this
            # reinsertion see placements the temperature-decayed cap hides
            if self._deep_p > 0.0 and rng.random() < self._deep_p:
                budget = old_cost + _DEEP_MULT * p.w1
                self._deep_used += 1
            else:
                budget = old_cost + self.cap_t * Tr + self.cap_w1 * p.w1
            # BALANCE HEADROOM. The cap is spent front-to-back, so a repair
            # whose payoff is a workload rebalance arriving on the LAST few
            # insertions dies on the first expensive one and is never seen --
            # the search is biased toward moves whose benefit shows up early.
            # obj2 is floor(max(u*L) - min(u*L)) and cannot go below zero, so
            # w2 * obj2 is an admissible bound on how much the w2 term can
            # still improve; granting exactly that much extra makes the prune
            # sound with respect to workload, and no more. It is small next to
            # w1 in practice (w2*obj2 is order 10^3 against w1 order 10^4), so
            # this widens the funnel without disabling it.
            if _BALHEAD:
                budget += p.w2 * st.obj2()
            ok = True
            spent = 0.0
            if _LBCAP:
                # lb_suf[idx] = structural minimum tard+pref cost of
                # order[idx+1:]. lbc ignores the signed w2 term, so on its own
                # it is a heuristic floor and NOT a valid bound -- the suffix
                # can come in under it by rebalancing load. Subtracting the
                # same admissible w2 * obj2 slack makes it one. Shrink each
                # scan's cap by it so doomed moves die early. The pre-scan
                # abort keeps the same bias headroom best_insertion grants its
                # winners -- a zero-headroom abort would reintroduce the
                # hard-reject pathology that measurably increased aborts.
                lb_slack = p.w2 * st.obj2() if _BALHEAD else 0.0
                lb_suf = [0.0] * len(order)
                acc = 0.0
                for idx in range(len(order) - 1, -1, -1):
                    lb_suf[idx] = max(0.0, acc - lb_slack)
                    acc += float(p.lbc[order[idx]])
                lb_head = (st.guide_w if guide_on else 0.0) + noise
            for idx, i in enumerate(order):
                cap_i = budget - spent - (lb_suf[idx] if _LBCAP else 0.0)
                if _LBCAP and cap_i + lb_head <= float(p.lbc[i]):
                    ok = False
                    self._aborts += 1
                    self._crit_hit(i)
                    break
                if _SIZECORNER:
                    xd_i = _size_xdir(p, i, self._size_med)
                elif _EXITCORNER:
                    xd_i = _exit_xdir(p, i, self._exit_med)
                else:
                    xd_i = xdir
                cand = st.best_insertion(i, bay_noise=noise,
                                         xdir=xd_i, ydir=ydir,
                                         best_cost_init=cap_i,
                                         use_guide=guide_on,
                                         use_dayguide=dayguide_on,
                                         blink=self.blink)
                if cand is None and _LASTCHANCE > 1:
                    # LAST-CHANCE RESCAN.
                    #
                    # day_scan is exhaustive over integer positions, but not
                    # over FEASIBILITY: a position whose raster screen is
                    # ambiguous needs an exact polygon test, and those are
                    # rationed by exact_budget (128). When the ration runs out
                    # the kernel scores the remaining ambiguous positions as
                    # INFEASIBLE. That is a false negative, not a tie-break --
                    # a genuinely placeable block can be reported unplaceable,
                    # and since this is the abort site, one such verdict throws
                    # away the whole repair. The existing code knows the
                    # verdict is unsound here: "when exact_budget exhausts
                    # within a day the verdict is scan-order dependent"
                    # (ogc_state.py), which is why the earliest-feasible-day
                    # memo is default-off.
                    #
                    # So before believing "nowhere to put this", spend a bigger
                    # ration and ask once more. It costs one extra scan of ONE
                    # block, only on repairs that were about to be discarded
                    # entirely, and it can only turn a rejection into a
                    # placement -- never the reverse.
                    self._lc_try += 1
                    cand = st.best_insertion(
                        i, bay_noise=noise, xdir=xd_i, ydir=ydir,
                        best_cost_init=cap_i, use_guide=guide_on,
                        use_dayguide=dayguide_on, blink=0.0,
                        exact_budget=_LASTCHANCE * st.exact_budget)
                    if cand is not None:
                        self._lc_save += 1
                if cand is None:
                    ok = False
                    self._aborts += 1
                    self._abort_pos += idx
                    self._abort_len += len(order)
                    self._ins_wasted += idx
                    # this is the binding constraint: the block the current
                    # arrangement leaves nowhere to put
                    self._crit_hit(i)
                    break
                if use_rlook and idx + 1 < len(order):
                    # roll forward over the TRUE continuation: the blocks
                    # still waiting in `order` are exactly what comes next,
                    # so unlike the construct-side arbiter this lookahead
                    # guesses nothing
                    cands = _look_views(st, i, cand, xd_i, dayguide_on, True,
                                        modes=_RLOOK_MODES,
                                        use_guide=guide_on, ydir=ydir)
                    if len(cands) > 1:
                        fut = [int(j) for j in
                               order[idx + 1:idx + 1 + _RLOOK_L]]
                        pick = _look_pick(
                            cands, _look_probe(st, i, cands, fut))
                        if pick:
                            cand = cands[pick]
                c, b, t, o, px, py = cand
                # `spent` is a running total of the repair's true cost, so an
                # insertion that IMPROVES the objective has to count as the
                # negative number it is. Clamping it at zero (the old
                # max(0.0, c)) threw away exactly the evidence that a compound
                # move is paying off: the block that landed in the least-loaded
                # bay and cut the imbalance earned the repair no room at all
                # for the blocks that follow it.
                spent += c if _BALHEAD else max(0.0, c)
                if _EXITPACK == "1":
                    ref = st.refine_position(i, b, t, o, xdir=xd_i,
                                             px0=px, py0=py)
                    if ref is not None:
                        px, py = ref
                elif _EXITPACK == "2":
                    px, py = st.refine_position_exact(i, b, t, o, px, py)
                elif _EXITPACK == "3":
                    px, py = st.refine_position_sliver(i, b, t, o, px, py)
                st.insert(i, b, t, o, px, py)

            accepted = False
            k_gain = 0.0
            k_best = False
            if ok:
                new_obj = st.objective()
                d = new_obj - cur_obj
                if _TIEPROBE:
                    self._tp[0] += 1
                    if abs(d) <= 1e-9:
                        self._tp[1] += 1
                    elif d > 0:
                        self._tp[2] += 1
                if lahc is not None:
                    ok_acc = d <= 0 or new_obj <= lahc[it % self.lahc_len]
                elif lam is None:
                    ok_acc = (d <= 0
                              or rng.random() < math.exp(-d / max(Ta, 1e-9)))
                elif lam.tuning:
                    # TUNING PHASE (paper Alg 6): record this candidate's cost
                    # difference. Acceptance is unchanged from the shipped
                    # schedule unless OGC_LAM_CAL_ACC=1 restores the paper's
                    # accept-all random walk -- see the CAL_ACC block for the
                    # measurement that turned that default off.
                    lam.sample(d)
                    if _LAM_CAL_ACC:
                        ok_acc = True
                    else:
                        ok_acc = (d <= 0
                                  or rng.random() < math.exp(-d
                                                             / max(Ta, 1e-9)))
                    lam.maybe_finish(time.time(), t_start)
                else:
                    ok_acc = (d <= 0
                              or rng.random() < math.exp(-d / max(Ta, 1e-9)))
                    # ONLY here. This is the single point in the island where a
                    # complete neighbouring solution has been priced and a
                    # Metropolis decision actually taken -- a repair or
                    # geometry abort never got that far, and counting one as a
                    # rejection would read a placement failure as "too cold"
                    # and heat the search for it.
                    now_l = time.time()
                    lam.observe(ok_acc, d, now_l,
                                (now_l - t_start) / total)
                if ok_acc:
                    accepted = True
                    self._accepts += 1
                    if use_rlook:
                        self._rl_acc += 1
                    cur_obj = new_obj
                    k_gain = max(0.0, -d)
                    self.op_score[oi] += k_gain
                    if new_obj < best_obj - 1e-9:
                        best_obj = new_obj
                        best_snap = st.snapshot()
                        k_best = True
                        self.op_score[oi] += 0.1 * (cur_obj - new_obj + 1)
                        if use_rlook:
                            # credit RECORDS, not acceptances: this move rides
                            # ordinary iterations, so its acceptance rate is
                            # not differential and would inflate the share to
                            # the ceiling whether or not it earns it
                            self._rl_best += 1
            if lahc is not None:
                lahc[it % self.lahc_len] = cur_obj
            if not accepted:
                # revert
                for i in list(reversed(order)):
                    if st.placed[i]:
                        st.remove(i)
                for (i, b, t, o, px, py, sq, t2o) in old:
                    st.insert(i, b, t, o, px, py, seq=sq, t2=t2o)
            self.op_cnt[oi] += 1

            # adaptive removal size: target iteration times in [5, 40] ms
            dt = time.time() - it_t0
            if _KADAPT:
                # credited here, after dt is known: the ruin size's reward is
                # per second, and the revert this iteration may have just paid
                # for is part of what the size cost
                self._k_credit(accepted, k_gain, dt, k_best)
            if dt > 0.04 and self.k_max > 4:
                self.k_max -= 1
            elif dt < 0.005 and self.k_max < _K_CEIL:
                self.k_max += 1

            if it % 512 == 0:
                # adapt operator weights
                sc = self.op_score / np.maximum(1, self.op_cnt)
                if sc.sum() > 0:
                    self.op_w = 0.7 * self.op_w + 0.3 * (0.05 + sc / (sc.max() + 1e-9))
                self.op_score[:] = 0
                self.op_cnt[:] = 0
                # decay the assignment-guide bias when the on-target count
                # stops improving (targets geometrically unreachable)
                if st.guide is not None and st.guide_w > 0:
                    on = sum(1 for i in range(self.n)
                             if st.placed[i] and int(st.bay[i]) == st.guide[i])
                    if on <= getattr(self, "_guide_on", -1):
                        st.guide_w *= 0.8
                        if st.guide_w < 0.5 * p.w3:
                            st.guide_w = 0.5 * p.w3
                    self._guide_on = max(on, getattr(self, "_guide_on", -1))
                # safety valve for the day targets, with hysteresis: an
                # all-time-max ratchet floors out on ANY plateau (including
                # full success), so decay only on SUSTAINED regression of
                # the on-day count -- the lift actively hurting adherence --
                # and recover toward the knob value when adherence improves.
                # A healthy plateau leaves day_p exactly where the knob set
                # it, which the OGC_DAY_P sweep depends on.
                if st.tday is not None:
                    on_d = sum(1 for i in range(self.n)
                               if st.placed[i] and st.tday[i] >= 0
                               and abs(int(st.t1[i]) - int(st.tday[i])) <= 2)
                    prev = getattr(self, "_day_prev", None)
                    if prev is not None:
                        if on_d < prev:
                            self._day_reg += 1
                            if self._day_reg >= 2:
                                self._day_p = max(0.1, self._day_p * 0.8)
                                self._day_reg = 0
                        else:
                            self._day_reg = 0
                            if on_d > prev and self._day_p < self._day_p0:
                                self._day_p = min(self._day_p0,
                                                  self._day_p * 1.25)
                    self._day_prev = on_d
                # one-shot day-target re-solve from the live incumbent
                # (long-TL main island only, behind OGC_DAY_RESOLVE): the
                # initial targets age as assignments drift, and the realized
                # occupancy is a far better capacity estimate than the
                # up-front eta guess
                if (self._day_resolve and not self._day_resolved
                        and st.tday is not None
                        and time.time() > t_start
                        + 0.45 * (self.t_end - t_start)
                        and self.t_end - time.time() > 30.0):
                    self._day_resolved = True
                    self._resolve_days()

        st.restore(best_snap)
        if self.log and _TIEPROBE and self._tp[0]:
            n, tie, up = self._tp
            self.log(f"tieprobe isl={island} valid={n} tie={tie / n:.4f} "
                     f"uphill={up / n:.4f}")
        if self.log and lam is not None:
            self.log(lam.report())
            if lam.trace:
                self.log("lam trace q,A,tgt,T: " + " ".join(
                    f"{q:.3f}/{a:.2f}/{g:.2f}/{tt:.3g}"
                    for (q, a, g, tt) in lam.trace))
        if self.log and self._sep_on:
            self.log(f"sepmove: tries={self._sep_try} "
                     f"commits={self._sep_commit} acc={self._sep_acc} "
                     f"dbg={self._sep_dbg}")
        if self.log:
            self.log(f"LNS isl={island} T0={T0:.0f} iters={it} "
                     f"best={best_obj:.0f} "
                     f"tard={st.sum_tard:.0f} pref={st.sum_pref:.0f} "
                     f"obj2={st.obj2():.0f} aborts={self._aborts} "
                     f"acc={self._accepts} kmax={self.k_max} "
                     + (f"kmean={self._k_mean():.1f} " if _KADAPT else "")
                     + (f"lastchance={self._lc_save}/{self._lc_try} "
                        if _LASTCHANCE > 1 else "")
                     + f"nest={self._nest_acc}/{self._nest_try} "
                     f"dp={self._dp_acc}/{self._dp_try} "
                     f"chain={self._chain_acc}/{self._chain_try}"
                     + (f" day_p={self._day_p:.2f} "
                        f"rs={self._rs_acc}/{self._rs_try} "
                        f"rsdbg={self._rs_dbg}"
                        if st.tday is not None else "")
                     + (f" efd={st._efd_hit}/{st._efd_q} "
                        f"skip={st._efd_skip}"
                        if st._efd_on else ""))
        return best_obj
