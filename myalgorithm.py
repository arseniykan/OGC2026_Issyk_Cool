# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""OGC 2026 submission entry point.

Unified single-engine solver: per-instance classifier sets knobs for one
raster+exact day_scan LNS engine (multi-process islands, CP-SAT/Gurobi
guides, SISR/LAHC portfolio arms), with an official-checker safety tail.
All logic lives in _ogc/driver.py.
"""
import os
import sys

# Plan-and-Realize disabled. The cumulative_plan gate (driver.py: group in
# overload-*, TL>=180, guides on, maxK<=2) hard-locks island 1 to an area-based
# capacity plan. Measured on the 2026-08-03 final-train set: it fires on 13/40
# instances there (vs 4/40 when it was tuned, peak_util up to 3.47 vs 1.41) and
# every loss to the reference solver was a gated instance -- the plan is
# unrealizable at those densities and costs 25%% of the island budget plus
# BRKGA construction search on island 1.
os.environ["OGC_PLAN"] = "0"

# Self-Tuning Lam controller on island 2, at the configuration that was actually
# measured. `_ogc/` is byte-identical to the benchmarked v10.6 build; the two
# knobs below are the arm labelled "cheap" in bench/lam/v106_cheap, which is the
# only Lam configuration with a win from a correctly-paired experiment:
# TL=420, n=10, all-X925 cores alternating between arms, geomean -1.34% against
# the shipped v10.3 (bench/lam/v106_cheap == 20260811_Team_v10.3.zip as arm A),
# better on 7/10, paired t = -2.14. That is p ~= 0.06, i.e. suggestive and NOT
# significant -- see the noise calibration in EVIDENCE.md before trusting it.
#
#   PLATEAU=0.20  Lam & Delosme's 0.44 was derived for single-variable moves;
#                 one iteration here ruins k<=13 blocks and rebuilds, so holding
#                 44% uphill acceptance through mid-run is too disruptive. 0.20
#                 keeps the trajectory's shape and halves its level.
#   OBS_MOVES=1   also observe the special moves' (try_nest/chain/roleswap/
#                 separate/dp_repack) Metropolis decisions. Same population, same
#                 Ta, same objective. Raises the controller's observation supply
#                 10-17x (prob_2: 13473 obs vs 811; prob_37: 14614 vs 1372),
#                 which is what closes the feedback loop on the starved
#                 instances instead of leaving it open-loop on the schedule.
#
# Set before importing _ogc: both are module-level constants read at import of
# _ogc.ogc_search, and the island workers inherit this environment under fork
# and spawn alike. Explicit environment still overrides, so every A/B stays
# runnable from the same build.
os.environ.setdefault("OGC_LAM", "2")
os.environ.setdefault("OGC_LAM_PLATEAU", "0.20")
os.environ.setdefault("OGC_LAM_OBS_MOVES", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _ogc.driver import algorithm  # noqa: F401,E402
