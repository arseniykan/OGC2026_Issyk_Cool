#!/usr/bin/env python
# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""Solve an OGC 2026 instance and verify it with the organizers' checker.

    python run.py data/finals/fin_4.json          # official TL for that instance
    python run.py data/finals/fin_4.json --tl 60  # override the budget
    python run.py --suite finals                  # all 8 finals instances
    python run.py --suite preliminary             # all 6 preliminary instances

Every run is verified before it is reported: a solution that fails the checker is
printed as INFEASIBLE with the failing stage, never as a score.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Official per-instance time limits, as published by the Organizing Committee
# after each round closed. These are the budgets the reported results used.
OFFICIAL_TL = {
    "fin_1": 180.0, "fin_2": 300.0, "fin_3": 480.0, "fin_4": 180.0,
    "fin_5": 240.0, "fin_6": 600.0, "fin_7": 360.0, "fin_8": 360.0,
    "prelim_1": 60.0, "prelim_2": 120.0, "prelim_3": 240.0,
    "prelim_4": 480.0, "prelim_5": 600.0, "prelim_6": 900.0,
}

SUITES = {
    "finals": [ROOT / "data" / "finals" / f"fin_{i}.json" for i in range(1, 9)],
    "preliminary": [ROOT / "data" / "preliminary" / f"prelim_{i}.json"
                    for i in range(1, 7)],
}


def solve_one(path, timelimit=None, quiet=False):
    """Solve one instance, verify it, and return a result dict."""
    from myalgorithm import algorithm
    from _ogc.official_checker import check_feasibility

    name = path.stem
    prob_info = json.loads(path.read_text(encoding="utf-8"))
    tl = timelimit if timelimit is not None else OFFICIAL_TL.get(name, 180.0)

    if not quiet:
        print(f"{name}: {len(prob_info['blocks'])} blocks, "
              f"{len(prob_info['bays'])} bays, TL {tl:.0f}s ... ", end="", flush=True)

    t0 = time.time()
    solution = algorithm(prob_info, tl)
    wall = time.time() - t0

    res = check_feasibility(prob_info, solution)
    res.update(name=name, wall=wall, tl=tl, solution=solution)

    if not quiet:
        if res.get("feasible"):
            print(f"obj {res['objective']:,.0f}  "
                  f"(Z1 {res['obj1']:,.0f}  Z2 {res['obj2']:,.2f}  "
                  f"Z3 {res['obj3']:,.0f})  {wall:.1f}s")
        else:
            print(f"INFEASIBLE at stage {res.get('stage')}: "
                  f"{'; '.join(res.get('violations', [])[:3])}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance", nargs="?", type=Path,
                    help="path to an instance JSON")
    ap.add_argument("--suite", choices=sorted(SUITES),
                    help="run a whole published suite at official time limits")
    ap.add_argument("--tl", type=float, default=None,
                    help="override the time limit, in seconds")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the solution JSON here (single instance only)")
    args = ap.parse_args(argv)

    if bool(args.instance) == bool(args.suite):
        ap.error("give exactly one of: an instance path, or --suite")

    if args.instance:
        if not args.instance.exists():
            ap.error(f"no such instance: {args.instance}")
        res = solve_one(args.instance, args.tl)
        if args.out:
            args.out.write_text(json.dumps(res["solution"], indent=1),
                                encoding="utf-8")
            print(f"solution written to {args.out}")
        return 0 if res.get("feasible") else 1

    paths = SUITES[args.suite]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"missing instances: {', '.join(p.name for p in missing)}",
              file=sys.stderr)
        return 2

    results = [solve_one(p, args.tl) for p in paths]
    ok = [r for r in results if r.get("feasible")]
    print(f"\n{len(ok)}/{len(results)} feasible, "
          f"{sum(r['wall'] for r in results):.0f}s total")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
