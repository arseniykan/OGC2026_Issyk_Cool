# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

# =============================================================================
#  OGC 2026 -- unified single-engine driver
#
#  One engine (the raster+exact day_scan LNS lineage, server-proven as
#  engine_v417) driven by a per-instance classifier that sets KNOBS instead of
#  switching engines:
#    - SISR dose        : "0" | "island3" | "all"   (string-removal arm)
#    - guide on/off     : CP-SAT bay guide + day guide (off for the pocket
#                         where guides measurably add tardiness)
#    - LAHC islands     : dose "all" runs LAHC acceptance on islands 1/3
#  The former engine_v47 pocket (moderate congestion x tardiness-dominant)
#  is reproduced inside this engine as dose="all" + guides off, i.e. the
#  fork's search portfolio expressed as knobs of the same code base.
#
#  Safety tail: official shapely checker when available, _ogc's own exact
#  integer checker otherwise; repair; guaranteed-feasible fallback.
# =============================================================================

import os
import sys
import time
import math
import warnings
import threading
import contextlib
from collections import deque

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -----------------------------------------------------------------------------
# Optional-component failure sink.
#
# DEFINED HERE, ABOVE ITS FIRST USE, AND NOT MOVED. The bundled-cache manifest
# block below runs at IMPORT time and reports problems through
# _optional_failure(). When this function was defined further down the file
# (after classify()), that except-handler referenced a name that did not exist
# yet, so ANY exception while reading the manifest -- truncated or non-JSON
# manifest, unexpected field types, a read failure after the exists() check, a
# metadata lookup failure, an unpicklable cache index -- was replaced by
#
#     NameError: name '_optional_failure' is not defined
#
# raised while importing _ogc.driver. myalgorithm.py could then not import
# `algorithm` at all: not a degraded solve, a total failure on every instance,
# from a dormant path that only opens when something else has already gone
# slightly wrong. Reproduced by writing a non-JSON manifest into the package.
# -----------------------------------------------------------------------------
_OPTIONAL_FAILURES = deque(maxlen=32)


def _optional_failure(stage, exc, log=None):
    """Retain and surface genuine optional-component exceptions.

    Returning None because a relaxation found no incumbent is expected and is
    not reported. Exceptions are different: v9 silently erased them unless
    OGC_LOG was enabled, making dependency/API regressions indistinguishable
    from a normal unavailable guide.
    """
    msg = f"{stage} failed: {type(exc).__name__}: {exc}"
    _OPTIONAL_FAILURES.append(msg)
    if log:
        log(msg)
    else:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


# -----------------------------------------------------------------------------
# Shipped numba cache: use it only if it can genuinely serve THIS machine.
#
# numba keys every cached kernel on (signature, (llvm_triple, cpu_name,
# cpu_features), code hashes) and discards the whole index unless the pickled
# numba version and the source (st_mtime, st_size) stamp also match. Pinning
# NUMBA_CPU_NAME is therefore the *price* of a portable cache, not a goal: it
# freezes codegen at whatever microarchitecture the cache was built for. If the
# cache cannot be used anyway -- different arch, different numba, skewed
# mtime -- paying that price buys nothing, we would JIT from scratch AND run
# unvectorised kernels. So the pin happens only after the probe says yes.
#
# The probe never imports numba (that would lock codegen in before we decide).
# -----------------------------------------------------------------------------
_MANIFEST = os.path.join(_PKG_DIR, "numba_cache_manifest.json")
_CACHE_LIVE = False


def _restore_pinned_mtimes(pin):
    """A zip stores DOS *local* time with no timezone field, so `unzip` on a
    machine that is not UTC restores source mtimes shifted by its UTC offset
    (measured: Berlin -3600, New York +18000, Tokyo -32400). That alone voids
    the whole cache index, so re-pin from the manifest before anything
    compiles."""
    for f in os.listdir(_PKG_DIR):
        if f.endswith(".py"):
            p = os.path.join(_PKG_DIR, f)
            try:
                if os.stat(p).st_mtime != pin:
                    os.utime(p, (pin, pin))
            except OSError:
                pass       # read-only extract: nothing to do, probe catches it


def _host_supports(flags):
    """Whether this CPU really has every feature the cache was built against.

    A cache built for e.g. 'x86-64-v3' is only loadable if we also pin
    NUMBA_CPU_NAME to it -- and that pin makes LLVM *emit* AVX2/FMA/BMI2. On a
    CPU without them that is a SIGILL, i.e. 'process terminated unexpectedly'
    on every instance. Refusing the cache costs ~15s of JIT; guessing wrong
    costs the entire submission, so an unreadable /proc/cpuinfo means no."""
    if not flags:
        return True                             # 'generic' needs nothing
    try:
        with open("/proc/cpuinfo") as fh:
            have = set()
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    have.update(line.split(":", 1)[1].split())
                    break
        return have and all(f in have for f in flags)
    except OSError:
        return False


def _cache_is_live(cpu, features):
    """Replicate numba's own index validation (numba/core/caching.py,
    IndexDataCacheFile._load_index) without importing numba: version, source
    stamp, and at least one overload compiled for this exact target triple and
    codegen identity. An aarch64-built cache cannot serve an x86_64 grader --
    NUMBA_CPU_NAME/NUMBA_CPU_FEATURES change the cpu and feature fields of the
    key, never the triple."""
    import pickle
    pyc = os.path.join(_PKG_DIR, "__pycache__")
    try:
        from importlib.metadata import version as _version
        import llvmlite.binding as _ll
        triple = _ll.get_process_triple()
        st = os.stat(os.path.join(_PKG_DIR, "ogc_kernels.py"))
        nbi = [f for f in os.listdir(pyc)
               if f.startswith("ogc_kernels.day_scan") and f.endswith(".nbi")]
        if not nbi:
            return False
        with open(os.path.join(pyc, nbi[0]), "rb") as fh:
            if pickle.load(fh) != _version("numba"):
                return False                    # built by a different numba
            stamp, overloads = pickle.loads(fh.read())
        if stamp != (st.st_mtime, st.st_size):
            return False                        # numba would discard the index
        return any(k[1] == (triple, cpu, features) for k in overloads)
    except Exception:
        return False


if os.path.exists(_MANIFEST):
    _cpu, _feat, _flags = "generic", "", ()
    try:
        import json as _json
        with open(_MANIFEST) as _fh:
            _mf = _json.load(_fh)
        _cpu = _mf.get("cpu") or "generic"
        _feat = _mf.get("features") or ""
        _flags = tuple(_mf.get("require_flags") or ())
        if _mf.get("mtime"):
            _restore_pinned_mtimes(float(_mf["mtime"]))
    except Exception as exc:
        _optional_failure("bundled feasibility checker", exc)
    # Pin BEFORE the probe: _cache_is_live() unpickles the .nbi index, and the
    # pickled overload keys contain numba types -- unpickling them IMPORTS
    # numba, which freezes numba.core.config from the env at that instant.
    # Worse, JITCPUCodegen freezes _tm_features at first-codegen init while
    # magic_tuple()'s cpu field is read live, so pinning after the probe
    # yields a lookup key (pinned cpu, host features) that can never match the
    # shipped (pinned cpu, "") key -- measured: probe says live, then a full
    # 15s recompile anyway. Pin first, so numba's first import already sees
    # the pinned codegen; if the probe then refuses the cache, unpin and force
    # a config reload -- nothing has compiled yet, so host codegen is fully
    # restored.
    _pin_name = "NUMBA_CPU_NAME" not in os.environ
    _pin_feat = "NUMBA_CPU_FEATURES" not in os.environ
    if _pin_name:
        os.environ["NUMBA_CPU_NAME"] = _cpu
    if _pin_feat:
        os.environ["NUMBA_CPU_FEATURES"] = _feat
    if _host_supports(_flags) and _cache_is_live(_cpu, _feat):
        _CACHE_LIVE = True
    else:
        if _pin_name:
            os.environ.pop("NUMBA_CPU_NAME", None)
        if _pin_feat:
            os.environ.pop("NUMBA_CPU_FEATURES", None)
        if "numba" in sys.modules:      # imported by the index unpickle
            try:
                from numba.core import config as _nbconfig
                _nbconfig.reload_config()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Instance classification (pure python, milliseconds). Verbatim thresholds from
# the server-proven v4.20 router; groups now select knob presets, not engines.
# -----------------------------------------------------------------------------

def _shoelace(verts):
    a = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


# user-provided OGC_BRKGA A/B override, snapshotted before the driver ever
# writes the knob itself (this module imports before any solve)
_BRK_ENV0 = os.environ.get("OGC_BRKGA")
_CAP_T_ENV0 = os.environ.get("OGC_CAP_T")
_CAP_W1_ENV0 = os.environ.get("OGC_CAP_W1")
# -----------------------------------------------------------------------------
# Solve-scoped configuration.
#
# The driver selects per-instance behaviour by writing environment variables,
# which are PROCESS-global and outlive the call that set them. Two things
# follow, and both are fixed here rather than by discipline at each call site.
#
# WHAT WAS ACTUALLY EXPOSED.  The common worry -- "the driver writes a knob,
# then ogc_search freezes it in a module constant at import, so instance B
# inherits instance A's value" -- does NOT apply to this code base, and it was
# checked rather than assumed: the variables the driver and ogc_solve write
# (OGC_GROUP, OGC_BRKGA, OGC_CAP_T, OGC_CAP_W1, OGC_V417_SISR, OGC_LAHC) and
# the variables ogc_search reads into module constants at import are disjoint
# sets. Every driver-written knob is already read per run, at LNS/State
# construction. The import-frozen constants are the experiment knobs, which the
# driver never writes.
#
# What WAS exposed is narrower but real:
#   * ogc_solve set OGC_LAHC and never cleared it, behind a guard that reads
#     "not already set". In-process (OGC_MP_START=single, or any platform
#     without fork) the first instance that took that branch turned LAHC on
#     for every later instance in the batch, permanently.
#   * OGC_GROUP and OGC_V417_SISR were written and never removed, so they
#     survived the call and leaked into whatever the caller did next.
#   * An exception between writing a knob and finishing the solve left the
#     environment holding that instance's settings.
#   * Nothing recorded which configuration a returned solution was produced
#     under, which makes a batch-only discrepancy near impossible to diagnose.
#
# solve_config() applies a knob dict for exactly the duration of one solve and
# restores the previous environment on the way out, including on exception, so
# a solve cannot influence its successor or its caller. _SOLVE_LOCK serialises
# the scoped region: environment variables cannot express two configurations at
# once, so concurrent algorithm() calls in one process are made to take turns
# instead of silently interleaving. True parallel multi-config solving needs a
# config object threaded through State/LNS instead of the environment; that is
# a larger change than this one and is not pretended here.
# -----------------------------------------------------------------------------
_SOLVE_LOCK = threading.RLock()
_EFFECTIVE_CONFIG = {}


@contextlib.contextmanager
def solve_config(knobs, log=None):
    """Apply `knobs` (name -> value, or None to unset) for one solve only."""
    global _EFFECTIVE_CONFIG
    with _SOLVE_LOCK:
        saved = {k: os.environ.get(k) for k in knobs}
        try:
            for k, v in knobs.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = str(v)
            _EFFECTIVE_CONFIG = {k: os.environ.get(k) for k in knobs}
            if log:
                log("config: " + " ".join(
                    f"{k}={v}" for k, v in sorted(_EFFECTIVE_CONFIG.items())
                    if v is not None))
            yield _EFFECTIVE_CONFIG
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def effective_config():
    """The knob set the most recent solve actually ran under."""
    return dict(_EFFECTIVE_CONFIG)


def optional_failures():
    """Recent bounded optional-component diagnostics for tests/embedders."""
    return tuple(_OPTIONAL_FAILURES)


def _bbox(layers):
    xs = [p[0] for v in layers for p in v]
    ys = [p[1] for v in layers for p in v]
    if not xs:
        return 0.0, 0.0
    return max(xs) - min(xs), max(ys) - min(ys)


def _fit_mask(block, bays):
    """Bit mask of bays that can physically host the block in SOME orientation.

    Bounding-box containment only, so it is a relaxation -- a bay outside the
    mask is definitely unusable, a bay inside it merely might be usable. That
    is the direction a capacity bound needs.
    """
    mask = 0
    for j, bay in enumerate(bays):
        for orient in block["shape"]:
            layers = [v for v in orient["layers"] if v]
            bw, bh = _bbox(layers)
            if bw <= bay["width"] + 1e-9 and bh <= bay["height"] + 1e-9:
                mask |= 1 << j
                break
    return mask


def _refined_features(blocks, bays, horizon):
    """Congestion statistics that do not assume the three things the legacy
    peak_util assumes: one layer, one shared pool of bay area, and every block
    starting on its release date.

    Returns (util_layers, util_hall, util_energetic, fit_min).

    LAYERS.  The legacy statistic measures layer 0. Layer 0 is not the
    footprint that matters: a block with a small base and a wide overhang
    occupies its upper-layer area against every other block's matching layer,
    and blocks the crane over all of it. Measured on the training set the
    max-layer to layer-0 area ratio averages 1.62 on prob_5 and reaches 20.2,
    so layer 0 can understate a block by an order of magnitude. (Orientation,
    by contrast, is NOT a source of error here: orientations are rigid
    transforms and layer areas across them are identical to four decimals on
    every training instance, so reading orientation 0 costs nothing.)

    HALL.  Summing all bay areas into one pool assumes every block can use
    every bay. Blocks restricted by bounding box to a subset S of bays must
    ALL fit within S, so each subset gives its own utilisation bound and the
    binding one is the max. With m <= 5 bays there are at most 32 subsets, so
    this is enumerated exactly rather than approximated. On the training set
    the tightest blocks reach only 40-50% of bays, so the pooled figure really
    does overstate what is available to them.

    ENERGETIC.  Charging every block over [R, R+P) prices a schedule nobody is
    obliged to follow; it is a guess, not a bound. The compulsory part -- the
    intersection of the earliest and latest placements, max(0, EF - LS) -- is
    the portion of a block that provably lands inside a window whatever the
    solver decides. Taking that over a ladder of window lengths turns the
    statistic into a genuine lower bound on congestion instead of an optimistic
    point estimate.
    """
    m = len(bays)
    bay_area = [float(b["width"] * b["height"]) for b in bays]
    items = []          # (area, mask, R, P, D)
    for b in blocks:
        layers = [v for v in b["shape"][0]["layers"] if v]
        area = max((_shoelace(v) for v in layers), default=0.0)
        items.append((area, _fit_mask(b, bays),
                      int(b["release_time"]), max(1, int(b["processing_time"])),
                      int(b["due_date"])))
    fit_min = min((bin(it[1]).count("1") for it in items), default=0) / max(m, 1)

    # --- layer-aware version of the legacy earliest-start profile -----------
    demand = [0.0] * (horizon + 2)
    for area, _mask, R, P, _D in items:
        for d in range(max(0, R), min(horizon, R + P)):
            demand[d] += area
    total_area = sum(bay_area)
    util_layers = (max(demand) if demand else 0.0) / max(total_area, 1e-9)

    # --- Hall-type subset bound --------------------------------------------
    util_hall = 0.0
    for sub in range(1, 1 << m):
        cap = sum(bay_area[j] for j in range(m) if sub >> j & 1)
        if cap <= 0.0:
            continue
        sub_dem = [0.0] * (horizon + 2)
        used = False
        for area, mask, R, P, _D in items:
            if mask and (mask & ~sub) == 0:      # confined to this subset
                used = True
                for d in range(max(0, R), min(horizon, R + P)):
                    sub_dem[d] += area
        if used:
            util_hall = max(util_hall, max(sub_dem) / cap)

    # --- energetic (compulsory-part) lower bound ---------------------------
    util_energetic = 0.0
    lengths = [1, 2, 4, 8, 16, 32, 64]
    # The window sweep is O(len(lengths) * horizon * n) in pure Python, which
    # is ~30ms at shipyard horizons but would grow without bound on a legal
    # long-horizon instance. Stride the window starts so the work stays flat;
    # a coarser grid still lower-bounds, it just bounds slightly less tightly.
    stride = max(1, horizon // 128)
    for L in lengths:
        if L > horizon:
            break
        for t1 in range(0, max(1, horizon - L + 1), stride):
            t2 = t1 + L
            need = 0.0
            for area, _mask, R, P, D in items:
                ls = max(R, D - P)               # latest start (tardy => R)
                ef = R + P                       # earliest finish
                # left-shift/right-shift intersection with [t1, t2)
                dur = min(P, t2 - t1, ef - t1, t2 - ls)
                if dur > 0:
                    need += area * dur
            util_energetic = max(util_energetic,
                                 need / (max(total_area, 1e-9) * L))
    return util_layers, util_hall, util_energetic, fit_min


def classify(prob_info):
    blocks = prob_info["blocks"]
    bays = prob_info["bays"]
    w = prob_info.get("weights", {}) or {}
    w1 = float(w.get("w1", 1.0))
    w3 = float(w.get("w3", 1.0))
    n = len(blocks)
    total_area = sum(b["width"] * b["height"] for b in bays)

    horizon = 0
    total_days = 0
    for b in blocks:
        horizon = max(horizon, b["release_time"] + b["processing_time"])
        total_days += b["processing_time"]
    if horizon > 200_000 or total_days > 5_000_000:
        raise ValueError("instance horizon too large to profile")
    demand = [0.0] * (horizon + 1)
    procs = []
    for b in blocks:
        layers = [v for v in b["shape"][0]["layers"] if v]
        a = _shoelace(layers[0]) if layers else 0.0
        procs.append(b["processing_time"])
        for d in range(max(0, b["release_time"]),
                       min(horizon, b["release_time"] + b["processing_time"])):
            demand[d] += a
    peak_util = (max(demand) if demand else 0.0) / max(total_area, 1e-9)
    procs.sort()
    p_med = procs[len(procs) // 2] if procs else 0
    r13 = w1 / max(w3, 1.0)

    # Refined congestion statistics. These are DIAGNOSTIC by default and do
    # not move the router: the thresholds below were tuned against the legacy
    # peak_util on the server, so swapping the statistic underneath them would
    # re-route instances wholesale on no evidence. OGC_CLASSIFY=v2 switches the
    # router onto max(legacy, refined) for the benchmark run that would earn
    # the change; either way the numbers are logged, so how far an instance
    # sits from a threshold is visible instead of implicit.
    util_layers = util_hall = util_energetic = fit_min = -1.0
    try:
        (util_layers, util_hall, util_energetic,
         fit_min) = _refined_features(blocks, bays, horizon)
    except Exception as exc:
        _optional_failure("refined congestion features", exc)
    route_util = peak_util
    if os.environ.get("OGC_CLASSIFY") == "v2" and util_layers >= 0.0:
        route_util = max(peak_util, util_layers, util_hall)

    if w1 <= 1000:
        group = "soft"
    elif route_util >= 0.95 or (p_med >= 20 and route_util >= 0.75):
        group = "overload-ext" if route_util >= 1.2 else "overload-mod"
    elif w3 >= 300 and r13 <= 12:
        group = "pref"
    elif route_util >= 0.45 or p_med >= 10:
        group = "congested"
    elif n <= 200:
        group = "light-small"
    else:
        group = "light-large"

    # How close this instance sits to flipping group. The thresholds are hard
    # edges -- 0.949 and 0.951 buy different portfolios for the same problem --
    # and nothing previously recorded when an instance was sitting on one, so a
    # hidden instance landing there was invisible. Logging the margin does not
    # remove the discontinuity, but it makes near-boundary cases identifiable
    # instead of silently high-variance.
    margin = min(abs(route_util - t) for t in (0.45, 0.75, 0.95, 1.20))

    feats = {"peak_util": peak_util, "p_med": p_med, "n": n,
             "w1": w1, "w3": w3, "r13": r13,
             "route_util": route_util, "util_layers": util_layers,
             "util_hall": util_hall, "util_energetic": util_energetic,
             "fit_min": fit_min, "margin": margin}
    return group, feats


def pick_knobs(prob_info, timelimit):
    """group -> knob preset for the single engine.

    Returns (group, feats, knobs) where knobs = dict(sisr=..., guides=bool).
    Uses the v4.20 routing evidence, but keeps a one-island SISR hedge for
    every non-pocket family so a novel hidden distribution cannot disable
    an entire search basin merely because one threshold fired.
      - overload + congested-heavy: SISR island3, guides on
      - the v47 pocket (non-pref, non-overload congested, peak>=0.70):
        SISR all + LAHC islands + guides OFF (the fork's portfolio as knobs)
    """
    try:
        group, feats = classify(prob_info)
    except Exception:
        return "unknown", {}, {"sisr": "hedge", "guides": True}

    # Robust default at 200-300 s: preserve three baseline islands and reserve
    # one island for the structurally different string-removal search. This
    # limits classifier regret on an unfamiliar hidden family to one core.
    knobs = {"sisr": "hedge", "guides": True}
    if group == "congested":
        pref_leaning = (feats["w3"] >= 300.0 or feats["r13"] <= 50.0)
        near_overload = feats["peak_util"] > 0.85
        pocket = (not pref_leaning and not near_overload
                  and feats["peak_util"] >= 0.70 and feats["n"] <= 300)
        if pocket and timelimit >= 45.0:
            knobs = {"sisr": "all", "guides": False}
        elif pref_leaning:
            knobs = {"sisr": "hedge", "guides": True}
        else:
            knobs = {"sisr": "island3", "guides": True}
    elif group in ("overload-mod", "overload-ext"):
        knobs = {"sisr": "island3", "guides": True}
    elif group == "pref":
        knobs = {"sisr": "island3", "guides": True}
    # light-small / light-large / soft keep the diversified default.
    return group, feats, knobs


# -----------------------------------------------------------------------------
# Feasibility checking: official shapely checker if importable, else _ogc's
# own exact-integer stack. Both return {"feasible": bool, "objective": float}.
# -----------------------------------------------------------------------------

def check_solution_dict(prob_info, sol):
    # The evaluation server overwrites/provides utils.py. It is the authority
    # for numerical polygon repair, rounding, crane replay, and objective
    # calculation, so use it before either shipped approximation.
    try:
        import utils
        res = utils.check_feasibility(prob_info, sol)
        if isinstance(res, dict):
            return res
        if isinstance(res, tuple):
            return {"feasible": bool(res[0]), "objective": None,
                    "violations": list(res[1:])}
        return {"feasible": bool(res), "objective": None,
                "violations": []}
    except ImportError:
        pass
    except Exception as exc:
        _optional_failure("organizer feasibility checker", exc)
    try:
        from _ogc.official_checker import check_feasibility
        return check_feasibility(prob_info, sol)
    except ImportError:
        pass
    except Exception:
        pass
    try:
        from _ogc.model import Problem
        from _ogc import feas
        prob = Problem(prob_info)
        s = feas.parse_output(prob, sol)
        ok, reason = feas.check_solution(prob, s)
        obj = float(s.objective()[0]) if ok else None
        return {"feasible": bool(ok), "objective": obj,
                "violations": [reason] if not ok else []}
    except Exception as exc:
        _optional_failure("internal feasibility checker", exc)
        return {"feasible": None, "objective": None, "violations": []}


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def algorithm(prob_info, timelimit=60):
    """Entry point: knobs + (at long limits) the best-of-2 time hedge.

    At timelimit >= 4*baseline_tl (default 220 s) run the solver once at
    baseline_tl, then again with the remaining budget, and return the
    checker-arbitrated best -- the server-proven variance insurance for long
    evaluations. Below that, a single run uses the whole budget."""
    t0 = time.time()
    try:
        timelimit = float(timelimit)
    except (TypeError, ValueError):
        timelimit = 60.0
    if not math.isfinite(timelimit) or timelimit <= 0.0:
        timelimit = 60.0

    log = None
    if os.environ.get("OGC_LOG"):
        name = prob_info.get("name", "?") if isinstance(prob_info, dict) else "?"
        log = lambda msg: print(f"[ogc {name}] {msg}", flush=True)

    try:
        baseline_tl = float(os.environ.get("OGC_BASELINE_TL", "55"))
        if not (baseline_tl == baseline_tl) or baseline_tl <= 0:
            baseline_tl = 55.0
    except (TypeError, ValueError):
        baseline_tl = 55.0
    # baseline hedge OFF by default: measured on data_test 4/5/6 at
    # official TLs (2 rolls) the baseline candidate never won and skipping
    # it was +1.68% total (prob_6 all-time best 26.34M); the fallback
    # ladder still guarantees feasibility. OGC_BASELINE=1 re-enables.
    use_baseline = (os.environ.get("OGC_BASELINE", "0") == "1"
                    and timelimit >= 4.0 * baseline_tl)

    # long-limit pocket instances adopt the fork's looser recreate budgets.
    # Measured on data_test/prob_4@480 (2 rolls/arm): CAP 4/2 closed a -14%
    # gap vs the v47 engine to -2%; disabling nest/chain made it worse.
    # These are read per run at LNS construction, so they only have to hold
    # for the duration of the solve -- solve_config() guarantees exactly that
    # and unwinds them afterwards, so a failure here cannot leak either.
    cap_t = _CAP_T_ENV0
    cap_w1 = _CAP_W1_ENV0
    try:
        _g0, _f0, _k0 = pick_knobs(prob_info, timelimit)
        # measured: 4/2 wins at 480s (data_test/prob_4 -14% -> -2%) but
        # LOSES at 600s on train pockets (prob_35 310k -> 267k with 2/1,
        # prob_21 296k -> 271k). Keep 4/2 only in its measured window.
        pocket_caps = (_k0.get("sisr") == "all"
                       and 220.0 <= timelimit <= 520.0)
        if cap_t is None and pocket_caps:
            cap_t = "4"
        if cap_w1 is None and pocket_caps:
            cap_w1 = "2"
    except Exception as exc:
        # a failed classifier must not select pocket caps by accident
        _optional_failure("pocket-cap classification", exc, log)

    with solve_config({"OGC_CAP_T": cap_t, "OGC_CAP_W1": cap_w1}, log):
        if not use_baseline:
            return _solve_once(prob_info, timelimit, t0, log)

        candidates = []
        try:
            if log:
                log(f"hedge: baseline phase tl={baseline_tl:.0f}s")
            candidates.append(_solve_once(prob_info, baseline_tl,
                                          time.time(), log))
        except Exception as exc:
            _optional_failure("final feasibility safety net", exc, log)
        try:
            remaining = timelimit - (time.time() - t0)
            candidates.append(_solve_once(prob_info, remaining,
                                          time.time(), log))
        except Exception:
            if not candidates:
                from _ogc.ogc_fallback import fallback_solve
                return fallback_solve(prob_info)
        if len(candidates) == 1:
            return candidates[0]
        best, best_obj = None, None
        for sol in candidates:
            try:
                res = check_solution_dict(prob_info, sol)
                if res["feasible"] and res["objective"] is not None:
                    if best_obj is None or res["objective"] < best_obj:
                        best, best_obj = sol, res["objective"]
            except Exception:
                continue
        if log and best_obj is not None:
            log(f"hedge: selected obj={best_obj:,.0f}")
    return best if best is not None else candidates[-1]


def _solve_once(prob_info, timelimit, t0, log):
    """Classify, then run one solve under a configuration scoped to it."""
    group, feats, knobs = pick_knobs(prob_info, timelimit)
    # BRKGA order-search on the congested/overload BOUNDARY (v5.9 adoption,
    # results/logs/v58c_laneA.log 2026-08-08): the group default leaves BRKGA
    # off for group=congested, but at peak_util just under the 0.95 overload
    # cutoff construction quality already decides the cell -- s2_10 @0.94:
    # gate won 3/3 paired rolls, mean -6.6%. Two cells DOWN-band refuted the
    # blanket gate (s2_33 @0.84: 0/3, up to +19% -- order search burns budget
    # the LNS needs), so the band stops at 0.90. Same contract as the other
    # knob bands: explicit env override (snapshotted at import) always wins.
    if _BRK_ENV0 is not None:
        brkga = _BRK_ENV0
    elif (group == "congested"
          and 0.90 <= feats.get("peak_util", 0.0) < 0.95):
        brkga = "1"
    else:
        brkga = None
    cfg = {
        "OGC_GROUP": group,
        "OGC_BRKGA": brkga,
        "OGC_V417_SISR": (os.environ.get("OGC_V417_SISR_FORCE")
                          or knobs["sisr"]),
    }
    # scoped, so these cannot survive the solve that selected them -- see
    # solve_config(). Every one of them is read per run by the engine.
    with solve_config(cfg, log):
        return _solve_body(prob_info, timelimit, t0, log, group, feats, knobs)


def _solve_body(prob_info, timelimit, t0, log, group, feats, knobs):
    guides_on = knobs["guides"] and not os.environ.get("OGC_NO_GUIDE")
    if log:
        log(f"group={group} sisr={knobs['sisr']} guides={guides_on} "
            f"peak={feats.get('peak_util', -1):.2f} n={feats.get('n', -1)} "
            f"tl={timelimit:.0f}s")
        log(f"congestion: legacy={feats.get('peak_util', -1):.3f} "
            f"layers={feats.get('util_layers', -1):.3f} "
            f"hall={feats.get('util_hall', -1):.3f} "
            f"energetic={feats.get('util_energetic', -1):.3f} "
            f"fit_min={feats.get('fit_min', -1):.2f} "
            f"margin={feats.get('margin', -1):.3f}")

    # Cache VALIDITY, not mere presence: a rejected cache must fall back to
    # the cold budget (min_main 25, no sprint auto-guide) instead of paying
    # ~15s of JIT while budgeting as if it had paid ~2s.
    cached = _CACHE_LIVE
    try:
        min_main = float(os.environ.get("OGC_MIN_MAIN",
                                        "12" if cached else "25"))
        if not (min_main == min_main):
            min_main = 12.0 if cached else 25.0
    except (TypeError, ValueError):
        min_main = 12.0 if cached else 25.0
    if timelimit < min_main:
        from _ogc.ogc_fallback import fallback_solve
        return fallback_solve(prob_info)

    # Dense day arrays are fast for normal shipyard horizons, but allocating
    # them up to an arbitrary billion-scale release date is neither safe nor
    # useful. Such legal extreme-date inputs take the sparse pure-Python
    # fallback; all ordinary inputs use an instance-derived dense horizon.
    try:
        _bs = prob_info.get("blocks", [])
        dense_need = (max((int(b["release_time"]) for b in _bs), default=0)
                      + sum(max(1, int(b["processing_time"])) for b in _bs)
                      + max((max(1, int(b["processing_time"])) for b in _bs),
                            default=1) + 16)
        dense_limit = max(8192, int(os.environ.get("OGC_DENSE_TCAP_MAX",
                                                   "2000000")))
        if dense_need > dense_limit:
            if log:
                log(f"sparse fallback: required horizon {dense_need:,} > "
                    f"dense limit {dense_limit:,}")
            from _ogc.ogc_fallback import fallback_solve
            return fallback_solve(prob_info)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _optional_failure("dense-horizon profiling", exc, log)
        from _ogc.ogc_fallback import fallback_solve
        return fallback_solve(prob_info)

    if timelimit < 40.0:
        return _sprint(prob_info, t0, timelimit, log, cached, guides_on)

    # Without a live cache, solve_parallel() pays ~15s of numba JIT serially
    # just before it forks. Everything between here and there is CP-SAT
    # (ortools holds no GIL), so start the JIT now and let it finish inside
    # that window instead of after it; warmup() in solve_parallel then blocks
    # on numba's own compile lock rather than compiling. Pure overlap, no
    # change to what gets compiled. Skipped when the cache serves the kernels
    # anyway. OGC_JIT_OVERLAP=0 restores the serial order.
    if not cached and os.environ.get("OGC_JIT_OVERLAP", "1") != "0":
        try:
            import threading
            from _ogc.ogc_kernels import warmup as _warm
            threading.Thread(target=_warm, daemon=True).start()
        except Exception:
            pass

    # reserve time for final feasibility check + output safety
    n = len(prob_info.get("blocks", []))
    reserve = min(25.0, max(4.0, 0.06 * timelimit + n / 100.0))
    t_end = t0 + timelimit - reserve

    # geometry-free assignment relaxation (CP-SAT): optimal target bays used
    # to bias insertion proposals. On spatially loose groups the EXACT
    # assignment optimum (no capacity term needed) replaces the capacity-
    # priced approximation; it also provides a certified lower bound and an
    # endgame candidate (see ogc_assign).
    targets = None
    assign_pack = None
    loose = (group in ("light-small", "light-large")
             and feats.get("peak_util", 1.0) < 0.45
             and os.environ.get("OGC_ASSIGN", "1") != "0")
    if loose:
        try:
            from _ogc.ogc_assign import exact_assignment
            bud = min(2.0, 0.03 * timelimit)
            pure = exact_assignment(prob_info, time_budget=bud, workers=4)
            # K-aware capacity arms: with 3-4 layers the crane liftability
            # rules slash achievable packing density far below the area
            # bound (measured on data_test/prob_3: eta 0.72 leaves 100+
            # tardy days, 0.52 still 22), so deep arms are needed
            try:
                maxK = max(len(s["layers"]) for b in prob_info["blocks"]
                           for s in b["shape"])
            except Exception:
                maxK = 2
            if maxK <= 2:
                etas = (0.85, 0.72)
            elif timelimit < 180:
                etas = (0.72, 0.62)
            else:
                etas = (0.72, 0.62, 0.52)
            arms = []
            for eta in etas:
                r = exact_assignment(prob_info, time_budget=bud, workers=4,
                                     eta=eta)
                if r is not None:
                    arms.append((r[0], r[3]))
            if pure is not None:
                assign_pack = (pure[1],
                               arms if arms else [(pure[0], None)])
                # guide with the most-achievable arm (tightest capacity);
                # fall back to the pure optimum
                targets = list(arms[-1][0] if arms else pure[0])
                if log:
                    log(f"exact assignment bound={pure[1]:,.0f} "
                        f"proven={pure[2]} arms={len(arms)} "
                        f"t={time.time()-t0:.1f}s")
        except Exception as exc:
            _optional_failure("exact assignment guide", exc, log)
            assign_pack = None
    # Plan-and-Realize for DENSE instances at long limits: solve the joint
    # assignment + cumulative area-scheduling plan (tardiness allowed) at two
    # density levels and feed (targets, sdays) to the locked-realization
    # island + endgame. Measured (data_test @ official TLs): incumbents
    # behave like eta~0.60 and the eta-0.70 plan schedule was 2-8x better
    # than the incumbent on prob_4/prob_5 -- the headroom lives here.
    # Gates: only when guides are ON (the plan IS a guide, and the pocket's
    # defining evidence is that guides add tardiness there -- measured:
    # plan on data_test/prob_4 (pocket) -4%, on prob_5 (overload, guided)
    # +7% and a new all-time best), and only for maxK <= 2 (K=4 area plans
    # are unrealizable under the crane rules: data_test/prob_6@900 -3.8%,
    # 2 rolls, and the prob_3 locked-island post-mortem).
    try:
        _maxK = max(len(s["layers"]) for b in prob_info["blocks"]
                    for s in b["shape"])
    except Exception:
        _maxK = 4
    # group gate: plan-off probe recovered congested prob_29/39 @600s to
    # exact Arseniy parity (+12.6%/+6.6%), while the plan's only proven win
    # is overload-mod (test file5 +8.9%) -> overload groups only
    if (group in ("overload-mod", "overload-ext")
            and timelimit >= 180.0 and guides_on and _maxK <= 2
            and os.environ.get("OGC_PLAN", "1") != "0"):
        try:
            from _ogc.ogc_assign import cumulative_plan
            petas = (0.68, 0.58)
            pbud = min(15.0, 0.04 * timelimit)
            # solve the arms in PARALLEL threads (CP-SAT releases the GIL):
            # sequential arms cost 2x pbud of pure prelude serialization
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as _ex:
                futs = [_ex.submit(cumulative_plan, prob_info, eta,
                                   pbud, 2) for eta in petas]
                results = [f.result() for f in futs]
            arms = []
            best_plan = None
            for r in results:
                if r is not None:
                    arms.append((r[0], r[3]))
                    if best_plan is None or r[1] < best_plan:
                        best_plan = r[1]
            if arms:
                assign_pack = (0.0, arms)
                if log:
                    log(f"cumulative plan arms={len(arms)} "
                        f"best_plan_obj={best_plan:,.0f} "
                        f"t={time.time()-t0:.1f}s")
        except Exception as exc:
            _optional_failure("cumulative plan guide", exc, log)

    if targets is None and guides_on:
        try:
            from _ogc.ogc_guide import compute_targets
            targets = compute_targets(
                prob_info, time_budget=min(6.0, 0.08 * timelimit), workers=4)
            if log:
                log(f"guide {'ok' if targets is not None else 'unavailable'} "
                    f"t={time.time()-t0:.1f}s")
        except Exception as exc:
            _optional_failure("bay guide", exc, log)
            targets = None

    try:
        isl_tl = float(os.environ.get("OGC_ISLANDS_TL", "55"))
        if not (isl_tl == isl_tl):
            isl_tl = 55.0
    except (TypeError, ValueError):
        isl_tl = 55.0

    # day-level tardiness targets (per-bay scheduling relaxation) on top of
    # the bay assignment
    day_targets = None
    if (targets is not None and timelimit >= isl_tl
            and os.environ.get("OGC_DAYGUIDE", "1") == "1"):
        if os.environ.get("OGC_GLOBALGUIDE", "1") == "1":
            try:
                from _ogc.ogc_globalguide import compute_global_targets
                day_targets = compute_global_targets(
                    prob_info, time_budget=min(6.0, 0.05 * timelimit),
                    deadline=t0 + min(15.0, 0.25 * timelimit))
                if log:
                    log(f"globalguide "
                        f"{'ok' if day_targets is not None else 'n/a'} "
                        f"t={time.time()-t0:.1f}s")
            except Exception as exc:
                _optional_failure("global day guide", exc, log)
                day_targets = None
        if day_targets is None:
            try:
                from _ogc.ogc_dayguide import compute_day_targets
                day_targets = compute_day_targets(
                    prob_info, targets,
                    time_budget=min(4.0, 0.03 * timelimit), workers=4)
                if log:
                    log(f"dayguide "
                        f"{'ok' if day_targets is not None else 'n/a'} "
                        f"t={time.time()-t0:.1f}s")
            except Exception as exc:
                _optional_failure("per-bay day guide", exc, log)
                day_targets = None

    try:
        _drtl = float(os.environ.get("OGC_DAY_RESOLVE_TL", "600"))
        if not (_drtl == _drtl):
            _drtl = 600.0
    except (TypeError, ValueError):
        _drtl = 600.0
    day_resolve = (day_targets is not None and timelimit >= _drtl
                   and os.environ.get("OGC_DAY_RESOLVE", "0") == "1")

    n_extra = 3 if timelimit >= isl_tl else 0
    try:
        if n_extra > 0:
            from _ogc.ogc_solve import solve_parallel
            obj, sol = solve_parallel(prob_info, t_end, n_extra, log=log,
                                      hard_deadline=t0 + timelimit - 2.0,
                                      targets=targets,
                                      day_targets=day_targets,
                                      day_resolve=day_resolve,
                                      assign_pack=assign_pack)
        else:
            from _ogc.ogc_solve import solve_single
            obj, sol = solve_single(prob_info, t_end, seed=20260711, log=log,
                                    targets=targets,
                                    day_targets=day_targets)
    except Exception:
        import traceback
        if log:
            log(traceback.format_exc()[-500:])
        from _ogc.ogc_fallback import fallback_solve
        sol = fallback_solve(prob_info)

    # final safety net: verify, repair, guaranteed-feasible fallback
    try:
        t_left = timelimit - (time.time() - t0)
        if t_left > 3.0:
            res = check_solution_dict(prob_info, sol)
            if log:
                log(f"final check feasible={res['feasible']} "
                    f"obj={res['objective']}")
            if res["feasible"] is False:
                sol, ok = _repair(prob_info, sol, res,
                                  deadline=t0 + timelimit - 2.0, log=log)
                if not ok and timelimit - (time.time() - t0) > 0.3:
                    from _ogc.ogc_fallback import fallback_solve
                    sol = fallback_solve(prob_info)
    except Exception as exc:
        _optional_failure("sprint feasibility safety net", exc, log)

    return sol


def _sprint(prob_info, t0, timelimit, log, cached=False, guides_on=True):
    """Short-time-limit pipeline: construct + whatever LNS fits."""
    n = len(prob_info.get("blocks", []))
    reserve = max(3.5, 0.12 * timelimit + n / 150.0)
    t_end = t0 + timelimit - reserve

    targets = None
    sg = os.environ.get("OGC_SPRINT_GUIDE", "auto")
    auto_ok = cached and timelimit >= 20.0 and n <= 500
    if guides_on and (sg == "1" or (sg != "0" and auto_ok)):
        try:
            import threading
            from _ogc.ogc_kernels import warmup as _warmup
            threading.Thread(target=_warmup, daemon=True).start()

            box = {}

            def _guide():
                try:
                    from _ogc.ogc_guide import compute_targets
                    box["t"] = compute_targets(
                        prob_info,
                        time_budget=min(2.0, 0.06 * timelimit), workers=4)
                except Exception as exc:
                    box["t"] = None
                    box["error"] = exc

            gt = threading.Thread(target=_guide, daemon=True)
            gt.start()
            gt.join(timeout=min(4.0, 0.15 * (timelimit - reserve)))
            targets = box.get("t")
            if "error" in box:
                _optional_failure("sprint bay guide", box["error"], log)
            if log:
                log(f"sprint guide {'ok' if targets is not None else 'n/a'} "
                    f"t={time.time()-t0:.1f}s")
        except Exception as exc:
            _optional_failure("sprint guide orchestration", exc, log)
            targets = None

    try:
        from _ogc.ogc_solve import solve_single
        obj, sol = solve_single(prob_info, t_end, seed=20260711, log=log,
                                targets=targets, jit_deadline=t_end - 1.0)
    except Exception:
        import traceback
        if log:
            log(traceback.format_exc()[-500:])
        from _ogc.ogc_fallback import fallback_solve
        return fallback_solve(prob_info)

    try:
        t_left = timelimit - (time.time() - t0)
        if t_left > 2.5:
            res = check_solution_dict(prob_info, sol)
            if log:
                log(f"sprint check feasible={res['feasible']} "
                    f"obj={res['objective']} t={time.time()-t0:.1f}s")
            if res["feasible"] is False:
                from _ogc.ogc_fallback import fallback_solve
                if timelimit - (time.time() - t0) > 4.5:
                    sol, ok = _repair(prob_info, sol, res,
                                      deadline=t0 + timelimit - 1.5, log=log)
                    if not ok:
                        sol = fallback_solve(prob_info)
                else:
                    sol = fallback_solve(prob_info)
    except Exception as exc:
        _optional_failure("solution repair", exc, log)
    return sol


def _repair(prob_info, sol, res, deadline, log=None):
    """Push violating blocks to guaranteed-feasible empty-bay windows using a
    fresh state rebuilt from the solution."""
    import numpy as np
    from _ogc.ogc_prep import Prep
    from _ogc.ogc_state import State

    ok = False
    try:
        if time.time() > deadline - 3.0:
            return sol, False
        prep = Prep(prob_info)
        state = State(prep, np.random.default_rng(1))
        ops = sol.get("operations", {})
        info = []
        exits = {}
        for t_str in sorted(ops, key=lambda s: int(s)):
            for op in ops[t_str]:
                if op["type"] == "ENTRY":
                    info.append((op["block_id"], op["bay_id"], int(t_str),
                                 op["x"], op["y"], op["orient_idx"]))
                else:
                    exits[op["block_id"]] = int(t_str)
        for (i, b, t, x, y, o) in info:
            state.insert(i, b, t, o, x, y, t2=exits.get(i))

        for _ in range(6):
            if time.time() > deadline:
                break
            bad = []
            seen = set()
            for v in res.get("violations", []):
                for tok in v.split("block ")[1:]:
                    try:
                        bid = int(tok.split()[0].rstrip(":,"))
                    except (ValueError, IndexError):
                        continue
                    if bid not in seen:
                        seen.add(bid)
                        bad.append(bid)
            if not bad:
                break
            for i in bad:
                if state.placed[i]:
                    state.remove(i)
            for i in bad:
                if time.time() > deadline:
                    return sol, ok
                done = False
                for b in range(prep.m):
                    if not prep.block_fits[i, b]:
                        continue
                    t = state.day_upper_bound(i, b)
                    r = state.try_day(i, b, t)
                    if r is not None:
                        o, px, py = r
                        state.insert(i, b, t, o, px, py)
                        done = True
                        break
                if not done:
                    return sol, ok
            if time.time() > deadline - 1.0:
                return sol, ok
            sol2 = state.build_operations()
            res = check_solution_dict(prob_info, sol2)
            if log:
                log(f"repair pass: feasible={res['feasible']}")
            sol = sol2
            ok = bool(res["feasible"])
            if ok:
                break
    except Exception:
        pass
    return sol, ok
