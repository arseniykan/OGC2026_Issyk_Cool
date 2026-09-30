# =============================================================================
#  OGC 2026 -- The Grand Shipyard Puzzle
#  Team   : Issyk Cool
#  Authors: Arseniy Kan, Alina Akhmetbek
#  Date   : 2026-08-12
# =============================================================================

"""Scalar area proxy shared by the assignment and day relaxations.

WHY THIS EXISTS.  All three relaxations (ogc_assign, ogc_dayguide,
ogc_globalguide) reduce the spatial problem to "area occupied on a day must
fit inside eta * bay area", and all three measured that area on LAYER 0 alone.
Layer 0 is the wrong layer to measure.

The layer collision constraint is per level: two blocks conflict when their
layer-l polygons overlap, for any l. So the level that binds is the LARGEST
one, not the lowest. A block with a narrow base and a wide upper deck is
charged almost nothing by a layer-0 model while consuming its full upper-deck
area against every other block's matching layer -- and blocking the crane over
all of it, since the crane rule tests a block's layers against every layer at
or above them.

Measured on the training set (max-layer area / layer-0 area, per block):
    prob_5   mean 1.62   max 20.20   (mean 2.99 layers)
    prob_12  mean 1.15   max  3.74   (mean 1.85 layers)
    prob_13  mean 1.16   max  4.36   (mean 1.83 layers)
So layer 0 can understate a single block by more than an order of magnitude,
and understates the typical dense instance by 15-60%. That is a systematic,
one-directional error: the relaxation always believes more fits than does,
which is exactly the shape of the observed failure -- a mathematically
attractive plan that the geometric constructor then cannot realise.

The code base already knew: ogc_solve picks the "most REALIZABLE" locked arm
by probing, commenting that "area-based eta overestimates what multi-layer
crane rules allow", and deeper eta discounts are applied to multi-layer
instances. Those are compensations for this measurement error. Measuring the
binding layer addresses the cause instead of discounting the symptom.

ORIENTATION is deliberately NOT searched here: orientations are rigid
transforms, and layer areas across them are identical to four decimal places
on every training instance (max/min ratio 1.0000), so orientation 0 is exact
for an area statistic. Shape-dependent packability differs by orientation, but
no scalar area proxy can express that either way.

MODE, via OGC_RELAX_AREA:
  "layer0"   (default) -- the shipped behaviour, kept because eta was tuned
                          against it and changing the demand underneath a
                          tuned capacity coefficient re-scales every guide at
                          once; that needs a benchmark run to adopt, not an
                          assertion that the new number is better founded.
  "maxlayer"            -- the binding layer, as argued above.
"""
import math
import os


def _shoelace(verts):
    a = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def mode():
    m = os.environ.get("OGC_RELAX_AREA", "layer0").strip().lower()
    return m if m in ("layer0", "maxlayer") else "layer0"


def layer_area(layers, how=None):
    """Scalar area for one orientation's layer list."""
    lays = [v for v in layers if v]
    if not lays:
        return 0.0
    if (how or mode()) == "maxlayer":
        return max(_shoelace(v) for v in lays)
    return _shoelace(lays[0])


def block_area(block, orient=0, how=None):
    """Scalar area proxy for a block, from one of its orientations."""
    try:
        return layer_area(block["shape"][orient]["layers"], how)
    except (KeyError, IndexError, TypeError):
        return 0.0


def demand(block, orient=0, how=None):
    """Integer cumulative-demand value (>= 1) for a CP-SAT area model."""
    return max(1, int(round(block_area(block, orient, how))))


__all__ = ["mode", "layer_area", "block_area", "demand", "_shoelace", "math"]
