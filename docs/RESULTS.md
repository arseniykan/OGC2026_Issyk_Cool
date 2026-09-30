# Results

## Final standing

**3rd of 958 teams, total score 284 — 2nd Prize.**

| rank | team | score |
|---:|---|---:|
| 1 | tishii24 | 311 |
| 2 | TheFactoryMustGrow | 297 |
| **3** | **Issyk Cool** | **284** |
| 4 | MSS | 278 |
| 5 | 최적화야호 | 277 |

Field: 958 teams / 1,448 participants / 55 countries entered; 40 advanced to the final
round, 39 were ranked. 462 submissions were made in the final round, 438 accepted. Our
rank over the last seven daily snapshots was 3→3→4→4→3→3→3 — never outside the top four.

Scoring is **per-instance rank**, not objective sum: each of the eight instances awards
points by your rank on that instance, so the total rewards being good everywhere rather
than excellent somewhere. Per-slot point breakdowns are in the Organizing Committee's
final results report.

In the preliminary round the same lineage placed **3rd of 408** with 2,432 points, 8
points behind first — a margin equal to exactly one instance slot.

## The scored board

Submission `20260812_Team_v10.6`, 8/8 feasible. Objective values:

| instance | TL (s) | bays | blocks | w1 | objective | gap to field best |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 180 | 3 | 200 | 3,333 | 1,794,442 | +14.3% |
| P2 | 300 | 3 | 250 | 6,667 | 16,415,669 | +11.4% |
| P3 | 480 | 3 | 250 | 3,333 | 4,265,813 | +8.0% |
| P4 | 180 | 3 | 150 | 13,559 | 3,288,730 | **+1.6%** |
| P5 | 240 | 2 | 150 | 6,667 | 5,171,244 | +18.8% |
| P6 | 600 | 5 | 300 | 333 | 794,938 | +8.5% |
| P7 | 360 | 5 | 300 | 6,667 | 14,365,981 | +17.3% |
| P8 | 360 | 4 | 200 | 13,333 | 14,916,568 | **+23.6%** |

Mean gap to the best value any team recorded on each instance: **12.9%**.

These instances are in [`data/finals/`](../data/finals) as `fin_1.json` … `fin_8.json`.

### Where the architecture is competitive — and where it isn't

**P4, +1.6%, our best slot.** TL 180 s, 150 blocks, and `w1 = 13,559` — the highest in
the set. That is the congested, tardiness-dominated, short-budget regime the compiled
kernel was built for. We owned the same slot in the preliminary round, where our P4 value
was the best of any team's final submission.

**P8, +23.6%, our worst slot.** TL 360 s, 4 bays, 200 blocks, `w1 = 13,333`. We
identified P8 as the weak slot internally a day before the close and never found the
lever. It correlates with no other instance in the set (|r| ≤ 0.22), which is why no
amount of work on the slots we *could* move ever reached it — and it is the single
clearest piece of unfinished business in this solver.

## The robustness re-run

After the round closed, every team's final submission was re-run on the same eight
instances at 0.70× and 1.33–1.50× the official time limits:

| time budget | our rank | our score |
|---|---:|---:|
| 0.70× | 3rd | 286 |
| 1.00× (official) | 3rd | 284 |
| 1.33–1.50× | 4th | 281 |

Our standing **rises as the budget tightens and falls as it loosens** — one competitor
passes us on the long set. Two readings, both of which we think are true:

1. The compiled kernel and the final-round speed work pay most where time is scarcest.
2. The remaining deficit is **quality per iteration, not iterations per second**. Extra
   time helps our competitors more than it helps us, so the neighborhood and the guidance
   are the gap, not the geometry engine.

This is consistent with what we measured internally: scheduling was nearly closed against
our own bounds, and packing density was the prize we didn't fully claim.

## Submissions do not buy score

We made 17 submissions (16 accepted), the 7th-most in the field. The winning team made
**10**. One team made 23 and finished 18th. The mean was 11.23 per ranked team.

The reason is a noise floor we measured directly: re-sending a *byte-identical* zip moved
our mean gap by 1.97 percentage points — larger than the spread across eight genuinely
different builds. Below roughly 2pp a single submission resolves nothing, so most of the
information in a resend is variance, not signal.

## Reproducing

```bash
python run.py --suite finals
```

**Use a 4-core Linux host.** The island pool requires `fork`; on spawn-only platforms
`_ogc/ogc_solve.py` falls back to a single in-process island by design, which costs
roughly a quarter of the search. Two measured points:

| instance | scored (4-island Linux) | single-island Windows | gap |
|---|---:|---:|---:|
| `prelim_1` | 11,280 | 11,280 | 0% |
| `fin_4` | 3,288,730 | 4,127,504 | +25% |

Light instances are insensitive — one island already closes `prelim_1`. Congested ones are
not, and the platform gap there dwarfs any algorithmic difference you might be trying to
measure.

Beyond that, expect run-to-run variation of 1–2% on the long overload instances and 9–16%
on the short preference-weighted ones. The solver is stochastic and multi-process, and
those coefficients of variation are measured, not hypothetical. Single runs do not
distinguish builds in this problem; see the noise-floor note above.
