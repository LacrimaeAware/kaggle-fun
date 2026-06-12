# UMUD Master Review (rewritten 2026-06-10)

The single clean source of truth for the project. Built from a 5-agent source-grounded audit of the
code, docs, CSVs, and experiments. If another doc disagrees with this one, this one is right.

---

## RESUME HERE (last touched 2026-06-11, before a multi-day break)

**Best LB: 0.61918 (safe baseline, already your live score).** Production defaults are safe
(`UMUD_FL_FACING=0`, `UMUD_FL_IDENTITY_BLEND=0`); a fresh run reproduces the baseline.

**What changed most recently:** built a human-in-loop labeling lab (`benchmark_lab/`) and hand-labeled
**19 of 24 real TEST images** (apo + fascicle masks, in gitignored `results/human_benchmark/`). This is
the first local oracle on the *actual test distribution* — the thing that fixes our core problem (the
35-expert benchmark is different devices and mispredicted the LB 4 times). Labels are "rough/first-pass"
(user's words), useful for triage, not yet final truth.

**Submission decision this session (no slot spent):** HOLD the blind probes. We are 0-for-4 on isolated
LB probes, and we just built the oracle that should gate the next one. Do NOT rush facing-per-gap before
the break. The one defensible single-slot play, if you want to use one, is the **IMG_00275 scale fix**
(double-confirmed wrong: OCR 2× scale anomaly + the human label disagrees -10.6mm FL / +9.9° PA with the
shipped value) — but it is one image, so expect a small move at most.

**#1 task when you return** (this is the workflow you already designed in `benchmark_lab/NEXT_LABEL_PACK.md`):
1. Score the 0.619 baseline AND the facing / facing-per-gap candidates against the 19 human test rows
   (`benchmark_lab/score_labels.py` + `review_server.py`).
2. Wire **facing-FL per gap** (per-gap = multi-muscle separation only; FL via `compute_facing_fl()`, NOT
   the wave trace — see §6). Submit only if it improves the human rows without breaking sanity rows.
3. Bigger lever, harder: a better fascicle segmentation model (the real FL bottleneck — see §7).

---

## 0. THE NUMBER CLARIFICATION (read this first — it is the thing everyone keeps conflating)

There are **three different score scales**. All are tolerance-normalized MAE, lower=better, but they
measure **different things on different data and are NOT comparable to each other.**

| | Scale | Range | What it is |
|---|---|---|---|
| **A** | **Public LEADERBOARD** | **~0.46–0.67** (ours **0.619** best) | The real hidden 309-image test. Scale must be RECOVERED, FL bias is EXPOSED. **The only number that counts.** |
| **B** | **35-image BENCHMARK** | **~0.19–0.23** (ours 0.227) | Local CPU score vs 7 experts, fed **TRUE scale** and **FL recentered to the true mean**. Looks ~3x better than A *by construction* — it bypasses scale and hides FL bias. **A sanity tool, NOT an oracle** (it has mispredicted the LB direction 4 times). |
| **C** | **REFERENCE points** | 0.307–0.679 | Other people's scores; some on scale B, some on scale A — tagged below. |

**Rule: only compare A-to-A.** A 0.227 (B) is NOT better than 0.619 (A). They are different exams.
The project's single recurring mistake was treating a B-scale win as a reason to spend an A submission.

---

## 1. Leaderboard ladder (Scale A — the real number)

| LB | Change that produced it | Status |
|---:|---|---|
| 1.23135 → 1.11066 → 1.09194 | early pipeline (FL/MT mostly constants) | superseded |
| **0.61918** | **scale router + inner-edge MT + fragment-FL** | **BEST / SAFE BASELINE** |
| 0.62561 | MT vertical-3 (MT only) | rejected (bench win, LB loss) |
| 0.63905 | FL identity-blend (FL only) | rejected (bench win, LB loss) |
| 0.66459 | facing FL (`UMUD_FL_FACING=1`) | rejected (multi-muscle outliers — see §3) |
| 0.66711 | bar-only scale tail | rejected (worst probe) |

**The scale router moved the score (1.09→0.619). Nothing has beaten 0.619 since.** Every isolated change submitted has regressed.

## 2. Reference points (Scale C — tagged)

| Reference | Score | Scale | Note |
|---|---:|---|---|
| Human-vs-human floor | 0.307 | B | the ceiling; ~0.3 is unbeatable |
| DL-Track (true scale) | 0.331 | B | a good pipeline is already human-level on the 35-set |
| Public leader | 0.378 | ~A | sits on the human floor |
| By-hand labeling the test set (Patrick, 3rd) | 0.459 | **A** | careful human on the REAL board |
| DL-Track Kaggle benchmark | 0.679 | **A** | different exam — do NOT compare to the 0.331 |
| **Our best** | **0.619** | **A** | beats the 0.679 ref; behind 0.459 (by-hand) and 0.378 (leader) |

The 0.331 and 0.679 are the same tool on different exams. Never subtract across scales.

---

## 3. What's WIRED and banked (this is what HELPS — it's in the 0.619 pipeline)

| dim | wired method | gain |
|---|---|---|
| **Scale** | per-family tick/ruler router, 295/309 coverage | the score-mover, 1.09→0.619 |
| **MT** | inner-edge aponeurosis gap (not band centroids) | bench MT 0.49→0.18 |
| **PA** | weighted-PCA/TLS orientation + min-6°/min-40px filters + length-weighted median | bench PA 0.225→0.164, near human floor; hand-drawn lines confirm fit within **1.8°** |
| **FL** | fragment extrapolation (safe 0.619 baseline; facing/minimize-extrapolation is opt-in and rejected as-is) | bench FL 0.48→0.35 |
| **Seg** | two ResNet34 U-Nets + TTA (mirror+multiscale) | bench 0.383→0.370 |

PA and MT are effectively solved on geometry. Scale is solved and now *verifiable* (§5).

## 4. What HARMS / was rejected (regressed the LB or the bench)

| change | result | why |
|---|---|---|
| facing FL submission | LB 0.619→**0.665** | bend is real but the gate is wrong 41%; gain under tolerance; outliers + ~13 multi-muscle images regress (§6) |
| FL identity-blend | LB 0.619→0.639 | recentered-mean trap (bench win, LB loss) |
| MT vertical-3 | LB 0.619→0.626 | protocol-aligned ≠ LB win |
| bar-only scale tail | LB 0.619→0.667 | unvalidated 3cm bar assumption |
| recall-bias seg retrain | bench 0.368→0.484 | more fragments = noise; can't fix FL by drawing more pixels |
| CLAHE at inference | bench worse | train/test mismatch |

## 5. Scale — solved and verifiable, NOT the bottleneck

- Router covers **295/309**; the OCR×tick cross-check partitions all 309: **48 verified, 99 text-confirmed, 147 tick-only, 1 flag, 14 mean.**
- **Where a second independent read exists (148 images), OCR and ticks agree 147/148.** Scale is right where checkable.
- The lone disagreement, **IMG_00275**, is a caught silent 2x tick error (tick 201 vs printed ruler 101) — quarantined to the prior instead of shipping half-size FL/MT. A provably-correct one-image fix.
- **15 genuine fallbacks** (14 `mean` + IMG_00275) have no readable scale and honestly fall to the prior (incl. the mirrored IMG_00251).
- **The FL-mean recenter is a measured NO-OP** on the shipped pipeline: regenerating with `UMUD_FL_RECENTER=0` changes 0/309 rows (the per-image FL already averages 74.4). We already ship honest per-image FL.

## 6. FL geometry — the contested area, settled

Side-by-side on the 35 experts (true scale), `bench_fl_methods.py`:

| FL method | FL mean | bias | term |
|---|---:|---:|---:|
| straight, all fragments | 84.7mm | **+24** | 2.02 (overshoots off-frame) |
| **facing (consensus + parabola + minimize-extrapolation)** | 61.5mm | **+0.7** | **0.26** (zero bias) |
| per-gap wave/bend | 84.5mm | +25 | 2.1 (same overshoot — dropped minimize-extrapolation) |

- **The overshoot (straight FL +24mm) is the disease; minimize-extrapolation is the cure.** Facing packages it with consensus angle + facing-parabola to reach zero bias.
- **Facing is GOOD on the bench (zero bias) but REGRESSED the LB** because on ~13 multi-muscle images its gate produces garbage, and its opposite-bend gate is wrong 41% of the time on independent (FALLMUD) apos.
- **THE CONCEPTUAL KEY:** the competition's FL ground truth is the human raters' **STRAIGHT-LINE extrapolation measurement** (a convention, with minimize-extrapolation), **NOT the true curved anatomy.** The fascicle bend is anatomically real (confirmed on FALLMUD: parabola fits GT apos 44% better) but it **diverges from the scored convention**, which is exactly why curving the wave doesn't help the metric and below-tolerance. **Chase the rater's convention (straight + low-extrapolation), not the anatomy.**

## 7. PA / MT / Segmentation (the dimensions that keep getting dropped)

- **PA: solved.** 0.15, near the human floor; hand-drawn lines confirm the fitted angle within 1.8°. Nothing left to pick.
- **MT: strong (0.18).** Inner-edge gap was the win. Residual error is *scale*, not geometry (the IMG_00275 2x bug is the one real MT scale error).
- **Segmentation: the real bottleneck.** The FL gap to DL-Track (0.47 vs 0.31) is **fascicle MASK quality**, not geometry. Post-processing can't fix it (recall-bias and CLAHE both failed). The "unseen-device domain shift" hypothesis was disproven (the test is NOT a separate image class). It needs a genuinely better fascicle model — better training/data/architecture.

---

## 8. VERDICT: what helps, what harms, what's neutral

- **HELPS (banked in 0.619):** scale router, inner-edge MT, weighted-PCA PA + filters, TTA, fragment-FL baseline.
- **HARMS (rejected):** facing-FL submission, FL blend, MT vertical-3, bar-only scale tail, recall-bias retrain, CLAHE inference.
- **NEUTRAL / no-op:** FL-mean recenter (no-op on shipped pipeline), mean-drop (no-op), per-gap wave/bend FL (worse than facing — the bend is real but diverges from the scored convention and is below tolerance).
- **UNTESTED with real potential:** facing + per-gap multi-muscle fix (could recover facing's zero-bias gain *without* the multi-muscle regression); a better fascicle segmentation model (the real bottleneck).

## 9. FORWARD PLAN

1. **Floor:** `Downloads/0P61918_submission_local.csv` (LB 0.619) is the safe fallback. The production code now defaults to `UMUD_FL_FACING=0` and `UMUD_FL_IDENTITY_BLEND=0`, so a fresh run keeps the safe fragment-FL baseline unless a rejected probe is explicitly enabled.
2. **Build more benchmarks before burning more geometry submissions.** `benchmark_lab/` now contains a manifest builder, Cintiq/browser labeler, and scorer. Seed manifests were generated under `results/human_benchmark/`: `public_seed_manifest.csv` for public/FALLMUD labels and `target_seed_manifest.csv` for declared human-in-loop target labels. This is the new decision layer: label enough cases that the 0.619 baseline and rejected 0.665 facing variant can be distinguished locally.
3. **The one geometry shot left:** wire **facing-FL per gap** (per-gap = multi-muscle separation only; FL = the facing method, NOT the wave trace). This targets exactly the ~13 multi-muscle images that regressed facing. Do not submit it until the new human benchmark says it fixes those failures.
4. **The real lever (bigger, harder):** better fascicle segmentation. FL is mask-limited, not geometry-limited. This needs model work (more/better fascicle training data, a stronger fascicle model), not more post-processing. The new benchmark labels are the validation set for that GPU work.
5. **Stop:** chasing the fascicle bend (real but diverges from the scored convention, below tolerance); submitting on benchmark improvement alone (the benchmark is a sanity tool, mispredicts the LB).
6. **Free wins available:** apply the OCR scale to IMG_00275 (provably-correct, one image); the verified/text-confirmed scale partition gives confidence the rest are right.
