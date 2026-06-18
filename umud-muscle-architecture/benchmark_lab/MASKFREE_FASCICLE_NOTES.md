# Mask-free fascicle measurement: exploration notes

> **(2026-06-17) Two corrections, read both.**
> (1) The 35-benchmark is leaked: 21/35 images are in the U-Net training set, so the pipeline's 0.90 deg
> PA there is inflated. Do not trust the 35-benchmark ranking.
> (2) I then over-claimed a "held-out flip" off the GM calf-raise video (pipeline 10.66 deg, raw walk
> 4.99 wins), THEN over-claimed "the GM labels are wrong." Both WALKED BACK. The GM Readme says the
> labels are careful MANUAL expert analysis (3 raters, custom Matlab, every 2nd frame). Pennation
> genuinely changes during a calf raise, PA is vs the moving deep apo (I used a fixed straight line),
> and band_tex_angle disagrees with raters by ~5 deg even on the TRUSTED 35-benchmark. So my texture
> measure is too crude to judge the GM labels, and I cannot use the GM video to rank methods either way.
> NET: the 35 is leaked, the GM video is unjudgeable with my tools, so mask-free-vs-pipeline on held-out
> data is UNRESOLVED. The only per-image TEST truth that can rank methods is the user's correction-UI
> labels. (I over-concluded 3x on this dataset; stop turning crude one-offs into verdicts.)



Goal: measure fascicle PA and FL inside the muscle band directly from the raw grayscale, instead of
from the sparse U-Net fascicle mask. The band (the two aponeurosis lines) still comes from the existing
pipeline; this only replaces the fascicle half. Motivation: PA/FL is the weak term and the user wanted a
first-principles geometry route ("follow the bright ridges like a river"). Code: `field_fascicle.py`,
harness `run_field_eval.py`. All scored on the 35-image expert benchmark (true scale, PA/FL truth).

## Methods tried and benchmark result (PA/FL error in tolerances, lower better)

| method | PA(tol) | FL(tol) | bias |
|---|---:|---:|---|
| pipeline (mask + PCA), reference | 0.150 | 0.522 | near the human floor |
| structure-tensor field + streamline | ~0.49 | ~1.1 | PA too FLAT (~12-15 vs truth 17-24) |
| windowed Radon field | ~0.49 | ~1.1 | too flat |
| blob (ridge -> threshold -> per-blob PCA), weighted | 0.486 | 1.423 | too flat |
| field span+ridge+exclude | 0.815 | 2.680 | flat + FL blow-up |
| brightness walk (block pool + quantize + momentum cone) | wMED PA ~31-38 | short FL | PA too STEEP |
| radar walk (fan-of-rays matched filter, wMED), full 35 | 0.512 | 1.207 | fixed the steep bias; ~3 deg PA error, same band as the flat blob |
| radar walk wMEAN, full 35 | 0.614 | 2.319 | |
| human floor | 0.245 | 0.403 | |

## What we learned (the durable findings)

- **The two classical extremes bracket the truth.** Windowed-orientation methods (structure tensor,
  Radon, blob PCA) read PA too flat; the brightness walk reads PA too steep; the truth sits between,
  which is where the existing mask-PCA already is. None of the mask-free methods beat the pipeline on
  the benchmark.
- **Why flat:** any window/blob that pools a 2D region averages across several stacked fascicles plus
  the dark gaps, so the pooled orientation comes out flatter than any single fascicle (the
  two-collinear-sticks-merged-flat effect).
- **Why steep:** the greedy "step to the brightest neighbor" walk climbs toward bright structure
  (speckle columns, aponeurosis edges) in short steep hops and locks steep early; momentum/cone did not
  save it, and size-weighting did not either (the big bright paths are steep too).
- **FL is downstream of PA.** FL = thickness / sin(PA), so a flat PA gives huge FL and a steep PA gives
  tiny FL. Get PA right and FL follows; there is no independent FL lever here.
- **The mask-free methods are not independent of each other** (all windowed-orientation cousins), so
  fusing them adds nothing; they share the flat bias. A genuinely different estimator (the walk, or the
  radar step below) is what would make fusion meaningful.

## Implementation bugs found (external audit) and status

- Harness did not test the intended configs (span/ridge/exclude). FIXED: `run_field_eval.py` now runs
  the blob method and the intended field config and scores them reproducibly.
- `apo_exclude` was applied AFTER the structure tensor, so it never removed apo texture from the tensor
  integration. Still true for the field method; the blob and walk methods do not use the tensor so it
  does not apply to them.
- Spanning FL was unbounded (flat PA -> explosive FL, lines off-screen). FIXED: every candidate is
  weighted by its on-screen fraction (a line 90% off-screen counts 10%), and overlay lines are clipped
  to the scan rectangle.
- Blind spots (a U-Net-segmented fascicle with no line). FIXED in the blob method: every U-Net fascicle
  blob is added as its own blob, kept separate so it cannot merge and flatten.
- Weighting: each candidate is weighted by 2D size (blob area, or walk path bright-mass) times its
  on-screen fraction; PA/FL are weighted medians (and weighted means for the walk).

## Validation discipline (unchanged)

The benchmark does not predict the leaderboard and is a different distribution. Use these methods as a
QA cross-check, not a submission gate. The only per-image test truth is the user's correction-UI labels.

## Radar step result (done)

The radar step (fan of candidate rays, integrate brightness along each over length L, lock onto the
max-average direction) DID fix the steep bias: on 4 spot images PA looked great (im_29 20.0 vs 20.4).
But on the FULL 35 it lands at PA 0.512 / FL 1.207, the same band as the flat blob (0.486), about 3 deg
of PA error and ~3x the pipeline (0.150). The spot images flattered it. So the radar is the cleanest
mask-free version but it does not break the ~0.5-tol ceiling the whole family hits, and it does not
reach the pipeline or the human floor.

**Robust conclusion (4 genuinely different methods, full set):** mask-free fascicle measurement
converges to ~0.5 tol PA error and does not beat the mask-PCA pipeline (0.150). As a standalone it
loses; as a fusion partner a 3-deg estimator only drags the 0.9-deg pipeline. Keep it as a QA
cross-check, not a replacement or fusion input. The real remaining levers are FL extrapolation and
test-domain segmentation, neither of which mask-free addresses.

## Radar refinements tried (full 35)

- **Continuity stop** (ray stops at the first dark cell, no jumping over gaps): cleaned up the walks
  (less hopping between fascicles) but did NOT move the aggregate (0.512 -> 0.521). The jumping was a
  secondary issue; the dominant error is a flat-structure lock on ~6 images.
- **Middle-band focus + curve rejection** (inset both apos by 30% of band height; drop walks whose
  chord/arc < 0.85): made it clearly WORSE (0.521 -> 0.925 PA, FL 4.3). The user predicted this. The
  failing images are misreads, not aponeurosis-contamination, so cutting the band to the middle removed
  signal. Both knobs reverted to off (params kept in `measure_walk`, default 0).
- Best mask-free result stays the continuity radar at ~0.521 PA / 1.43 FL, still ~3x the pipeline.

## The combine (segmentation + walk) -- TESTED, it works

The walk is soft hand-segmentation and it is bad at exactly what learning is good at: connecting broken
fascicle stretches and ignoring non-fascicle structure. Combine = let the trained net say what is a
fascicle, let the walk connect/extrapolate.

**No-GPU proxy tested (full 35):** run the walk on raw brightness GATED by the existing fascicle U-Net
mask (gray * (0.15 + 0.85 * blurred-fascicle-mask)). Result: **PA 0.521 -> 0.397 (3.13 -> 2.38 deg),
FL 1.434 -> 0.931**, and the flat-failure tail collapsed (images >6 deg: 6 -> 1; <=2 deg: 13 -> 17).
Gating starved the connective tissue, exactly the hypothesis. Still behind the pipeline (0.150) and the
benchmark does not predict the LB, so it is a measurement-quality result, not a submission. But it is
the first time combining with segmentation produced a real jump, which validates the direction.

**Next (GPU):** replace the binary-mask gate with a trained per-pixel direction + confidence field
(cos 2theta, sin 2theta, conf) on the fascicle masks, and run this same walk on it. The gating proxy
already closed ~40% of the raw-walk-to-pipeline gap; the learned field should do better. Code path:
`measure_walk` already accepts any field as its `gray` arg, so only the field-builder changes.

## Ideas parked

- **Orientation-filter bank (learned endpoint).** The radar step is hand-built oriented matched filtering;
  the learned version is a small conv layer / oriented-filter bank that outputs (cos 2theta, sin 2theta,
  confidence) per pixel, trained on the fascicle masks. Cleaner than hand-tuning, and the natural "ML
  beyond segmentation" play, but note the mask-PCA orientation is already near the floor, so the gain
  would be test-domain robustness, not a benchmark win.
- Global non-crossing / coherent-river constraint: a separate, harder global optimization; not part of
  the per-fascicle radar idea.
