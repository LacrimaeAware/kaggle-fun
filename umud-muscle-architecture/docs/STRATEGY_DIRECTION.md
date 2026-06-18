# UMUD — strategy direction and data catalog (user-led rebuild)

Captured 2026-06-15 from the user's own direction, in the user's framing. This is the clean-methodology
rebuild that replaces the leaderboard-tuning approach (which is bracketed out at the 0.46041 floor, see
`CURRENT_STATE.md`). Datasets are kept here in full and are NOT discounted. Pair with `HANDOFF.md`.

## Stance

Stop tuning global multipliers. Build a genuine measurement method: read the image like a human,
recover the fascicle geometry directly as a coherent field, and classify what is being looked at. The
user is driving the design; this records it so it is not lost or second-guessed.

## Core ideas (user)

1. **Muscle / condition classification (user's #1, their specialty).**
   Train a model to identify the muscle (VL, GM, GL, TA, RF, ...) and likely condition subclass (e.g.
   cerebral-palsy features) from the ultrasound. Use the class to route measurement and to flag deviant
   features. Hypothesis: a condition like CP may share features across muscles (calf vs bicep), so
   subclass detection could surface the deviations that break naive measurement. This is where external
   data helps most.

2. **Read the image like a human (scale / OCR).**
   Recover scale by reading the printed numbers and ticks robustly. We got family_b scale ~10% wrong;
   scale is foundational (MT is mostly scale).

3. **Curvature via the "river flow" model (the central idea).**
   Treat every detected fascicle fragment as a streamline in a flow field. Reconstruct a global field
   where streamlines flow coherently and do not intersect (a river). Read curvature / the wave from the
   field, not from isolated fragments. Incorporate per-band logic: each aponeurosis band defines its own
   flow region; the field must respect band boundaries. This also answers "what is FL when there are
   several fascicles": the field's representative streamline (median/mean), not a hand-picked fragment.

4. **Force maximum extrapolation, then enforce consistency.**
   Detect every plausible bright ridge/groove (greedy, not picky), extrapolate them all, then constrain
   them to be mutually consistent within the field. Segmentation becomes a heuristic input, not the truth.

5. **Ridge-following / random-walk orientation estimate.**
   Within the muscle band only: sample seed points, step toward the brightest nearby pixel, repeat to
   trace a path, and accumulate the path's local angle. Median/mean the traversed angles for the dominant
   fascicle orientation. Fascicles are the bright structures, so the traced paths land on them. Stop at an
   aponeurosis or when brightness drops; avoid immediate backtracking.
   NOTE (real alignment, not flattery): this is the classical structure-tensor / Frangi-Sato ridge /
   Radon-Hough orientation extractor, which the repo's own research already ranked the #1 GPU-free lever
   ("reaches the manual noise floor; prefer Radon/Hough over a bare structure tensor"). The user arrived
   at it independently.

6. **Center-of-band measurement region.**
   Measure only in the central 30-50% of the muscle band (shrink the inter-aponeurosis region and fit it
   inside itself; a circle or rectangle also works). Fascicle behavior near the aponeuroses is deviant;
   the center is cleaner.

7. **Build our own segmentation; do not just consume the provided masks.**
   Compare our segmentation against the provided/expert masks rather than treating their masks or outputs
   as the target. The provided fascicle masks are sparse; our own ridge/flow detection may be a better
   front-end.

## Measurement-target note

FL with multiple fascicles = the host wants a representative value (median/mean over ~3 fascicles). The
flow field's representative streamline is the principled estimator, and per-band fields resolve the
multi-muscle ambiguity.

## Needs domain knowledge (user-flagged open questions)

- Which fascicle angles are trustworthy vs deviant (edge vs center).
- Band logic / multi-muscle separation.
- Texture/motion features that signal curving, flexing, or condition.

## Data catalog (all kept; nothing discounted)

Already have:
- **#1 UMUD 35-image benchmark** — xlsx scalars (7 raters PA/FL/MT) + true scale. No masks.
- **#2 DL-Track training masks** — these ARE our 2761 fascicle + 1048 aponeurosis expert masks.
- **#6 FALLMUD** — lower-leg, with expert aponeurosis masks (`data/dropoff/FALLMUD/NeilCronin/`).
- Unextracted: GM_dynamic and CSA_RF benchmark zips (`data/osfstorage-archive/Expert Analysed...`).

To fetch if pursued:
- **#3 Pohle-Fröhlich / Dalitz** — 425 frames, 3-expert fascicle ORIENTATION ground truth. Directly
  trains/validates the orientation-field idea (#3, #5). Highest-value new set for this direction.
- **#4 AnkleImage** — PA/FL/MT scalars; verify the annotation method (manual vs algorithm) before trusting.
- **#5 DUSTrack** — FL+PA, 3 raters x 3 days, medial gastrocnemius. Good rater-noise reference.

Two legitimate uses for data, both worth keeping all sets for:
- **Muscle/condition classifier (idea #1)**: more labeled muscles/conditions across ALL these sets help,
  even out-of-distribution ones, because the classifier learns class features.
- **Orientation / segmentation front-end (ideas 3-5, 7)**: #3 (orientation truth) and the mask sets
  (#2, #6) are the training/validation material.

Caveat to hold (not a reason to discount): these sets are mostly lower-leg + VL, while the competition
test adds rectus femoris, cerebral palsy, and Philips Lumify. They add volume and class coverage, not
direct coverage of those test-only domains. That argues for ALSO growing the correction-UI hand-labels on
the real test images, not for skipping the external data.

## Correction to the earlier framing (recorded)

The claim "we haven't checked whether segmentation is the bottleneck" was wrong-headed: fascicle-mask
Dice is ~0.29 (low) and the scale crop was ~10% off, both concrete evidence the front-end is weak. The
orientation-field rebuild attacks that directly and does not depend on first proving it with a
decomposition. The decomposition stays a cheap sanity check, not a gate.
