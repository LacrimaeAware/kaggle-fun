# UMUD expert benchmark: what the measured ground truth tells us

You downloaded the OSF expert-analysed benchmark (`data/osfstorage-archive/`, gitignored).
The useful payload is `Expert Analysed Benchmark Image Datasets/benchmark_dataset_architecture_v0.1.0.zip`
(extracted to `data/osf_arch_benchmark/`, also gitignored): **35 muscle-architecture images** plus
`Results_benchmark_architecture_v0.1.0.xlsx`. The aponeurosis/fascicle benchmark *model* folders in
the archive came down empty, so we did not get DL-Track's pretrained segmenters from this download.

## What the spreadsheet contains (per image, 35 rows)

- `Scale_pixel_per_cm` - the **true scale** for each image (range 74-126, so scale varies ~1.7x).
- `R1..R7_{MT,FL,PA}` - all **seven experts'** thickness/length/angle measurements (in mm/mm/deg).
- `DLTrack_{MT,FL,PA}` - **DL-Track's own automated output** on these images (with correct scale).
- `SMA_{MT,FL,PA}` - the SMA tool's automated output.
- Methodology (Readme): ImageJ, 3 thickness lines + 3 fascicles + 3 angles per image, averaged;
  muscles GM/soleus/VL; four devices (Hitachi Aloka, Telemed EchoBlaster, Philips HD11, Telemed ArtUs).

## The scored picture (tolerance-normalized error vs the expert consensus, on these 35)

| | UMUD-style score | per-target detail |
| --- | ---: | --- |
| Human floor (one careful expert vs the other six) | **0.307** | this is the irreducible noise |
| DL-Track, correct scale | **0.331** | PA 1.45 deg, FL 3.74 mm, MT 1.31 mm |
| SMA tool | 0.409 | PA 2.39 deg, FL 7.01 mm, MT 0.74 mm |
| Kaggle DL-Track benchmark (reference) | 0.679 | DL-Track run WITHOUT per-image scale |
| Public leader (reference) | 0.378 | |
| Us (reference) | 1.092 | |

Inter-expert spread (the noise): PA SD ~1.6 deg, FL SD ~5.4 mm, MT SD ~1.0 mm. FL is the noisiest
even for experts; MT the most reliable. This matches the reliability literature, now measured.

## What this establishes (evidence, not speculation)

1. **The ceiling is about 0.3.** A careful human expert "scores" 0.307 against the other experts.
   You essentially cannot beat ~0.3, because that is how much the human ground truth disagrees with
   itself. The public leader at 0.378 is sitting right on this floor. (Caveat: the hidden test is
   different images, same methodology, so these are strong estimates, not exact transfers.)
2. **DL-Track with correct scale is already human-level (0.331).** The tool, done right, is at the
   floor. So the leader is plausibly just a well-calibrated DL-Track-style pipeline, nothing exotic.
3. **The Kaggle benchmark (0.679) is DL-Track done WITHOUT scale.** The factor-of-two gap between
   0.331 (correct scale) and 0.679 (Kaggle) is almost entirely **per-image scale**. This is now the
   grounded version of the earlier calibration claim: calibration is the dominant lever, because the
   same tool moves from 0.68 to 0.33 just by getting pixels-to-mm right.
4. **We are at 1.092 because we have not built scale or FL.** FL is constant on all 309 rows, MT on
   251. The road from 1.09 toward ~0.4 is scale + real FL/MT, exactly what DL-Track-with-scale shows
   is reachable.
5. **Our tick detector does not yet generalize.** On these 35 it returned a scale on 17/35 and was
   about 2x off (it assumes 5 mm side ticks; these devices use 1 cm ticks), at low confidence. The
   value is that we now have the true scale to fix it against.

## Why this also answers the "can we even iterate locally" worry

Yes. We now have **35 images with true measured PA/FL/MT and the true scale**, plus DL-Track's and
SMA's outputs for reference. That is a local validation bench: run our geometry/calibration on these
images, score against the experts, and iterate - on CPU, in seconds, no training per loop. The slow
GPU training is a once-in-a-while step; the oracle/visual/measurement iteration runs locally against
real numbers. The loop the project needs does not depend on fast local training.

## Immediate uses

- Build `benchmark_validate.py`: given our predicted PA/FL/MT for these 35, print the UMUD-style
  score vs experts next to the DL-Track and human-floor references. That is our local scoreboard.
- Fix calibration against truth: the `Scale_pixel_per_cm` column lets us learn the real tick spacing
  per device and stop assuming 5 mm. The detector being 2x off is a solvable, now-measurable bug.
- Study DL-Track's per-image outputs (already in the sheet) to see where the benchmark tool matches
  or misses experts, without running anything.
