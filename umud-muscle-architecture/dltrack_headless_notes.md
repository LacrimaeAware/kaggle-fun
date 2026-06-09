# DL-Track-US: scale handling, headless batch path, and data/labels

Research note (my lane in the current split; Codex owns calibration code in
`tick_calibration.py` / `mt_calibration_submission.py`). All line numbers are from a shallow
clone of `github.com/PaulRitsche/DL_Track_US` (main, read 2026-06-08); the example/model
bundle is `DL_Track_US_example0.3.0.zip`, i.e. the 0.3.x line that the 0.67944 benchmark
("DLTrack_0.3.1") comes from. Quotes are verbatim from the source.

## TL;DR for the team

- DL-Track's pixels-to-mm step is dead simple and worth copying exactly:
  `mm = px * spacing / calib_dist`, where `calib_dist` is the detected pixel gap of a scale
  and `spacing` is the known mm value. If `calib_dist` is falsy, **outputs stay in pixels**.
- Its *automatic* calibration only looks at the **right 15%** of the image and needs the
  operator to declare `spacing` (5/10/15/20 mm). Our `.png` family has its ruler on the
  **left**, so DL-Track's auto-scale reads the wrong region there. That is a concrete,
  code-level reason the benchmark cannot calibrate all 309 images and leaves score on the table.
- It *can* run headless (no GUI) via `calculateBatch(...)`, but the function is GUI-coupled
  (needs a stub `gui` object and `tkinter.messagebox` neutralized) and only does straight-line
  fascicles. Output is an `Results.xlsx` with one sheet per parameter, not a tidy per-image row.
- The public DL-Track release ships **only segmentation masks** as labels (apo + fascicle).
  There is **no public table of measured PA/FL/MT**. That weakens the "leader trained a direct
  mm-regressor on real labels" hypothesis and shifts weight onto calibration.

## (a) How the scale is set and used

Conversion (the formula to reuse), `gui_helpers/do_calculations.py:696-701`:

```python
unit = "pix"
# scale data
if calib_dist:
    fasc_l = fasc_l / (calib_dist / int(spacing))
    midthick = midthick / (calib_dist / int(spacing))
    unit = "mm"
```

So `px_per_mm = calib_dist / spacing`, and `value_mm = value_px / px_per_mm`. If `calib_dist`
is `None`/0 (scaling set to "None"), fascicle length and thickness are reported **in pixels** —
which on this leaderboard would be catastrophic, so the benchmark almost certainly used a
non-None scaling.

Three scaling modes, `DL_Track_US_GUI.py:406-447` (default is `None`, spacing default 10 mm,
the Calibrate button is enabled only for Manual):

- **Bar (automatic)** — `gui_helpers/calibrate.py:57-75`, `calibrateDistanceStatic(img, spacing)`:

  ```python
  imgscale = img[int(height*0.4):height, (width - int(width*0.15)):width]   # right 15%, lower 60%
  calib_dist = np.max(np.diff(np.argwhere(imgscale.max(axis=1) > 150)))      # max gap of bright rows
  ```

  It assumes scale bars on the **right** side, takes the largest pixel gap between bright rows,
  and pairs it with the user-declared `spacing`. (Note: docstring says "median" gap but the code
  uses `np.max` — a quirk, not important for us.)

- **Manual** — `DL_Track_US_GUI.py:651-685`: the user clicks two points, Euclidean pixel
  distance becomes `calib_dist` for the declared `spacing`:

  ```python
  self.calib_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
  ```

- **None** — no scaling; results stay in pixels.

Implication for us: the conversion math is trivial and identical to what Codex's tick detector
needs. The only hard part is getting `px_per_mm` per image automatically, which DL-Track does
*not* solve well for heterogeneous layouts (right-side-only, fixed spacing). Our recon shows the
`.png` images carry an explicit depth readout ("Tiefe 4.0 cm") plus a left ruler, and the `.tif`
images have right-edge ticks — so per-family detection beats DL-Track's single fixed rule.

## (b) Headless batch path

Entry point, `gui_helpers/calculate_architecture.py:523-535`:

```python
def calculateBatch(
    rootpath, apo_modelpath, fasc_modelpath, flip_file_path,
    file_type, scaling, spacing, filter_fasc, settings, gui, image_frame=None,
) -> None:
```

Its own docstring: "designed to be executed from a GUI," scope limited to vastus lateralis,
tibialis anterior, soleus, gastrocnemius. The UMUD test images are vastus lateralis
("vl rechts" / "Rectus VL" overlays), so they are **in scope**.

To run it without a GUI:

- Pass a **stub `gui`** object exposing the attributes/methods it touches (`should_stop`,
  `is_running`, `do_break`, and any progress hooks). Confirmed coupling at
  `calculate_architecture.py:720-747`.
- **Neutralize `tkinter.messagebox`** (monkeypatch `showinfo`/`showerror` to no-ops, or run under
  a virtual display). The curve mode calls `showinfo("...not yet fully implemented")` and error
  paths pop dialogs; on a headless Kaggle box those would crash without a display.
- Only **`approach="linear_extrapolation"`** runs; `curve_polyfitting` aborts via `gui.do_break`.
  So the benchmark is straight-line geometry, and curved-fascicle modeling remains a genuine,
  unclaimed improvement (matches `strategy_brief.md` lever 4).
- Two **Keras `.h5`** models are loaded with `tf.keras.models.load_model` in the `ImageProcessor`
  (`calculate_architecture.py:99-114`): one aponeurosis U-Net, one VGG16-encoder fascicle U-Net.
- Output: `exportToExcel` writes `Results.xlsx` with **one sheet per parameter** indexed by
  filename, **columns per fascicle** (`do_calculations.py:740-747`) — not a tidy
  `image_id,pa_deg,fl_mm,mt_mm` row. A reproduction would need to aggregate (median per image)
  into the 309-row submission.

Practical verdict: a DL-Track reproduction on Kaggle is feasible (TF/Keras + the OSF `.h5`
models + stubbed `gui` + messagebox monkeypatch + `scaling="Bar"`), but it would **reproduce
~0.679, not beat it**, because the calibration is the weak link and it is exactly what the Bar
mode gets wrong on our images. Reproduction is only worth it as a sanity-check baseline; the
score lever is fixing calibration ourselves.

## (c) Models, training data, and labels

- **Hosting:** models, example media, and training image+mask pairs are all on a single OSF
  project, **osf.io/7mjsc**, as a downloadable zip (no figshare/Zenodo/Drive, no auto-download);
  the user manually selects the apo `.h5` and the VGG16 fascicle model. Cited:
  `docs/installation.md:14-15`, `docs/automated_image_analysis.md:58-69`.
- **Labels available = masks only.** The only training labels are paired images plus
  aponeurosis/fascicle **segmentation masks** (`docs/training_your_own_networks.md:23-38,158-200`);
  the ImageJ labeling macro outputs masks. There is **no public table of measured PA/FL/MT**
  (`docs/news.md:70-78`, `docs/index.md:9-13`); those numbers are computed at inference. The only
  place measured architecture values appear is an external **UMUD** benchmark dataset, i.e. the
  competition itself, not DL-Track.
- **License:** package code is Apache-2.0 (`pyproject.toml:16-21`); the **OSF data/model bundle
  has no explicit in-repo license**, so reuse terms are unstated (confidence medium). Training
  data covers only four lower-limb muscles on four ultrasound devices (`docs/index.md:52-58`) —
  a real domain-shift risk against UMUD's `.png` (different device) family.

## What this changes for our plan

1. **Validates the calibration-first track.** The px→mm math is `value_px * spacing / calib_dist`;
   Codex's detector only has to produce `px_per_mm` per image and we plug into the same formula.
   MT-first is right: thickness comes straight from the two aponeurosis bands we already segment.
2. **Down-weights the "external measured labels" hypothesis.** DL-Track exposes masks, not
   measured PA/FL/MT, so a direct mm-regressor trained on real labels is not available from the
   public DL-Track data. The leader's edge is more likely calibration (+ solid segmentation),
   not secret labels. (Caveat: he could have measured/sourced labels elsewhere; unverified.)
3. **Extra segmentation data, maybe.** DL-Track's apo/fascicle masks on OSF could augment our
   U-Net training, but only if the OSF bundle license permits and we document it. License is
   currently unstated — verify before use.
4. **Curved fascicles are open.** The benchmark only does straight-line extrapolation, so
   curve/spline fascicle length is a legitimate novelty lever, not just a reimplementation.

## Open / unverified

- Did not read the leader's actual UMUD notebook (Kaggle behind a browser check); the
  calibration story remains the leading hypothesis, not a confirmed fact.
- Whether DL-Track's models transfer to the UMUD `.png` device without fine-tuning is untested.
- OSF bundle license needs a direct check before any reuse of its masks/models.
- A reference clone sits outside the repo at `../_ref_DL_Track_US` (not committed) if anyone
  wants to re-verify line numbers.
