# UMUD Benchmark Lab

This folder is for building our own local validation sets instead of treating Kaggle submissions as
the only oracle.

The goal is to create small, clean, auditable benchmark packs:

- **public/external packs** from already-local public datasets such as FALLMUD or the OSF benchmark;
- **declared human-in-loop packs** from competition target images, if we choose that route and
  disclose it as external data per `competition_reference.md`.

These are different modes. Public/external labels are ordinary validation data. Target-image human
labels are not "secretly automated" just because a model proposed the case first; they are human-created
target information and must be logged as a declared human-in-loop branch.

## Quick Start

Create a public/external manifest:

```powershell
python umud-muscle-architecture\benchmark_lab\make_manifest.py `
  --fallmud 24 `
  --target 0 `
  --out umud-muscle-architecture\results\human_benchmark\public_manifest.csv
```

Create a target-image manifest, only if we intentionally choose declared human-in-loop mode:

```powershell
python umud-muscle-architecture\benchmark_lab\make_manifest.py `
  --fallmud 0 `
  --target 24 `
  --out umud-muscle-architecture\results\human_benchmark\target_manifest.csv
```

Start the labeling app:

```powershell
python umud-muscle-architecture\benchmark_lab\label_server.py `
  --manifest umud-muscle-architecture\results\human_benchmark\public_manifest.csv `
  --out-dir umud-muscle-architecture\results\human_benchmark\labels `
  --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Score saved labels:

```powershell
python umud-muscle-architecture\benchmark_lab\score_labels.py `
  --manifest umud-muscle-architecture\results\human_benchmark\public_manifest.csv `
  --labels-dir umud-muscle-architecture\results\human_benchmark\labels `
  --out umud-muscle-architecture\results\human_benchmark\scores.csv
```

Review labels against candidate submissions:

```powershell
python umud-muscle-architecture\benchmark_lab\review_server.py --port 8767
```

Open:

```text
http://127.0.0.1:8767
```

## Labeling Protocol

Use the same convention every time:

1. Draw **aponeurosis** on the `apo` layer.
   Mark the visible aponeurosis structure that bounds the measured muscle region. If the visible
   structure is a band rather than a one-pixel line, fill the visible band. The measurement code will
   use the muscle-facing inner edge.

   Practical shortcut: drawing the gap-facing boundary line is fine. You do not need a perfect fill of
   the whole band. Multiple strokes are acceptable; the scorer groups split strokes into upper/lower
   boundaries by height. The main thing to avoid is accidentally connecting the upper and lower
   boundaries into one blob, or drawing extra unrelated boundary pieces.

2. Draw **fascicles** on the `fasc` layer.
   Trace only visible fascicle fragments. Do not extrapolate them to the aponeuroses by hand; the
   geometry code should do that. If there are many fragments, trace the clearest low-extrapolation
   ones first.

3. Use **ignore** for ambiguity.
   Mark overlay text, unresolvable shadow, or regions where the correct structure is genuinely unclear.
   The first scorer records ignore coverage but does not yet mask it out of every metric. Ignore is
   optional; use it only when you want to warn later tools not to trust a region.

4. Leave manual measurement boxes blank unless you intentionally measured them.
   PA = angle, FL = length, MT = thickness. The scorer derives these from your masks. The optional
   boxes are only for rare overrides or notes from another measurement tool.

5. Save notes.
   Put uncertainty, multi-gap cases, scale oddities, or anything you want a later model to know in the
   notes field. This matters more than it feels like in the moment.

6. Keep target labels separate.
   If an image comes from `test_images_v2`, keep the manifest and saved label folder separate from
   public/external packs. We should be able to delete or ignore the target-human branch and still have
   a clean automated/no-oracle project state.

## Drawing Tools

- **brush**: freehand drawing with pen or mouse.
- **dot line**: click a point, then click the next point; each click connects to the previous point.
  This is usually the cleanest tool for straight visible fragments.
- **curve chain**: click anchor points along the visible path. The active curve is redrawn as a smooth
  spline whenever you add a point, so a new point can gently re-bend the previous segment instead of
  creating a hard kink. Pending anchor markers are preview only; they are not saved into the mask.
- **eraser**: removes pixels from the active layer only.
- **save**: writes the current masks and notes to the selected labels folder. It does not submit
  anything to Kaggle. Reloading the page restores saved masks.
- **zoom**: use the top-bar `-`, `100%`, and `+` controls, or Ctrl/Cmd plus, minus, and zero.

## What This Enables

- Mask-quality benchmarks: predicted masks vs human masks.
- Measurement benchmarks: run the same geometry on human masks and predicted masks, then compare
  PA/FL/MT in pixel or millimeter space where scale is known.
- Active learning, honestly named: choose the next images by uncertainty, but record the resulting
  corrections as human-in-loop data.
- GPU fine-tuning with real validation: train on public/external masks, validate on a human pack, and
  only submit if it improves the benchmark we actually care about.

## Current Next Pack

Recommended first pack:

- 12 target images from the known failure families: multi-gap/facing failures, scale anomalies, low raw
  support, and odd rulers;
- 12 FALLMUD/public images with existing masks, to sanity-check label convention and scorer behavior;
- no leaderboard submission until the pack can distinguish the 0.619 baseline from the rejected
  0.665 facing-FL variant.
