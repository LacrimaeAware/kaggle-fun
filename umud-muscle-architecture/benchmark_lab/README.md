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

## Labeling Protocol

Use the same convention every time:

1. Draw **aponeurosis** on the `apo` layer.
   Mark the visible aponeurosis structure that bounds the measured muscle region. If the visible
   structure is a band rather than a one-pixel line, fill the visible band. The measurement code will
   use the muscle-facing inner edge.

2. Draw **fascicles** on the `fasc` layer.
   Trace only visible fascicle fragments. Do not extrapolate them to the aponeuroses by hand; the
   geometry code should do that. If there are many fragments, trace the clearest low-extrapolation
   ones first.

3. Use **ignore** for ambiguity.
   Mark overlay text, unresolvable shadow, or regions where the correct structure is genuinely unclear.
   The first scorer records ignore coverage but does not yet mask it out of every metric.

4. Save notes.
   Put uncertainty, multi-gap cases, scale oddities, or anything you want a later model to know in the
   notes field. This matters more than it feels like in the moment.

5. Keep target labels separate.
   If an image comes from `test_images_v2`, keep the manifest and saved label folder separate from
   public/external packs. We should be able to delete or ignore the target-human branch and still have
   a clean automated/no-oracle project state.

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
