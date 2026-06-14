# Calibration verification (before running): the tick detector reads UI text, not the ruler

Verification of the `tick_calibration.py` + calibrated-MT path before spending a Kaggle run.
Done by re-running the detector locally (203/309 detected, 53 at confidence >= 0.7,
px/mm median 15.3) and reading the QA overlays. Conclusion: **not ready to run/submit** — the
calibration is unreliable on the PNG family, which is where ~43 of the ~53 calibrated-MT rows
come from. The segmentation half and the wiring (confidence gate, prior fallback, clip,
audit CSV) are fine.

## Evidence (PNG family, the German-UI device)

- `IMG_00252.png`, depth readout "Tiefe **4.0 cm**": detector picked `side_ticks/right`,
  `spacing=67px`, `tick=5mm`, `px/mm=13.40`, `conf=0.83`. The detection strip sits on the
  **right-side UI text panel** (Bildfrequenz / Verstärk / Tiefe / L12-4 / BewApp / dB), whose
  text rows are ~67px apart. The actual ruler is on the **left** ("0 … 2 … 4cm").
- `IMG_00280.png`, depth "Tiefe **3.0 cm**": same device, detector again returns the identical
  `67px -> px/mm=13.40`.
- The identical value across a 4cm scan and a 3cm scan is the giveaway. True scales differ
  (eyeballing the left rulers: ~15 px/mm at 4cm depth, ~20 px/mm at 3cm depth). A constant
  13.40 means the detector keys off the depth-INDEPENDENT text panel, not the ruler. Per-family
  numbers agree: PNG high-confidence px/mm clusters exactly at 13.40.

## Why it would hurt the score

`MT_mm = mt_px / px_per_mm`. 13.40 is too low, so calibrated MT is biased high: roughly +13% on
4cm images and **+50% on 3cm images**. MT tolerance is only 3mm, so on the shallow-depth PNGs the
calibrated MT is likely WORSE than the 18.628 prior. "Confidence" does not protect against this:
it scores tick regularity, and regular UI text rows score high while being wrong scale.

## Concrete fixes (Codex owns tick_calibration.py)

1. **Exclude the right-side UI text panel** from PNG side-candidates, or reject candidates whose
   "ticks" coincide with the text column. Right now both sides are searched and the most regular
   wins, so the text panel beats the ruler.
2. **Use the two clean PNG signals instead:**
   - Parse the **"Tiefe X.X cm" depth text** (OCR or template match) -> depth_mm. Then
     `px/mm = ultrasound_panel_height_px / depth_mm`.
   - And/or detect the **left numbered ruler** span (0 to N cm) -> `px/mm = span_px / (N*10)`.
3. **Make it depth-aware / validate scale:** a px/mm that is constant across images of different
   depth is a red flag. Cross-check any candidate against the depth text; reject if the implied
   panel height disagrees with depth by more than a tolerance.
4. **Confidence should reflect scale plausibility**, not only tick regularity.

## TIFF family

Only 10/251 reached conf >= 0.7 (one bottom-tick value, 13.45). Most TIFFs are cropped and lack a
visible ruler UI, so they correctly stay at the prior MT. That family needs a different scale
source (sequence borrowing, or the DL-Track-scale research) and is lower priority than fixing PNG.

## Recommendation

Hold the Kaggle run until PNG calibration reads the ruler/depth rather than the text panel, then
re-check the overlays (the red tick lines should land on the left ruler and px/mm should track the
depth readout). The segmentation training itself is unaffected and ready; only the MT-calibration
input needs the fix. Overlays for inspection: `results/calibration_debug/overlays/`.

## Follow-up fix applied

`tick_calibration.py` now has a PNG-specific `png_left_ruler` path. For `.png` images it ignores
the right-side UI text panel and reads the extreme-left numbered ruler instead. Re-running the
local diagnostics gives the expected depth-aware behavior:

- `IMG_00252.png` (4.0 cm): `75 px / 5 mm = 15.0 px/mm`, method `png_left_ruler`.
- `IMG_00280.png` (3.0 cm): `100 px / 5 mm = 20.0 px/mm`, method `png_left_ruler`.
- PNG split: `58/58` detected, `58/58` above confidence `0.7`.
- TIFF split remains cautious: only `10/251` above confidence `0.7`, so most TIFFs still fall
  back to the prior MT until a separate scale source is found.

The current calibration status is therefore: PNG MT calibration is ready for a cautious
confidence-gated Kaggle run; TIFF calibration is still mostly fallback.
