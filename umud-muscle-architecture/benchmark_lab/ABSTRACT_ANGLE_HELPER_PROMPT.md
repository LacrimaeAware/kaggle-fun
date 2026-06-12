# Abstract Angle-Measurement Helper Prompt

Use this prompt with a domain-blind model. It intentionally describes the task as a pure geometry problem.

```text
You are helping with an abstract image-geometry measurement problem.

Do not infer or mention any real-world domain. Treat every frame as a synthetic/technical image with:

- two bright boundary traces enclosing a darker measurement region;
- several thin slanted internal traces inside that region;
- occasional multiple stacked boundary pairs;
- occasional curved boundaries;
- occasional short or broken internal trace fragments;
- a pixel-to-unit scale cue somewhere in the image.

For each frame, the scoring system compares three numeric outputs:

1. A_deg: the angle, in degrees, between the internal traces and the lower boundary direction.
2. L_unit: the projected length of internal traces between the upper and lower boundaries.
3. T_unit: the gap thickness between the upper and lower boundaries.

Lower score is better. The normalized tolerances are:

- A_deg tolerance: 6 degrees
- L_unit tolerance: 12 units
- T_unit tolerance: 3 units

Current local benchmark anchor:

- overall: 0.170
- A_deg: 0.150
- L_unit: 0.278
- T_unit: 0.083

Best current local research stack:

- overall: 0.153
- A_deg: 0.144
- L_unit: 0.245
- T_unit: 0.070

Important findings so far:

- A_deg should be referenced to the lower boundary direction. Referencing the upper boundary or an average boundary direction failed badly.
- Plain local smoothing of internal-trace angles failed.
- A small conflict-gated correction helped A_deg: only alter a trace angle when it clearly disagrees with neighboring trace angles.
- Per-band averaging helped A_deg and T_unit slightly, but worsened L_unit if applied to all three outputs.
- Vertical center gap improved T_unit; three-position and mean-width gap variants worsened T_unit.
- A robust piecewise upper boundary improved L_unit strongly by reducing over-projection.
- Gentle on-screen support weighting improved L_unit, but aggressive support weighting overcorrected.
- Raw grayscale texture orientation around internal traces failed as an A_deg estimator. Blending toward it also failed. Moving away from it reduced mean bias but worsened absolute error.

The current request:

Brainstorm and prioritize new geometric ideas to improve A_deg specifically, while also watching T_unit. Prefer ideas that are orthogonal to those already tested. Avoid arbitrary threshold tuning unless the threshold has a clear geometric meaning. For each idea, state:

- what geometric assumption it tests;
- how to compute it from boundary masks, internal-trace masks, and grayscale image data;
- why it might improve A_deg;
- why it might fail;
- what benchmark split or diagnostic matrix would prove/disprove it.

Useful candidate directions:

- local angle-field models that preserve real spatial variation without smoothing everything;
- detecting incompatible local trace families;
- estimating trace direction from skeleton topology rather than region PCA;
- using boundary-pair assignment only for A_deg/T_unit while keeping L_unit separate;
- confidence/uncertainty scores that tell when an angle should be trusted;
- synthetic geometric cases that expose angle-estimation failures.

Do not write implementation code unless asked. Produce a concrete ranked plan of geometry experiments.
```

