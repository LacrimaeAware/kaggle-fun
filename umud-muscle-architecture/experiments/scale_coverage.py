"""Coverage of the per-family scale router (scale_ticks.recover_for_image) over all 309 test images."""
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402

base = ROOT / "data" / "test_images_v2" / "test_set_v2"
fam = collections.defaultdict(lambda: {"n": 0, "ok": 0, "scales": [], "methods": collections.Counter()})
for f in sorted(base.iterdir()):
    a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if a is None:
        continue
    if f.suffix.lower() == ".png":
        key = "PNG(800x1200)"
    elif a.shape in [(800, 1200), (644, 1088)]:
        key = str(a.shape)
    else:
        key = "cropped" if a.shape[0] < 600 else "other-tif"
    r = fam[key]
    r["n"] += 1
    s, m, c = ST.recover_for_image(a, f.name)
    if s is not None:
        r["ok"] += 1
        r["scales"].append(s)
        r["methods"][m] += 1

print(f"{'family':>16} {'n':>4} {'scaled':>6} {'med px/cm':>10}  methods")
tot = to = 0
for k, r in sorted(fam.items(), key=lambda kv: -kv[1]["n"]):
    med = f"{np.median(r['scales']):.0f}" if r["scales"] else "--"
    print(f"{str(k):>16} {r['n']:>4} {r['ok']:>6} {med:>10}  {dict(r['methods'])}")
    tot += r["n"]; to += r["ok"]
print(f"{'TOTAL':>16} {tot:>4} {to:>6}  ({to * 100 // tot}% of test set scaled)")
