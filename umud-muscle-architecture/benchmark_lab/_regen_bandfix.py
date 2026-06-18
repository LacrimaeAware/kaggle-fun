import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "umud-muscle-architecture")
import segment_then_measure as M

BEST = "umud-muscle-architecture/results/submission_familyb_bottomticks_scale_fix.csv"
OUT  = "umud-muscle-architecture/results/submission_bandfix.csv"

def load(t):
    m = M.build_model(encoder_weights=None); m.load_state_dict(M.checkpoint_state(torch.load(M.weights_path(t), map_location="cpu"))); return m.eval().to(M.DEVICE)
apo, fasc = load("apo"), load("fasc")
best = pd.read_csv(BEST).set_index("image_id")
files = sorted(p for p in M.DIRS["test"].iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)

changed = 0
for p in files:
    img = M.read_rgb(p); am = M.predict_mask(apo, img, "apo"); fm = M.predict_mask(fasc, img, "fasc")
    M.USE_BAND_FIX = False; off = M.measure(am, fm)
    M.USE_BAND_FIX = True;  on  = M.measure(am, fm)
    if off is None or on is None or p.name not in best.index: continue
    # detect a real geometry change
    def g(d,k): return d.get(k) or 0
    if abs(g(on,"pa_deg")-g(off,"pa_deg"))<0.05 and abs(g(on,"fl_px")-g(off,"fl_px"))<1 and abs(g(on,"mt_px")-g(off,"mt_px"))<1:
        continue
    row = best.loc[p.name]
    # PA: only transform is +2.5 then clip; replace measured part
    if on.get("pa_deg") is not None:
        best.at[p.name,"pa_deg"] = round(float(np.clip(on["pa_deg"]+2.5, M.PA_MIN, M.PA_MAX)),3)
    # FL/MT: scale the row's final value by the band-fixed/old px ratio (all downstream transforms are multiplicative)
    if g(off,"fl_px")>0 and g(on,"fl_px")>0:
        best.at[p.name,"fl_mm"] = round(float(np.clip(row["fl_mm"]*on["fl_px"]/off["fl_px"], M.FL_MIN, M.FL_MAX)),3)
    if g(off,"mt_px")>0 and g(on,"mt_px")>0:
        best.at[p.name,"mt_mm"] = round(float(np.clip(row["mt_mm"]*on["mt_px"]/off["mt_px"], M.MT_MIN, M.MT_MAX)),3)
    changed += 1

best.reset_index().to_csv(OUT, index=False)
print(f"band-fix splice: changed {changed} rows; wrote {OUT}")
# show a few changed rows old vs new
old = pd.read_csv(BEST).set_index("image_id"); new = pd.read_csv(OUT).set_index("image_id")
diff = (old[["pa_deg","fl_mm","mt_mm"]] - new[["pa_deg","fl_mm","mt_mm"]]).abs().sum(axis=1).sort_values(ascending=False)
for iid in diff.head(8).index:
    print("  %-15s OLD pa=%.1f fl=%.0f mt=%.1f  ->  NEW pa=%.1f fl=%.0f mt=%.1f"%(iid, old.at[iid,"pa_deg"],old.at[iid,"fl_mm"],old.at[iid,"mt_mm"], new.at[iid,"pa_deg"],new.at[iid,"fl_mm"],new.at[iid,"mt_mm"]))
