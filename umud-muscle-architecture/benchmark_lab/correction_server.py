"""Server for the typed-correction labeling UI.

Serves the reproducible pre-fill (from bake_prefill.py), the test images (TIFF -> PNG, mtime-cached),
and per-image saved corrections (one file per image, so a fast save never clobbers another image).
Live-scores corrections via correction_score.score_corrections (single engine).

    python umud-muscle-architecture/benchmark_lab/bake_prefill.py    # once, to generate pre-fills
    python umud-muscle-architecture/benchmark_lab/correction_server.py
    open http://127.0.0.1:8790
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_lab"))
import segment_then_measure as M
import correction_score as CS

PREFILL_DIR = Path(os.environ.get("UMUD_PREFILL_DIR", str(ROOT / "results" / "correction_prefill")))
LABELS_DIR = Path(os.environ.get("UMUD_LABELS_DIR", str(ROOT / "results" / "correction_labels")))
CACHE_DIR = PREFILL_DIR / "_cache"
UI_DIR = Path(__file__).resolve().parent / "correction_ui"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_PATHS = {}  # image_id -> source path


def manifest_rows():
    import csv
    rows = []
    man = PREFILL_DIR / "manifest.csv"
    if not man.exists():
        return rows
    with man.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = Path(r["image_id"]).stem
            lf = LABELS_DIR / f"{stem}.json"
            r["done"] = lf.exists()
            r["multiband"] = False
            if lf.exists():
                try:
                    r["multiband"] = bool(json.loads(lf.read_text(encoding="utf-8")).get("corrections", {}).get("multiband"))
                except Exception:
                    pass
            _PATHS[r["image_id"]] = ROOT / r["image_path"]
            rows.append(r)
    return rows


def image_png(image_id):
    """M.read_rgb -> PNG so canvas pixels match the geometry's coordinate space. mtime-cached."""
    src = _PATHS.get(image_id)
    if src is None:
        for r in manifest_rows():
            if r["image_id"] == image_id:
                src = _PATHS.get(image_id); break
    if src is None or not Path(src).exists():
        return None
    cache = CACHE_DIR / f"{Path(image_id).stem}.png"
    if cache.exists() and cache.stat().st_mtime >= Path(src).stat().st_mtime:
        return cache.read_bytes()
    rgb = M.read_rgb(Path(src))
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    cache.write_bytes(buf.tobytes())
    return buf.tobytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, (UI_DIR / "index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if u.path == "/app.js":
            return self._send(200, (UI_DIR / "app.js").read_text(encoding="utf-8"), "application/javascript")
        if u.path == "/styles.css":
            return self._send(200, (UI_DIR / "styles.css").read_text(encoding="utf-8"), "text/css")
        if u.path == "/api/manifest":
            return self._send(200, {"rows": manifest_rows()})
        if u.path == "/api/prefill":
            pf = PREFILL_DIR / f"{Path(q.get('id', [''])[0]).stem}.json"
            return self._send(200, json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {})
        if u.path == "/api/corrections":
            lf = LABELS_DIR / f"{Path(q.get('id', [''])[0]).stem}.json"
            return self._send(200, json.loads(lf.read_text(encoding="utf-8")) if lf.exists() else {})
        if u.path == "/api/score":
            iid = q.get("id", [""])[0]
            pf = PREFILL_DIR / f"{Path(iid).stem}.json"
            lf = LABELS_DIR / f"{Path(iid).stem}.json"
            if not pf.exists():
                return self._send(404, {"error": "no prefill"})
            corr = json.loads(lf.read_text(encoding="utf-8")).get("corrections", {}) if lf.exists() else {}
            return self._send(200, CS.score_corrections(json.loads(pf.read_text(encoding="utf-8")), corr))
        if u.path == "/image":
            png = image_png(q.get("id", [""])[0])
            return self._send(200, png, "image/png") if png else self._send(404, b"", "image/png")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/corrections":
            iid = body.get("image_id", "")
            if not iid:
                return self._send(400, {"error": "no image_id"})
            stem = Path(iid).stem
            rec = {"image_id": iid, "corrections": body.get("corrections", {}),
                   "blind_angle": body.get("blind_angle", False), "updated_at": body.get("updated_at", "")}
            with _LOCK:
                tmp = LABELS_DIR / f".{stem}.tmp"
                tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                tmp.replace(LABELS_DIR / f"{stem}.json")
            # live score after save
            pf = PREFILL_DIR / f"{stem}.json"
            score = CS.score_corrections(json.loads(pf.read_text(encoding="utf-8")), rec["corrections"]) if pf.exists() else {}
            return self._send(200, {"ok": True, "score": score})
        return self._send(404, {"error": "not found"})


def main():
    manifest_rows()  # warm _PATHS
    port = int(os.environ.get("UMUD_PORT", "8790"))
    print(f"correction UI at http://127.0.0.1:{port}  (prefills: {PREFILL_DIR})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
