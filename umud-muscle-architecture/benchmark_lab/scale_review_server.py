"""Local scale-oracle review server.

Example:
    python benchmark_lab/scale_review_server.py --port 8773 --pack start
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "scale_review_v1"
RESULTS_DIR = ROOT / "results" / "scale_oracle_review"
TEST_IMAGES = ROOT / "data" / "test_images_v2" / "test_set_v2"
NOTES_PATH = RESULTS_DIR / "oracle_notes.json"
PNG_CACHE = RESULTS_DIR / "png_cache"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_notes() -> dict[str, dict]:
    if not NOTES_PATH.exists():
        return {}
    try:
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_notes(notes: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(json.dumps(notes, indent=2, sort_keys=True), encoding="utf-8")


class ScaleReviewHandler(BaseHTTPRequestHandler):
    server_version = "ScaleReview/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/scale-review", "/scale-review/"}:
            self._send_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/scale-review/"):
            rel = path.removeprefix("/scale-review/") or "index.html"
            self._send_file(STATIC_DIR / rel)
            return
        if path == "/api/manifest":
            query = parse_qs(parsed.query)
            pack = query.get("pack", [self.server.pack])[0]
            self._send_manifest(pack)
            return
        if path == "/api/notes":
            self._send_json({"notes": _read_notes()})
            return
        if path.startswith("/image/"):
            image_id = Path(path.removeprefix("/image/")).name
            self._send_image(image_id)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = self.rfile.read(length)
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        image_id = str(data.get("image_id", "")).strip()
        if not image_id:
            self.send_error(400, "missing image_id")
            return
        notes = _read_notes()
        clean = {
            "status": str(data.get("status", "")),
            "oracle_scale_px_per_cm": str(data.get("oracle_scale_px_per_cm", "")),
            "oracle_depth_mm": str(data.get("oracle_depth_mm", "")),
            "oracle_ticks": str(data.get("oracle_ticks", "")),
            "comment": str(data.get("comment", "")),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        notes[image_id] = clean
        _write_notes(notes)
        self._send_json({"ok": True, "note": clean})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send_manifest(self, pack: str) -> None:
        pack_name = str(pack or self.server.pack).strip().lower()
        path = RESULTS_DIR / ("start_pack.csv" if pack_name == "start" else "manifest.csv")
        if not path.exists():
            self.send_error(404, f"missing {path}; run experiments/exp60_scale_oracle_review_pack.py")
            return
        rows = _read_csv(path)
        notes = _read_notes()
        if pack_name in {"remaining", "unreviewed"}:
            reviewed = {image_id for image_id, note in notes.items() if note.get("status")}
            rows = [row for row in rows if row.get("image_id", "") not in reviewed]
        self._send_json(
            {
                "pack": pack_name,
                "row_count": len(rows),
                "rows": rows,
                "notes": notes,
                "notes_path": str(NOTES_PATH),
            }
        )

    def _send_image(self, image_id: str) -> None:
        path = TEST_IMAGES / image_id
        if not path.exists():
            self.send_error(404, "missing image")
            return
        if path.suffix.lower() in {".tif", ".tiff"}:
            self._send_tiff_as_png(path)
            return
        self._send_file(path)

    def _send_tiff_as_png(self, path: Path) -> None:
        PNG_CACHE.mkdir(parents=True, exist_ok=True)
        out = PNG_CACHE / f"{path.stem}.png"
        if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
            with Image.open(path) as im:
                if im.mode not in {"RGB", "RGBA"}:
                    im = im.convert("RGB")
                im.save(out, format="PNG")
        body = out.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        path = path.resolve()
        static_root = STATIC_DIR.resolve()
        image_root = TEST_IMAGES.resolve()
        if not (str(path).startswith(str(static_root)) or str(path).startswith(str(image_root))):
            self.send_error(403, "forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(404, "not found")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--pack", choices=["start", "remaining", "all"], default="remaining")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ScaleReviewHandler)
    server.pack = args.pack
    print(f"scale review viewer: http://127.0.0.1:{args.port}/scale-review/")
    print(f"pack: {args.pack}")
    print(f"notes: {NOTES_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
