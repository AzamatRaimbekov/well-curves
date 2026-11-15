"""Редактор оцифровки в браузере: скан + кривые + сомнительные участки, правка мышью, сохранение в .improj/LAS.

python -m wellcurves edit <файл.improj> [--image скан] [--port 8765]
Сохраняет рядом: <имя>.edited.improj, edited/curves.las, edited/curves.csv, <имя>.doubts.json.
Сервер слушает только 127.0.0.1.
"""
from __future__ import annotations

import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import doubts as D
from . import improj, improj_tools

Image.MAX_IMAGE_PIXELS = None
HTML = Path(__file__).with_name("editor.html")


class State:
    def __init__(self, proj_path: Path, image: Path | None):
        self.path = proj_path
        edited = proj_path.with_suffix(".edited.improj")
        self.src = edited if edited.exists() else proj_path     # продолжаем с прошлой правки
        self.project = improj.load(self.src)
        self.image_path = image or improj_tools.find_image(self.project, proj_path)
        if self.image_path is None or not Path(self.image_path).exists():
            raise SystemExit("Не найден скан: укажите --image")
        self.img = Image.open(self.image_path)
        self.img.load()
        self.lock = threading.Lock()
        self.doubts_path = proj_path.with_suffix(".doubts.json")
        if self.doubts_path.exists():
            self.doubts = json.loads(self.doubts_path.read_text(encoding="utf-8"))
        else:
            self.refresh_doubts()

    def refresh_doubts(self):
        self.doubts = D.find_doubts(self.project, self.image_path)
        self.doubts_path.write_text(json.dumps(self.doubts, ensure_ascii=False, indent=1), encoding="utf-8")

    def payload(self) -> dict:
        grids = []
        for g in self.project["Grids"]:
            geo = improj.GridGeom.from_grid(g)
            curves = []
            for c in g["Curves"][0]["Items"]:
                col = c["Color"]
                curves.append({"name": c["Name"].strip(), "min": c["MinX"], "max": c["MaxX"], "log": c.get("ScaleType", 0) == 1,
                               "color": [col["R"], col["G"], col["B"]],
                               "levels": [[m["MultiplierLevel"], m["MinX"], m["MaxX"]] for m in c.get("MultiplierLevels") or []],
                               "points": [[round(q["X"], 1), round(q["Y"], 1), q.get("MultiplierLevel") or 1,
                                           1 if q.get("NullsAfterThat") else 0] for q in c["Points"]]})
            grids.append({"name": g["Name"], "top_y": geo.top_y, "bottom_y": geo.bottom_y, "min_d": geo.min_depth,
                          "max_d": geo.max_depth, "edge_y": geo.edge_y.tolist(), "left_x": geo.left_x.tolist(),
                          "right_x": geo.right_x.tolist(), "curves": curves})
        return {"file": str(self.path), "source": str(self.src), "image": str(self.image_path),
                "width": self.img.width, "height": self.img.height, "grids": grids, "doubts": self.doubts,
                "well": self.project.get("WELL", ""), "api": self.project.get("API", "")}

    def tile(self, y0: int, y1: int, scale: float) -> bytes:
        y0, y1 = max(0, y0), min(self.img.height, y1)
        with self.lock:
            c = self.img.crop((0, y0, self.img.width, y1)).convert("L")
        if scale != 1:
            c = c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))),
                         Image.LANCZOS if scale < 1 else Image.NEAREST)
        buf = io.BytesIO()
        c.save(buf, "PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    def save(self, data: dict) -> dict:
        for gi, g in enumerate(self.project["Grids"]):
            for ci, c in enumerate(g["Curves"][0]["Items"]):
                pts = data["grids"][gi][ci]
                uid = c.get("Uid")
                old = {(round(q["X"], 1), round(q["Y"], 1)): q for q in c["Points"]}
                new = []
                for x, y, lv, nulls in pts:
                    q = old.get((round(x, 1), round(y, 1)))
                    if q is None:                      # новая точка — ручная (Type 1)
                        q = improj._pt(x, y, 1, int(lv), bool(nulls), uid)
                    else:
                        q = dict(q, MultiplierLevel=int(lv), NullsAfterThat=bool(nulls))
                    new.append(q)
                c["Points"] = new
        out = self.path.with_suffix(".edited.improj")
        improj.save(self.project, out)
        files = improj_tools.export(self.project, self.path.parent / "edited")
        if "doubts" in data:
            self.doubts = data["doubts"]
            self.doubts_path.write_text(json.dumps(self.doubts, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"saved": str(out), "files": [str(f) for f in files]}


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тихо
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    self._send(HTML.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/project":
                    self._send(json.dumps(st.payload(), ensure_ascii=False).encode("utf-8"), "application/json")
                elif u.path == "/api/tile":
                    self._send(st.tile(int(q["y0"][0]), int(q["y1"][0]), float(q.get("s", ["1"])[0])), "image/png")
                elif u.path == "/api/doubts":
                    st.refresh_doubts()
                    self._send(json.dumps(st.doubts, ensure_ascii=False).encode("utf-8"), "application/json")
                else:
                    self._send(b"not found", "text/plain", 404)
            except Exception as e:  # noqa: BLE001
                self._send(str(e).encode("utf-8"), "text/plain; charset=utf-8", 500)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                if self.path == "/api/save":
                    res = st.save(data)
                    if data.get("recheck"):
                        st.refresh_doubts()
                        res["doubts"] = st.doubts
                    self._send(json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json")
                else:
                    self._send(b"not found", "text/plain", 404)
            except Exception as e:  # noqa: BLE001
                self._send(str(e).encode("utf-8"), "text/plain; charset=utf-8", 500)

    return H


def serve(proj_path: str | Path, image: str | Path | None = None, port: int = 8765, open_browser: bool = True):
    st = State(Path(proj_path).resolve(), Path(image).resolve() if image else None)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    url = f"http://127.0.0.1:{port}/"
    print(f"Редактор: {url}\nФайл: {st.src}\nСкан: {st.image_path}\nСомнительных участков: {len(st.doubts)}\nОстановить: Ctrl+C", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
