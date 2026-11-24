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
ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class State:
    def __init__(self, proj_path: Path | None, image: Path | None, note: str = ""):
        """proj_path=None — только скан, без кривых (задания обводки для него ещё нет)."""
        self.path = proj_path
        self.note = note
        self.lock = threading.Lock()
        if proj_path is None:
            self.src = None
            self.project = {"Grids": []}
            self.image_path = image
        else:
            edited = proj_path.with_suffix(".edited.improj")
            self.src = edited if edited.exists() else proj_path     # продолжаем с прошлой правки
            self.project = improj.load(self.src)
            self.image_path = image or improj_tools.find_image(self.project, proj_path)
        if self.image_path is None or not Path(self.image_path).exists():
            raise ValueError("Не найден скан: укажите --image")
        self.img = Image.open(self.image_path)
        self.img.load()
        if proj_path is None:
            self.doubts_path, self.doubts = None, []
            return
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
        return {"file": str(self.path or ""), "source": str(self.src or self.image_path),
                "image": str(self.image_path), "note": self.note,
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

    def export_png(self) -> bytes:
        """Скан в полном размере с наложенными кривыми — пиксель в пиксель со сканом."""
        out = (self.path.parent if self.path else ROOT / "output" / "uploaded") / "overlay_full.png"
        with self.lock:
            improj_tools.overlay(self.project, Path(self.image_path), out, max_height=10 ** 9)
        return out.read_bytes()

    def save(self, data: dict) -> dict:
        if self.path is None:
            raise ValueError("Кривых нет: для этого скана ещё не сделана обводка")
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


def store_upload(name: str, body: bytes) -> Path:
    """Скан → data/inbox, .improj → output/uploaded. Существующие файлы не перезаписываются."""
    name = Path(name.replace("\\", "/")).name
    ext = Path(name).suffix.lower()
    if not name or name.startswith("."):
        raise ValueError("Плохое имя файла")
    if ext in IMG_EXT:
        folder = ROOT / "data" / "inbox"
    elif ext == ".improj":
        folder = ROOT / "output" / "uploaded"
    else:
        raise ValueError(f"Тип файла не поддерживается: {ext or name}")
    folder.mkdir(parents=True, exist_ok=True)
    dst, n = folder / name, 1
    while dst.exists() and dst.read_bytes() != body:      # тот же файл — берём как есть, другой — кладём рядом
        n += 1
        dst = folder / f"{Path(name).stem}_{n}{ext}"
    if not dst.exists():
        dst.write_bytes(body)
    return dst


def find_jobs(image: Path) -> list[Path]:
    """Задания автообводки, написанные именно для этого скана (байт в байт тот же файл)."""
    found = []
    size = image.stat().st_size
    for jp in sorted((ROOT / "data" / "jobs").glob("*.json")):
        if jp.name.startswith("_"):
            continue
        try:
            ji = Path(json.loads(jp.read_text(encoding="utf-8")).get("image", ""))
        except (ValueError, OSError, AttributeError):
            continue
        if not ji.is_absolute():
            ji = ROOT / ji
        if ji.is_file() and ji.stat().st_size == size and (ji.resolve() == image.resolve()
                                                           or ji.read_bytes() == image.read_bytes()):
            found.append(jp)
    return found


def open_upload(image: Path | None, proj: Path | None, job: str = "") -> State:
    """Что открыть после загрузки: готовый .improj, либо автообводку по заданию, либо один скан."""
    if proj is not None:
        return State(proj, image, note="Открыт загруженный проект .improj")
    if image is None:
        raise ValueError("Нужен скан (TIFF/PNG/JPG)")
    jobs = find_jobs(image)
    if job:
        jobs = [j for j in jobs if j.stem == job]
    if not jobs:
        return State(None, image, note=f"Скан сохранён в {image}. Задания обводки для него ещё нет: шкалы и глубины "
                                       f"из шапки сначала читает агент — попросите Claude «оцифруй {image.name}».")
    jp = jobs[0]
    out = ROOT / "output" / jp.stem
    if not (out / "auto.improj").exists():
        from . import pipeline
        pipeline.run(jp)
    more = f" Для этого скана есть ещё задания: {', '.join(j.stem for j in jobs[1:])}." if len(jobs) > 1 else ""
    return State(out / "auto.improj", image, note=f"Обводка наложена по заданию {jp.stem}.{more}")


def _uploaded(v: str | None) -> Path | None:
    if not v:
        return None
    p = Path(v).resolve()
    if not any(p.is_relative_to(d) for d in (ROOT / "data" / "inbox", ROOT / "output" / "uploaded")):
        raise ValueError("Файл вне папки загрузок")
    return p


def make_handler(box: dict):
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
            st = box["st"]
            try:
                if u.path == "/":
                    self._send(HTML.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/project":
                    self._send(json.dumps(st.payload(), ensure_ascii=False).encode("utf-8"), "application/json")
                elif u.path == "/api/tile":
                    self._send(st.tile(int(q["y0"][0]), int(q["y1"][0]), float(q.get("s", ["1"])[0])), "image/png")
                elif u.path == "/api/export.png":
                    body = st.export_png()
                    fname = f"{Path(st.image_path).stem}_overlay.png".encode("ascii", "replace").decode("ascii")
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/doubts":
                    st.refresh_doubts()
                    self._send(json.dumps(st.doubts, ensure_ascii=False).encode("utf-8"), "application/json")
                else:
                    self._send(b"not found", "text/plain", 404)
            except Exception as e:  # noqa: BLE001
                self._send(str(e).encode("utf-8"), "text/plain; charset=utf-8", 500)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            u = urlparse(self.path)
            st = box["st"]
            try:
                raw = self.rfile.read(n)
                if u.path == "/api/upload":      # тело запроса — сам файл, имя в ?name=
                    dst = store_upload(parse_qs(u.query)["name"][0], raw)
                    self._send(json.dumps({"path": str(dst)}, ensure_ascii=False).encode("utf-8"), "application/json")
                    return
                data = json.loads(raw.decode("utf-8"))
                if u.path == "/api/open":        # показать загруженное: проект, автообводку или один скан
                    with box["lock"]:
                        box["st"] = open_upload(_uploaded(data.get("image")), _uploaded(data.get("improj")),
                                                data.get("job", ""))
                    self._send(json.dumps(box["st"].payload(), ensure_ascii=False).encode("utf-8"), "application/json")
                elif u.path == "/api/save":
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
    try:
        st = State(Path(proj_path).resolve(), Path(image).resolve() if image else None)
    except ValueError as e:
        raise SystemExit(str(e))
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler({"st": st, "lock": threading.Lock()}))
    url = f"http://127.0.0.1:{port}/"
    print(f"Редактор: {url}\nФайл: {st.src}\nСкан: {st.image_path}\nСомнительных участков: {len(st.doubts)}\nОстановить: Ctrl+C", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
