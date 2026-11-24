"""Один скан → все кривые: автообводка, цветная перерисовка на белом фоне, LAS/CSV, .improj, проверочные наложения.

Задание (JSON), которое агент составляет, посмотрев на скан (шапку с названиями и шкалами):
{
  "image": "data/inbox/2.tif",
  "well": "Hosek Unit #1 ST #1", "api": "42-493-32710",
  "depth": {"unit": "ft", "ref": [[Y1, 6500], [Y2, 8500]], "snap": true},   # 2 подписанные глубины → строки скана
  "interval": [6950, 8950],                                                 # что оцифровать (необязательно)
  "step": 0.5,                                                              # шаг LAS, ft
  "tracks": [
    {"name": "Track 1", "left": 145, "right": 636,                          # края дорожки (пиксели)
     "curves": [
       {"name": "Gamma API", "unit": "API", "min": 0, "max": 150,
        "wrap": [[150, 300], [300, 450]],      # шкалы перехлёста (необязательно)
        "color": null,                         # null = чёрная линия; [r,g,b] = цвет из легенды
        "seed_x": 300,                         # где кривая в начале интервала (необязательно)
        "stroke": 3, "clean": true,            # 3 = убирать тонкую сетку; пунктир/тонкие → 1 и clean=false
        "draw": [230, 30, 30]}                 # цвет на итоговой картинке (необязательно)
     ]}
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import improj, tracer

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]

PALETTE = [(220, 30, 30), (20, 110, 230), (20, 160, 60), (230, 130, 0), (150, 40, 190),
           (0, 160, 170), (200, 0, 120), (110, 80, 30), (90, 90, 90), (240, 190, 0)]


# ---------------------------------------------------------------- глубина
def snap_rows(dark: np.ndarray, x0: int, x1: int, y: float, radius: int = 25) -> float:
    """Подтягивает строку к ближайшей толстой горизонтальной линии сетки (центр линии)."""
    a, b = max(0, int(y) - radius), min(dark.shape[0], int(y) + radius + 1)
    frac = dark[a:b, x0:x1].mean(1)
    if frac.max() < 0.5:
        return y
    rows = np.flatnonzero(frac >= 0.5) + a
    # группы подряд идущих строк → центры линий
    groups = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    centers = [g.mean() for g in groups]
    return float(min(centers, key=lambda c: abs(c - y)))


class DepthMap:
    """Строка скана ↔ глубина. Две опорные точки — прямая; больше — кусочно-линейно (скан растянут
    неравномерно: на длинной ленте отметки 50 ft «уплывают» на 1–3 ft от прямой)."""

    def __init__(self, ref):
        pts = sorted({(float(y), float(d)) for y, d in ref})
        self.ys = np.array([p[0] for p in pts]); self.ds = np.array([p[1] for p in pts])
        (self.y1, self.d1), (self.y2, self.d2) = pts[0], pts[-1]

    @staticmethod
    def _interp(v, xs, fs):
        v = np.asarray(v, float)
        out = np.interp(v, xs, fs)
        lo, hi = v < xs[0], v > xs[-1]          # за крайними точками — продолжаем крайний отрезок
        out = np.where(lo, fs[0] + (v - xs[0]) * (fs[1] - fs[0]) / (xs[1] - xs[0]), out)
        out = np.where(hi, fs[-1] + (v - xs[-1]) * (fs[-1] - fs[-2]) / (xs[-1] - xs[-2]), out)
        return out

    def depth(self, y):
        return self._interp(y, self.ys, self.ds)

    def row(self, d):
        return self._interp(d, self.ds, self.ys)

    @property
    def px_per_ft(self):
        return (self.y2 - self.y1) / (self.d2 - self.d1)


def detect_layout(rgb: np.ndarray, y0: int, y1: int, min_frac: float = 0.85) -> dict:
    """Подсказка агенту: x длинных вертикальных линий (края дорожек) и y толстых горизонтальных линий."""
    dark = tracer.dark_mask(rgb[y0:y1].min(2))
    colf = dark.mean(0)
    xs = np.flatnonzero(colf >= min_frac)
    vlines = [(int(g[0]), int(g[-1])) for g in np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)] if xs.size else []
    rowf = dark.mean(1)
    ys = np.flatnonzero(rowf >= 0.5)
    hlines = [round(float(g.mean()) + y0, 1) for g in np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)] if ys.size else []
    spacing = float(np.median(np.diff(hlines))) if len(hlines) > 2 else None
    return {"vertical_lines_x": vlines, "horizontal_lines_y": hlines[:40], "h_spacing_px": spacing}


def track_edges(dark: np.ndarray, left0: float, right0: float, y0: int, y1: int, block: int = 400,
                search: int = 14):
    """Края дорожки по всей глубине: скан бывает перекошен и рамка «уезжает» на десятки px.
    В каждом блоке строк ищем вертикальную линию рамки рядом с предыдущим положением (центр линии).
    Возвращает (ys_узлов, left, right) — узлы для кусочно-линейной интерполяции."""
    H, W = dark.shape
    ys, ls, rs = [], [], []
    cl, cr = float(left0), float(right0)
    for a in range(y0, y1, block):
        b = min(y1, a + block)
        if b - a < 40:
            break
        res = []
        for c in (cl, cr):
            x0, x1 = max(0, int(round(c)) - search), min(W, int(round(c)) + search + 1)
            frac = dark[a:b, x0:x1].mean(0)
            idx = np.flatnonzero(frac >= 0.6)
            if not idx.size:
                res.append(None)
                continue
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            best = min(runs, key=lambda r: abs(r.mean() + x0 - c))
            res.append(float(best.mean()) + x0)
        # принимаем сдвиг, только если ширина дорожки сохранилась (иначе поймали линию сетки)
        nl = res[0] if res[0] is not None else cl
        nr = res[1] if res[1] is not None else cr
        if abs((nr - nl) - (right0 - left0)) <= 4:
            cl, cr = nl, nr
        elif res[0] is not None and res[1] is None:
            cl, cr = nl, nl + (right0 - left0)
        elif res[1] is not None and res[0] is None:
            cl, cr = nr - (right0 - left0), nr
        ys.append((a + b) / 2)
        ls.append(cl)
        rs.append(cr)
    if not ys:
        return np.array([y0, y1], float), np.array([left0, left0], float), np.array([right0, right0], float)
    return np.array(ys), np.array(ls), np.array(rs)


# ---------------------------------------------------------------- значения
def _wrap_lists(cv: dict):
    """Уровни шкалы как в SoloImageAnalyzer: 1 — основной; 2, 3… — выход за правый край; −1, −2… — за левый
    (уровня 0 нет). "wrap"/"wrap_low" — список диапазонов [[a, b], …] или число уровней (диапазоны считаются сами:
    каждый следующий уровень сдвинут на ширину шкалы)."""
    lo, hi = cv["min"], cv["max"]
    span = hi - lo
    up, low = cv.get("wrap") or [], cv.get("wrap_low") or []
    if isinstance(up, bool):
        up = 1
    if isinstance(up, int):
        up = [[hi + span * k, hi + span * (k + 1)] for k in range(up)]
    if isinstance(low, bool):
        low = 1
    if isinstance(low, int):
        low = [[lo - span * (k + 1), lo - span * k] for k in range(low)]
    return up, low


def _levels(cv: dict):
    up, low = _wrap_lists(cv)
    lv = [(1, cv["min"], cv["max"])]
    for i, (a, b) in enumerate(up):
        lv.append((i + 2, a, b))
    for i, (a, b) in enumerate(low):
        lv.append((-(i + 1), a, b))
    return lv


def _improj_level(t: int) -> int:
    """Уровень трассировщика (0 — основной, +1 вправо, −1 влево) → уровень .improj (1, 2… / −1, −2…)."""
    return t + 1 if t >= 0 else t


def points_to_values(points, xl_fn, xr_fn, depth_map: DepthMap, cv: dict):
    levels = {k: (a, b) for k, a, b in _levels(cv)}
    log = cv.get("scale") == "log"
    out = []
    for x, y, lv, nulls in points:
        l, r = xl_fn(y), xr_fn(y)
        t = (x - l) / (r - l)
        a, b = levels.get(lv, levels[1])
        if log:
            v = 10 ** (np.log10(a) + t * (np.log10(b) - np.log10(a)))
        else:
            v = a + t * (b - a)
        out.append((float(depth_map.depth(y)), float(v), nulls))
    return out


def resample(vals, step, d0, d1):
    d = np.array([v[0] for v in vals]); x = np.array([v[1] for v in vals])
    nulls = np.array([v[2] for v in vals])
    order = np.argsort(d, kind="stable")
    d, x, nulls = d[order], x[order], nulls[order]
    grid = np.round(np.arange(d0, d1 + step / 2, step), 3)
    # на одной глубине может быть несколько точек (горизонтальный пик) — берём крайнее значение пика
    uniq, idx = np.unique(d, return_index=True)
    xv = np.array([x[i:j][np.argmax(np.abs(x[i:j] - np.median(x)))] if j - i > 1 else x[i]
                   for i, j in zip(idx, list(idx[1:]) + [len(d)])])
    out = np.interp(grid, uniq, xv, left=np.nan, right=np.nan)
    for k in np.flatnonzero(nulls[:-1]):
        out[(grid > d[k]) & (grid < d[k + 1])] = np.nan
    return grid, out


# ---------------------------------------------------------------- основной прогон
def run(job_path: str | Path, rgb: np.ndarray | None = None, out_dir: str | Path | None = None) -> dict:
    """rgb / out_dir — для пакетной сверки: уже загруженный (обрезанный) скан и своя папка результата."""
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    img_path = Path(job["image"])
    if not img_path.is_absolute():
        img_path = ROOT / img_path
    if rgb is None:
        rgb = np.array(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]
    gray_dark = tracer.dark_mask(rgb.min(2))

    # глубина: 2 опорные строки, по желанию подтягиваем к линиям сетки
    tr0 = job["tracks"][0]
    ref = [list(r) for r in job["depth"]["ref"]]
    if job["depth"].get("snap", True):
        for r in ref:
            r[0] = snap_rows(gray_dark, int(tr0["left"]), int(tr0["right"]), r[0])
    dm = DepthMap(ref)
    d_from, d_to = job.get("interval") or [float(dm.depth(0)), float(dm.depth(H - 1))]
    y0 = int(max(0, np.floor(dm.row(d_from))))
    y1 = int(min(H, np.ceil(dm.row(d_to)) + 1))

    out_dir = Path(out_dir) if out_dir else ROOT / "output" / job_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    color_i = 0
    for tr in job["tracks"]:
        ys = np.arange(y0, y1)
        left = np.full(ys.size, float(tr["left"]))
        right = np.full(ys.size, float(tr["right"]))
        if "left_bottom" in tr:   # перекос скана: край внизу другой
            left = np.linspace(tr["left"], tr["left_bottom"], ys.size)
            right = np.linspace(tr["right"], tr["right_bottom"], ys.size)
        elif tr.get("auto_edges", True):   # отслеживаем рамку дорожки по глубине (перекос скана)
            ny, nl, nr = track_edges(gray_dark, tr["left"], tr["right"], y0, y1)
            left, right = np.interp(ys, ny, nl), np.interp(ys, ny, nr)
        xl_fn = lambda y, ys=ys, left=left: float(np.interp(y, ys, left))
        xr_fn = lambda y, ys=ys, right=right: float(np.interp(y, ys, right))
        used = np.zeros((y1 - y0, W), bool) if len(tr["curves"]) > 1 else None
        for cv in tr["curves"]:
            mask = tracer.curve_mask(rgb, cv.get("color"), cv.get("tol", 45), y0=y0, y1=y1,
                                     x0=max(0, int(left.min()) - 5), x1=min(W, int(right.max()) + 6))
            res = tracer.trace_track(mask, left, right, y0, y1, seed_x=cv.get("seed_x"),
                                     stroke=cv.get("stroke", 0), clean=cv.get("clean", True),
                                     wrap=bool(cv.get("wrap") or cv.get("wrap_low")), zigzag=cv.get("zigzag", dm.px_per_ft < 4),   # сжатый лог → режим «гармошки»
                                     max_level=1 + len(_wrap_lists(cv)[0]), style=cv.get("style"),
                                     used=used, jump_cost=cv.get("jump_cost", 1.0), x_range=cv.get("x_range"),
                                     slope_cost=cv.get("smooth", 0.0),
                                     centerline=cv.get("centerline", True),
                                     min_level=-len(_wrap_lists(cv)[1]), protect=bool(cv.get("straight")),
                                     ink_reward=cv.get("ink_reward", 0.0))
            # уровни трассировщика: 0 = основной → 1 в .improj
            pts = [(x, y, _improj_level(lv), n) for x, y, lv, n in res.points]
            vals = points_to_values(pts, xl_fn, xr_fn, dm, cv)
            draw = tuple(cv.get("draw") or PALETTE[color_i % len(PALETTE)])
            color_i += 1
            results.append({"track": tr["name"], "curve": cv, "points": pts, "values": vals,
                            "coverage": res.coverage, "draw": draw, "left": left, "right": right, "ys": ys})

    step = job.get("step", 0.5)
    files = _write_outputs(job, results, dm, d_from, d_to, step, out_dir, rgb)
    report = {"image": str(img_path), "depth_ref": ref, "px_per_ft": round(dm.px_per_ft, 3),
              "interval": [d_from, d_to], "out": str(out_dir), "files": [str(f) for f in files], "curves": []}
    for r in results:
        v = np.array([x[1] for x in r["values"]]) if r["values"] else np.array([])
        report["curves"].append({"track": r["track"], "curve": r["curve"]["name"],
                                 "coverage": round(r["coverage"], 3), "points": len(r["points"]),
                                 "min": round(float(v.min()), 2) if v.size else None,
                                 "max": round(float(v.max()), 2) if v.size else None,
                                 "draw_rgb": list(r["draw"]),
                                 "warning": "покрытие < 90% — проверь overlay" if r["coverage"] < 0.9 else None})
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


# ---------------------------------------------------------------- выходные файлы
def _font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _write_outputs(job, results, dm, d0, d1, step, out_dir, rgb):
    import csv
    import lasio

    files = []
    # 1) CSV + LAS — из чисел
    cols = []
    grid = None
    for r in results:
        grid, v = resample(r["values"], step, d0, d1)
        cols.append((r["curve"], v))
    with open(out_dir / "curves.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["DEPTH_FT"] + [c["name"] for c, _ in cols])
        for i, d in enumerate(grid):
            w.writerow([f"{d:.2f}"] + ["" if np.isnan(v[i]) else f"{v[i]:.3f}" for _, v in cols])
    files.append(out_dir / "curves.csv")
    las = lasio.LASFile()
    las.well["WELL"].value = job.get("well", "")
    las.well["API"].value = job.get("api", "")
    las.well["NULL"].value = -999.25
    las.append_curve("DEPT", grid, unit="FT")
    used = set()
    for c, v in cols:
        mn = "".join(ch for ch in c["name"].upper() if ch.isalnum())[:12] or "CURVE"
        while mn in used:
            mn += "_"
        used.add(mn)
        las.append_curve(mn, np.where(np.isnan(v), -999.25, v), unit=c.get("unit", ""), descr=c["name"])
    las.write(str(out_dir / "curves.las"), version=2.0)
    files.append(out_dir / "curves.las")

    # 2) цветная перерисовка на белом фоне — строится ИЗ ЧИСЕЛ (проверка математики)
    files += render_colored(job, results, cols, grid, d0, d1, out_dir)

    # 3) .improj для SoloImageAnalyzer
    grids = []
    for tr in job["tracks"]:
        rs = [r for r in results if r["track"] == tr["name"]]
        if not rs:
            continue
        ys, left, right = rs[0]["ys"], rs[0]["left"], rs[0]["right"]
        corners = {"TopLeft": (left[0], ys[0]), "TopRight": (right[0], ys[0]),
                   "BottomRight": (right[-1], ys[-1]), "BottomLeft": (left[-1], ys[-1])}
        curves = [improj.CurveSpec(r["curve"]["name"], r["curve"]["min"], r["curve"]["max"], r["points"],
                                   r["draw"], r["curve"].get("scale") == "log",
                                   [(k, a, b) for k, a, b in _levels(r["curve"])]) for r in rs]
        inter = []
        step_i = max(1, len(ys) // 12)             # промежуточные пары рамки — перекос скана
        if float(np.ptp(left)) > 1.5 or float(np.ptp(right)) > 1.5:
            inter = [((float(left[i]), float(ys[i])), (float(right[i]), float(ys[i])))
                     for i in range(step_i, len(ys) - step_i // 2, step_i)]
        grids.append(improj.GridSpec(tr["name"], round(float(dm.depth(ys[0])), 2), round(float(dm.depth(ys[-1])), 2),
                                     corners, curves, inter))
    proj = improj.build_project(str(Path(job["image"]).resolve() if Path(job["image"]).is_absolute() else ROOT / job["image"]),
                                grids, job.get("well", ""), job.get("api", ""))
    improj.save(proj, out_dir / "auto.improj")
    files.append(out_dir / "auto.improj")

    # 4) наложение на скан (окна по 200 ft) — для глазной проверки
    files += render_overlays(results, dm, d0, d1, rgb, out_dir)
    return files


def render_colored(job, results, cols, grid, d0, d1, out_dir, px_per_ft=4, track_w=360):
    tracks = [t["name"] for t in job["tracks"]]
    n = len(tracks)
    depth_w, head_h, pad = 70, 40 + 34 * max(len(t["curves"]) for t in job["tracks"]), 10
    H = int((d1 - d0) * px_per_ft) + head_h + pad * 2
    W = depth_w + n * (track_w + pad) + pad
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    f, fb = _font(13), _font(15)
    y_of = lambda d: head_h + pad + (d - d0) * px_per_ft
    # глубины и сетка
    for d in np.arange(np.ceil(d0 / 10) * 10, d1 + 0.01, 10):
        y = y_of(d)
        heavy = d % 50 == 0
        for k in range(n):
            x0 = depth_w + k * (track_w + pad)
            dr.line([(x0, y), (x0 + track_w, y)], fill=(170, 170, 170) if heavy else (225, 225, 225), width=1)
        if d % 50 == 0:
            dr.text((5, y - 7), f"{d:.0f}", fill=(0, 0, 0), font=f)
    for k, tname in enumerate(tracks):
        x0 = depth_w + k * (track_w + pad)
        for i in range(1, 10):
            x = x0 + track_w * i / 10
            dr.line([(x, head_h + pad), (x, H - pad)], fill=(235, 235, 235) if i != 5 else (210, 210, 210))
        dr.rectangle([x0, head_h + pad, x0 + track_w, H - pad], outline=(60, 60, 60))
        dr.text((x0 + 4, 4), tname, fill=(0, 0, 0), font=fb)
        tr_res = [(r, v) for r, (c, v) in zip(results, cols) if r["track"] == tname]
        for j, (r, v) in enumerate(tr_res):
            cv = r["curve"]
            yh = 26 + j * 34
            dr.line([(x0, yh + 22), (x0 + track_w, yh + 22)], fill=r["draw"], width=3)
            dr.text((x0, yh + 4), f'{cv["min"]:g}', fill=r["draw"], font=f)
            label = f'{cv["name"]} ({cv.get("unit", "")})'
            tw = dr.textlength(label, font=f)
            dr.text((x0 + (track_w - tw) / 2, yh + 4), label, fill=r["draw"], font=f)
            mx = f'{cv["max"]:g}'
            dr.text((x0 + track_w - dr.textlength(mx, font=f), yh + 4), mx, fill=r["draw"], font=f)
            # кривая из чисел: значение → x по основной шкале, перехлёст заворачивается как на бланке
            lo, hi = cv["min"], cv["max"]
            span = hi - lo
            seg = []
            for d, val in zip(grid, v):
                if np.isnan(val):
                    if len(seg) > 1:
                        dr.line(seg, fill=r["draw"], width=2)
                    seg = []
                    continue
                t = (val - lo) / span
                wrapped = t - np.floor(t) if (cv.get("wrap") and t > 1) else min(max(t, 0), 1)
                x = x0 + wrapped * track_w
                if seg and abs(x - seg[-1][0]) > track_w * 0.8:   # переход на другой уровень шкалы
                    if len(seg) > 1:
                        dr.line(seg, fill=r["draw"], width=2)
                    seg = []
                seg.append((x, y_of(d)))
            if len(seg) > 1:
                dr.line(seg, fill=r["draw"], width=2)
    files = []
    full = out_dir / "colored.png"
    im.save(full)
    files.append(full)
    # страницы по 250 ft для просмотра
    pages = out_dir / "colored_pages"
    pages.mkdir(exist_ok=True)
    for a in np.arange(d0, d1, 250):
        b = min(d1, a + 250)
        top = int(y_of(a)) - 2
        page = Image.new("RGB", (W, head_h + int((b - a) * px_per_ft) + pad * 2), "white")
        page.paste(im.crop((0, 0, W, head_h + pad)), (0, 0))
        page.paste(im.crop((0, top, W, int(y_of(b)) + pad)), (0, head_h + pad))
        p = pages / f"{a:.0f}-{b:.0f}.png"
        page.save(p)
        files.append(p)
    return files


def render_overlays(results, dm, d0, d1, rgb, out_dir, window=200):
    od = out_dir / "overlay"
    od.mkdir(exist_ok=True)
    files = []
    H, W = rgb.shape[:2]
    xs_all = [x for r in results for x in (r["left"].min(), r["right"].max())]
    xa, xb = max(0, int(min(xs_all)) - 30), min(W, int(max(xs_all)) + 30)
    for a in np.arange(d0, d1, window):
        b = min(d1, a + window)
        ya, yb = int(max(0, dm.row(a))), int(min(H, dm.row(b)))
        if yb - ya < 5:
            continue
        crop = Image.fromarray(rgb[ya:yb, xa:xb])
        crop = Image.blend(crop, Image.new("RGB", crop.size, "white"), 0.5)
        dr = ImageDraw.Draw(crop)
        for r in results:
            seg, prev_lv = [], None
            for x, y, lv, n in r["points"]:
                if not (ya <= y <= yb):
                    continue
                if prev_lv is not None and lv != prev_lv and len(seg) > 1:
                    dr.line(seg, fill=r["draw"], width=2); seg = []
                seg.append((x - xa, y - ya)); prev_lv = lv
                if n and len(seg) > 1:
                    dr.line(seg, fill=r["draw"], width=2); seg = []
            if len(seg) > 1:
                dr.line(seg, fill=r["draw"], width=2)
        p = od / f"{a:.0f}-{b:.0f}.png"
        crop.save(p)
        files.append(p)
    return files
