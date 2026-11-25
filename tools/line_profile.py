"""Как выглядит линия каждой кривой на скане — по обводке исполнителя (база готовых задач).
Для каждой кривой в окне глубин (первые --win ft) идём вдоль её линии с шагом 1 px и меряем:
  cover   — доля пути, где под линией есть чернила (сплошная ≈ 1, пунктир 0.5–0.85, точки < 0.5);
  dash    — медианная длина штриха, px; gap — медианная длина разрыва, px;
  width   — медианная толщина штриха под линией, px (strokes.stroke_width);
  color   — средний цвет чернил под линией (на цветных сканах).
python tools/line_profile.py [API ...] [--win 400]  → output/report/line_profile.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _runs(b):
    """Длины серий True и False в булевом векторе."""
    if b.size == 0:
        return [], []
    d = np.flatnonzero(np.diff(b.astype(np.int8))) + 1
    seg = np.split(b, d)
    on = [len(s) for s in seg if s[0]]
    off = [len(s) for s in seg if not s[0]]
    return on, off


def profile_curve(dark, W, rgb, pts, y0, r=3, lab=None, ext=None):
    """pts — [(x, y)] в координатах окна. Возвращает признаки линии."""
    pts = sorted(pts, key=lambda p: p[1])
    xs, ys = [], []
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        n = max(1, int(np.ceil(np.hypot(xb - xa, yb - ya))))
        t = np.arange(n) / n
        xs.append(xa + (xb - xa) * t); ys.append(ya + (yb - ya) * t)
    if not xs:
        return None
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    H, Wd = dark.shape
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    keep = (yi >= 0) & (yi < H) & (xi >= r) & (xi < Wd - r)
    xi, yi = xi[keep], yi[keep]
    if xi.size < 50:
        return None
    hit = np.zeros(xi.size, bool)
    wv = []
    cols = []
    for k in range(-r, r + 1):
        m = dark[yi, xi + k]
        hit |= m
    # толщина: максимум W в окрестности ±r (центр штриха), только где есть чернила
    wn = np.max(np.stack([W[yi, xi + k] for k in range(-r, r + 1)]), 0)
    wv = wn[hit]
    if rgb is not None:
        for k in range(-r, r + 1):
            m = dark[yi, xi + k]
            cols.append(rgb[yi[m], xi[m] + k])
        cols = np.concatenate(cols) if cols else np.zeros((0, 3))
    long_frac = comp_med = None
    if lab is not None:
        # компонента чернил (без сетки) под линией: у сплошной — длинная, у пунктира — короткие штрихи
        L = np.zeros(xi.size, np.int64)
        for k in range(-r, r + 1):
            lk = lab[yi, xi + k]
            L = np.where((L == 0) & (lk > 0), lk, L)
        sizes = ext[L[L > 0]]
        long_frac = round(float((sizes >= 60).mean()), 3) if sizes.size else 0.0
        comp_med = float(np.median(sizes)) if sizes.size else 0.0
    on, off = _runs(hit)
    on = [v for v in on if v >= 2]
    off = [v for v in off if v >= 2]
    res = {"n": int(xi.size), "cover": round(float(hit.mean()), 3),
           "dash": float(np.median(on)) if on else 0.0, "gap": float(np.median(off)) if off else 0.0,
           "n_gaps_per_100": round(100 * len(off) / xi.size, 2),
           "width": round(float(np.median(wv)), 2) if wv.size else 0.0,
           "long_frac": long_frac, "comp_med": comp_med,
           "width_p75": round(float(np.percentile(wv, 75)), 2) if wv.size else 0.0}
    if rgb is not None and len(cols):
        c = np.median(cols, 0)
        res["color"] = [int(v) for v in c]
        res["chroma"] = int(c.max() - c.min())
    return res


def one_scan(name, win):
    from PIL import Image
    from wellcurves import improj, strokes
    Image.MAX_IMAGE_PIXELS = None
    d = ROOT / "data" / "training" / name
    proj, tif = next(d.glob("*.improj")), next(d.glob("*.tif"))
    p = improj.load(proj)
    im = Image.open(tif)
    color = im.mode not in ("1", "L", "LA", "I;16", "I")
    out = []
    for gi, g in enumerate(p["Grids"]):
        items = [c for c in g["Curves"][0]["Items"] if len(c["Points"]) >= 20]
        if not items:
            continue
        try:
            geo = improj.GridGeom.from_grid(g)
        except Exception:  # noqa: BLE001
            continue
        ys_all = [q["Y"] for c in items for q in c["Points"]]
        ya = int(max(0, min(ys_all)))
        rows_per_ft = abs(geo.bottom_y - geo.top_y) / max(1e-6, abs(geo.max_depth - geo.min_depth))
        yb = int(min(im.size[1], ya + win * rows_per_ft))
        xa = int(max(0, min(geo.left_x) - 20)); xb = int(min(im.size[0], max(geo.right_x) + 20))
        if yb - ya < 50:
            continue
        crop = im.crop((xa, ya, xb, yb))
        rgb = np.array(crop.convert("RGB")) if color else None
        gray = np.array(crop.convert("L"))
        dark = gray < 128
        W, grid = strokes.width_and_grid(dark)
        import cv2
        n_l, lab, st, _ = cv2.connectedComponentsWithStats((dark & ~grid).astype(np.uint8), connectivity=8)
        ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        for c in items:
            pts = [(q["X"] - xa, q["Y"] - ya) for q in c["Points"] if ya <= q["Y"] < yb]
            f = profile_curve(dark, W, rgb, pts, ya, lab=lab, ext=ext)
            if f:
                out.append({"api": name, "g": gi, "grid": g["Name"], "curve": c["Name"].strip(),
                            "n_curves": len(items), "px_per_ft": round(rows_per_ft, 2), "color_scan": color, **f})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("apis", nargs="*"); ap.add_argument("--win", type=float, default=400)
    a = ap.parse_args()
    names = [d.name for d in sorted((ROOT / "data" / "training").iterdir())
             if d.is_dir() and list(d.glob("*.improj")) and list(d.glob("*.tif")) and (not a.apis or d.name in a.apis)]
    rows = []
    for n in names:
        try:
            r = one_scan(n, a.win)
        except Exception as e:  # noqa: BLE001
            print(n, "ошибка", e, flush=True); continue
        rows += r
        for x in r:
            print(f'{n} g{x["g"]} {x["curve"][:32]:32} cover {x["cover"]:.2f} dash {x["dash"]:5.0f} gap {x["gap"]:4.0f} '
                  f'w {x["width"]:4.1f} long {x["long_frac"]:.2f} comp {x["comp_med"]:5.0f} {x.get("color", "")}', flush=True)
    out = ROOT / "output" / "report" / "line_profile.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", out, len(rows))
