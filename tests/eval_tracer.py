"""Сравнение автообводки с ручной оцифровкой из эталонного .improj.

python tests/eval_tracer.py data/reference/tz/2done.improj [grid_index curve_index]
Печатает ошибку в единицах кривой (медиана, 90-й и 95-й перцентиль) и долю точек с ошибкой > 2% шкалы.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellcurves import improj, improj_tools, tracer  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def evaluate(proj_path, gi=0, ci=0, verbose=True, color=None, stroke=3, clean=True, tol=45):
    p = improj.load(proj_path)
    img = improj_tools.find_image(p, Path(proj_path))
    rgb = np.array(Image.open(img).convert("RGB"))
    g = p["Grids"][gi]
    c = g["Curves"][0]["Items"][ci]
    geo = improj.GridGeom.from_grid(g)
    y0, y1 = int(np.ceil(geo.top_y)), int(np.floor(geo.bottom_y))
    ys = np.arange(y0, y1)
    xl, xr = geo.edges(ys)
    dark = tracer.curve_mask(rgb, color, tol, y0=y0, y1=y1, x0=max(0, int(xl.min()) - 5), x1=int(xr.max()) + 5)
    gray = rgb[..., 0]
    ref_first = c["Points"][0]
    t = time.time()
    res = tracer.trace_track(dark, xl, xr, y0, y1, seed_x=ref_first["X"], stroke=stroke, clean=clean)
    dt = time.time() - t
    # значения автообводки
    levels = {m["MultiplierLevel"]: (m["MinX"], m["MaxX"]) for m in c["MultiplierLevels"]}
    base = min(levels)
    ok = ~np.isnan(res.xs)
    frac = (res.xs - xl) / (xr - xl)
    lv = res.level + base
    lo = np.array([levels.get(int(k), (c["MinX"], c["MaxX"]))[0] for k in lv])
    hi = np.array([levels.get(int(k), (c["MinX"], c["MaxX"]))[1] for k in lv])
    auto_val = lo + frac * (hi - lo)
    depth = geo.depth(ys)
    ref = improj.curve_values(g, c)
    order = np.argsort(ref["depth"])
    ref_val = np.interp(depth, ref["depth"][order], ref["value"][order], left=np.nan, right=np.nan)
    m = ok & ~np.isnan(ref_val)
    err = np.abs(auto_val[m] - ref_val[m])
    span = abs(c["MaxX"] - c["MinX"])
    out = {"curve": c["Name"], "rows": len(ys), "coverage": round(res.coverage, 3), "sec": round(dt, 1),
           "median": round(float(np.median(err)), 2), "p90": round(float(np.percentile(err, 90)), 2),
           "p95": round(float(np.percentile(err, 95)), 2),
           "bad_%(>2%span)": round(100 * float(np.mean(err > 0.02 * span)), 2), "span": span}
    out.update(geometric(res.points, c["Points"], gray.shape, float(np.mean(xr - xl)) / span))
    if verbose:
        print(out)
        bad = np.flatnonzero(m)[err > 0.05 * span]
        if bad.size:
            # сгруппировать плохие участки по глубине
            d = depth[bad]
            groups = np.split(d, np.where(np.diff(d) > 2)[0] + 1)
            print("  плохие участки (ft):", [(round(gr[0], 1), round(gr[-1], 1)) for gr in groups[:15]], "всего", len(groups))
    return out, res, depth, auto_val, ref_val


def _polyline_dist(points_a, points_b, shape):
    """Расстояние (px) от вершин линии B до ломаной A. Перехлёст (смена уровня) рвёт линию."""
    import cv2
    canvas = np.full(shape, 255, np.uint8)
    seg = []
    prev_lv = None
    for x, y, lv, nl in points_a:
        if prev_lv is not None and lv != prev_lv and len(seg) > 1:
            cv2.polylines(canvas, [np.array(seg, np.int32).reshape(-1, 1, 2)], False, 0, 1); seg = []
        seg.append((int(round(x)), int(round(y)))); prev_lv = lv
        if nl and len(seg) > 1:
            cv2.polylines(canvas, [np.array(seg, np.int32).reshape(-1, 1, 2)], False, 0, 1); seg = []
    if len(seg) > 1:
        cv2.polylines(canvas, [np.array(seg, np.int32).reshape(-1, 1, 2)], False, 0, 1)
    dt = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    xs = np.clip(np.round([p[0] for p in points_b]).astype(int), 0, shape[1] - 1)
    ys = np.clip(np.round([p[1] for p in points_b]).astype(int), 0, shape[0] - 1)
    return dt[ys, xs]


def geometric(auto_pts, ref_pts, shape, px_per_unit):
    ref = [(q["X"], q["Y"], q["MultiplierLevel"], q["NullsAfterThat"]) for q in ref_pts]
    y0 = max(min(p[1] for p in ref), min(p[1] for p in auto_pts))
    y1 = min(max(p[1] for p in ref), max(p[1] for p in auto_pts))
    ref_in = [p for p in ref if y0 <= p[1] <= y1]
    auto_in = [p for p in auto_pts if y0 <= p[1] <= y1]
    d_ref = _polyline_dist(auto_pts, ref_in, shape) / px_per_unit   # эталон → авто: пропущенные детали
    d_auto = _polyline_dist(ref, auto_in, shape) / px_per_unit      # авто → эталон: лишнее/ложное
    return {"geo_ref→auto_med": round(float(np.median(d_ref)), 2), "geo_ref→auto_p95": round(float(np.percentile(d_ref, 95)), 2),
            "geo_auto→ref_med": round(float(np.median(d_auto)), 2), "geo_auto→ref_p95": round(float(np.percentile(d_auto, 95)), 2),
            "geo_miss_%(>3u)": round(100 * float(np.mean(d_ref > 3)), 2), "geo_false_%(>3u)": round(100 * float(np.mean(d_auto > 3)), 2)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("proj"); ap.add_argument("gi", type=int, nargs="?", default=0); ap.add_argument("ci", type=int, nargs="?", default=0)
    ap.add_argument("--color", help="r,g,b"); ap.add_argument("--stroke", type=int, default=3)
    ap.add_argument("--noclean", action="store_true"); ap.add_argument("--tol", type=float, default=45)
    a = ap.parse_args()
    col = tuple(int(v) for v in a.color.split(",")) if a.color else None
    evaluate(a.proj, a.gi, a.ci, color=col, stroke=a.stroke, clean=not a.noclean, tol=a.tol)
