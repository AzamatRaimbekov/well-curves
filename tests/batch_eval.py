"""Пакетная оценка автообводки на скачанных задачах Kaiten (data/training/<API>/: скан + .improj).

Для каждой кривой: грид и шкала — из .improj исполнителя; цвет — по пикселям скана в точках исполнителя;
обводим окно (по умолчанию 4000 строк из середины интервала) и сравниваем геометрически с обводкой исполнителя.

python tests/batch_eval.py [--window 4000] [--workers 4]
→ output/batch_eval/results.json, summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

TRAIN = ROOT / "data" / "training"
OUT = ROOT / "output" / "batch_eval"


def sample_color(rgb, pts, max_n=400):
    """Цвет кривой на скане: в окрестности точек исполнителя берём самый «чернильный» пиксель."""
    h, w = rgb.shape[:2]
    cols = []
    for q in pts[:: max(1, len(pts) // max_n)]:
        x, y = int(round(q["X"])), int(round(q["Y"]))
        if not (2 <= x < w - 3 and 2 <= y < h - 3):
            continue
        patch = rgb[y - 1:y + 2, x - 2:x + 3].reshape(-1, 3).astype(int)
        sat = patch.max(1) - patch.min(1)
        ink = (255 - patch.mean(1)) + 2 * sat
        cols.append(patch[np.argmax(ink)])
    if not cols:
        return None, 0.0
    c = np.array(cols)
    sat = c.max(1) - c.min(1)
    frac_col = float(np.mean(sat > 60))
    if frac_col > 0.5:
        cc = c[sat > 60]
        return tuple(int(v) for v in np.median(cc, 0)), frac_col
    return None, frac_col


def eval_curve(args):
    proj_path, gi, ci, window = args
    from PIL import Image

    from wellcurves import improj, tracer
    from eval_tracer import geometric

    Image.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    p = improj.load(proj_path)
    g = p["Grids"][gi]
    c = g["Curves"][0]["Items"][ci]
    base = {"task": Path(proj_path).parent.name, "grid": g["Name"], "curve": c["Name"],
            "range": f'{c["MinX"]}..{c["MaxX"]}', "levels": len(c.get("MultiplierLevels") or []),
            "points": len(c["Points"])}
    try:
        if len(c["Points"]) < 20 or c["MinX"] == c["MaxX"]:
            return {**base, "status": "skip", "reason": "мало точек или шкала 0..0"}
        imgs = [f for f in Path(proj_path).parent.iterdir() if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")]
        if not imgs:
            return {**base, "status": "skip", "reason": "нет скана"}
        im = Image.open(imgs[0])
        rgb = np.array(im.convert("RGB"))
        geo = improj.GridGeom.from_grid(g)
        ys_ref = np.array([q["Y"] for q in c["Points"]])
        lo, hi = max(int(geo.top_y), int(ys_ref.min())), min(int(geo.bottom_y), int(ys_ref.max()))
        mid = (lo + hi) // 2
        y0, y1 = max(lo, mid - window // 2), min(hi, mid + window // 2)
        if y1 - y0 < 200:
            return {**base, "status": "skip", "reason": "короткий интервал"}
        ys = np.arange(y0, y1)
        xl, xr = geo.edges(ys)
        if rgb.shape[1] < xr.max() or rgb.shape[0] < y1:
            return {**base, "status": "skip", "reason": f"скан {rgb.shape[1]}x{rgb.shape[0]} не совпадает с improj"}
        color, frac_col = sample_color(rgb, [q for q in c["Points"] if y0 <= q["Y"] <= y1])
        mask = tracer.curve_mask(rgb, color, 45, y0=y0, y1=y1, x0=max(0, int(xl.min()) - 5), x1=int(xr.max()) + 6)
        # стартовая точка — первая точка исполнителя в окне (как «seed» от агента)
        first = next(q for q in c["Points"] if q["Y"] >= y0)
        thin = color is not None
        res = tracer.trace_track(mask, xl, xr, y0, y1, seed_x=first["X"], stroke=1 if thin else 0,
                                 clean=not thin, wrap=len(c.get("MultiplierLevels") or []) > 1)
        span = abs(c["MaxX"] - c["MinX"])
        ref_pts = [q for q in c["Points"] if y0 <= q["Y"] <= y1]
        if len(res.points) < 5 or len(ref_pts) < 5:
            return {**base, "status": "fail", "reason": "обводка не нашла кривую", "color": color,
                    "coverage": round(res.coverage, 3)}
        px_per_unit = float(np.mean(xr - xl)) / span
        geo_m = geometric(res.points, ref_pts, rgb.shape[:2], px_per_unit)
        # в % ширины шкалы — сравнимо между кривыми
        to_pct = 100 / span
        return {**base, "status": "ok", "color": color, "color_frac": round(frac_col, 2),
                "window_rows": int(y1 - y0), "coverage": round(res.coverage, 3), "sec": round(time.time() - t0, 1),
                "med_pct": round(geo_m["geo_ref→auto_med"] * to_pct, 2),
                "p95_pct": round(geo_m["geo_ref→auto_p95"] * to_pct, 2),
                "false_p95_pct": round(geo_m["geo_auto→ref_p95"] * to_pct, 2),
                "miss_%": geo_m["geo_miss_%(>3u)"], "false_%": geo_m["geo_false_%(>3u)"], **geo_m}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "error", "reason": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from wellcurves import improj

    jobs = []
    for d in sorted(TRAIN.iterdir()):
        projs = sorted(d.glob("*.improj"))
        if not projs:
            continue
        p = improj.load(projs[0])
        for gi, g in enumerate(p["Grids"]):
            for ci, _ in enumerate(g["Curves"][0]["Items"]):
                jobs.append((str(projs[0]), gi, ci, a.window))
    print("кривых для оценки:", len(jobs), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(eval_curve, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            results.append(r)
            print(f"[{i + 1}/{len(jobs)}] {r['task']} | {r['curve'][:35]} | {r['status']} "
                  f"{r.get('med_pct', '')} {r.get('p95_pct', '')} {r.get('reason', '')}", flush=True)
            (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("готово", OUT / "results.json")


if __name__ == "__main__":
    main()
