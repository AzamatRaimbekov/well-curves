"""Быстрая сверка на всех эталонах: у каждой дорожки берётся окно глубин (по умолчанию первые 400 ft),
скан обрезается до окна — прогон всех пар занимает минуты, а не часы. Для сравнения версий алгоритма.
python tools/quick_eval.py <метка> [--win 400] [--jobs 4] [API ...]  → output/report/quick_<метка>.json"""
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def one_scan(args):
    name, tag, win = args
    import numpy as np
    from PIL import Image
    from wellcurves import improj, pipeline, verify
    import job_from_improj
    Image.MAX_IMAGE_PIXELS = None
    d = ROOT / "data" / "training" / name
    proj, tif = next(d.glob("*.improj")), next(d.glob("*.tif"))
    rows = []
    try:
        n = len(improj.load(proj)["Grids"])
        full = np.array(Image.open(tif).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        return [{"api": name, "status": f"не открылось: {e}"}]
    for gi in range(n):
        t0 = time.time()
        try:
            job = job_from_improj.make(proj, tif.resolve(), gi)
            if not job["tracks"][0]["curves"]:
                continue
            (y1, d1), (y2, d2) = job["depth"]["ref"]
            if d1 == d2 or y1 == y2:
                rows.append({"api": name, "g": gi, "status": "рамка эталона без глубин"})
                continue
            a, b = job["interval"]
            b = min(b, a + win)
            job["interval"] = [a, b]
            row = lambda dd: y1 + (dd - d1) * (y2 - y1) / (d2 - d1)
            ya = int(max(0, min(row(a), row(b)) - 80))
            yb = int(min(full.shape[0], max(row(a), row(b)) + 80))
            job["depth"]["ref"] = [[y1 - ya, d1], [y2 - ya, d2]]
            out = ROOT / "output" / "quick" / tag / f"{name}_g{gi}"
            out.mkdir(parents=True, exist_ok=True)
            jp = out / "job.json"
            jp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            pipeline.run(jp, rgb=full[ya:yb], out_dir=out)
            res = verify.compare(out / "auto.improj", proj, 0.5, 1.0)
            want = {c["name"] for c in job["tracks"][0]["curves"]}
            for r in res:
                if r.get("curve") in want and r.get("grid") == job["tracks"][0]["name"]:
                    rows.append({"api": name, "g": gi, "ncurves": len(want), "sec": round(time.time() - t0),
                                 **{k: v for k, v in r.items() if k != "worst"}})
        except Exception as e:  # noqa: BLE001
            rows.append({"api": name, "g": gi, "status": f"ошибка: {e}", "tb": traceback.format_exc()[-600:]})
    return rows


def summary(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    g = sorted(r["good_%"] for r in ok)
    if not g:
        return {}
    return {"curves": len(g), "errors": len(rows) - len(g), "mean": round(sum(g) / len(g), 1),
            "median": g[len(g) // 2], ">=95": sum(v >= 95 for v in g), ">=80": sum(v >= 80 for v in g),
            "<20": sum(v < 20 for v in g)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("apis", nargs="*")
    ap.add_argument("--win", type=float, default=400); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    names = [d.name for d in sorted((ROOT / "data" / "training").iterdir())
             if list(d.glob("*.improj")) and list(d.glob("*.tif")) and (not a.apis or d.name in a.apis)]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, rs in zip(names, ex.map(one_scan, [(n, a.tag, a.win) for n in names])):
            rows += rs
            print(name, summary(rs), flush=True)
    out = ROOT / "output" / "report" / f"quick_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ИТОГ", summary(rows), "→", out)
