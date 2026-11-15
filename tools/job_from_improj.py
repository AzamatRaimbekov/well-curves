"""Задание автообводки из .improj исполнителя (для обучения/проверки алгоритма на готовых задачах).
Берём у исполнителя только геометрию: рамку, глубины, шкалы, уровни и стартовую точку каждой кривой.
python tools/job_from_improj.py <improj> <scan> [grid_index ...]  → data/jobs/ref_<API>_g<i>.json"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellcurves import improj  # noqa: E402


def make(proj_path, image, gi):
    p = improj.load(proj_path)
    g = p["Grids"][gi]
    geo = improj.GridGeom.from_grid(g)
    curves = []
    for c in g["Curves"][0]["Items"]:
        if len(c["Points"]) < 20 or c["MinX"] == c["MaxX"]:
            continue
        up = sorted([m for m in c["MultiplierLevels"] if m["MultiplierLevel"] >= 2], key=lambda m: m["MultiplierLevel"])
        low = sorted([m for m in c["MultiplierLevels"] if m["MultiplierLevel"] <= -1], key=lambda m: -m["MultiplierLevel"])
        first = min(c["Points"], key=lambda q: q["Y"])
        cv = {"name": c["Name"].strip(), "min": c["MinX"], "max": c["MaxX"], "seed_x": round(first["X"], 1)}
        if up:
            cv["wrap"] = [[m["MinX"], m["MaxX"]] for m in up]
        if low:
            cv["wrap_low"] = [[m["MinX"], m["MaxX"]] for m in low]
        if c.get("ScaleType") == 1:
            cv["scale"] = "log"
        curves.append(cv)
    ys = [q["Y"] for c in g["Curves"][0]["Items"] for q in c["Points"]]
    d0, d1 = float(geo.depth(min(ys))), float(geo.depth(max(ys)))
    return {"image": str(image), "well": p.get("WELL", ""), "api": p.get("API", ""),
            "depth": {"unit": "ft", "ref": [[geo.top_y, geo.min_depth], [geo.bottom_y, geo.max_depth]], "snap": False},
            "interval": [round(min(d0, d1), 1), round(max(d0, d1), 1)], "step": 0.1,
            "tracks": [{"name": g["Name"], "left": float(geo.left_x[0]), "right": float(geo.right_x[0]), "curves": curves}]}


if __name__ == "__main__":
    proj, image = Path(sys.argv[1]), Path(sys.argv[2])
    p = improj.load(proj)
    idx = [int(v) for v in sys.argv[3:]] or range(len(p["Grids"]))
    for gi in idx:
        job = make(proj, image.resolve(), gi)
        out = ROOT / "data" / "jobs" / f"ref_{proj.stem}_g{gi}.json"
        out.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        print(out, [c["name"] for c in job["tracks"][0]["curves"]], job["interval"])
