"""Сверка режима «по толщине штриха» с эталоном исполнителя на окне глубин.
python tools/eval_width.py <API> <grid> <имя кривой>=<solid|dashed>[:lo[:core]] ... [--win 400] [--from D]
Пример: python tools/eval_width.py 10539239 1 "Density Porosity_pu=solid" "Neutron Porosity_pu=dashed"
Порядок кривых в командной строке = порядок обводки (сначала сплошная, потом пунктир)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def main():
    import numpy as np
    from PIL import Image
    from wellcurves import pipeline, verify
    import job_from_improj
    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser()
    ap.add_argument("api"); ap.add_argument("grid", type=int); ap.add_argument("curves", nargs="+")
    ap.add_argument("--win", type=float, default=400); ap.add_argument("--from", dest="d0", type=float)
    ap.add_argument("--plain", action="store_true", help="старый режим (без толщины) — для сравнения")
    a = ap.parse_args()
    d = ROOT / "data" / "training" / a.api
    proj, tif = next(d.glob("*.improj")), next(d.glob("*.tif"))
    job = job_from_improj.make(proj, tif.resolve(), a.grid)
    by = {c["name"]: c for c in job["tracks"][0]["curves"]}
    order = []
    for spec in a.curves:
        name, _, st = spec.partition("=")
        style, *nums = st.split(":")
        c = by[name]
        if not a.plain:
            lo = float(nums[0]) if nums else 4
            core = float(nums[1]) if len(nums) > 1 else lo + 1
            c.update(style=style, width=[lo], core=core, ink_reward=0.5 if style == "solid" else 1.5,
                     core_gain=1.0 if style == "solid" else 2.0)
        order.append(c)
    job["tracks"][0]["curves"] = order
    (y1, d1), (y2, d2) = job["depth"]["ref"]
    lo_d = a.d0 if a.d0 is not None else job["interval"][0]
    job["interval"] = [lo_d, min(job["interval"][1], lo_d + a.win)]
    row = lambda dd: y1 + (dd - d1) * (y2 - y1) / (d2 - d1)
    full = Image.open(tif)
    ya = int(max(0, min(row(job["interval"][0]), row(job["interval"][1])) - 80))
    yb = int(min(full.size[1], max(row(job["interval"][0]), row(job["interval"][1])) + 80))
    rgb = np.array(full.crop((0, ya, full.size[0], yb)).convert("RGB"))
    job["depth"]["ref"] = [[y1 - ya, d1], [y2 - ya, d2]]
    # стартовая точка: положение эталона в начале окна (в полном эталоне она относится к началу кривой)
    from wellcurves import improj
    g = improj.load(proj)["Grids"][a.grid]
    for c in g["Curves"][0]["Items"]:
        if c["Name"].strip() in by:
            pts = sorted(c["Points"], key=lambda q: abs(q["Y"] - (ya + 80)))
            by[c["Name"].strip()]["seed_x"] = round(pts[0]["X"], 1)
    out = ROOT / "output" / "eval_width" / f"{a.api}_g{a.grid}{'_plain' if a.plain else ''}"
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "job.json"
    jp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    pipeline.run(jp, rgb=rgb, out_dir=out)
    for r in verify.compare(out / "auto.improj", proj, 0.5, 1.0):
        if r.get("curve") in by and r.get("grid") == job["tracks"][0]["name"]:
            print(f"{a.api} g{a.grid} {r['curve'][:40]:40} совпало {r.get('good_%')}%  p95 {r.get('geo_p95_pct')}%  "
                  f"эталон {r.get('ref_ft')} ft, не покрыто {r.get('missing_ft')} ft")
    print("→", out)


if __name__ == "__main__":
    main()
