"""Точность распознавания типа линии (lines.py) на базе: что распознала программа от стартовой точки
против того, что видно под обводкой исполнителя (output/report/line_profile.json).
python tools/eval_lines.py [API ...] [--win 400]  → таблица и итог: совпало kind (solid/dashed/thin/straight)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))


def main():
    from PIL import Image
    from wellcurves import improj, lines, tracer, pipeline
    import job_from_improj
    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser(); ap.add_argument("apis", nargs="*"); ap.add_argument("--win", type=float, default=400)
    ap.add_argument("--joint", default="{}")
    a = ap.parse_args()
    lines.JOINT.update(json.loads(a.joint))
    prof = {(r["api"], r["g"], r["curve"]): r for r in
            json.loads((ROOT / "output/report/line_profile.json").read_text(encoding="utf-8"))}
    names = sorted({k[0] for k in prof if not a.apis or k[0] in a.apis})
    ok = tot = 0
    conf = {}
    for name in names:
        d = ROOT / "data/training" / name
        proj, tif = next(d.glob("*.improj")), next(d.glob("*.tif"))
        p = improj.load(proj); im = Image.open(tif)
        for gi, g in enumerate(p["Grids"]):
            try:
                job = job_from_improj.make(proj, tif, gi)
            except Exception:  # noqa: BLE001
                continue
            cvs = job["tracks"][0]["curves"]
            if not cvs:
                continue
            (y1, d1), (y2, d2) = job["depth"]["ref"]
            if d1 == d2:
                continue
            row = lambda dd: y1 + (dd - d1) * (y2 - y1) / (d2 - d1)
            ya = int(max(0, row(job["interval"][0])))
            yb = int(min(im.size[1], ya + 1500))
            tr = job["tracks"][0]
            xl = int(np.floor(tr["left"])) + 2; xr = int(np.ceil(tr["right"])) - 2
            rgb = np.array(im.crop((max(0, xl - 5), ya, min(im.size[0], xr + 6), yb)).convert("RGB"))
            mask = tracer.curve_mask(rgb, None)
            sub = mask[:, 5: 5 + xr - xl + 1]
            seeds = []
            for c in cvs:
                it = [q for q in g["Curves"][0]["Items"] if q["Name"].strip() == c["name"]]
                q = min(it[0]["Points"], key=lambda q: abs(q["Y"] - ya)) if it else None
                seeds.append(None if q is None or abs(q["Y"] - ya) > 40 else q["X"] - xl)
            h = sub.shape[0]
            feats = lines.recognize(sub, seeds, np.full(h, 0.0), np.full(h, sub.shape[1] - 1.0))
            for c, f in zip(cvs, feats):
                r = prof.get((name, gi, c["name"]))
                if not r:
                    continue
                want = lines.classify({**r, "dash_len": 0}).get("_kind", "-")
                got = lines.classify(f).get("_kind", "-") if f else "-"
                tot += 1; ok += want == got
                conf[(want, got)] = conf.get((want, got), 0) + 1
                print(f"{name} g{gi} {c['name'][:26]:26} эталон {want:8} программа {got:8} "
                      f"[{lines.recognize.why[cvs.index(c)]}] (ref long {r['long_frac']} w {r['width']} | got {f and f['long_frac']} w {f and f['width']})", flush=True)
    print(f"ИТОГ: совпало {ok} из {tot} ({100 * ok / max(1, tot):.0f} %)")
    for k, v in sorted(conf.items()):
        print("  эталон", k[0], "→ программа", k[1], ":", v)


if __name__ == "__main__":
    main()
