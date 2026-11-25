"""Картинка: эталон исполнителя (зелёный) и автообводка (красный) одной кривой поверх скана.
python tools/ref_vs_auto.py <эталон.improj> <auto.improj> <грид-индекс эталона> "<кривая>" <y0> <y1> <out.png> [--scale 0.5]"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellcurves import improj  # noqa: E402
Image.MAX_IMAGE_PIXELS = None


def pts(p, name, gi=None):
    for i, g in enumerate(p["Grids"]):
        if gi is not None and i != gi:
            continue
        for c in g["Curves"][0]["Items"]:
            if c["Name"].strip() == name:
                return [(q["X"], q["Y"]) for q in sorted(c["Points"], key=lambda q: q["Y"])]
    return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("auto"); ap.add_argument("gi", type=int); ap.add_argument("curve")
    ap.add_argument("y0", type=int); ap.add_argument("y1", type=int); ap.add_argument("out")
    ap.add_argument("--yoff", type=int, default=0, help="сдвиг строк авто-проекта (если он считался по обрезке)")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    ref = improj.load(a.ref); auto = improj.load(a.auto)
    tif = next(Path(a.ref).parent.glob("*.tif"))
    im = Image.open(tif).crop((0, a.y0, Image.open(tif).size[0], a.y1)).convert("RGB")
    im = Image.blend(im, Image.new("RGB", im.size, "white"), 0.55)
    d = ImageDraw.Draw(im)
    for P, col in ((pts(ref, a.curve, a.gi), (0, 170, 0)), (pts(auto, a.curve), (230, 0, 0))):
        off = a.yoff if col[0] == 230 else 0
        q = [(x, y + off - a.y0) for x, y in P if a.y0 <= y + off < a.y1]
        for u, v in zip(q, q[1:]):
            if abs(v[1] - u[1]) < 30:
                d.line([u, v], fill=col, width=3)
    im = im.resize((int(im.size[0] * a.scale), int(im.size[1] * a.scale)))
    im.save(a.out); print(a.out)
