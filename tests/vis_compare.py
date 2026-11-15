"""Рисует скан + эталон (красный) + автообводку (синий) за интервал глубин."""
import sys, pickle
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from wellcurves import improj, improj_tools
from eval_tracer import evaluate
Image.MAX_IMAGE_PIXELS = None

def vis(proj, gi, ci, windows, out_prefix):
    o, res, depth, av, rv = evaluate(proj, gi, ci, verbose=True)
    p = improj.load(proj); g = p["Grids"][gi]; c = g["Curves"][0]["Items"][ci]
    geo = improj.GridGeom.from_grid(g)
    img = Image.open(improj_tools.find_image(p, Path(proj))).convert("RGB")
    for k, (d0, d1) in enumerate(windows):
        ya, yb = int(geo.y_of_depth(d0)), int(geo.y_of_depth(d1))
        xa, xb = int(geo.left_x.min()) - 20, int(geo.right_x.max()) + 20
        crop = Image.blend(img.crop((xa, ya, xb, yb)), Image.new("RGB", (xb - xa, yb - ya), "white"), 0.5)
        dr = ImageDraw.Draw(crop)
        ref = [(q["X"] - xa, q["Y"] - ya) for q in c["Points"] if ya <= q["Y"] <= yb]
        if len(ref) > 1: dr.line(ref, fill=(230, 0, 0), width=3)
        au = [(x - xa, y - ya) for x, y, lv, n in res.points if ya <= y <= yb]
        if len(au) > 1: dr.line(au, fill=(0, 60, 255), width=2)
        crop.save(f"{out_prefix}_{k}.png")
        print("saved", f"{out_prefix}_{k}.png", crop.size)

if __name__ == "__main__":
    a = sys.argv
    wins = [tuple(map(float, w.split("-"))) for w in a[4:]]
    vis(a[1], int(a[2]), int(a[3]), wins, str(ROOT / "output" / "look" / Path(a[1]).stem))
