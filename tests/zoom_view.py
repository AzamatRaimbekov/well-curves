"""Крупный вид: скан + точки .improj. python tests/zoom_view.py <improj> <scan> x0 y0 x1 y1 [Z] [out]"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellcurves import improj
Image.MAX_IMAGE_PIXELS = None
pr = improj.load(sys.argv[1]); x0, y0, x1, y1 = map(int, sys.argv[3:7]); Z = int(sys.argv[7]) if len(sys.argv) > 7 else 4
im = Image.open(sys.argv[2]).convert("RGB").crop((x0, y0, x1, y1))
im = Image.blend(im, Image.new("RGB", im.size, "white"), 0.55).resize((im.width * Z, im.height * Z), Image.NEAREST)
d = ImageDraw.Draw(im)
for g in pr["Grids"]:
    for c in g["Curves"][0]["Items"]:
        col = (c["Color"]["R"], c["Color"]["G"], c["Color"]["B"])
        pts = [q for q in c["Points"] if y0 - 40 <= q["Y"] <= y1 + 40]
        seg = []
        for i, q in enumerate(pts):
            seg.append(((q["X"] - x0 + 0.5) * Z, (q["Y"] - y0 + 0.5) * Z))
            nxt = pts[i + 1] if i + 1 < len(pts) else None
            if nxt is None or q.get("NullsAfterThat") or nxt["MultiplierLevel"] != q["MultiplierLevel"]:
                if len(seg) > 1: d.line(seg, fill=col, width=2)
                seg = []
        for q in pts:
            x, y = (q["X"] - x0 + 0.5) * Z, (q["Y"] - y0 + 0.5) * Z
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 0, 0))
out = sys.argv[8] if len(sys.argv) > 8 else str(ROOT / "output" / "look" / "zoom.png"); im.save(out); print(out, im.size)
