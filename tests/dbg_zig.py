"""Отладка: чистая маска + точки автообводки + эталон на куске дорожки.
python tests/dbg_zig.py <job.json> <track_i> <y0> <y1> [ref.improj grid_i]"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellcurves import tracer, improj
Image.MAX_IMAGE_PIXELS = None
job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); ti = int(sys.argv[2]); y0, y1 = int(sys.argv[3]), int(sys.argv[4])
tr = job["tracks"][ti]; cv = tr["curves"][0]
rgb = np.array(Image.open(ROOT / job["image"]).convert("RGB"))
ys = np.arange(y0, y1); left = np.full(ys.size, float(tr["left"])); right = np.full(ys.size, float(tr["right"]))
mask = tracer.curve_mask(rgb, cv.get("color"), 45, y0=y0, y1=y1, x0=int(left.min()) - 5, x1=int(right.max()) + 6)
xl = int(np.floor(left.min())) + 2; xr = int(np.ceil(right.max())) - 2
sub = mask[y0:y1, xl:xr + 1]
st = cv.get("stroke", 0) or tracer.auto_stroke(sub)
m, gr, gc = tracer.clean_mask(sub, stroke=st)
res = tracer.trace_track(mask, left, right, y0, y1, seed_x=cv.get("seed_x"), stroke=cv.get("stroke", 0), wrap=bool(cv.get("wrap")), zigzag=cv.get("zigzag"), max_level=1 + len(cv.get("wrap") or []))
Z = int(sys.argv[7]) if len(sys.argv) > 7 else 3
im = np.full((y1 - y0, xr - xl + 1, 3), 255, np.uint8); im[sub] = (200, 200, 200); im[m] = (90, 90, 90)
im = Image.fromarray(im).resize(((xr - xl + 1) * Z, (y1 - y0) * Z), Image.NEAREST); d = ImageDraw.Draw(im)
if len(sys.argv) > 5:
    p = improj.load(sys.argv[5]); c = p["Grids"][int(sys.argv[6])]["Curves"][0]["Items"][0]
    rp = [((q["X"] - xl) * Z, (q["Y"] - y0) * Z) for q in c["Points"] if y0 <= q["Y"] <= y1]
    if len(rp) > 1: d.line(rp, fill=(255, 0, 0), width=1)
ap = [((x - xl) * Z, (y - y0) * Z) for x, y, lv, n in res.points]
if len(ap) > 1: d.line(ap, fill=(0, 80, 255), width=2)
for x, y in ap: d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 160, 0))
out = ROOT / "output" / "look" / "dbg_zig.png"; im.save(out); print(out, "stroke", st, "pts", len(ap), "cov", round(res.coverage, 3))
