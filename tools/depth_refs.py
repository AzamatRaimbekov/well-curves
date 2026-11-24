"""Опорные точки глубины по толстым горизонтальным линиям сетки.
python tools/depth_refs.py <скан> <y0> <y1> <x0> <x1> <строка> <глубина> <px_на_ft> [--step 50]
<строка>,<глубина> — одна подписанная отметка (примерно); остальные линии ищутся от неё с шагом step.
Печатает JSON-список [[строка, глубина], ...] для поля depth.ref задания."""
import argparse, json, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def find(image, y0, y1, x0, x1, row, depth, ppf, step=50.0, radius=14, min_thick=3):
    im = Image.open(image)
    dark = np.array(im.crop((x0, y0, x1, y1)).convert("L")) < 128
    frac = dark.mean(1)
    thick = np.convolve(frac >= 0.6, np.ones(min_thick), "same") >= min_thick     # линия толщиной ≥ min_thick строк
    rows = np.flatnonzero(thick)
    groups = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1) if len(rows) else []
    centers = np.array([g.mean() + y0 for g in groups if len(g)])
    refs, cur_row, cur_d = [], float(row), float(depth)
    for sign in (1, -1):
        r, d = float(row), float(depth)
        if len(centers):
            j = np.argmin(abs(centers - r))
            if abs(centers[j] - r) <= radius:
                r = float(centers[j])
        if sign == 1 and y0 <= r < y1:
            refs.append([round(r, 1), d])
        while True:
            exp, d = r + sign * step * ppf, d + sign * step
            if not (y0 <= exp < y1):
                break
            j = np.argmin(abs(centers - exp)) if len(centers) else None
            if j is not None and abs(centers[j] - exp) <= radius:
                r = float(centers[j]); refs.append([round(r, 1), d])
            else:
                r = exp                                         # линии нет (разрыв листа) — идём по расчёту
    return sorted(refs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); [ap.add_argument(n, type=float) for n in ("y0", "y1", "x0", "x1", "row", "depth", "ppf")]
    ap.add_argument("--step", type=float, default=50.0); ap.add_argument("--thick", type=int, default=3)
    a = ap.parse_args()
    r = find(a.image, int(a.y0), int(a.y1), int(a.x0), int(a.x1), a.row, a.depth, a.ppf, a.step, min_thick=a.thick)
    print(json.dumps(r)); print(len(r), "точек", file=sys.stderr)
