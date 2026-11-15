"""Сквозная проверка: рисуем «фото» с известными кривыми, оцифровываем, сравниваем."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from wellcurves.cli import run_job  # noqa: E402


def make_chart(path: Path):
    """Бланк 400x1000: глубина 1000..1500 м по Y (пиксели 0..999), значение 0..150 по X (пиксели 0..399)."""
    h, w = 1000, 400
    img = np.full((h, w, 3), 255, np.uint8)
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), (190, 190, 190), 1)  # серая сетка
    for y in range(0, h, 50):
        cv2.line(img, (0, y), (w, y), (190, 190, 190), 1)
    ys = np.arange(h)
    depth = 1000 + ys * 0.5
    gr = 75 + 40 * np.sin(depth / 25)          # «ГК», зелёная
    sp = 40 + 20 * np.cos(depth / 40)          # «ПС», красная
    for vals, bgr in [(gr, (0, 160, 0)), (sp, (0, 0, 220))]:
        px = (vals / 150 * (w - 1)).astype(np.int32)
        cv2.polylines(img, [np.column_stack([px, ys]).reshape(-1, 1, 2)], False, bgr, 2)
    cv2.imwrite(str(path), img)
    return depth, {"GR": gr, "SP": sp}


def test_roundtrip(tmp_path=None):
    tmp = Path(tmp_path or ROOT / "output" / "_test")
    tmp.mkdir(parents=True, exist_ok=True)
    img = tmp / "synthetic.png"
    depth, truth = make_chart(img)
    job = {
        "image": str(img), "orientation": "vertical", "las_step": 0.5,
        "meta": {"well": "TEST-1", "field": "Synthetic"},
        "axes": {"x": {"p1": [0, 0], "p2": [399, 150]}, "y": {"p1": [0, 1000], "p2": [999, 1499.5]}},
        "curves": [
            {"name": "GR", "unit": "API", "color_rgb": [0, 160, 0], "tol": 35},
            {"name": "SP", "unit": "mV", "color_rgb": [220, 0, 0], "tol": 35},
        ],
    }
    job_path = tmp / "synthetic_job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    report = run_job(job_path)
    for c in report["curves"]:
        assert c["coverage"] > 0.95, c
    out = Path(report["out"])
    for f in ["clean.png", "overlay.png", "curves.csv", "curves.las", "report.json"]:
        assert (out / f).exists(), f
    # точность: сравниваем с истинной кривой
    import csv
    rows = list(csv.DictReader(open(out / "curves.csv", encoding="utf-8-sig"), delimiter=";"))
    for name, vals in truth.items():
        got = [(float(r["DEPT"]), float(r["value"])) for r in rows if r["curve"] == name]
        d, v = np.array(got).T
        err = np.abs(v - np.interp(d, depth, vals))
        print(name, "median err", round(float(np.median(err)), 3), "p95", round(float(np.percentile(err, 95)), 3))
        assert np.median(err) < 1.0  # < 0.7% шкалы


if __name__ == "__main__":
    test_roundtrip()
    print("OK")
