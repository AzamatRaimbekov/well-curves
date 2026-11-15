"""Командная строка.

  python -m wellcurves colors  data/inbox/photo.jpg        # подсказать цвета кривых
  python -m wellcurves grid    data/inbox/photo.jpg        # сетка пикселей для калибровки
  python -m wellcurves run     data/jobs/<job>.json        # оцифровать по заданию
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .extract import dominant_colors, extract_curve, load_image
from .render import render_clean, render_overlay, write_csv, write_las

ROOT = Path(__file__).resolve().parents[2]


def cmd_colors(a):
    img = load_image(a.image)
    print(json.dumps(dominant_colors(img, a.k), ensure_ascii=False, indent=1))


def cmd_grid(a):
    """Сохраняет копию снимка с сеткой и номерами пикселей — по ней читаем опорные точки осей."""
    img = load_image(a.image)
    h, w = img.shape[:2]
    out = img.copy()
    step = a.step
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (200, 0, 200), 1)
        cv2.putText(out, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (200, 0, 200), 1)
        cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)
    dst = Path(a.out) if a.out else ROOT / "output" / f"{Path(a.image).stem}_grid.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", out)[1].tofile(str(dst))
    print(f"size: {w}x{h}\ngrid: {dst}")


def run_job(job_path: Path) -> dict:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    img_path = Path(job["image"])
    if not img_path.is_absolute():
        img_path = ROOT / img_path
    img = load_image(str(img_path), job.get("crop"))
    orientation = job.get("orientation", "vertical")
    curves = []
    for c in job["curves"]:
        axes = dict(job["axes"])
        if "axis" in c:  # у кривой своя шкала значений (своя дорожка/диапазон)
            axes["x" if orientation == "vertical" else "y"] = c["axis"]
        curves.append(extract_curve(img, c, axes, orientation))

    out_dir = ROOT / "output" / job_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    render_clean(curves, job, out_dir / "clean.png")
    render_overlay(img, curves, job, out_dir / "overlay.png")
    write_csv(curves, out_dir / "curves.csv", job.get("arg_name", "DEPT"))
    if orientation == "vertical" and job.get("las_step"):
        write_las(curves, out_dir / "curves.las", job["las_step"], job.get("meta", {}))

    report = {"job": job_path.name, "out": str(out_dir), "curves": []}
    for c in curves:
        r = {"name": c["name"], "coverage": round(c["coverage"], 3), "points": int(c["val"].size),
             "mask_pixels": c["mask_pixels"]}
        if c["val"].size:
            r.update(min=round(float(np.min(c["val"])), 3), max=round(float(np.max(c["val"])), 3),
                     arg_from=round(float(np.min(c["arg"])), 2), arg_to=round(float(np.max(c["arg"])), 2))
        r["warning"] = ("кривая не найдена — проверь цвет/tol" if c["val"].size == 0 else
                        "покрытие < 70% — проверь цвет, tol или разрывы" if c["coverage"] < 0.7 else None)
        report["curves"].append(r)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def cmd_run(a):
    print(json.dumps(run_job(Path(a.job)), ensure_ascii=False, indent=1))


def cmd_improj(a):
    """Разбор готового .improj: сводка, LAS/CSV, наложение на скан."""
    from . import improj, improj_tools

    path = Path(a.project)
    p = improj.load(path)
    out = ROOT / "output" / path.stem
    rows = improj.summary(p)
    files = improj_tools.export(p, out)
    img = Path(a.image) if a.image else improj_tools.find_image(p, path)
    if img and img.exists():
        files.append(improj_tools.overlay(p, img, out / "overlay.png", a.d_from, a.d_to))
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"well": p.get("WELL"), "api": p.get("API"), "image": str(img) if img else None,
                      "curves": rows, "files": [str(f) for f in files],
                      "roundtrip_ok": improj_tools.roundtrip_check(path)}, ensure_ascii=False, indent=1))


def cmd_check(a):
    """Автопроверка .improj: ошибки, пропуски, масштаб, оплата."""
    from PIL import Image

    from . import check, improj, improj_tools

    Image.MAX_IMAGE_PIXELS = None
    path = Path(a.project)
    p = improj.load(path)
    dpi = a.dpi
    img = improj_tools.find_image(p, path)
    if not dpi and img:
        info = Image.open(img).info.get("dpi")
        dpi = float(info[1]) if info else None
    res = check.check_project(p, dpi)
    res["dpi"] = dpi
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_auto(a):
    """Автообводка всех кривых скана по заданию."""
    from . import pipeline

    print(json.dumps(pipeline.run(a.job), ensure_ascii=False, indent=1))


def cmd_verify(a):
    """Сверка автообводки с эталонным .improj (в % шкалы)."""
    from . import verify

    print(json.dumps(verify.compare(a.auto, a.ref, a.step, a.tol), ensure_ascii=False, indent=1))


def cmd_doubts(a):
    """Сомнительные участки .improj: где обводка могла ошибиться."""
    from . import doubts, improj, improj_tools

    pr = improj.load(a.project)
    img = a.image or improj_tools.find_image(pr, Path(a.project))
    d = doubts.find_doubts(pr, img)
    Path(a.project).with_suffix(".doubts.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"total": len(d), "serious": sum(x["severity"] >= 2 for x in d),
                      "by_curve": doubts.summary(pr, d)}, ensure_ascii=False, indent=1))


def cmd_edit(a):
    """Редактор в браузере: правка кривых по сомнительным местам."""
    from . import editor

    editor.serve(a.project, a.image, a.port, not a.no_browser)


def cmd_layout(a):
    from PIL import Image

    from . import pipeline

    Image.MAX_IMAGE_PIXELS = None
    rgb = np.array(Image.open(a.image).convert("RGB"))
    y0, y1 = (int(v) for v in a.rows.split(":"))
    res = pipeline.detect_layout(rgb, y0, y1)
    res["image_size"] = [rgb.shape[1], rgb.shape[0]]
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_crop(a):
    """Вырезает кусок скана для просмотра (длинные сканы целиком не открыть)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(a.image)
    y0, y1 = (int(v) for v in a.rows.split(":"))
    c = im.crop((0, y0, im.width, min(im.height, y1))).convert("RGB")
    if c.width > a.max_w:
        c = c.resize((a.max_w, int(c.height * a.max_w / c.width)))
    dst = Path(a.out) if a.out else ROOT / "output" / "look" / f"{Path(a.image).stem}_{y0}_{y1}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    c.save(dst)
    print(f"{dst}  (масштаб {c.width / im.width:.3f})")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="wellcurves")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("colors"); s.add_argument("image"); s.add_argument("-k", type=int, default=8); s.set_defaults(f=cmd_colors)
    s = sub.add_parser("grid"); s.add_argument("image"); s.add_argument("--step", type=int, default=50); s.add_argument("--out"); s.set_defaults(f=cmd_grid)
    s = sub.add_parser("run"); s.add_argument("job"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("improj"); s.add_argument("project"); s.add_argument("--image")
    s.add_argument("--from", dest="d_from", type=float); s.add_argument("--to", dest="d_to", type=float)
    s.set_defaults(f=cmd_improj)
    s = sub.add_parser("auto"); s.add_argument("job"); s.set_defaults(f=cmd_auto)
    s = sub.add_parser("verify"); s.add_argument("auto"); s.add_argument("ref")
    s.add_argument("--step", type=float, default=0.5); s.add_argument("--tol", type=float, default=1.0); s.set_defaults(f=cmd_verify)
    s = sub.add_parser("doubts"); s.add_argument("project"); s.add_argument("--image"); s.set_defaults(f=cmd_doubts)
    s = sub.add_parser("edit"); s.add_argument("project"); s.add_argument("--image"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true"); s.set_defaults(f=cmd_edit)
    s = sub.add_parser("layout"); s.add_argument("image"); s.add_argument("--rows", required=True); s.set_defaults(f=cmd_layout)
    s = sub.add_parser("crop"); s.add_argument("image"); s.add_argument("--rows", required=True)
    s.add_argument("--max-w", dest="max_w", type=int, default=1000); s.add_argument("--out"); s.set_defaults(f=cmd_crop)
    s = sub.add_parser("check"); s.add_argument("project"); s.add_argument("--dpi", type=float); s.set_defaults(f=cmd_check)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
