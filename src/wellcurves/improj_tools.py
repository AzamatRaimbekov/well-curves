"""Инструменты поверх .improj: сводка, выгрузка LAS/CSV, наложение на скан, круговая проверка записи."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import improj

Image.MAX_IMAGE_PIXELS = None


def find_image(project: dict, proj_path: Path) -> Path | None:
    """ImagePath в проекте — путь на чужом компьютере; ищем файл с тем же именем рядом с проектом."""
    p = Path(project["ImagePath"])
    if p.exists():
        return p
    name = project["ImagePath"].replace("\\", "/").split("/")[-1]
    cand = proj_path.parent / name
    if cand.exists():
        return cand
    # скачанные из Kaiten сканы переименованы в <API>.tif — берём единственный скан в папке
    imgs = [f for f in proj_path.parent.iterdir()
            if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg") and "underlay" not in f.name and "_grid" not in f.name]
    return imgs[0] if len(imgs) == 1 else None


def export(project: dict, out_dir: Path, step: float | None = None) -> list[Path]:
    """CSV (длинный формат) и LAS 2.0 на шаге проекта (по умолчанию 0.1 ft)."""
    import lasio

    out_dir.mkdir(parents=True, exist_ok=True)
    step = step or project.get("Step") or 0.1
    files = []
    with open(out_dir / "curves.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["grid", "curve", "depth_ft", "value", "x_px", "y_px", "type"])
        for g in project["Grids"]:
            for c in g["Curves"][0]["Items"]:
                v = improj.curve_values(g, c)
                for d, val, x, y, p in zip(v["depth"], v["value"], v["x"], v["y"], c["Points"]):
                    w.writerow([g["Name"], c["Name"], f"{d:.2f}", f"{val:.4f}", int(x), int(y),
                                "manual" if p["Type"] == 1 else "auto"])
    files.append(out_dir / "curves.csv")

    las = lasio.LASFile()
    las.well["WELL"].value = project.get("WELL", "")
    las.well["API"].value = project.get("API", "")
    las.well["NULL"].value = -999.25
    curves = []
    for g in project["Grids"]:
        for c in g["Curves"][0]["Items"]:
            v = improj.curve_values(g, c)
            if v["depth"].size > 1:
                curves.append((c["Name"], v))
    if curves:
        d0 = min(v["depth"].min() for _, v in curves)
        d1 = max(v["depth"].max() for _, v in curves)
        d0, d1 = np.floor(d0 / step) * step, np.ceil(d1 / step) * step
        grid = None
        used = set()
        for name, v in curves:
            grid, col = improj.resample(v["depth"], v["value"], v["nulls_after"], step, d0, d1)
            if grid is not None and "DEPT" not in las.curves.keys():
                las.append_curve("DEPT", grid, unit="FT")
            mnem = "".join(ch for ch in name if ch.isalnum())[:16] or "CURVE"
            while mnem in used:
                mnem += "_"
            used.add(mnem)
            las.append_curve(mnem, col, descr=name)
        las.write(str(out_dir / "curves.las"), version=2.0)
        files.append(out_dir / "curves.las")
    return files


def overlay(project: dict, image_path: Path, out_png: Path, d_from=None, d_to=None,
            max_height: int = 6000) -> Path:
    """Скан + оцифрованные кривые + рамки дорожек. Длинные сканы режутся по глубине (d_from..d_to)."""
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    y0, y1 = 0, H
    if d_from is not None or d_to is not None:
        g = improj.GridGeom.from_grid(project["Grids"][0])
        if d_from is not None:
            y0 = max(0, int(g.y_of_depth(d_from)) - 50)
        if d_to is not None:
            y1 = min(H, int(g.y_of_depth(d_to)) + 50)
    crop = im.crop((0, y0, W, y1))
    faded = Image.blend(crop, Image.new("RGB", crop.size, "white"), 0.45)
    dr = ImageDraw.Draw(faded)
    for g in project["Grids"]:
        geo = improj.GridGeom.from_grid(g)
        ys = np.linspace(geo.top_y, geo.bottom_y, 50)
        l, r = geo.edges(ys)
        for xs in (l, r):
            dr.line([(float(x), float(y - y0)) for x, y in zip(xs, ys)], fill=(0, 120, 255), width=2)
        dr.line([(l[0], geo.top_y - y0), (r[0], geo.top_y - y0)], fill=(0, 120, 255), width=2)
        dr.line([(l[-1], geo.bottom_y - y0), (r[-1], geo.bottom_y - y0)], fill=(0, 120, 255), width=2)
        for c in g["Curves"][0]["Items"]:
            col = (c["Color"]["R"], c["Color"]["G"], c["Color"]["B"])
            seg = []
            for p in c["Points"]:
                if y0 <= p["Y"] <= y1:
                    seg.append((p["X"], p["Y"] - y0))
                if p.get("NullsAfterThat") and seg:
                    if len(seg) > 1:
                        dr.line(seg, fill=col, width=3)
                    seg = []
            if len(seg) > 1:
                dr.line(seg, fill=col, width=3)
    if faded.height > max_height:
        k = max_height / faded.height
        faded = faded.resize((max(1, int(faded.width * k)), max_height))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    faded.save(out_png)
    return out_png


def roundtrip_check(path: Path) -> bool:
    """Пересобираем проект через build_project и сверяем значения — проверка, что запись верна."""
    p = improj.load(path)
    specs = []
    for g in p["Grids"]:
        items = g["Bounds"][0]["Items"]
        corners = {b["Name"]: (b["ImagePoint"]["X"], b["ImagePoint"]["Y"]) for b in items if b["BoundType"] in (1, 2, 3, 4)}
        inter = [b for b in items if b["BoundType"] == 5]
        pairs = [((a["ImagePoint"]["X"], a["ImagePoint"]["Y"]), (b["ImagePoint"]["X"], b["ImagePoint"]["Y"]))
                 for a, b in zip(inter[::2], inter[1::2])]
        curves = [improj.CurveSpec(c["Name"], c["MinX"], c["MaxX"],
                                   [(q["X"], q["Y"], q["MultiplierLevel"], q["NullsAfterThat"]) for q in c["Points"]],
                                   (c["Color"]["R"], c["Color"]["G"], c["Color"]["B"]), c["ScaleType"] == 1,
                                   [(m["MultiplierLevel"], m["MinX"], m["MaxX"]) for m in c["MultiplierLevels"]])
                  for c in g["Curves"][0]["Items"]]
        specs.append(improj.GridSpec(g["Name"], g["MinY"], g["MaxY"], corners, curves, pairs))
    q = json.loads(json.dumps(improj.build_project(p["ImagePath"], specs, p.get("WELL", ""), p.get("API", ""))))
    for ga, gb in zip(p["Grids"], q["Grids"]):
        for ca, cb in zip(ga["Curves"][0]["Items"], gb["Curves"][0]["Items"]):
            va, vb = improj.curve_values(ga, ca), improj.curve_values(gb, cb)
            if va["value"].size and not np.allclose(va["value"], vb["value"], equal_nan=True):
                return False
    return set(p.keys()) == set(q.keys())
