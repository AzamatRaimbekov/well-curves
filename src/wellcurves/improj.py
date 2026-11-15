"""Чтение и запись проектов SoloImageAnalyzer (.improj) — формат сдачи работы в РЕ-БУЛАК.

.improj — это JSON (UTF-8 с BOM). Устройство, выясненное по эталонам из data/reference/tz:

Project
  ImagePath, WELL, API, Unit (0 = ft), Step (шаг выгрузки, 0.1)
  Grids[]                      — рамка одной дорожки (track) в заданном интервале глубин
    MinY / MaxY                — глубина верха / низа рамки (ft)
    Bounds[0].Items[]          — опорные точки рамки в пикселях скана:
        BoundType 1 TopLeft, 2 TopRight, 3 BottomRight, 4 BottomLeft,
        5 Intermediate — пары левый/правый край на промежуточных строках
        (компенсация перекоса/«уезжания» дорожки на длинном скане)
    Curves[0].Items[]          — кривые дорожки
        Name, MinX / MaxX      — значения на левом / правом краю дорожки (может быть MinX > MaxX)
        ScaleType 0 lin, 1 log
        MultiplierLevels[]     — уровни «перехлёста» (backup scale): уровень 1 = 0..150,
                                 уровень 2 = 150..300 и т.д., у каждого свой MinX/MaxX
        Points[]               — точки кривой в ПИКСЕЛЯХ скана (X, Y), Y по возрастанию
            Type 1 = поставлена вручную, 2 = снята автоматически
            MultiplierLevel     — на каком уровне шкалы точка
            NullsAfterThat      — после этой точки разрыв (нет данных)

Глубина точки — линейно по Y между верхом и низом рамки.
Значение — по X между левым и правым краем рамки на этой строке (с учётом Intermediate).
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOUND_NAMES = {1: "TopLeft", 2: "TopRight", 3: "BottomRight", 4: "BottomLeft"}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(project: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8-sig")


# ---------------------------------------------------------------- геометрия рамки
@dataclass
class GridGeom:
    top_y: float
    bottom_y: float
    min_depth: float
    max_depth: float
    edge_y: np.ndarray      # строки, где известны края
    left_x: np.ndarray
    right_x: np.ndarray

    @classmethod
    def from_grid(cls, g: dict) -> "GridGeom":
        items = g["Bounds"][0]["Items"]
        pts = {}
        inter = []
        for b in items:
            ip = b["ImagePoint"]
            if b["BoundType"] in BOUND_NAMES:
                pts[b["BoundType"]] = (ip["X"], ip["Y"])
            else:
                inter.append((ip["X"], ip["Y"]))
        top_y = (pts[1][1] + pts[2][1]) / 2
        bottom_y = (pts[3][1] + pts[4][1]) / 2
        rows = [(pts[1][1], pts[1][0], pts[2][0]), (pts[4][1], pts[4][0], pts[3][0])]
        # промежуточные точки идут парами (левый, правый) на одной строке
        inter.sort(key=lambda p: (round(p[1], 1), p[0]))
        for i in range(0, len(inter) - 1, 2):
            (x1, y1), (x2, y2) = inter[i], inter[i + 1]
            rows.append(((y1 + y2) / 2, min(x1, x2), max(x1, x2)))
        rows.sort()
        r = np.array(rows)
        return cls(top_y, bottom_y, float(g["MinY"]), float(g["MaxY"]), r[:, 0], r[:, 1], r[:, 2])

    def depth(self, y):
        t = (np.asarray(y, float) - self.top_y) / (self.bottom_y - self.top_y)
        return self.min_depth + t * (self.max_depth - self.min_depth)

    def y_of_depth(self, d):
        t = (np.asarray(d, float) - self.min_depth) / (self.max_depth - self.min_depth)
        return self.top_y + t * (self.bottom_y - self.top_y)

    def edges(self, y):
        y = np.asarray(y, float)
        return np.interp(y, self.edge_y, self.left_x), np.interp(y, self.edge_y, self.right_x)


def _scale(v0, v1, t, log):
    if log:
        return 10 ** (math.log10(v0) + t * (math.log10(v1) - math.log10(v0)))
    return v0 + t * (v1 - v0)


def curve_values(g: dict, c: dict) -> dict:
    """Точки кривой → глубина (ft) и значение в единицах кривой."""
    geo = GridGeom.from_grid(g)
    pts = c["Points"]
    if not pts:
        return {"depth": np.array([]), "value": np.array([]), "x": np.array([]), "y": np.array([]),
                "nulls_after": np.array([], bool)}
    x = np.array([p["X"] for p in pts], float)
    y = np.array([p["Y"] for p in pts], float)
    lvl = np.array([p.get("MultiplierLevel") or 1 for p in pts])
    nulls = np.array([bool(p.get("NullsAfterThat")) for p in pts])
    levels = {m["MultiplierLevel"]: (m["MinX"], m["MaxX"]) for m in c.get("MultiplierLevels") or []}
    left, right = geo.edges(y)
    t = (x - left) / (right - left)
    log = c.get("ScaleType", 0) == 1
    val = np.array([_scale(*levels.get(int(l), (c["MinX"], c["MaxX"])), ti, log) for ti, l in zip(t, lvl)])
    return {"depth": geo.depth(y), "value": val, "x": x, "y": y, "nulls_after": nulls, "level": lvl}


def resample(depth, value, nulls_after, step: float, d0: float, d1: float, null=-999.25):
    """Равномерная сетка глубин; разрывы (NullsAfterThat) → null."""
    grid = np.round(np.arange(d0, d1 + step / 2, step), 4)
    out = np.interp(grid, depth, value, left=np.nan, right=np.nan)
    for i in np.flatnonzero(nulls_after[:-1]):
        out[(grid > depth[i]) & (grid < depth[i + 1])] = np.nan
    out[np.isnan(out)] = null
    return grid, out


def summary(project: dict) -> list[dict]:
    rows = []
    for g in project["Grids"]:
        for c in g["Curves"][0]["Items"]:
            v = curve_values(g, c)
            n = len(c["Points"])
            manual = sum(1 for p in c["Points"] if p["Type"] == 1)
            rows.append({
                "grid": g["Name"], "curve": c["Name"], "range": f'{c["MinX"]}..{c["MaxX"]}',
                "levels": len(c.get("MultiplierLevels") or []),
                "depth": f'{g["MinY"]}..{g["MaxY"]}',
                "points": n, "manual_%": round(100 * manual / n, 1) if n else 0,
                "from": round(float(v["depth"].min()), 1) if n else None,
                "to": round(float(v["depth"].max()), 1) if n else None,
                "vmin": round(float(np.nanmin(v["value"])), 2) if n else None,
                "vmax": round(float(np.nanmax(v["value"])), 2) if n else None,
            })
    return rows


# ---------------------------------------------------------------- запись нового проекта
def _uid() -> str:
    return str(uuid.uuid4())


def _pt(x, y, typ=0, level=0, nulls=False, curve_uid=None):
    return {"Type": typ, "X": float(x), "Y": float(y), "OriginalY": 0, "RealX": 0, "RealY": 0,
            "ParentCurveUid": curve_uid, "NullsAfterThat": nulls, "MultiplierLevel": level,
            "SelectedColor": None, "IsLeft": False}


DEFAULT_EXTRACT = {"Type": 0, "Threshold": 40, "Persistence": 5, "GridSizeX": 1, "GridSizeY": 20,
                   "SizeX": 3, "SizeY": 3, "Tolerance": 30, "SimpleExtract": True, "Pixelization": False,
                   "RemoveGrid": False, "FilterRepeates": True, "FilterPeaks": True, "Resample": True,
                   "WrapPoints": False, "AutoLeveling": False, "WrapShiftThreshold": 100,
                   "MultiplierLevelThreshold": 2, "CleanWrapTails": False, "ShiftFromBound": 3,
                   "EvaluatePeak": True}


@dataclass
class CurveSpec:
    name: str
    min_x: float
    max_x: float
    points: list            # [(x_px, y_px, level, nulls_after)]
    color: tuple = (255, 0, 0)
    log: bool = False
    levels: list = field(default_factory=list)   # [(level, min_x, max_x)]


@dataclass
class GridSpec:
    name: str
    min_depth: float
    max_depth: float
    corners: dict           # {"TopLeft":(x,y), "TopRight":..., "BottomRight":..., "BottomLeft":...}
    curves: list
    intermediate: list = field(default_factory=list)   # [((xl,y),(xr,y)), ...]


def build_project(image_path: str, grids: list[GridSpec], well: str = "", api: str = "",
                  save_path: str = "") -> dict:
    """Собирает .improj в том же виде, что сохраняет SoloImageAnalyzer."""
    out_grids = []
    for gs in grids:
        guid = _uid()
        items = []
        for bt, name in BOUND_NAMES.items():
            x, y = gs.corners[name]
            items.append({"Uid": _uid(), "Name": name, "BoundType": bt, "ImagePoint": _pt(x, y), "DeepValue": 0})
        for i, (l, r) in enumerate(gs.intermediate):
            for k, (x, y) in enumerate((l, r)):
                items.append({"Uid": _uid(), "Name": f"{2 * i + k + 1}_Intermediate", "BoundType": 5,
                              "ImagePoint": _pt(x, y), "DeepValue": 0})
        curves = []
        for cs in gs.curves:
            cuid = _uid()
            col = {"B": cs.color[2], "G": cs.color[1], "R": cs.color[0]}
            levels = cs.levels or [(1, cs.min_x, cs.max_x)]
            curves.append({
                "Uid": cuid, "Name": cs.name, "Unit": None, "ScaleType": 1 if cs.log else 0,
                "MinX": cs.min_x, "MaxX": cs.max_x, "ParentGridUid": guid,
                "Points": [_pt(x, y, 2, lv, nl, cuid) for x, y, lv, nl in cs.points],
                "RegionPoints": [], "Color": col, "Thickness": 2, "Opacity": 100,
                "ExtractParams": dict(DEFAULT_EXTRACT),
                "CustomPointsSettings": {"ExponentialSmoothing": 30},
                "SelectedColor": {"B": 0, "G": 0, "R": 0}, "Description": None,
                "MultiplierLevels": [{"Uid": _uid(), "Name": None, "IsVisible": True, "Color": col,
                                      "MultiplierLevel": lv, "ValuesRange": f"{a:g} : {b:g}",
                                      "MinX": a, "MaxX": b} for lv, a, b in levels],
                "IsSelectedForExport": True, "SelectedColorStatus": "#FF000000",
                "ColorAtPoint": {"ColorContext": None, "A": 255, "R": 0, "G": 0, "B": 0,
                                 "ScA": 1, "ScR": 0, "ScG": 0, "ScB": 0},
            })
        out_grids.append({"Uid": guid, "Name": gs.name, "Unit": 0, "MinY": gs.min_depth, "MaxY": gs.max_depth,
                          "GridStepX": None, "GridStepY": None,
                          "Bounds": [{"Uid": _uid(), "Name": "Bounds", "ParentGridUid": guid, "Items": items}],
                          "Curves": [{"Uid": _uid(), "Name": "Curves", "ParentGridUid": guid, "Items": curves}],
                          "Scale": 0})
    first = out_grids[0] if out_grids else None
    return {
        "Zoom": 1.0, "ZoomStatus": "100%", "SelectedPoints": [], "ImagePath": image_path,
        "SelectedColor": None, "ColorAtPoint": None, "SelectedColorStatus": "", "Status": "",
        "SaveProjectPath": save_path, "Grids": out_grids,
        "SelectedGridUid": first["Uid"] if first else None, "AutosaveInterval": 0, "DoNotChangeState": False,
        "SelectedCurveUid": first["Curves"][0]["Items"][0]["Uid"] if first and first["Curves"][0]["Items"] else None,
        "WELL": well, "API": api, "Unit": 0, "Step": 0.1,
    }
