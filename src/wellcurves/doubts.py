"""Поиск сомнительных участков оцифровки: где программа (или человек) могла ошибиться.

Работает по готовому .improj и скану, поэтому годится и для автообводки, и для проверки чужой работы.
Причины:
  gap      — между соседними точками большой пропуск по глубине (линии нет или она потеряна)
  no_ink   — отрезок между точками идёт по пустому месту скана
  off_ink  — точки стоят не на чернилах
  together — кривая долго идёт вплотную к другой кривой той же дорожки (могли перепутаться)
  edge     — кривая прижата к краю дорожки (возможен срез или пропущенный перехлёст)
  level    — смена уровня шкалы (проверить, что перехлёст настоящий)
  uncovered — на скане в дорожке есть чернила, которые не покрыла ни одна кривая (срезанный пик, пропущенный кусок)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import improj, tracer

Image.MAX_IMAGE_PIXELS = None
REASONS = {"gap": "пропуск: линии нет или она потеряна", "no_ink": "линия проведена по пустому месту",
           "off_ink": "точки стоят не на линии скана", "together": "идёт вплотную к другой кривой",
           "edge": "прижата к краю дорожки", "level": "смена уровня шкалы",
           "uncovered": "на скане есть линия, которую обводка не покрыла (срезан пик или пропущен кусок)"}
SEVERITY = {"gap": 2, "no_ink": 3, "off_ink": 3, "together": 2, "edge": 1, "level": 1, "uncovered": 3}


def _runs(flags: np.ndarray, min_len: int):
    idx = np.flatnonzero(flags)
    if not idx.size:
        return []
    out = []
    for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if g.size >= min_len:
            out.append((int(g[0]), int(g[-1])))
    return out


def find_doubts(project: dict, image_path: str | Path, gap_px: int = 45, seg_px: int = 22) -> list[dict]:
    gray = np.array(Image.open(image_path).convert("L"))
    dark = tracer.dark_mask(gray)
    near = cv2.dilate(dark.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    H, W = dark.shape
    res = []
    for gi, g in enumerate(project["Grids"]):
        geo = improj.GridGeom.from_grid(g)
        curves = g["Curves"][0]["Items"]
        tracks = []
        for c in curves:
            p = c["Points"]
            x = np.array([q["X"] for q in p], float)
            y = np.array([q["Y"] for q in p], float)
            tracks.append((x, y))
        # непокрытые чернила дорожки: чистим сетку, рисуем все кривые толстой линией, смотрим остаток
        unc = {}                                         # curve index -> [(y0, y1)]
        have = [i for i, (x, y) in enumerate(tracks) if x.size >= 3]
        if have:
            ya = int(max(0, min(tracks[i][1].min() for i in have)))
            yb = int(min(H, max(tracks[i][1].max() for i in have) + 1))
            xa_ = int(max(0, np.floor(geo.left_x.min()) + 2))
            xb_ = int(min(W, np.ceil(geo.right_x.max()) - 1))
            if yb - ya > 50 and xb_ - xa_ > 20:
                sub = dark[ya:yb, xa_:xb_]
                ink, _, _ = tracer.clean_mask(sub, stroke=tracer.auto_stroke(sub))
                cov = np.zeros(ink.shape, np.uint8)
                for i in have:
                    x, y = tracks[i]
                    lv = np.array([q.get("MultiplierLevel") or 1 for q in curves[i]["Points"]])
                    pts = np.column_stack([x - xa_, y - ya]).astype(np.int32)
                    brk = np.flatnonzero((lv[:-1] != lv[1:]) | (np.diff(y) > gap_px)) + 1
                    for seg_ in np.split(pts, brk):
                        if len(seg_) > 1:
                            cv2.polylines(cov, [seg_.reshape(-1, 1, 2)], False, 1, thickness=9)
                left_over = ink & ~cov.astype(bool)
                left_over = cv2.morphologyEx(left_over.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                rows_bad = left_over.sum(1) >= 6
                for a, b in _runs(cv2.dilate(rows_bad.astype(np.uint8)[:, None], np.ones((5, 1), np.uint8))[:, 0] > 0, 3):
                    cols = np.flatnonzero(left_over[a:b + 1].any(0))
                    if not cols.size:
                        continue
                    xm, ym = cols.mean() + xa_, (a + b) / 2 + ya
                    # ближайшая по x кривая на этой глубине
                    best, bd = None, 1e9
                    for i in have:
                        x, y = tracks[i]
                        k = int(np.argmin(np.abs(y - ym)))
                        if abs(y[k] - ym) < 80 and abs(x[k] - xm) < bd:
                            best, bd = i, abs(x[k] - xm)
                    if best is not None:
                        unc.setdefault(best, []).append((a + ya, b + ya))
        for ci, c in enumerate(curves):
            p = c["Points"]
            x, y = tracks[ci]
            if x.size < 3:
                continue
            found = [(a, b, "uncovered") for a, b in unc.get(ci, [])]   # (y0, y1, reason)
            nulls = np.array([bool(q.get("NullsAfterThat")) for q in p])
            lvl = np.array([q.get("MultiplierLevel") or 1 for q in p])
            dy, dx = np.diff(y), np.diff(x)
            # 1) пропуски
            for i in np.flatnonzero((dy > gap_px) & ~nulls[:-1]):
                found.append((y[i], y[i + 1], "gap"))
            # 2) отрезки по пустому месту
            seg = np.hypot(dx, dy)
            for i in np.flatnonzero((seg >= seg_px) & (dy <= gap_px) & (lvl[:-1] == lvl[1:])):
                n = int(seg[i]) + 1
                xs = np.clip(np.round(np.linspace(x[i], x[i + 1], n)).astype(int), 0, W - 1)
                ys = np.clip(np.round(np.linspace(y[i], y[i + 1], n)).astype(int), 0, H - 1)
                if near[ys, xs].mean() < 0.45:
                    found.append((min(y[i], y[i + 1]), max(y[i], y[i + 1]), "no_ink"))
            # 3) точки не на чернилах
            xi = np.clip(np.round(x).astype(int), 0, W - 1)
            yi = np.clip(np.round(y).astype(int), 0, H - 1)
            for a, b in _runs(~near[yi, xi], 5):
                found.append((y[a], y[b], "off_ink"))
            # 4) прижата к краю
            left, right = geo.edges(y)
            for a, b in _runs((np.abs(x - left) <= 3) | (np.abs(x - right) <= 3), 12):
                found.append((y[a], y[b], "edge"))
            # 5) смена уровня
            for i in np.flatnonzero(lvl[:-1] != lvl[1:]):
                found.append((y[i] - 5, y[i + 1] + 5, "level"))
            # 6) вплотную к другой кривой
            o = np.argsort(y, kind="stable")
            yu, ui = np.unique(y[o], return_index=True)
            xu = x[o][ui]
            if yu.size > 2:
                rows = np.arange(int(yu[0]), int(yu[-1]) + 1)
                xa = np.interp(rows, yu, xu)
                for cj, (x2, y2) in enumerate(tracks):
                    if cj == ci or x2.size < 3:
                        continue
                    o2 = np.argsort(y2, kind="stable")
                    y2u, u2 = np.unique(y2[o2], return_index=True)
                    xb = np.interp(rows, y2u, x2[o2][u2], left=np.nan, right=np.nan)
                    for a, b in _runs(np.abs(xa - xb) <= 4, 25):
                        found.append((rows[a], rows[b], "together"))
            if not found:
                continue
            # склейка близких участков
            found.sort()
            merged = []
            for y0, y1, r in found:
                if merged and y0 <= merged[-1][1] + 30:
                    merged[-1][1] = max(merged[-1][1], y1)
                    merged[-1][2].add(r)
                else:
                    merged.append([y0, y1, {r}])
            for y0, y1, rs in merged:
                rs = sorted(rs, key=lambda r: -SEVERITY[r])
                res.append({"grid": gi, "curve": ci, "name": c["Name"].strip(), "y0": round(float(y0) - 8, 1),
                            "y1": round(float(y1) + 8, 1), "d0": round(float(geo.depth(y0)), 1),
                            "d1": round(float(geo.depth(y1)), 1), "reasons": rs,
                            "text": "; ".join(REASONS[r] for r in rs), "severity": max(SEVERITY[r] for r in rs)})
    res.sort(key=lambda d: (d["grid"], d["curve"], d["y0"]))
    return res


def summary(project: dict, doubts: list[dict]) -> list[dict]:
    """По кривым: сколько сомнительных участков и какая доля длины кривой под сомнением."""
    out = []
    for gi, g in enumerate(project["Grids"]):
        for ci, c in enumerate(g["Curves"][0]["Items"]):
            ys = [q["Y"] for q in c["Points"]]
            if len(ys) < 2:
                continue
            span = max(ys) - min(ys)
            ds = [d for d in doubts if d["grid"] == gi and d["curve"] == ci and d["severity"] >= 2]
            bad = sum(d["y1"] - d["y0"] for d in ds)
            out.append({"curve": c["Name"].strip(), "doubts": len(ds),
                        "doubt_share_%": round(100 * min(1.0, bad / span), 1) if span else 0.0})
    return out
