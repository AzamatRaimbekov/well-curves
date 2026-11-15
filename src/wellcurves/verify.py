"""Сверка автообводки с эталоном (.improj исполнителя) в единицах кривой.

Обе кривые переводятся в (глубина, значение) и сравниваются на общей сетке глубин.
Два показателя:
  * «по глубине» — |авто − эталон| на той же глубине (строго: на крутом пике сдвиг в 0.5 ft даёт большую разницу);
  * «геометрический» — расстояние от точки эталона до ближайшего места автокривой в плоскости
    (значение в % шкалы, глубина в окне ±tol_ft) — так сравнивает глаз проверяющего.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import improj


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _series(g, c, step, d0, d1):
    v = improj.curve_values(g, c)
    if v["depth"].size < 2:
        return None
    o = np.argsort(v["depth"], kind="stable")
    grid, col = improj.resample(v["depth"][o], v["value"][o], v["nulls_after"][o], step, d0, d1)
    col = col.astype(float)
    col[col == -999.25] = np.nan
    return grid, col


def _curves(project):
    out = {}
    for g in project["Grids"]:
        for c in g["Curves"][0]["Items"]:
            out[(_norm(g["Name"]), _norm(c["Name"]))] = (g, c)
    return out


def compare(auto_path: str | Path, ref_path: str | Path, step: float = 0.5, tol_ft: float = 1.0,
            bad_pct: float = 3.0) -> list[dict]:
    """Список результатов по кривым: медиана/p95 ошибки, доля хороших точек, худшие интервалы глубин."""
    a, r = improj.load(auto_path), improj.load(ref_path)
    ac, rc = _curves(a), _curves(r)
    res = []
    for key, (rg, rcv) in rc.items():
        row = {"grid": rg["Name"], "curve": rcv["Name"].strip()}
        match = ac.get(key) or next((v for k, v in ac.items() if k[1] == key[1]), None)
        if match is None:      # имена различаются хвостом (единицы, номер трека) — сравниваем по началу
            match = next((v for k, v in ac.items() if key[1].startswith(k[1]) or k[1].startswith(key[1])), None)
        if match is None:
            res.append({**row, "status": "нет в автообводке"})
            continue
        ag, acv = match
        span = abs(rcv["MaxX"] - rcv["MinX"])
        if span == 0 or len(rcv["Points"]) < 2:
            res.append({**row, "status": "эталон без шкалы/точек"})
            continue
        rv = improj.curve_values(rg, rcv)
        d0, d1 = float(np.floor(rv["depth"].min())), float(np.ceil(rv["depth"].max()))
        sa, sr = _series(ag, acv, step, d0, d1), _series(rg, rcv, step, d0, d1)
        if sa is None:
            res.append({**row, "status": "автообводка пустая"})
            continue
        grid, va = sa
        _, vr = sr
        both = ~np.isnan(va) & ~np.isnan(vr)
        if both.sum() < 10:
            res.append({**row, "status": "нет общих глубин"})
            continue
        diff = np.abs(va - vr) / span * 100            # % шкалы на той же глубине
        # геометрическая: минимум по окну ±tol_ft от |авто(d+δ) − эталон(d)| с учётом сдвига по глубине
        k = int(round(tol_ft / step))
        geo = np.full(grid.size, np.inf)
        for s in range(-k, k + 1):
            sh = np.roll(va, s)
            if s > 0:
                sh[:s] = np.nan
            elif s < 0:
                sh[s:] = np.nan
            d = np.abs(sh - vr) / span * 100
            geo = np.fmin(geo, d)
        # между соседними отсчётами автокривая проходит все промежуточные значения
        lo = np.fmin(va, np.roll(va, 1)); hi = np.fmax(va, np.roll(va, 1))
        for s in range(-k, k + 1):
            l2, h2 = np.roll(lo, s), np.roll(hi, s)
            inside = (vr >= l2) & (vr <= h2)
            geo[inside] = 0.0
        ok = both & np.isfinite(geo)
        bad = ok & (geo > bad_pct)
        # худшие интервалы глубин (склеиваем соседние плохие отсчёты, разрыв до 5 ft)
        ivs, idx = [], np.flatnonzero(bad)
        if idx.size:
            start = prev = idx[0]
            for i in idx[1:]:
                if (i - prev) * step > 5:
                    ivs.append((start, prev)); start = i
                prev = i
            ivs.append((start, prev))
        ivs = sorted(ivs, key=lambda t: -(t[1] - t[0]))[:8]
        ref_only = ~np.isnan(vr) & np.isnan(va)
        auto_only = np.isnan(vr) & ~np.isnan(va)
        res.append({**row, "status": "ok", "scale": f'{rcv["MinX"]}..{rcv["MaxX"]}',
                    "ref_ft": round(float((~np.isnan(vr)).sum() * step), 1),
                    "auto_ft": round(float((~np.isnan(va)).sum() * step), 1),
                    "missing_ft": round(float(ref_only.sum() * step), 1),
                    "extra_ft": round(float(auto_only.sum() * step), 1),
                    "same_depth_med_pct": round(float(np.median(diff[both])), 2),
                    "same_depth_p95_pct": round(float(np.percentile(diff[both], 95)), 2),
                    "geo_med_pct": round(float(np.median(geo[ok])), 2),
                    "geo_p95_pct": round(float(np.percentile(geo[ok], 95)), 2),
                    "geo_p95_units": round(float(np.percentile(geo[ok], 95)) * span / 100, 2),
                    "good_%": round(float(100 - bad.sum() / ok.sum() * 100), 2),
                    "bad_ft": round(float(bad.sum() * step), 1),
                    "worst": [[round(float(grid[s]), 1), round(float(grid[e]), 1)] for s, e in ivs]})
    return res
