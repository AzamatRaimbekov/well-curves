"""Распознавание типа линии кривой по скану: сплошная / пунктир и толщина штриха.

Зачем: на ч/б логах кривые одной дорожки различаются только линией. Раньше тип линии и толщину в задание
вписывал человек (`style`, `width`, `core`), и при ошибке обводка уходила на соседнюю линию.
Теперь при `"style": "auto"` программа сама смотрит, какая линия у кривой в начале интервала.

Как: от стартовой точки (seed_x) проводится короткий пробный путь (первые ~700 строк) по чернилам без сетки.
Вдоль пути меряем:
  long_frac — доля пути, где чернила под ним принадлежат длинному куску (≥ 60 px): сплошная ≈ 1, пунктир < 0.7;
  width     — медианная толщина штриха под путём, px.
Признаки и пороги подобраны по базе готовых задач (`tools/line_profile.py`, отчёт output/report/line_profile.json).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import strokes

LONG = 60          # кусок чернил длиннее LONG px считается частью сплошной линии
SOLID_MIN = 0.5    # long_frac ≥ SOLID_MIN → сплошная (по базе: ошибка «сплошная→пунктир» 6 из 62, она самая вредная)
THIN = 3.5         # штрих тоньше THIN px — тонкая линия


def path_features(dark: np.ndarray, W: np.ndarray, lab: np.ndarray, ext: np.ndarray, xs: np.ndarray,
                  r: int = 3, grid: np.ndarray | None = None) -> dict | None:
    """xs[i] — x пути в строке i (nan — нет точки). Признаки линии под путём."""
    ok = ~np.isnan(xs)
    yi = np.flatnonzero(ok)
    if yi.size < 30:
        return None
    xi = np.round(xs[ok]).astype(int)
    H, Wd = dark.shape
    keep = (xi >= r) & (xi < Wd - r)
    yi, xi = yi[keep], xi[keep]
    on_grid = 0.0
    if grid is not None:
        g = np.zeros(yi.size, bool)
        for k in range(-1, 2):
            g |= grid[yi, np.clip(xi + k, 0, Wd - 1)]
        on_grid = float(g.mean())
    L = np.zeros(yi.size, np.int64)
    wmax = np.zeros(yi.size, np.float32)
    for k in range(-r, r + 1):
        lk = lab[yi, xi + k]
        L = np.where((L == 0) & (lk > 0), lk, L)
        wmax = np.maximum(wmax, np.where(dark[yi, xi + k], W[yi, xi + k], 0))
    hit = L > 0
    if hit.sum() < 20:
        return None
    sizes = ext[L[hit]]
    short = sizes[sizes < LONG]
    return {"cover": round(float(hit.mean()), 3), "on_grid": round(on_grid, 3),
            "dash_len": float(np.median(short)) if short.size else 0.0,
            "long_frac": round(float((sizes >= LONG).mean()), 3),
            "width": round(float(np.median(wmax[hit])), 2)}


def classify(f: dict | None) -> dict:
    """Признаки → параметры обводки. Пустой словарь = оставить старый режим."""
    if not f:
        return {}
    if f.get("on_grid", 0) >= 0.5:      # прямая вдоль линии сетки (Bit Size и т.п.): режим толщины её снимет
        return {"_kind": "straight", **_f(f)}
    w = f["width"]
    if w < THIN:                         # тонкая линия: режим толщины её теряет — старый режим
        return {"_kind": "thin", **_f(f)}
    lo = float(np.clip(np.floor(w) - 1, 2, 4))
    if f["long_frac"] >= SOLID_MIN:
        return {"style": "solid", "width": [lo], "core": lo + 1, "stroke_w": w, "ink_reward": 0.5, "_kind": "solid", **_f(f)}
    return {"style": "dashed", "width": [lo], "core": lo + 1, "ink_reward": 1.5, "core_gain": 2.0,
            "dash_max": int(max(24, 2 * f.get("dash_len", 0))), "_kind": "dashed", **_f(f)}


def _f(f):
    return {"_features": f}


def follow(dark: np.ndarray, x0: float, left=None, right=None, step: int = 12, max_gap: int = 40) -> np.ndarray:
    """Пробный путь «ближайший штрих»: в каждой строке берём чернила, ближайшие к x предыдущей строки (не дальше
    step px + по 1 px на каждую пустую строку). Путь держится своей линии и не перескакивает на жирную соседнюю,
    как это делает путь с выигрышем за чернила."""
    h, w = dark.shape
    xs = np.full(h, np.nan)
    x, gap = float(x0), 0
    for y in range(h):
        lo = int(max(0, x - step - gap)); hi = int(min(w, x + step + gap + 1))
        if left is not None:
            lo = max(lo, int(left[y])); hi = min(hi, int(right[y]) + 1)
        row = np.flatnonzero(dark[y, lo:hi]) + lo
        if row.size:
            # отрезок чернил, ближайший к x; точка — его центр
            runs = np.split(row, np.flatnonzero(np.diff(row) > 1) + 1)
            c = min(runs, key=lambda r: min(abs(r[0] - x), abs(r[-1] - x)))
            if c.size <= 3 * step:            # длинный горизонтальный отрезок — линия сетки/подпись, не берём
                x = float(c.mean()); xs[y] = x; gap = 0
                continue
        gap += 1
        if gap > max_gap:
            break
    return xs


def bridge_grid(ink: np.ndarray, grid: np.ndarray, reach: int = 4) -> np.ndarray:
    """Склеивает линию, разрезанную снятой сеткой: пиксель сетки возвращается, если чернила есть и выше, и ниже
    (в пределах reach строк) — или левее и правее. Иначе сплошная кривая на густой сетке распадается на кусочки
    между линиями сетки и выглядит как пунктир."""
    def near(a, axis, sign):
        out = np.zeros_like(a)
        for k in range(1, reach + 1):
            out |= np.roll(a, sign * k, axis=axis)
        return out
    vert = near(ink, 0, 1) & near(ink, 0, -1)
    horz = near(ink, 1, 1) & near(ink, 1, -1)
    return ink | (grid & (vert | horz))


def recognize(sub: np.ndarray, seeds: list, left: np.ndarray, right: np.ndarray, rows: int = 700) -> list:
    """sub — маска чернил дорожки (строки интервала, столбцы дорожки); seeds — x старта каждой кривой (в координатах
    sub) или None. left/right — края дорожки по строкам (в координатах sub). Возвращает признаки для каждой кривой."""
    top = sub[: min(rows, sub.shape[0])]
    W, grid = strokes.width_and_grid(top)
    ink = top & ~grid
    _, lab, st, _ = cv2.connectedComponentsWithStats(bridge_grid(ink, grid).astype(np.uint8), connectivity=8)
    ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    out, paths = [], []
    for s in seeds:
        if s is None:
            out.append(None); paths.append(None)
            continue
        lt, rt = left[: top.shape[0]], right[: top.shape[0]]
        # «щуп» идёт по штрихам толщиной от 2 px без сетки: тонкая сетка в 1 px (точечная, серая) его не уводит
        xs = follow(ink & (W >= 2.0), s, lt, rt)
        if np.isfinite(xs).mean() < 0.3:
            xs = follow(ink, s, lt, rt)
        if np.isfinite(xs).mean() < 0.3:            # своей линии без сетки нет: прямая вдоль сетки (Bit Size)
            xs = follow(top, s, lt, rt)
        out.append(path_features(top, W, lab, ext, xs, grid=grid)); paths.append(xs)
    # «щупы» двух кривых сошлись на одну линию (кривые стартуют рядом) — признаки относятся не к своей кривой.
    # Такое распознавание ненадёжно: оставляем этим кривым старый режим
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i], paths[j]
            if a is None or b is None:
                continue
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() > 30 and (np.abs(a[both] - b[both]) < 4).mean() > 0.5:
                out[i] = out[j] = None
    return out
