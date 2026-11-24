"""Толщина штриха в каждой точке чернил — признак, по которому разделяются чёрные линии на 1-битном скане:
толстая кривая (4–5 px), тонкая кривая (2–3 px), сетка (1–2 px). Квадратное открытие 3×3 этого не различает:
после него толстая кривая, тонкая линия температуры и жирные линии сетки остаются одной связной маской."""
from __future__ import annotations

import numpy as np


def _run_axis0(a: np.ndarray) -> np.ndarray:
    """Длина непрерывного отрезка чернил вдоль оси 0, проходящего через пиксель."""
    h = a.shape[0]
    f = np.zeros(a.shape, np.int32)
    f[0] = a[0]
    for y in range(1, h):
        f[y] = (f[y - 1] + 1) * a[y]
    b = np.zeros(a.shape, np.int32)
    b[-1] = a[-1]
    for y in range(h - 2, -1, -1):
        b[y] = (b[y + 1] + 1) * a[y]
    return np.maximum(f + b - 1, 0)


def _run_diag(a: np.ndarray, dx: int) -> np.ndarray:
    """То же вдоль диагонали (dx = +1: вправо-вниз, −1: влево-вниз)."""
    h, w = a.shape
    f = np.zeros(a.shape, np.int32)
    f[0] = a[0]
    b = np.zeros(a.shape, np.int32)
    b[-1] = a[-1]
    for y in range(1, h):
        prev = np.zeros(w, np.int32)
        if dx > 0:
            prev[1:] = f[y - 1, :-1]
        else:
            prev[:-1] = f[y - 1, 1:]
        f[y] = (prev + 1) * a[y]
    for y in range(h - 2, -1, -1):
        nxt = np.zeros(w, np.int32)
        if dx > 0:
            nxt[:-1] = b[y + 1, 1:]
        else:
            nxt[1:] = b[y + 1, :-1]
        b[y] = (nxt + 1) * a[y]
    return np.maximum(f + b - 1, 0)


def run_lengths(dark: np.ndarray):
    """(по горизонтали, по вертикали) — длины отрезков чернил через каждый пиксель."""
    a = dark.astype(np.int32)
    return _run_axis0(a.T.copy()).T, _run_axis0(a)


def stroke_width(dark: np.ndarray, runs=None) -> np.ndarray:
    """Толщина штриха: минимум из длин отрезков чернил по 4 направлениям (диагональные × √2).
    Не зависит от наклона линии: для линии толщиной t под любым углом даёт ≈ t."""
    a = dark.astype(np.int32)
    hr, vr = runs if runs is not None else run_lengths(dark)
    d1 = _run_diag(a, 1) * 1.4142
    d2 = _run_diag(a, -1) * 1.4142
    return np.minimum(np.minimum(hr, vr), np.minimum(d1, d2)).astype(np.float32)


def grid_lines(dark: np.ndarray, runs=None, min_len: int = 150, max_thick: int = 6, col_fill: float = 0.5,
               block: int = 600) -> np.ndarray:
    """Маска прямых линий сетки. Горизонтальная: отрезок чернил длиннее min_len при малой высоте.
    Вертикальная: столбец, заполненный больше col_fill в блоке строк (кривая столько прямо не стоит,
    а если стоит — её защищает protect в задании)."""
    hr, vr = runs if runs is not None else run_lengths(dark)
    g = dark & (hr >= min_len) & (vr <= max_thick)
    h = dark.shape[0]
    for y in range(0, h, block):
        blk = dark[y:y + block]
        if blk.shape[0] < 100:
            break
        cols = blk.mean(0) > col_fill
        if cols.any():
            g[y:y + block] |= blk & cols[None, :] & (vr[y:y + block] >= 40) & (hr[y:y + block] <= max_thick)
    return g


def width_and_grid(dark: np.ndarray, chunk: int = 3000, ov: int = 200):
    """Толщина штриха и маска сетки для высокой дорожки — кусками по строкам (память), с перекрытием."""
    h = dark.shape[0]
    W = np.zeros(dark.shape, np.float32)
    G = np.zeros(dark.shape, bool)
    for y in range(0, h, chunk):
        a, b = max(0, y - ov), min(h, y + chunk + ov)
        part = dark[a:b]
        runs = run_lengths(part)
        w = stroke_width(part, runs)
        g = grid_lines(part, runs)
        e = min(h, y + chunk)
        W[y:e] = w[y - a:e - a]
        G[y:e] = g[y - a:e - a]
    return W, G


def width_mask(dark: np.ndarray, lo: float, hi: float = 99.0, W=None, grid=None) -> np.ndarray:
    """Чернила линий толщиной lo..hi px без прямых линий сетки. Вокруг толстого ядра штрих восстанавливается
    целиком (края штриха по диагональным отрезкам выглядят тоньше ядра)."""
    import cv2
    if W is None or grid is None:
        W, grid = width_and_grid(dark)
    ink = dark & ~grid
    core = ink & (W >= lo)
    if hi < 90:
        too = cv2.dilate((ink & (W > hi)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        core &= ~too
    k = max(3, int(lo) | 1)
    return cv2.dilate(core.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ink


def dash_filter(m: np.ndarray, max_len: int = 24, min_area: int = 10, need: np.ndarray | None = None) -> np.ndarray:
    """Только короткие штрихи пунктира: компоненты не длиннее max_len по обеим осям. Сплошные линии
    (температура, рамка, остатки сетки) длиннее штриха и отсеиваются. need — маска «толстого ядра»:
    штрих берётся, только если в нём есть такие пиксели (тонкий пунктир соседней кривой не проходит)."""
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = (st[:, cv2.CC_STAT_WIDTH] <= max_len) & (st[:, cv2.CC_STAT_HEIGHT] <= max_len) & (st[:, cv2.CC_STAT_AREA] >= min_area)
    if need is not None:
        keep &= np.bincount(lab[need & m], minlength=n) >= 4
    keep[0] = False
    return keep[lab]
