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
THIN_MODE = True
JOINT = {}
JOINT_GUARD = "start"  # проверка совместного распознавания: "seed" | "partner"
JOINT_CUT = 25       # обрезать совместный «щуп» на первой общей полосе такой длины (0 — не обрезать)
USE_JOINT = True     # нераспознанным по своему «щупу» — совместные «щупы»           # настройки совместных «щупов» (follow_joint)   # тонкие 2–3 px — тоже режим толщины (порог 2 px); False — старый режим


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
    if w < THIN:
        if not THIN_MODE or w < 1.9:     # 1–1.5 px — как сетка: режим толщины её не отделит — старый режим
            return {"_kind": "thin", **_f(f)}
        # тонкая линия 2–3 px: порог толщины 2 px отсекает сетку в 1 px; различаем по рисунку штриха
        if f["long_frac"] >= SOLID_MIN:
            return {"style": "solid", "width": [2.0], "core": 3.0, "stroke_w": w, "ink_reward": 0.5,
                    "_kind": "thin-solid", **_f(f)}
        return {"style": "dashed", "width": [2.0], "core": 3.0, "ink_reward": 1.5, "core_gain": 2.0,
                "dash_max": int(max(24, 2 * f.get("dash_len", 0))), "_kind": "thin-dashed", **_f(f)}
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


def follow_joint(dark: np.ndarray, seeds: list, left=None, right=None, step: int = 12, max_gap: int = 40,
                 share_pen: float = 12.0, miss_cost: float | None = None):
    """Пробные пути всех кривых дорожки сразу. В каждой строке отрезки чернил распределяются между путями
    (венгерский алгоритм): один отрезок на двоих можно взять только со штрафом share_pen — если рядом есть
    свободный отрезок, второй путь уходит на него. Так после пересечения кривые расходятся по своим линиям,
    а не идут дальше вдвоём по одной. Возвращает (пути, маски строк, где отрезок был общий)."""
    from scipy.optimize import linear_sum_assignment
    h, w = dark.shape
    n = len(seeds)
    xs = np.full((n, h), np.nan)
    shared = np.zeros((n, h), bool)
    x = np.array([np.nan if s is None else float(s) for s in seeds])
    v = np.zeros(n); gap = np.zeros(n); alive = ~np.isnan(x)
    BIG = 1e6
    for y in range(h):
        act = np.flatnonzero(alive)
        if act.size == 0:
            break
        lo_t = int(left[y]) if left is not None else 0
        hi_t = int(right[y]) + 1 if right is not None else w
        row = np.flatnonzero(dark[y, max(0, lo_t):min(w, hi_t)]) + max(0, lo_t)
        runs = np.split(row, np.flatnonzero(np.diff(row) > 1) + 1) if row.size else []
        runs = [r for r in runs if r.size <= 3 * step]
        if runs:
            c = np.array([r.mean() for r in runs]); half = np.array([(r[-1] - r[0]) / 2 for r in runs])
            pred = x[act] + v[act]
            dc = np.abs(c[None, :] - pred[:, None])
            d = np.maximum(0, dc - half[None, :]) + 0.1 * dc
            reach = step + gap[act]
            d = np.where(d <= reach[:, None], d, BIG)
            parts = [d, np.where(d < BIG, d + share_pen, BIG)]
            if miss_cost is not None:          # «пропустить строку» дешевле, чем взять далёкий/чужой отрезок
                parts.append(np.where(np.eye(len(act), dtype=bool), miss_cost, BIG))
            cost = np.concatenate(parts, axis=1)
            ri, ci = linear_sum_assignment(cost)
            taken = {}
            for a, b in zip(ri, ci):
                if cost[a, b] >= BIG or b >= 2 * len(runs):
                    continue
                taken.setdefault(b % len(runs), []).append(act[a])
            got = set()
            for k, ids in taken.items():
                for i in ids:
                    nx = float(c[k])
                    if gap[i] == 0:
                        v[i] = float(np.clip(0.7 * v[i] + 0.3 * (nx - x[i]), -step / 2, step / 2))
                    x[i] = nx; xs[i, y] = nx; gap[i] = 0; got.add(i)
                    shared[i, y] = len(ids) > 1
            miss = [i for i in act if i not in got]
        else:
            miss = list(act)
        for i in miss:
            gap[i] += 1
            if gap[i] > max_gap:
                alive[i] = False
    return list(xs), list(shared)


def _own_rows(paths, shared=None):
    """Строки, где путь кривой не совпадает (|dx| ≥ 4) с путём другой кривой и не делит с ней отрезок."""
    own = [None if p is None else np.isfinite(p) & (True if shared is None else ~shared[i]) for i, p in enumerate(paths)]
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i == j or paths[i] is None or paths[j] is None:
                continue
            both = np.isfinite(paths[i]) & np.isfinite(paths[j])
            same = np.zeros_like(both)
            same[both] = np.abs(paths[i][both] - paths[j][both]) < 4
            own[i] &= ~same
    return own


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


def recognize(sub: np.ndarray, seeds: list, left: np.ndarray, right: np.ndarray, rows: int = 1500) -> list:
    """sub — маска чернил дорожки (строки интервала, столбцы дорожки); seeds — x старта каждой кривой (в координатах
    sub) или None. left/right — края дорожки по строкам (в координатах sub). Возвращает признаки для каждой кривой."""
    top = sub[: min(rows, sub.shape[0])]
    W, grid = strokes.width_and_grid(top)
    ink = top & ~grid
    _, lab, st, _ = cv2.connectedComponentsWithStats(bridge_grid(ink, grid).astype(np.uint8), connectivity=8)
    ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    out = [None] * len(seeds)
    lt, rt = left[: top.shape[0]], right[: top.shape[0]]
    # 1) у каждой кривой свой «щуп» (ближайший штрих) — по штрихам толщиной от 2 px без сетки
    paths = []
    for s in seeds:
        if s is None:
            paths.append(None); continue
        xs = follow(ink & (W >= 2.0), s, lt, rt)
        if np.isfinite(xs).mean() < 0.3:
            xs = follow(ink, s, lt, rt)
        if np.isfinite(xs).mean() < 0.3:            # своей линии без сетки нет: прямая вдоль сетки (Bit Size)
            xs = follow(top, s, lt, rt)
        paths.append(xs)
    # «щупы» кривых, стартующих рядом, сходятся на одну линию; общие строки — не про эту кривую
    own = _own_rows(paths)
    # 2) кому своих строк не хватило — совместные «щупы» всей дорожки: одну линию на двоих делят только со
    #    штрафом, после пересечения пути расходятся по своим линиям (раньше такие кривые шли в старый режим)
    jp, jsh = follow_joint(ink & (W >= 2.0), seeds, lt, rt, **JOINT)
    jown = _own_rows([None if s is None else q for s, q in zip(seeds, jp)], jsh)
    if JOINT_CUT:
        # после того как два «щупа» прошли по одной линии ≥ JOINT_CUT строк, кто из них кто — уже не наверняка
        # (на пересечении они могли поменяться): верим только участку до первой такой общей полосы
        for i, sh in enumerate(jsh):
            if jown[i] is None:
                continue
            run = np.convolve(sh.astype(int), np.ones(JOINT_CUT, int), "valid")
            k = np.flatnonzero(run >= JOINT_CUT)
            if k.size:
                jown[i] = jown[i].copy(); jown[i][k[0]:] = False
    why = []
    for i, p in enumerate(paths):
        if p is None:
            why.append("нет старта"); continue
        if own[i].sum() >= 60:
            out[i] = path_features(top, W, lab, ext, np.where(own[i], p, np.nan), grid=grid)
            why.append("ok"); continue
        if USE_JOINT and jown[i] is not None and jown[i].sum() >= 60:
            out[i] = path_features(top, W, lab, ext, np.where(jown[i], jp[i], np.nan), grid=grid)
            why.append("ok-joint"); continue
        why.append(f"своих строк {int(own[i].sum())} из {int(np.isfinite(p).sum())}")
    # распознанное совместными «щупами» не верим, если у другой кривой дорожки та же линия (тот же тип, толщина
    # ±1 px): такие кривые обводка всё равно не различит, а «щупы» могли поменяться местами на пересечении
    for i in range(len(out)):
        if why[i] != "ok-joint" or not out[i]:
            continue
        ki = classify(out[i]).get("_kind")
        for j in range(len(out)):
            if j != i and out[j] and classify(out[j]).get("_kind") == ki and abs(out[j]["width"] - out[i]["width"]) <= 1:
                out[i] = None; why[i] = "joint: такая же линия у соседней"; break
    # и не верим, если «напарник» (кривая, чей одиночный «щуп» шёл по той же линии) так и остался без своей линии:
    # значит, совместный «щуп» скорее отдал эту кривой линию напарника (тонкая 1 px кривая «забирает» толстую)
    for i in range(len(out)):
        if why[i] != "ok-joint" or not out[i]:
            continue
        if JOINT_GUARD == "start":
            # кривые стартуют из одной точки (ближе 10 px): какая линия чья, на старте не понять — «щупы» могли
            # поменяться местами
            if any(j != i and s2 is not None and abs(s2 - seeds[i]) < 10 for j, s2 in enumerate(seeds)):
                out[i] = None; why[i] = "joint: общий старт"
            continue
        if JOINT_GUARD == "seed":
            # старт кривой не на штрихе от 2 px — своя линия тонкая (1 px), совместный «щуп» взял чужую
            x0 = int(round(seeds[i])); lo, hi = max(0, x0 - 2), min(W.shape[1], x0 + 3)
            near = ((W[:20, lo:hi] >= 2.0) & ink[:20, lo:hi]).any(1).mean()
            if near < 0.5:
                out[i] = None; why[i] = "joint: старт на тонкой линии"
            continue
        for j in range(len(out)):
            if j == i or paths[j] is None or out[j]:
                continue
            both = np.isfinite(paths[i]) & np.isfinite(paths[j])
            close = both.copy(); close[both] = np.abs(paths[i][both] - paths[j][both]) < 4
            if close.sum() >= 0.3 * np.isfinite(paths[i]).sum():
                out[i] = None; why[i] = "joint: напарник не распознан"; break
    recognize.why = why
    return out
