"""Автообводка кривой на ч/б (или цветном) скане каротажа.

Шаги для одной кривой в одной дорожке:
1. Маска тёмных пикселей внутри дорожки (или маска по цвету, если кривая цветная).
2. Чистка: тонкая сетка (1–2 px) и пунктир убираются морфологическим открытием;
   толстые горизонтальные/вертикальные линии сетки находятся по доле тёмных пикселей
   и «залечиваются» — пиксели строки сетки берутся из соседних строк (кривая, пересекающая
   линию, остаётся непрерывной, а сама линия исчезает).
3. В каждой строке — интервалы тёмных пикселей (кандидаты). Оптимальный путь сверху вниз
   (Витерби): штраф = разрыв между интервалами соседних строк, с поддержкой перехлёста через
   край дорожки (кривая ушла за правый край → появилась слева на следующем уровне шкалы).
4. Точки: центр штриха; широкий интервал (горизонтальный выброс/пик) даёт дополнительную
   точку на конце пика, чтобы амплитуда пиков не терялась.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TraceResult:
    xs: np.ndarray        # x-пиксель центра кривой (абсолютный), NaN = нет данных
    ys: np.ndarray        # строки (абсолютные)
    level: np.ndarray     # уровень шкалы (1 = основной, 2 = перехлёст вправо, 0 = влево)
    points: list          # [(x, y, level, nulls_after)] для .improj / отрисовки
    coverage: float


def _line_groups(frac: np.ndarray, thr: float):
    """Подряд идущие индексы с frac > thr → [(начало, конец, центр, сила)]."""
    idx = np.flatnonzero(frac > thr)
    if idx.size == 0:
        return []
    out = []
    for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        out.append((int(g[0]), int(g[-1]), float((g * frac[g]).sum() / frac[g].sum()), float(frac[g].max())))
    return out


def periodic_cols(dark: np.ndarray, return_found: bool = False):
    """Вертикальные линии сетки = кандидаты, стоящие на равномерной решётке (дорожка делится на N частей).
    Прямые кривые (Bit Size, базовая линия CCL) обычно не на решётке и сохраняются.
    Если решётка не найдена (лог-шкала) — сеткой считаются только почти сплошные линии (> 85%)."""
    h, w = dark.shape
    frac = dark.mean(0)
    cands = _line_groups(frac, 0.2)
    is_grid = np.zeros(w, bool)
    # края дорожки
    for a, b, c, s in cands:
        if (c < 8 or c > w - 9) and s > 0.4:
            is_grid[a:b + 1] = True
    best = None
    centers = np.array([c for _, _, c, _ in cands]) if cands else np.array([])
    # опоры решётки — реальные крайние линии дорожки (если видны), иначе края окна
    strong = [c for _, _, c, s in cands if s > 0.5]
    left = strong[0] if strong and strong[0] < 0.1 * w else 0.0
    right = strong[-1] if strong and strong[-1] > 0.9 * w else w - 1.0
    for n in (2, 4, 5, 6, 8, 10, 12, 15, 20):
        step = (right - left) / n
        if step < 12 or centers.size == 0:
            continue
        tol = max(3.0, 0.12 * step)
        pos = left + np.arange(1, n) * step
        hit = np.array([np.min(np.abs(centers - p)) <= tol for p in pos])
        score = hit.mean()
        if score >= 0.7 and (best is None or hit.sum() > best[1]):
            best = (n, hit.sum(), pos, tol)
    if best:
        _, _, pos, tol = best
        for a, b, c, s in cands:
            if np.min(np.abs(pos - c)) <= tol:
                is_grid[a:b + 1] = True
    else:
        for a, b, c, s in cands:
            if s > 0.85:
                is_grid[a:b + 1] = True
    return (is_grid, best is not None) if return_found else is_grid


def _block_lattice(d: np.ndarray):
    """Для блока строк: кандидаты линий, опоры (левый/правый край) и лучшее число делений N."""
    w = d.shape[1]
    frac = d.mean(0)
    cands = _line_groups(frac, 0.2)
    strong = [c for _, _, c, s in cands if s > 0.5]
    left = strong[0] if strong and strong[0] < 0.1 * w else 0.0
    right = strong[-1] if strong and strong[-1] > 0.9 * w else w - 1.0
    centers = np.array([c for _, _, c, _ in cands]) if cands else np.array([])
    best = None
    for n in (2, 4, 5, 6, 8, 10, 12, 15, 20):
        step = (right - left) / n
        if step < 12 or centers.size == 0:
            continue
        tol = max(3.0, 0.12 * step)
        pos = left + np.arange(1, n) * step
        hit = np.array([np.min(np.abs(centers - p)) <= tol for p in pos])
        if hit.mean() >= 0.7 and (best is None or hit.sum() > best[1]):
            best = (n, int(hit.sum()))
    return cands, left, right, (best[0] if best else None)


def lattice_cols(dark: np.ndarray, block: int = 1000, min_share: float = 0.7):
    """Решётка вертикальных линий сетки по всему окну с учётом перекоса.
    Позиция k решётки считается сеткой, только если линия на ней найдена в >= min_share блоков —
    кривая не стоит на одном месте по всей длине, сетка стоит."""
    h, w = dark.shape
    blocks = [(b0, min(h, b0 + block)) for b0 in range(0, h, block) if min(h, b0 + block) - b0 >= 200 or b0 == 0]
    info = [_block_lattice(dark[a:b]) for a, b in blocks]
    ns = [n for *_, n in info if n]
    out = np.zeros((h, w), bool)
    found = np.zeros(h, bool)
    if not ns or len(ns) < 0.5 * len(blocks):
        return out, found
    n = max(set(ns), key=ns.count)
    hits = np.zeros(n + 1, int)
    per_block = []
    for (a, b), (cands, left, right, _) in zip(blocks, info):
        step = (right - left) / n
        tol = max(3.0, 0.12 * step)
        marks = {}
        for ca, cb, c, s in cands:
            k = int(round((c - left) / step))
            if 0 <= k <= n and abs(c - (left + k * step)) <= tol:
                marks.setdefault(k, []).append((ca, cb))
        for k in marks:
            hits[k] += 1
        per_block.append(((a, b), marks))
    good = {k for k in range(n + 1) if hits[k] >= min_share * len(blocks)}
    slots = np.zeros((h, w), bool)    # окрестность любой позиции решётки (для сплошных линий)
    for (a, b), (cands, left, right, _) in zip(blocks, info):
        step = (right - left) / n
        tol = int(np.ceil(max(3.0, 0.12 * step)))
        for k in range(n + 1):
            c = int(round(left + k * step))
            slots[a:b, max(0, c - tol):min(w, c + tol + 1)] = True
    lattice_cols.slots = slots
    for (a, b), marks in per_block:
        found[a:b] = True
        for k, spans in marks.items():
            if k in good:
                for ca, cb in spans:
                    out[a:b, ca:cb + 1] = True
    # хвост, не попавший в блоки
    last = blocks[-1][1]
    if last < h:
        out[last:] = out[last - 1]
        found[last:] = True
    return out, found


def periodic_rows(dark: np.ndarray) -> np.ndarray:
    """Горизонтальные линии сетки = строки с долей тёмного > 0.3, повторяющиеся с постоянным шагом.
    Одиночный горизонтальный выброс кривой (пик) шагу не подчиняется и не удаляется."""
    h, w = dark.shape
    frac = dark.mean(1)
    cands = _line_groups(frac, 0.3)
    is_grid = np.zeros(h, bool)
    if len(cands) < 3:
        for a, b, c, s in cands:
            if s > 0.6:
                is_grid[a:b + 1] = True
        return is_grid
    centers = np.array([c for _, _, c, _ in cands])
    d = np.diff(centers)
    step = np.median(d[d > 3]) if (d > 3).any() else 0
    for (a, b, c, s), cc in zip(cands, centers):
        if s > 0.75:             # почти сплошная линия во всю ширину — сетка в любом случае
            is_grid[a:b + 1] = True
            continue
        if step:
            near = np.abs(np.abs(centers - cc) - step) <= max(2.0, 0.1 * step)
            if near.any():
                is_grid[a:b + 1] = True
    return is_grid


def clean_mask(dark: np.ndarray, stroke: int = 3, line_frac: float = 0.5, col_frac: float = 0.6, protect_x: float | None = None,
               col_window: int = 400, track_protect: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dark — bool-маска (кусок дорожки). Возвращает очищенную маску и маски строк/столбцов сетки."""
    m = dark.astype(np.uint8)
    if stroke > 1:
        k = np.ones((stroke, stroke), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    h, w = m.shape
    rmean = dark.mean(1)
    # чёткая сетка (линии во всю ширину): порог выше, иначе длинные горизонтальные штрихи кривой
    # (сжатый лог, «гармошка») принимаются за линии сетки
    crisp = int((rmean > 0.92).sum()) >= max(3, h // 150)
    grid_rows = rmean > (0.88 if crisp else line_frac)
    # stroke=2 — кривая 2 px на тонкой (1 px) сетке: тонкие линии уже снял проход 2x2, залечивать надо только
    # толстые (которые его пережили). Иначе каждая линия сетки ±2 строки стирает пологие склоны пиков.
    thin_grid = stroke == 2
    if thin_grid:
        grid_rows = m.mean(1) > 0.35
    core_rows = grid_rows.copy()
    dense = bool((~grid_rows).any()) and float(dark[~grid_rows].mean()) > 0.10
    # скан бывает перекошен (дорожка «уезжает» на десятки px) — решётку ищем по блокам строк
    # 1) сплошные вертикальные линии: доля тёмного в скользящем окне строк (как раньше)
    # тонкая сетка (stroke=2): проход 2x2 рвёт вертикальную линию на обрывки, и её доля падает ниже порога —
    # обрывки остаются и путь «едет» по ним. Долю считаем по исходным чернилам.
    cs = np.cumsum(np.vstack([np.zeros((1, w)), dark.astype(np.uint8) if stroke == 2 else m]), axis=0)
    win = min(col_window, h)
    frac = np.zeros((h, w))
    for y0 in range(0, h, win // 2):
        y1 = min(h, y0 + win)
        a = max(0, y1 - win)
        frac[y0:y1] = ((cs[y1] - cs[a]) / (y1 - a))[None, :]
    solid = frac > col_frac
    # 2) решётка по блокам: добавляет тонкие/пунктирные линии сетки и защищает прямые кривые вне решётки
    lattice_cols.slots = None
    lattice, found = lattice_cols(dark)
    if dense or crisp:             # чёткая сетка ловится как сплошные линии; плотная «гармошка» — решётка по чернилам кривой ненадёжна, хватает сплошных линий
        lattice = np.zeros_like(lattice)
    # прямая кривая (Bit Size, база CCL) — сплошная линия, как сетка; защищаем ту, на которой старт кривой
    grid_cols = solid | (lattice if stroke <= 1 else False)
    # края толстых линий темнеют не полностью — расширяем найденные линии на grow px
    grow = 2
    if grid_rows.any():
        # расширяем линии только если их края действительно размыты: медиана заполнения строки
        # на расстоянии k от линии заметно выше обычной строки
        base = float(np.median(rmean[~core_rows])) if (~core_rows).any() else 0.0
        idx = np.flatnonzero(core_rows)
        bands = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        grow_r = 0
        for k in (1, 2):
            v = [rmean[b[0] - k] for b in bands if b[0] - k >= 0] + [rmean[b[-1] + k] for b in bands if b[-1] + k < h]
            if v and float(np.median(v)) > base + 0.08:
                grow_r = k
            else:
                break
        if not crisp:
            grow_r = grow
        if thin_grid:
            grow_r = 1
        if grow_r:
            grid_rows = cv2.dilate(grid_rows.astype(np.uint8)[:, None], np.ones((2 * grow_r + 1, 1), np.uint8))[:, 0].astype(bool)
    if grid_cols.any():
        grid_cols = cv2.dilate(grid_cols.astype(np.uint8), np.ones((1, 2 * grow + 1), np.uint8)).astype(bool)
    if protect_x is not None:
        lo, hi = int(max(0, protect_x - 4)), int(min(w, protect_x + 5))
        start = dark[: min(h, 300), lo:hi].mean(0) > 0.6
        if start.any():
            idx = np.flatnonzero(start) + lo
            grid_cols[:, max(0, idx.min() - 1):min(w, idx.max() + 2)] = False
            # на перекошенном скане прямая кривая (CCL, Bit Size) уезжает вбок на несколько px —
            # ведём защиту за линией блоками по 300 строк, иначе ниже её снимет чистка как линию сетки
            cur = float(idx.mean())
            for a in (range(300, h, 300) if track_protect else ()):
                b = min(h, a + 300)
                lo, hi = int(max(0, cur - 5)), int(min(w, cur + 6))
                st = dark[a:b, lo:hi].mean(0) > 0.6
                if st.any():
                    idx = np.flatnonzero(st) + lo
                    grid_cols[a:b, max(0, idx.min() - 1):min(w, idx.max() + 2)] = False
                    cur = float(idx.mean())
    out = m.astype(bool).copy()
    # залечиваем строки сетки
    rows = np.flatnonzero(grid_rows)
    if rows.size:
        good = np.flatnonzero(~grid_rows)
        if good.size:
            for r in rows:
                i = np.searchsorted(good, r)
                up = good[i - 1] if i > 0 else None
                dn = good[i] if i < good.size else None
                src = np.zeros(w, bool)
                if up is not None:
                    src |= out[up]
                if dn is not None:
                    src |= out[dn]
                if up is not None and dn is not None:
                    src = out[up] | out[dn]
                out[r] = src
    # пики кривой, лежащие на толстой горизонтальной линии: линия в этих столбцах локально толще
    raw = dark.astype(bool)
    rows_idx = np.flatnonzero(grid_rows)
    if rows_idx.size:
        for band in np.split(rows_idx, np.where(np.diff(rows_idx) > 1)[0] + 1):
            a, b = max(0, band[0] - 3), min(h, band[-1] + 4)
            thick = raw[a:b].sum(0)
            base = np.median(thick)
            spike = thick >= base + (2 if thin_grid else 3)   # кривая 2 px, лежащая вдоль линии, добавляет 2 строки
            spike &= ~grid_cols[band[0]] if grid_cols.any() else spike
            if spike.any():
                # убираем одиночные столбцы-шум, оставляем отрезки >= 3 px
                idx = np.flatnonzero(spike)
                for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                    if run.size >= 3:
                        out[band[0]:band[-1] + 1, run[0]:run[-1] + 1] = True
    # залечиваем столбцы сетки (по каждой строке берём соседние не-сеточные столбцы)
    if grid_cols.any():
        for y in np.flatnonzero(grid_cols.any(1)):
            gc = grid_cols[y]
            row = out[y].copy()
            idx = np.flatnonzero(gc)
            for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                a, b = run[0], run[-1]
                left = row[a - 1] if a > 0 else False
                right = row[b + 1] if b < w - 1 else False
                row[a:b + 1] = left or right
            out[y] = row
    return out, grid_rows, grid_cols


def _run_span(m: np.ndarray):
    """Для каждого пикселя чернил — начало и конец (включительно) горизонтального отрезка чернил, в который он входит."""
    h, w = m.shape
    pad = np.zeros((h, 1), np.int8)
    d = np.diff(np.concatenate([pad, m.astype(np.int8), pad], 1), axis=1)
    st, en = np.argwhere(d == 1), np.argwhere(d == -1)
    out = []
    for val in (st[:, 1].astype(np.int32), (en[:, 1] - 1).astype(np.int32)):
        acc = np.zeros((h, w + 1), np.int32)
        acc[st[:, 0], st[:, 1]] = val
        np.subtract.at(acc, (en[:, 0], en[:, 1]), val)
        out.append(np.cumsum(acc, 1)[:, :w])
    return out[0], out[1]


def _line_bands(dark: np.ndarray, blk: int, pair_thr: float, tol: int, min_support: float):
    """Горизонтальные линии сетки: для каждого пикселя — границы полосы линии (первая и последняя строка),
    −1 там, где линии нет.
    Дорожка режется на блоки по blk столбцов; строка блока — кандидат, если она вместе с соседней заполнена
    на pair_thr (линия на перекошенном скане «переступает» со строки на строку). Линия сетки идёт через всю
    дорожку: кандидат принимается, если в ±tol строках он есть в min_support доле блоков. Пологий склон пика
    занимает лишь часть ширины и не проходит."""
    h, w = dark.shape
    nb = max(1, int(np.ceil(w / blk)))
    P = np.stack([dark[:, b * blk:min(w, (b + 1) * blk)].mean(1) for b in range(nb)], 1)      # h × nb
    nxt = np.vstack([P[1:], np.zeros((1, nb))])
    prv = np.vstack([np.zeros((1, nb)), P[:-1]])
    pair = (P + nxt >= pair_thr) | (P + prv >= pair_thr)
    member = pair & (P >= 0.15)
    near = cv2.dilate(member.astype(np.uint8), np.ones((2 * tol + 1, 1), np.uint8)).astype(bool)
    member &= (near.mean(1) >= min_support)[:, None]
    # группы соседних строк-участников в каждом блоке
    groups = []
    for b in range(nb):
        idx = np.flatnonzero(member[:, b])
        if not idx.size:
            continue
        for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            groups.append((0.5 * (g[0] + g[-1]), b, int(g[0]), int(g[-1])))
    top = np.full((h, nb), -1, np.int32)
    bot = np.full((h, nb), -1, np.int32)
    if groups:
        groups.sort()
        cen = np.array([g[0] for g in groups])
        cuts = np.flatnonzero(np.diff(cen) > tol + 2) + 1
        for line in np.split(np.arange(len(groups)), cuts):          # одна линия сетки = группы с близким центром
            gs = [groups[i] for i in line]
            size = np.array([g[3] - g[2] + 1 for g in gs])
            med = int(np.median(size))
            fat = size > med + 1
            if fat.any() and (~fat).sum() >= 2:
                # в «толстых» блоках рядом с линией идёт склон кривой и раздувает полосу — берём полосу
                # обычной толщины там, где линия проходит по соседним блокам
                bb = np.array([g[1] for g in gs], float)
                k, c0 = np.polyfit(bb[~fat], np.array([g[0] for g in gs])[~fat], 1) if (~fat).sum() >= 3                     else (0.0, float(np.median(np.array([g[0] for g in gs])[~fat])))
            for g, f in zip(gs, fat):
                t0, t1 = g[2], g[3]
                if f and (~fat).sum() >= 2:
                    c = k * g[1] + c0
                    t0 = int(np.clip(np.floor(c - med / 2 + 0.5), g[2], g[3]))
                    t1 = int(min(g[3], t0 + med))
                top[t0:t1 + 1, g[1]] = t0
                bot[t0:t1 + 1, g[1]] = t1
    return np.repeat(top, blk, axis=1)[:, :w], np.repeat(bot, blk, axis=1)[:, :w]


def _bridge_bands(dark: np.ndarray, out: np.ndarray, band: np.ndarray, max_gap: int, min_area: int = 12) -> None:
    """Кривая, которая полого входит в линию сетки и какое-то время идёт внутри неё, после чистки рвётся:
    кусок внутри полосы неотличим от линии. Кривая непрерывна — значит, если к полосе с одной стороны подходит
    конец штриха (чернила касаются полосы только сверху или только снизу), а рядом, не дальше max_gap, такой же
    конец, то между ними кривая шла по линии: возвращаем чернила полосы на этом отрезке (правит out на месте).
    Крутое пересечение касается полосы и сверху, и снизу в одном месте — его не трогаем."""
    h, w = dark.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    big = st[:, cv2.CC_STAT_AREA] >= min_area
    big[0] = False
    solid = big[lab]                                   # чернила настоящих штрихов (не пыль)
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
    for k in range(1, nb):
        c0, r0, cw, ch = bst[k, 0], bst[k, 1], bst[k, 2], bst[k, 3]
        if cw < 8:
            continue
        sub = blab[r0:r0 + ch, c0:c0 + cw] == k
        has = sub.any(0)
        top = r0 + sub.argmax(0)
        bot = r0 + ch - 1 - sub[::-1].argmax(0)
        cols = np.arange(c0, c0 + cw)
        up = np.zeros(cw, bool)
        dn = np.zeros(cw, bool)
        ok = has & (top > 0)
        up[ok] = solid[top[ok] - 1, cols[ok]]
        ok = has & (bot < h - 1)
        dn[ok] = solid[bot[ok] + 1, cols[ok]]
        inb = (solid[r0:r0 + ch, c0:c0 + cw] & sub).any(0)
        up |= inb
        dn |= inb
        touch = np.flatnonzero(up | dn)
        if touch.size < 2:
            continue
        runs = np.split(touch, np.where(np.diff(touch) > 3)[0] + 1)
        info = [(int(r[0]), int(r[-1]), bool(up[r].any()), bool(dn[r].any())) for r in runs]
        i = 0
        while i + 1 < len(info):
            a0, a1, ua, da = info[i]
            b0, b1, ub, db = info[i + 1]
            if not (ua and da) and not (ub and db) and 0 < b0 - a1 - 1 <= max_gap:
                seg = slice(a1 + 1, b0)
                blk = sub[:, seg] & dark[r0:r0 + ch, c0 + a1 + 1:c0 + b0]
                out[r0:r0 + ch, c0 + a1 + 1:c0 + b0] |= blk
                i += 2
            else:
                i += 1


def remove_grid_lines(dark: np.ndarray, protect_x: float | None = None, blk: int = 32, chunk: int = 1280) -> np.ndarray:
    """Чистка тонкой сетки без морфологии (как снятие нотных линеек): в полосе линии сетки убираются чернила,
    поперечный отрезок которых целиком лежит внутри полосы. Кривая, пересекающая линию или касающаяся её,
    выходит за полосу и остаётся целой; пологий склон пика толщиной 1 px, не лежащий на линии, тоже остаётся
    (открытие 2×2 его стирало). protect_x — столбец прямой кривой (CCL), которую нельзя снять как линию."""
    h, w = dark.shape
    top, bot = _line_bands(dark, blk, 0.6, 2 + w // 400, 0.6)
    hband = top >= 0
    vs, ve = _run_span(dark.T)
    kill = hband & (vs.T >= top) & (ve.T <= bot) & dark
    del vs, ve, top, bot
    vband = np.zeros((h, w), bool)
    cur = protect_x
    for a in range(0, h, chunk):
        b = min(h, a + chunk)
        if b - a < 8 * blk and a > 0:
            a = max(0, b - chunk)
        part = dark[a:b]
        lt, rt = _line_bands(part.T, blk, 0.5, 2, 0.6)
        lt, rt = lt.T, rt.T
        if cur is not None:                       # прямая кривая: ведём защиту за линией по блокам
            lo, hi = int(max(0, cur - 5)), int(min(w, cur + 6))
            cols = np.flatnonzero((lt[:, lo:hi] >= 0).mean(0) > 0.3) + lo
            if cols.size:
                c0, c1 = max(0, cols.min() - 1), min(w, cols.max() + 2)
                lt[:, c0:c1] = -1
                cur = float(cols.mean())
        hs, he = _run_span(part)
        vband[a:b] |= lt >= 0
        kill[a:b] |= (lt >= 0) & (hs >= lt) & (he <= rt) & part
    kill |= hband & vband & dark                   # перекрестья сетки
    out = dark & ~kill
    # пыль: одиночные пиксели и пары
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    small = st[:, cv2.CC_STAT_AREA] <= 2
    small[0] = False
    out[small[lab]] = False
    _bridge_bands(dark, out, hband, max_gap=220)
    ot = np.ascontiguousarray(out.T)
    _bridge_bands(np.ascontiguousarray(dark.T), ot, np.ascontiguousarray(vband.T), max_gap=120)
    return np.ascontiguousarray(ot.T)


def _intervals(row: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return np.empty((0, 2), int)
    br = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return np.column_stack([starts, ends])


def viterbi(mask: np.ndarray, seed_x: float | None = None, wrap: bool = True, edge: int = 12,
            jump_cost: float = 1.0, max_skip: int = 60, skip_cost: float = 3.0,
            width_cost: float = 0.02, slope_cost: float = 0.0, left=None, right=None, weight=None,
            weight_gain: float = 0.0):
    """Оптимальный путь по интервалам строк. left/right — положение краёв дорожки в каждой строке (в координатах
    маски): на перекошенном скане рамка уезжает, и «кривая у края» надо мерить от края в этой строке.
    Возвращает для каждой строки выбранный интервал (или None) и сдвиг уровня шкалы."""
    h, w = mask.shape
    left = np.zeros(h) if left is None else np.asarray(left, float)
    right = np.full(h, w - 1.0) if right is None else np.asarray(right, float)
    cands = [_intervals(mask[y]) for y in range(h)]
    # weight — «ценность» чернил (толщина штриха сверх порога): путь получает выигрыш за толстую линию
    # и поэтому не уходит на соседнюю тонкую, даже если та глаже
    gains = None
    if weight is not None and weight_gain:
        gains = []
        for y in range(h):
            c = cands[y]
            if len(c) == 0:
                gains.append(np.zeros(0)); continue
            cs = np.concatenate([[0.0], np.cumsum(weight[y], dtype=float)])
            gains.append((cs[c[:, 1] + 1] - cs[c[:, 0]]) * weight_gain)
    # состояние: (строка, индекс интервала). Переход может перепрыгнуть до max_skip пустых строк.
    INF = 1e18
    cost = [np.full(len(c), INF) for c in cands]
    back = [np.full((len(c), 2), -1, int) for c in cands]
    lvl = [np.zeros(len(c), int) for c in cands]
    # старт: первые строки с кандидатами
    started = False
    last_rows: list[int] = []
    for y in range(h):
        c = cands[y]
        if len(c) == 0:
            continue
        cen = c.mean(1)
        wid = c[:, 1] - c[:, 0]
        em = wid * width_cost - (gains[y] if gains is not None else 0.0)
        if not started:
            base = np.abs(cen - seed_x) * jump_cost if seed_x is not None else np.zeros(len(c))
            cost[y] = base + em + y * skip_cost
            started = True
            last_rows = [y]
            continue
        best = np.full(len(c), INF)
        bk = np.full((len(c), 2), -1, int)
        bl = np.zeros(len(c), int)
        for py in last_rows:
            pc = cands[py]
            gap_rows = y - py - 1
            # разрыв между интервалами (0, если перекрываются/соприкасаются)
            g = np.maximum(0, np.maximum(c[None, :, 0] - pc[:, None, 1], pc[:, None, 0] - c[None, :, 1]) - 1)
            dl = np.zeros_like(g)
            if wrap:
                # перехлёст: предыдущий у правого края, текущий у левого → уровень +1 (и наоборот)
                right_to_left = (pc[:, None, 1] >= right[py] - edge) & (c[None, :, 0] <= left[y] + edge)
                left_to_right = (pc[:, None, 0] <= left[py] + edge) & (c[None, :, 1] >= right[y] - edge)
                gw1 = np.maximum(0, right[py] - pc[:, None, 1]) + np.maximum(0, c[None, :, 0] - left[y])
                gw2 = np.maximum(0, pc[:, None, 0] - left[py]) + np.maximum(0, right[y] - c[None, :, 1])
                g = np.where(right_to_left & (gw1 < g), gw1, g)
                dl = np.where(right_to_left & (gw1 <= g), 1, dl)
                g2 = np.where(left_to_right & (gw2 < g), gw2, g)
                dl = np.where(left_to_right & (gw2 < g), -1, dl)
                g = g2
            total = cost[py][:, None] + g * jump_cost + gap_rows * skip_cost + em[None, :]
            if slope_cost:        # гладкая кривая: штраф за сдвиг центра между строками
                total = total + slope_cost * np.minimum(np.abs(cen[None, :] - pc.mean(1)[:, None]) / (gap_rows + 1), 40)
            j = np.argmin(total, axis=0)
            t = total[j, np.arange(len(c))]
            upd = t < best
            best = np.where(upd, t, best)
            bk[upd, 0] = py
            bk[upd, 1] = j[upd]
            bl[upd] = (lvl[py][j] + dl[j, np.arange(len(c))])[upd]
        cost[y], back[y], lvl[y] = best, bk, bl
        # окно назад: все строки в пределах max_skip + всегда 2 последние непустые строки
        # (иначе длинный разрыв пунктира/пропуск записи обрывает путь)
        keep = [r for r in last_rows if y - r < max_skip]
        for r in last_rows[-2:]:
            if r not in keep:
                keep.append(r)
        last_rows = sorted(keep) + [y]
    # конец пути — в последней строке с кандидатами (с учётом штрафа за недотянутые строки)
    ends = [(cost[y].min() + (h - 1 - y) * skip_cost, y, int(cost[y].argmin())) for y in range(h) if len(cands[y])]
    if not ends:
        return [None] * h, np.zeros(h, int), cands
    _, y, j = min(ends)
    chosen = [None] * h
    level = np.zeros(h, int)
    while y >= 0 and j >= 0:
        chosen[y] = tuple(cands[y][j])
        level[y] = lvl[y][j]
        y, j = back[y][j]
    return chosen, level, cands


def trace_track(dark_full: np.ndarray, x_left: np.ndarray, x_right: np.ndarray, y0: int, y1: int,
                seed_x: float | None = None, stroke: int = 0, pad: int = 2, wrap: bool = True,
                clean: bool = True, zigzag: bool | None = None, max_level: int | None = None,
                style: str | None = None, used: np.ndarray | None = None, jump_cost: float = 1.0,
                x_range: tuple | None = None, slope_cost: float = 0.0, centerline: bool = False,
                min_level: int = 0, protect: bool = False, ink_reward: float = 0.0,
                width: tuple | None = None, core: float = 0.0, core_gain: float = 1.0, dash_max: int = 24) -> TraceResult:
    """dark_full — маска всего скана; x_left/x_right — края дорожки для строк y0..y1 (абсолютные)."""
    xl = int(np.floor(x_left.min())) + pad
    xr = int(np.ceil(x_right.max())) - pad
    sub = dark_full[y0:y1, xl:xr + 1]
    if not stroke:
        stroke = auto_stroke(sub)
    wt = None
    if width:
        # разделение по толщине штриха (strokes.py): берём только линии нужной толщины, сетка снята как прямые
        from . import strokes
        W, grid = strokes.width_and_grid(sub)
        m = strokes.width_mask(sub, width[0], width[1] if len(width) > 1 else 99.0, W=W, grid=grid)
        grid_rows = np.zeros(sub.shape[0], bool)
        if core:
            wt = np.clip(W - (core - 1.0), 0.0, 3.0) * m
        if style == "dashed":
            # пунктир ищем среди компонент маски с низким порогом: там сплошные линии (температура, рамка,
            # соседняя кривая) связны и длинны, а штрихи пунктира остаются короткими отдельными компонентами
            low = strokes.width_mask(sub, min(3.0, width[0]), W=W, grid=grid)
            m = strokes.dash_filter(low, max_len=dash_max, need=m)
        del W, grid
    elif style and style.startswith("thin"):
        # тонкие линии: всё, что не толстое; сетку снимаем по решётке (без морфологии)
        thick = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        thick = cv2.dilate(thick, np.ones((5, 5), np.uint8)).astype(bool)
        m, grid_rows, _ = clean_mask(sub & ~thick, stroke=1, protect_x=None if seed_x is None else seed_x - xl)
        m &= ~thick
        if style in ("thin-solid", "thin-dashed"):
            m = style_filter(m, "solid" if style == "thin-solid" else "dashed", min_h=40, min_area=120, min_dash=3)
    elif clean == "lines":                    # тонкая сетка 1 px: снимаем сами линии, без морфологии
        m = remove_grid_lines(sub, protect_x=(seed_x - xl) if (seed_x is not None and protect) else None)
        grid_rows = np.zeros(sub.shape[0], bool)
    elif clean:
        m, grid_rows, _ = clean_mask(sub, stroke=stroke, protect_x=None if seed_x is None else seed_x - xl,
                                     track_protect=protect)
    else:
        m, grid_rows = sub.copy(), np.zeros(sub.shape[0], bool)
    if style in ("solid", "dashed") and not width:
        m = style_filter(m, style)
    if used is not None:                      # чернила уже обведённых кривых этой дорожки
        m &= ~used[:, xl:xr + 1]
    if x_range is not None:                   # кривая живёт только в части дорожки (полудорожка)
        m[:, : max(0, int(x_range[0]) - xl)] = False
        m[:, max(0, int(x_range[1]) - xl + 1):] = False
    # убираем всё, что за краями дорожки на конкретной строке (перекос скана)
    cols = np.arange(xl, xr + 1)
    for i in range(m.shape[0]):
        m[i, (cols < x_left[i] + pad) | (cols > x_right[i] - pad)] = False
    if centerline and not zigzag:
        # осевая линия штриха: точки ложатся по центру кривой, а не по краю/середине пятна
        from skimage.morphology import skeletonize
        path_mask = skeletonize(m)
    else:
        path_mask = m
    # ink_reward > 0: путь получает выигрыш за каждый пройденный пиксель чернил — иначе ему дешевле перепрыгнуть
    # узкий пик (пропуск десятка строк), чем идти по склону, в котором после чистки сетки остались разрывы
    chosen, level, _ = viterbi(path_mask, None if seed_x is None else seed_x - xl, wrap=wrap, jump_cost=jump_cost,
                               slope_cost=slope_cost, width_cost=(-ink_reward if ink_reward else 0.02), left=np.asarray(x_left[:m.shape[0]]) - xl + pad,
                               right=np.asarray(x_right[:m.shape[0]]) - xl - pad, weight=wt, weight_gain=core_gain)
    h = m.shape[0]
    if wrap and not zigzag:
        level = relevel(chosen, m.shape[1], min_level, (max_level - 1) if max_level else 9)
    if used is not None:                      # отмечаем найденную кривую, чтобы следующая её не взяла
        mark = np.zeros(m.shape, np.uint8)
        for i, iv in enumerate(chosen):
            if iv is not None:
                mark[i, iv[0]:iv[1] + 1] = 1
        # вместе с соседними чернилами того же штриха (связность в пределах 3 px)
        mark = cv2.dilate(mark, np.ones((5, 7), np.uint8)).astype(bool) & sub
        used[:, xl:xr + 1] |= cv2.dilate(mark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    xs = np.full(h, np.nan)
    pts = []
    run_wide = []
    for i in range(h):
        iv = chosen[i]
        if iv is None:
            continue
        a, b = iv
        xs[i] = (a + b) / 2
    # точки: центр, а у широких интервалов (пики) — ещё и дальний конец
    prev = None
    i = 0
    while i < h:
        iv = chosen[i]
        if iv is None:
            if pts:
                x, y, lv, _ = pts[-1]
                pts[-1] = (x, y, lv, True) if _gap_after(chosen, i) else pts[-1]
            i += 1
            continue
        a, b = iv
        if b - a + 1 <= 3 * stroke:
            pts.append((xl + (a + b) / 2, y0 + i, int(level[i]), False))
            prev = (a + b) / 2
            i += 1
            continue
        # блок подряд идущих широких строк — один пик
        j = i
        while j + 1 < h and chosen[j + 1] is not None and chosen[j + 1][1] - chosen[j + 1][0] + 1 > 3 * stroke:
            j += 1
        mid = (i + j) // 2
        a, b = chosen[mid]
        entry = prev if prev is not None else (a + b) / 2
        nxt = next(((chosen[k][0] + chosen[k][1]) / 2 for k in range(j + 1, min(h, j + 40)) if chosen[k] is not None), entry)
        far = b - stroke / 2 if abs(b - entry) >= abs(a - entry) else a + stroke / 2
        y = y0 + mid
        lv = int(level[mid])
        pts.append((xl + np.clip(entry, a, b), y, lv, False))
        pts.append((xl + far, y, lv, False))
        pts.append((xl + np.clip(nxt, a, b), y, lv, False))
        prev = np.clip(nxt, a, b)
        i = j + 1
    cov = float(np.mean([c is not None for c in chosen])) if h else 0.0
    if centerline and not zigzag:
        pts = centerline_points(chosen, level, xl, y0)
    if zigzag:
        pts, level = zigzag_points(m, chosen, level, xl, y0, max(stroke, 2), max_level=max_level,
                                    min_level=min_level)
    return TraceResult(xs + xl, np.arange(y0, y1), level, pts, cov)


def relevel(chosen, w: int, bot: int, top: int, min_jump: float = 0.3, max_rows: int = 80) -> np.ndarray:
    """Уровни шкалы по скачкам пути. Перехлёст на скане — это разрыв: кривая исчезает в одной части дорожки
    и продолжается далеко в другой (не обязательно касаясь края). Скачок вправо = кривая ушла за левый край
    (уровень −1), скачок влево = ушла за правый край (уровень +1). Уровень не выходит за подписанные шкалы:
    если менять некуда — это не перехлёст, а обычный разрыв."""
    level = np.zeros(len(chosen), int)
    cur, prev, prev_i = 0, None, None
    thr = min_jump * w
    for i, iv in enumerate(chosen):
        if iv is None:
            continue
        if prev is not None and i - prev_i <= max_rows:
            right = iv[0] - prev[1]               # > 0: продолжение правее
            left = prev[0] - iv[1]                # > 0: продолжение левее
            if right > thr and cur - 1 >= bot:
                cur -= 1
            elif left > thr and cur + 1 <= top:
                cur += 1
        level[i] = cur
        prev, prev_i = iv, i
    return level


def centerline_points(chosen, level, xl: int, y0: int, gap_rows: int = 60):
    """Точки по осевой линии (скелету). В строке, где кривая идёт горизонтально, скелет — отрезок [a, b]:
    ставим оба конца в порядке движения; если следующая строка продолжается от того же конца — это пик
    «туда и обратно», добавляем возврат."""
    rows = [i for i, c in enumerate(chosen) if c is not None]
    pts, prev_x = [], None
    for k, i in enumerate(rows):
        a, b = chosen[i]
        lv = int(level[i])
        if pts and i - rows[k - 1] >= gap_rows:
            x_, y_, l_, _ = pts[-1]
            pts[-1] = (x_, y_, l_, True)
            prev_x = None
        if pts and pts[-1][2] != lv:
            prev_x = None                         # сменился уровень — продолжение с другого края
        y = y0 + i
        if b - a <= 1:
            x = (a + b) / 2
            pts.append((xl + x, y, lv, False))
            prev_x = x
            continue
        nxt = None
        if k + 1 < len(rows) and rows[k + 1] - i <= 3 and int(level[rows[k + 1]]) == lv:
            na, nb = chosen[rows[k + 1]]
            nxt = (na + nb) / 2
        if prev_x is None:
            e1, e2 = (a, b) if nxt is None or abs(b - nxt) <= abs(a - nxt) else (b, a)
        else:
            e1, e2 = (a, b) if abs(a - prev_x) <= abs(b - prev_x) else (b, a)
        pts.append((xl + e1, y, lv, False))
        pts.append((xl + e2, y, lv, False))
        prev_x = e2
        if nxt is not None and abs(nxt - e1) + 2 < abs(nxt - e2) and abs(e2 - e1) >= 4:
            pts.append((xl + e1, y, lv, False))   # пик туда и обратно
            prev_x = e1
    return pts


def local_extent(m: np.ndarray, chosen, reach: int = 12, return_mask: bool = False):
    """Для каждой строки — крайние x чернил кривой: всё, что связано с выбранным путём не дальше reach строк
    по вертикали (складки «гармошки», вершины пиков). Возвращает lo, hi (nan — нет данных)."""
    h, w = m.shape
    mm = m.astype(bool)
    # номера горизонтальных отрезков (run) — чтобы за один шаг заливать отрезок целиком
    start = mm & ~np.pad(mm, ((0, 0), (1, 0)))[:, :-1]
    run_id = np.cumsum(start.ravel()).reshape(h, w)
    run_id[~mm] = 0
    n = int(run_id.max()) + 1
    R = np.zeros((h, w), bool)
    for i, iv in enumerate(chosen):
        if iv is not None:
            R[i, iv[0]:iv[1] + 1] = True
    R &= mm
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(reach):
        flag = np.zeros(n, bool)
        flag[run_id[R]] = True
        flag[0] = False
        R = flag[run_id]
        R = cv2.dilate(R.astype(np.uint8), k3).astype(bool) & mm
    flag = np.zeros(n, bool)
    flag[run_id[R]] = True
    flag[0] = False
    R = flag[run_id]
    any_ = R.any(1)
    lo = np.where(any_, R.argmax(1), np.nan).astype(float)
    hi = np.where(any_, w - 1 - R[:, ::-1].argmax(1), np.nan).astype(float)
    return (lo, hi, R) if return_mask else (lo, hi)


def _edge_pieces(R: np.ndarray, edge: int):
    """По строкам: конец куска, начинающегося у левого края (lend), и начало куска, кончающегося у правого (rstart).
    -1 — такого куска нет. Отрезки с разрывом до 3 px склеиваются."""
    h, w = R.shape
    lend = np.full(h, -1)
    rstart = np.full(h, -1)
    for y in np.flatnonzero(R[:, :edge].any(1) | R[:, w - edge:].any(1)):
        iv = _intervals(R[y])
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s - merged[-1][1] <= 3:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        if merged[0][0] < edge:
            lend[y] = merged[0][1]
        if merged[-1][1] >= w - edge:
            rstart[y] = merged[-1][0]
    return lend, rstart


def zigzag_points(m: np.ndarray, chosen, level, xl: int, y0: int, stroke: int, prom: float | None = None,
                  max_level: int | None = None, edge: int = 8, min_level: int = 0):
    """Точки для сжатого лога: в узких местах — центр линии, в «гармошке» — вершины каждого изгиба
    (локальный минимум левого края и локальный максимум правого края), по порядку глубины.
    Уровни шкалы (перехлёст) считаются по чернилам: строка, где кривая касается обоих краёв с дырой
    посередине, — выход за край; куда кривая продолжается после такого блока строк — такой и уровень."""
    h, w = m.shape
    lo, hi, R = local_extent(m, chosen, return_mask=True)
    prom = prom or max(5.0, 2.0 * stroke)
    narrow_w = 3 * stroke
    valid = ~np.isnan(lo)
    M = m.astype(bool)                           # все чернила дорожки — для поиска выходов за край
    lend, rstart = _edge_pieces(M, edge)
    min_hole = max(12, 4 * stroke)
    # строка-событие: левый и правый куски существуют в этой или соседней строке и между ними дыра
    event = np.zeros(h, bool)
    for y in np.flatnonzero(lend >= 0):
        for d in (0, 1, -1):
            y2 = y + d
            if 0 <= y2 < h and rstart[y2] >= 0 and rstart[y2] - lend[y] >= min_hole:
                if d == 0 or not M[min(y, y2):max(y, y2) + 1, lend[y] + 1:rstart[y2]].any(0).all():
                    # между кусками есть пустые столбцы → куски не соединены
                    if not M[min(y, y2):max(y, y2) + 1, lend[y] + 1:rstart[y2]].any(0)[2:-2].all():
                        event[y] = event[y2] = True
    blocks, idx = [], np.flatnonzero(event)
    if idx.size:
        for g in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1):
            blocks.append((int(g[0]), int(g[-1])))

    lvl = np.zeros(h, int)
    cur = 0
    top = (max_level - 1) if max_level else 10 ** 6
    bot = min_level                              # нижний допустимый уровень (−1 = выход за левый край)
    ev = []                                      # (строка, x, уровень, порядок в строке; -1 = свободный)
    blocked = np.zeros(h, bool)
    pos = 0
    # «скачки»: соседние строки кривой не пересекаются по x (связь потеряна, напр. штрих лёг на линию сетки).
    # Если перед скачком кривая была у края — она ушла за него: уровень меняется.
    in_block = np.zeros(h, bool)
    for s, e in blocks:
        in_block[s:e + 1] = True
    vr = np.flatnonzero(valid)
    for yp, y in zip(vr[:-1], vr[1:]):
        if y - yp > 6 or in_block[yp] or in_block[y]:
            continue
        if hi[yp] < lo[y] - min_hole and lo[yp] <= edge:
            blocks.append((int(y), -1))          # ушла за левый край → уровень −1
        elif lo[yp] > hi[y] + min_hole and hi[yp] >= w - 1 - edge:
            blocks.append((int(y), -2))          # ушла за правый край → уровень +1
    blocks.sort()
    for s, e in blocks:
        if s < pos:
            continue
        lvl[pos:s] = cur
        pos = s
        if e < 0:
            d = -1 if e == -1 else 1
            if bot <= cur + d <= top:
                yp = int(vr[np.searchsorted(vr, s) - 1])
                ev += [(yp, 0 if d < 0 else w - 1, cur, 5), (s, w - 1 if d < 0 else 0, cur + d, -2)]
                cur += d
            continue
        rows_l = [y for y in range(s, e + 1) if lend[y] >= 0 and (rstart[y] < 0 or rstart[y] - lend[y] >= min_hole)]
        rows_r = [y for y in range(s, e + 1) if rstart[y] >= 0 and (lend[y] < 0 or rstart[y] - lend[y] >= min_hole)]
        if not rows_l or not rows_r or s < pos:
            continue
        lmax = max(lend[y] for y in rows_l)
        rmin = min(rstart[y] for y in rows_r)

        def came_from():
            """С какой стороны кривая подошла к блоку: чернила строки перед блоком примыкают к левому или правому куску."""
            l0, r0 = rows_l[0], rows_r[0]
            for y in range(s - 1, max(-1, s - 7), -1):
                iv = _intervals(M[y])
                if not len(iv):
                    continue
                tl = bool(((iv[:, 0] <= lend[l0] + 1)).any())
                tr = bool(((iv[:, 1] >= rstart[r0] - 1)).any())
                if tl != tr:
                    return "L" if tl else "R"
                return None
            return None

        def follow(side):
            """Идём вниз по куску-заходу (side = 'L': левый кусок). Возвращает (вернулась ли кривая, последняя строка
            захода, {строка: дальний конец куска})."""
            y0_ = rows_l[-1] if side == "L" else rows_r[-1]
            rng = [0, lend[y0_]] if side == "L" else [rstart[y0_], w - 1]
            ends = {y: (lend[y] if side == "L" else rstart[y]) for y in (rows_l if side == "L" else rows_r)}
            last, miss = y0_, 0
            for y in range(e + 1, min(h, e + 1 + 400)):
                iv = _intervals(M[y])
                over = [(a_, b_) for a_, b_ in iv if a_ <= rng[1] + 2 and b_ >= rng[0] - 2]
                foreign = [(a_, b_) for a_, b_ in iv if not (a_ <= rng[1] + 2 and b_ >= rng[0] - 2)
                           and (a_ - rng[1] >= min_hole or rng[0] - b_ >= min_hole)]
                if foreign:
                    return True, last, ends          # кривая появилась в другом месте → заход закончился
                if over:
                    rng = [min(o[0] for o in over), max(o[1] for o in over)]
                    ends[y] = rng[1] if side == "L" else rng[0]
                    last, miss = y, 0
                    if y - e >= 14:
                        return False, last, ends     # кусок живёт долго один → кривая осталась на этом уровне
                else:
                    miss += 1
                    if miss > 8:
                        return True, last, ends
            return False, last, ends

        before = came_from() or ("R" if cur < top else "L")
        if before == "R":
            lr, ll = cur, cur + 1                # правый кусок — текущий уровень, левый — следующий
        else:
            ll, lr = cur, cur - 1
        if not (bot <= ll <= top and bot <= lr <= top):
            before = "L" if before == "R" else "R"   # уровень вне шкалы — значит подошла с другой стороны
            if before == "R":
                lr, ll = cur, cur + 1
            else:
                ll, lr = cur, cur - 1
            if not (bot <= ll <= top and bot <= lr <= top):
                continue
        back, e2, ends = follow("L" if before == "R" else "R")
        if back:
            blocked[s:e2 + 1] = True
            ys_ = sorted(ends)
            if before == "R":                    # пик за правый край и обратно
                tip = max(ys_, key=lambda y: ends[y])
                ev += [(ys_[0], w - 1, lr, 0), (ys_[0], 0, ll, 1), (tip, ends[tip] - stroke / 2, ll, 2),
                       (ys_[-1], 0, ll, 3), (ys_[-1], w - 1, lr, 4)]
            else:                                # провал за левый край и обратно
                tip = min(ys_, key=lambda y: ends[y])
                ev += [(ys_[0], 0, ll, 0), (ys_[0], w - 1, lr, 1), (tip, ends[tip] + stroke / 2, lr, 2),
                       (ys_[-1], w - 1, lr, 3), (ys_[-1], 0, ll, 4)]
            e = max(e, e2)
        else:
            blocked[s:e + 1] = True
            if before == "R":                    # ушла на следующий уровень и осталась
                ev += [(rows_l[0], w - 1, lr, 0), (rows_l[0], 0, ll, 1)]
                cur = ll
            else:                                # вернулась на предыдущий уровень
                ev += [(rows_r[0], 0, ll, 0), (rows_r[0], w - 1, lr, 1)]
                cur = lr
        lvl[s:e + 1] = cur
        pos = e + 1
    lvl[pos:] = cur
    free = valid & ~blocked

    def tips(arr, sign):
        """Локальные экстремумы с выступом >= prom относительно соседних экстремумов."""
        a = np.where(free, arr * sign, -np.inf)
        out = []
        i = 0
        while i < h:
            if not free[i]:
                i += 1
                continue
            j = i
            while j + 1 < h and free[j + 1] and a[j + 1] == a[i]:
                j += 1                           # плато
            left = a[i - 1] if i > 0 else -np.inf
            right = a[j + 1] if j + 1 < h else -np.inf
            if a[i] > left and a[i] > right:
                def saddle(rng):
                    mn = a[i]
                    for k in rng:
                        if not free[k] or a[k] > a[i]:
                            return mn
                        mn = min(mn, a[k])
                    return mn
                pl = a[i] - saddle(range(i - 1, max(-1, i - 40), -1))
                pr = a[i] - saddle(range(j + 1, min(h, j + 40)))
                if min(pl, pr) >= prom:
                    out.append(((i + j) // 2, arr[i]))
            i = j + 1
        return out

    tip_rows = set()
    for y, x in tips(hi, 1):
        ev.append((y, x - stroke / 2, int(lvl[y]), -1))
        tip_rows.add(y)
    for y, x in tips(lo, -1):
        ev.append((y, x + stroke / 2, int(lvl[y]), -1))
        tip_rows.add(y)
    for i in range(h):
        if free[i] and i not in tip_rows and hi[i] - lo[i] + 1 <= narrow_w:
            ev.append((i, (lo[i] + hi[i]) / 2, int(lvl[i]), -1))
    ev.sort(key=lambda t: t[0])
    pts, prev_x, i = [], None, 0
    while i < len(ev):
        j = i
        while j + 1 < len(ev) and ev[j + 1][0] == ev[i][0]:
            j += 1
        grp = list(ev[i:j + 1])
        first = [g for g in grp if g[3] == -2]
        fixed = sorted([g for g in grp if g[3] >= 0], key=lambda g: g[3])
        loose = [g for g in grp if g[3] == -1]
        if first:
            prev_x = first[-1][1]
        if prev_x is not None:
            loose.sort(key=lambda g: abs(g[1] - prev_x))
        for y, x, lv, _ in first + loose + fixed:
            if pts and y - (pts[-1][1] - y0) >= 60 and not valid[pts[-1][1] - y0 + 1:y].any():
                px, py, pl_, _n = pts[-1]
                pts[-1] = (px, py, pl_, True)
            pts.append((xl + float(x), y0 + y, int(lv), False))
            prev_x = x
        i = j + 1
    return pts, lvl



def style_filter(m: np.ndarray, style: str, min_h: int = 60, min_area: int = 500, min_dash: int = 10) -> np.ndarray:
    """Разделение чёрных кривых по типу линии (после чистки сетки): сплошная линия — длинная связная
    компонента, пунктир — много коротких штрихов. style: 'solid' | 'dashed'."""
    # связность считаем по слегка расширенной маске: царапина/линия сгиба/остаток сетки в 1–3 px
    # не должны рвать сплошную кривую на «штрихи»
    mm = m.astype(np.uint8)
    # пунктир: штрихи разделены паузами 6–10 px — расширение их склеит, поэтому считаем по исходной маске
    wide = cv2.dilate(mm, np.ones((5, 5), np.uint8)) if style == "solid" else mm
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wide, 8)
    hgt = stats[:, cv2.CC_STAT_HEIGHT]
    area = np.bincount(lab[mm.astype(bool)], minlength=n)        # площадь по исходным чернилам
    big = (hgt >= min_h + (4 if style == "solid" else 0)) | (area >= min_area)
    keep = big if style == "solid" else (~big & (area >= min_dash))
    keep[0] = False
    return keep[lab] & mm.astype(bool)


def auto_stroke(sub: np.ndarray) -> int:
    """Выбор чистки по толщине штрихов вне линий сетки: если большинство штрихов (шириной >= 2 px)
    шире 4 px — кривая толстая, а сетка тонкая/пунктирная → открытие 3x3 (stroke=3).
    Иначе кривая тонкая, как сетка → без морфологии (stroke=1), сетка снимается только по решётке.
    Эталоны: 2.tif — 79% толстых (stroke 3), 10539247 — 17% (stroke 1)."""
    h, w = sub.shape
    rows = sub.mean(1)
    thick = total = 0
    for y in np.flatnonzero(rows < 0.3)[:: max(1, h // 1500)]:
        r = _intervals(sub[y])
        if len(r):
            ww = r[:, 1] - r[:, 0] + 1
            ww = ww[(ww >= 2) & (ww < 12)]
            thick += int((ww >= 4).sum())
            total += len(ww)
    return 3 if total and thick / total > 0.5 else 1


def _gap_after(chosen, i, min_gap: int = 60) -> bool:
    k = i
    while k < len(chosen) and chosen[k] is None:
        k += 1
    return (k - i) >= min_gap


def curve_mask(img_rgb: np.ndarray, color=None, tol: float = 45.0, dark_thresh: int | None = None,
               max_sat: int = 70, y0: int = 0, y1: int | None = None, x0: int = 0, x1: int | None = None) -> np.ndarray:
    """Маска пикселей кривой в окне [y0:y1, x0:x1] (возвращается маска размером со всё изображение,
    заполненная только в окне — экономит память на длинных сканах).

    color=None — чёрная кривая: тёмные и НЕнасыщенные пиксели (цветные кривые не попадают).
    color=(r,g,b) — цветная кривая: пиксели ближе tol по ΔE (Lab) к цвету из легенды.
    """
    h, w = img_rgb.shape[:2]
    y1 = h if y1 is None else y1
    x1 = w if x1 is None else x1
    win = img_rgb[y0:y1, x0:x1]
    out = np.zeros((h, w), bool)
    if color is None:
        gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
        thr = dark_thresh
        if thr is None:
            # на сером/цветном скане сетка серая (100–200), а кривые чёрные — не пускаем серое в маску
            sample = gray[:: max(1, gray.shape[0] // 400), :: 3]
            mid = np.mean((sample > 60) & (sample < 200))
            if mid > 0.05:
                thr = 90
        m = dark_mask(gray, thr)
        if win.ndim == 3:
            sat = win.max(2).astype(int) - win.min(2)
            m &= sat <= max_sat
    else:
        target_hsv = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_RGB2HSV_FULL)[0, 0].astype(int)
        if target_hsv[1] >= 80:
            # насыщенный цвет: сравниваем оттенок (устойчиво к пережатым/выцветшим сканам)
            hsv = cv2.cvtColor(win, cv2.COLOR_RGB2HSV_FULL).astype(int)
            dh = np.abs(hsv[..., 0] - target_hsv[0])
            dh = np.minimum(dh, 256 - dh)
            m = (dh <= tol * 0.4) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 50)
        else:
            lab = cv2.cvtColor(win, cv2.COLOR_RGB2LAB).astype(np.float32)
            t = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
            m = np.linalg.norm(lab - t, axis=2) <= tol
    out[y0:y1, x0:x1] = m
    return out


def dark_mask(img_gray: np.ndarray, thresh: int | None = None) -> np.ndarray:
    if thresh is None:
        thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = min(max(thresh, 64), 200)  # двухцветный скан даёт порог 0
    return img_gray < thresh
