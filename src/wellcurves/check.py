"""Автопроверка .improj перед сдачей — то, что ищет проверяющий, плюс масштаб и оплата по регламенту.

Масштаб печати лога определяется из скана: пиксели на фут × 100 / DPI = дюймов на 100 футов.
  ~1"  → 1" = 100'            30 сом / 1000 ft
  ~2"  → 1" = 50' (2"=100')   30 сом / 1000 ft
  ~5"  → 1" = 20' (5"=100')   45 сом / 1000 ft
Оплата = длина лога / 1000 × ставка × число кривых (каждый лог отдельно).
"""
from __future__ import annotations

import numpy as np

from . import improj

SCALES = [(1.0, '1" = 100\'', 30), (2.0, '1" = 50\' (2" = 100\')', 30), (5.0, '1" = 20\' (5" = 100\')', 45)]
DEFAULT_NAMES = {"", "curve", "curve 0", "curve 1", "new curve"}


def detect_scale(g: dict, dpi: float | None):
    geo = improj.GridGeom.from_grid(g)
    span_ft = geo.max_depth - geo.min_depth
    if not dpi or span_ft <= 0:
        return None
    inch_per_100 = (geo.bottom_y - geo.top_y) / span_ft * 100 / dpi
    best = min(SCALES, key=lambda s: abs(np.log(s[0] / inch_per_100)))
    return {"inch_per_100ft": round(inch_per_100, 2), "scale": best[1], "rate_som": best[2],
            "confident": abs(np.log(best[0] / inch_per_100)) < 0.15}


def check_project(p: dict, dpi: float | None = None, max_gap_ft: float = 10.0, edge_ft: float = 5.0) -> dict:
    issues, curves_out, pay = [], [], 0.0

    def add(level, where, msg):
        issues.append({"level": level, "where": where, "msg": msg})

    for g in p["Grids"]:
        gname = g["Name"]
        if g["MinY"] >= g["MaxY"]:
            add("error", gname, f"Top ({g['MinY']}) не меньше Bottom ({g['MaxY']})")
            continue
        geo = improj.GridGeom.from_grid(g)
        sc = detect_scale(g, dpi)
        if sc and not sc["confident"]:
            add("warn", gname, f"масштаб не похож на стандартный: {sc['inch_per_100ft']}\" на 100 ft — проверь Top/Bottom")
        for c in g["Curves"][0]["Items"]:
            where = f"{gname} / {c['Name']}"
            pts = c["Points"]
            if (c["Name"] or "").strip().lower() in DEFAULT_NAMES:
                add("error", where, "кривая не переименована (имя по умолчанию)")
            if c["MinX"] == c["MaxX"]:
                add("error", where, f"Left = Right = {c['MinX']}: шкала не задана, все значения будут одинаковыми")
            # замечания проверяющих Kaiten: «нет единиц измерения», «уровни с неравномерным диапазоном»
            if "(" not in (c["Name"] or ""):
                add("warn", where, "в названии нет единицы измерения в скобках, напр. «GR (GAPI)» — частое замечание проверяющих")
            ml = sorted(c.get("MultiplierLevels") or [], key=lambda m: m["MultiplierLevel"])
            for prev, nxt in zip(ml, ml[1:]):
                if abs(nxt["MinX"] - prev["MaxX"]) > 1e-9:
                    add("warn", where, f"уровень {nxt['MultiplierLevel']} имеет диапазон {nxt['MinX']}..{nxt['MaxX']}, "
                                       f"не продолжает уровень {prev['MultiplierLevel']} ({prev['MinX']}..{prev['MaxX']}) — "
                                       "сверь с подписью на изображении")
            if not pts:
                add("error", where, "в кривой нет точек")
                continue
            ys = np.array([q["Y"] for q in pts])
            if np.any(np.diff(ys) < 0):
                add("error", where, "точки идут не по порядку глубины (есть возвраты вверх)")
            levels = {m["MultiplierLevel"] for m in c.get("MultiplierLevels") or []}
            used = {q.get("MultiplierLevel") or 1 for q in pts}
            if levels and used - levels:
                add("error", where, f"использованы уровни перехлёста {sorted(used - levels)}, которых нет в MultiplierLevels")
            v = improj.curve_values(g, c)
            d = v["depth"]
            # покрытие интервала
            if d.min() - g["MinY"] > edge_ft:
                add("warn", where, f"начало кривой {d.min():.1f} ft, а рамка с {g['MinY']} ft — не обведено {d.min() - g['MinY']:.1f} ft")
            if g["MaxY"] - d.max() > edge_ft:
                add("warn", where, f"кривая кончается на {d.max():.1f} ft, а рамка до {g['MaxY']} ft — не обведено {g['MaxY'] - d.max():.1f} ft")
            # разрывы без отметки NullsAfterThat
            dd = np.diff(d)
            gaps = [i for i in np.flatnonzero(dd > max_gap_ft) if not pts[i].get("NullsAfterThat")]
            if gaps:
                big = max(gaps, key=lambda i: dd[i])
                add("warn", where, f"{len(gaps)} пропуск(ов) > {max_gap_ft:g} ft без отметки разрыва; "
                                   f"самый большой {dd[big]:.1f} ft ({d[big]:.1f}–{d[big + 1]:.1f} ft). "
                                   "Для точечных замеров (инклинометрия) это норма")
            # точки за пределами дорожки
            left, right = geo.edges(v["y"])
            t = (v["x"] - left) / (right - left)
            out = int(np.sum((t < -0.03) | (t > 1.03)))
            if out:
                add("warn", where, f"{out} точек вне границ дорожки — возможно нужен уровень перехлёста")
            length = float(d.max() - d.min())
            row = {"grid": gname, "curve": c["Name"], "from_ft": round(float(d.min()), 1),
                   "to_ft": round(float(d.max()), 1), "length_ft": round(length, 1), "points": len(pts)}
            if sc:
                row.update(scale=sc["scale"], pay_som=round(length / 1000 * sc["rate_som"], 1))
                pay += row["pay_som"]
            curves_out.append(row)
    import re
    for g in p["Grids"]:
        if not re.fullmatch(r"Grid \d+", g["Name"].strip()) and "repeat" not in g["Name"].lower():
            add("warn", g["Name"], "грид переименован — проверяющие просят не переименовывать гриды")
    if not any("repeat" in (c["Name"] or "").lower() for g in p["Grids"] for c in g["Curves"][0]["Items"]):
        add("info", "проект", "нет REPEAT-кривых: если на изображении есть Repeat Section (Top/Bottom) — её тоже надо оцифровать")
    errors = sum(1 for i in issues if i["level"] == "error")
    return {"well": p.get("WELL"), "api": p.get("API"), "ok": errors == 0, "errors": errors,
            "warnings": len(issues) - errors, "issues": issues, "curves": curves_out,
            "pay_som_total": round(pay, 1) if dpi else None}
