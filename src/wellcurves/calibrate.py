"""Пересчёт пикселей фотографии в реальные единицы (глубина, API, Ом·м, т и т.д.).

Калибровка оси задаётся двумя опорными точками: пиксель и значение,
прочитанное с подписи шкалы. Поддерживаются линейная и логарифмическая шкалы.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Axis:
    p1_px: float
    p1_val: float
    p2_px: float
    p2_val: float
    scale: str = "linear"  # "linear" | "log"

    @classmethod
    def from_dict(cls, d: dict) -> "Axis":
        (a_px, a_val), (b_px, b_val) = d["p1"], d["p2"]
        if a_px == b_px:
            raise ValueError("Опорные точки оси должны быть в разных пикселях")
        scale = d.get("scale", "linear")
        if scale == "log" and (a_val <= 0 or b_val <= 0):
            raise ValueError("Для логарифмической шкалы значения должны быть > 0")
        return cls(float(a_px), float(a_val), float(b_px), float(b_val), scale)

    def _f(self, v: float) -> float:
        return math.log10(v) if self.scale == "log" else v

    def _finv(self, v: float) -> float:
        return 10 ** v if self.scale == "log" else v

    def to_value(self, px: float) -> float:
        a, b = self._f(self.p1_val), self._f(self.p2_val)
        t = (px - self.p1_px) / (self.p2_px - self.p1_px)
        return self._finv(a + t * (b - a))

    def to_pixel(self, val: float) -> float:
        a, b = self._f(self.p1_val), self._f(self.p2_val)
        t = (self._f(val) - a) / (b - a)
        return self.p1_px + t * (self.p2_px - self.p1_px)
