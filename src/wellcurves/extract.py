"""Выделение кривых с фотографии по цвету.

Идея: каждая кривая на бланке нарисована своим цветом. Строим маску пикселей,
близких к цвету кривой (в пространстве Lab — оно ближе к восприятию глаза),
затем для каждой строки (или столбца) берём точку кривой. Если в строке
несколько кусков одного цвета, выбираем ближайший к предыдущей точке —
так кривая не «прыгает» на подписи и соседние линии того же цвета.
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibrate import Axis


def load_image(path: str, crop: list[int] | None = None) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)  # работает с кириллицей в путях
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
    if crop:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    return img


def color_mask(img_bgr: np.ndarray, rgb: list[int], tol: float = 30.0) -> np.ndarray:
    """Маска пикселей, отличающихся от цвета rgb не больше чем на tol (ΔE в Lab)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - target, axis=2)
    mask = (dist <= tol).astype(np.uint8)
    # убираем одиночные шумовые пиксели и склеиваем разрывы толщиной 1px
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _runs(idx: np.ndarray) -> list[np.ndarray]:
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)


def trace(mask: np.ndarray, along: str = "y") -> np.ndarray:
    """Возвращает массив координат кривой по оси «поперёк».

    along="y": кривая идёт сверху вниз (каротаж: глубина по вертикали),
               результат — x-пиксель для каждой строки.
    along="x": кривая идёт слева направо (время/глубина по горизонтали),
               результат — y-пиксель для каждого столбца.
    Нет данных → NaN.
    """
    m = mask if along == "y" else mask.T
    out = np.full(m.shape[0], np.nan)
    prev = np.nan
    for i in range(m.shape[0]):
        runs = _runs(np.flatnonzero(m[i]))
        if not runs:
            continue
        centers = np.array([r.mean() for r in runs])
        if np.isnan(prev):
            pick = centers[np.argmax([len(r) for r in runs])]
        else:
            pick = centers[np.argmin(np.abs(centers - prev))]
        out[i] = prev = pick
    return out


def fill_gaps(arr: np.ndarray, max_gap: int = 15) -> np.ndarray:
    """Линейно заполняет короткие разрывы (<= max_gap пикселей)."""
    a = arr.copy()
    isn = np.isnan(a)
    if isn.all():
        return a
    idx = np.arange(a.size)
    filled = np.interp(idx, idx[~isn], a[~isn])
    for run in _runs(np.flatnonzero(isn)):
        inner = run[0] > 0 and run[-1] < a.size - 1
        if inner and len(run) <= max_gap:
            a[run] = filled[run]
    return a


def extract_curve(img_bgr: np.ndarray, curve: dict, axes: dict, orientation: str) -> dict:
    """Извлекает одну кривую и переводит её в реальные единицы.

    orientation="vertical": глубина по Y, значение по X.
    orientation="horizontal": аргумент (время/глубина) по X, значение по Y.
    """
    ax_x, ax_y = Axis.from_dict(axes["x"]), Axis.from_dict(axes["y"])
    mask = color_mask(img_bgr, curve["color_rgb"], curve.get("tol", 30))
    along = "y" if orientation == "vertical" else "x"
    px = fill_gaps(trace(mask, along), curve.get("max_gap", 15))
    pos = np.arange(px.size, dtype=float)
    ok = ~np.isnan(px)
    if orientation == "vertical":
        arg = np.array([ax_y.to_value(p) for p in pos[ok]])
        val = np.array([ax_x.to_value(p) for p in px[ok]])
        pts = np.column_stack([px[ok], pos[ok]])
    else:
        arg = np.array([ax_x.to_value(p) for p in pos[ok]])
        val = np.array([ax_y.to_value(p) for p in px[ok]])
        pts = np.column_stack([pos[ok], px[ok]])
    coverage = float(ok.mean()) if px.size else 0.0
    return {"name": curve["name"], "unit": curve.get("unit", ""), "arg": arg,
            "val": val, "pixels": pts, "coverage": coverage,
            "mask_pixels": int(mask.sum())}


def dominant_colors(img_bgr: np.ndarray, k: int = 8, min_sat: int = 60) -> list[dict]:
    """Подсказка: самые частые насыщенные (не серые/не чёрные) цвета на картинке."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sel = (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= 50)
    pix = img_bgr[sel].astype(np.float32)
    if len(pix) < k:
        return []
    if len(pix) > 200_000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 200_000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pix, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    res = [{"rgb": [int(c) for c in centers[i][::-1]], "share": round(counts[i] / len(pix), 3)}
           for i in np.argsort(-counts) if counts[i] / len(pix) >= 0.005]
    return res
