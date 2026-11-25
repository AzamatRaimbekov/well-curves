"""Распознавание типа линии (lines.py) на синтетических дорожках."""
import numpy as np

from wellcurves import lines


def _track(h=600, w=400):
    a = np.zeros((h, w), bool)
    a[:, ::50] = True            # тонкая вертикальная сетка 1 px
    a[::40, :] = True            # тонкая горизонтальная сетка 1 px
    return a


def _wave(h, x0, amp=25, period=300):
    y = np.arange(h)
    return (x0 + amp * np.sin(2 * np.pi * y / period)).astype(int)


def _draw(a, xs, width, dash=None):
    h = a.shape[0]
    for y in range(h):
        if dash and (y // dash[0]) % 2:          # dash = (длина штриха, ...) — через одну пропуск
            continue
        x = xs[y]
        a[y, x - width // 2: x - width // 2 + width] = True
        if y:                                     # соединяем соседние строки (крутые участки)
            lo, hi = sorted((xs[y - 1], x))
            a[y, lo:hi + 1] = True if not dash else a[y, lo:hi + 1]


def _kinds(a, seeds):
    h, w = a.shape
    f = lines.recognize(a, seeds, np.zeros(h), np.full(h, w - 1.0))
    return [lines.classify(x).get("_kind") for x in f], f


def test_solid_and_dashed_in_one_track():
    a = _track()
    s1, s2 = _wave(600, 120), _wave(600, 280, period=260)
    _draw(a, s1, 6)
    _draw(a, s2, 4, dash=(12,))
    kinds, f = _kinds(a, [float(s1[0]), float(s2[0])])
    assert kinds == ["solid", "dashed"], f


def test_thin_line():
    a = _track()
    s = _wave(600, 200)
    _draw(a, s, 2)
    kinds, f = _kinds(a, [float(s[0])])
    assert kinds == ["thin"], f


def test_probe_does_not_stick_to_grid():
    """Стартовая точка рядом с вертикальной линией сетки: «щуп» должен уйти по кривой, а не по сетке."""
    a = _track()
    s = _wave(600, 152, amp=30)
    _draw(a, s, 6)
    h, w = a.shape
    f = lines.recognize(a, [float(s[0])], np.zeros(h), np.full(h, w - 1.0))[0]
    assert f["width"] >= 4 and f["long_frac"] > 0.8, f
