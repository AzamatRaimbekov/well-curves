"""Чистая перерисовка кривых и наложение для проверки."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def render_clean(curves: list[dict], job: dict, out_png: Path) -> None:
    """Аккуратный график: каждая кривая своим цветом из палитры, подписи, сетка."""
    orientation = job.get("orientation", "vertical")
    meta = job.get("meta", {})
    tracks = job.get("tracks")  # [[ "GR" ], ["RT","RXO"]] — кривые по дорожкам
    if not tracks:
        tracks = [[c["name"]] for c in curves] if orientation == "vertical" else [[c["name"] for c in curves]]
    by_name = {c["name"]: c for c in curves}
    colors = {c["name"]: c.get("plot_rgb") or c["color_rgb"] for c in job["curves"]}
    scales = {c["name"]: c.get("scale", job["axes"]["x" if orientation == "vertical" else "y"].get("scale", "linear"))
              for c in job["curves"]}

    if orientation == "vertical":
        fig, axs = plt.subplots(1, len(tracks), figsize=(2.6 * len(tracks) + 1, 11), sharey=True, squeeze=False)
        for ax, names in zip(axs[0], tracks):
            for n in names:
                c = by_name.get(n)
                if c is None or c["val"].size == 0:
                    continue
                ax.plot(c["val"], c["arg"], color=_hex(colors[n]), lw=1.3, label=f'{n} {c["unit"]}'.strip())
                if scales[n] == "log":
                    ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=8, frameon=False)
        axs[0][0].invert_yaxis()
        axs[0][0].set_ylabel(job.get("arg_label", "Глубина, м"))
    else:
        fig, ax = plt.subplots(figsize=(13, 6))
        for names in tracks:
            for n in names:
                c = by_name.get(n)
                if c is None or c["val"].size == 0:
                    continue
                ax.plot(c["arg"], c["val"], color=_hex(colors[n]), lw=1.3, label=f'{n} {c["unit"]}'.strip())
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(job.get("arg_label", "Время / глубина"))
        ax.legend(fontsize=8)
    title = " · ".join(str(v) for v in [meta.get("field"), meta.get("well"), meta.get("date")] if v)
    fig.suptitle(title or Path(job["image"]).stem)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_overlay(img_bgr: np.ndarray, curves: list[dict], job: dict, out_png: Path) -> None:
    """Оригинал + извлечённые точки поверх — чтобы глазами проверить совпадение."""
    over = (img_bgr * 0.55 + 255 * 0.45).astype(np.uint8)  # бледный оригинал
    colors = {c["name"]: c.get("plot_rgb") or c["color_rgb"] for c in job["curves"]}
    for c in curves:
        pts = c["pixels"].astype(np.int32)
        if len(pts) > 1:
            bgr = tuple(int(v) for v in colors[c["name"]][::-1])
            cv2.polylines(over, [pts.reshape(-1, 1, 2)], False, (0, 0, 0), 4)
            cv2.polylines(over, [pts.reshape(-1, 1, 2)], False, bgr, 2)
    ok, buf = cv2.imencode(".png", over)
    buf.tofile(str(out_png))


def write_csv(curves: list[dict], out_csv: Path, arg_name: str = "DEPT") -> None:
    """Длинный формат: аргумент, кривая, значение — без потерь, при любой сетке."""
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([arg_name, "curve", "value", "unit"])
        for c in curves:
            for a, v in zip(c["arg"], c["val"]):
                w.writerow([f"{a:.3f}", c["name"], f"{v:.4f}", c["unit"]])


def write_las(curves: list[dict], out_las: Path, step: float, meta: dict) -> None:
    """LAS 2.0 на равномерной сетке глубин (для каротажа, ориентация vertical)."""
    import lasio

    valid = [c for c in curves if c["arg"].size > 1]
    if not valid:
        return
    lo = min(c["arg"].min() for c in valid)
    hi = max(c["arg"].max() for c in valid)
    depth = np.arange(lo, hi + step / 2, step)
    las = lasio.LASFile()
    las.well["WELL"].value = meta.get("well", "")
    las.well["FLD"].value = meta.get("field", "")
    las.append_curve("DEPT", depth, unit="M")
    for c in valid:
        order = np.argsort(c["arg"])
        a, v = c["arg"][order], c["val"][order]
        col = np.interp(depth, a, v, left=np.nan, right=np.nan)
        las.append_curve(c["name"], col, unit=c["unit"])
    las.write(str(out_las), version=2.0)
