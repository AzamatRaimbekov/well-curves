"""MCP-сервер автообводки каротажных логов (well-curves).

Порядок работы агента с новым сканом:
  1. scan_info(image)                    — размер, режим, профиль секций по высоте
  2. scan_crop(image, y0, y1)            — картинка куска скана (шапка, подписи глубин) → посмотреть Read
  3. scan_layout(image, y0, y1)          — x вертикальных линий (края дорожек), y горизонтальных линий (шаг глубины)
  4. job_template()                      — формат задания; агент пишет data/jobs/<API>.json
  5. autotrace(job)                      — обводка всех кривых → output/<API>/ (improj, LAS, CSV, colored, overlay)
  6. overlay_list(api)                   — проверочные картинки (обводка поверх скана) → посмотреть Read
  7. check_improj(path)                  — автопроверка перед сдачей
  8. verify(auto, ref)                   — сверка с эталонным .improj (если есть), в % шкалы
  9. publish(api)                        — собрать итоговую папку results/<API>/
Длинные операции (autotrace) идут минуты: сканы по 50–100 тыс. строк.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
mcp = MCPServer("wellcurves")


def _p(path: str) -> Path:
    q = Path(path)
    return q if q.is_absolute() else ROOT / q


@mcp.tool()
def scan_info(image: str) -> str:
    """Размер и режим скана + доля чернил по блокам 500 строк (видно шапки и секции лога)."""
    im = Image.open(_p(image))
    a = np.array(im.convert("L")) < 128
    r = a.mean(1)
    blk = r[: len(r) // 500 * 500].reshape(-1, 500).mean(1)
    return json.dumps({"size": im.size, "mode": im.mode, "dpi": [float(v) for v in im.info.get("dpi", (0, 0))],
                       "ink_per_500_rows": [round(float(v), 2) for v in blk]})


@mcp.tool()
def scan_crop(image: str, y0: int, y1: int, max_w: int = 1400) -> str:
    """Вырезать строки y0..y1 скана в PNG (output/look/). Возвращает путь и масштаб — картинку открыть через Read."""
    im = Image.open(_p(image))
    c = im.crop((0, y0, im.width, min(im.height, y1))).convert("RGB")
    k = 1.0
    if c.width > max_w:
        k = max_w / c.width
        c = c.resize((max_w, int(c.height * k)))
    dst = ROOT / "output" / "look" / f"{Path(image).stem}_{y0}_{y1}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    c.save(dst)
    return json.dumps({"png": str(dst), "scale": round(k, 3)})


@mcp.tool()
def scan_layout(image: str, y0: int, y1: int) -> str:
    """Края дорожек (x длинных вертикальных линий) и горизонтальные линии сетки (y, шаг) на строках y0..y1."""
    from wellcurves import pipeline

    rgb = np.array(Image.open(_p(image)).crop((0, y0, 10 ** 6, y1)).convert("RGB"))
    res = pipeline.detect_layout(rgb, 0, rgb.shape[0])
    res["horizontal_lines_y"] = [round(v + y0, 1) for v in res["horizontal_lines_y"]]
    return json.dumps(res)


@mcp.tool()
def job_template() -> str:
    """Формат задания автообводки (JSON) с пояснением полей."""
    from wellcurves import pipeline

    return (pipeline.__doc__ or "") + (
        "\nДоп. поля кривой: \"style\": \"solid\"|\"dashed\" — тип линии, когда в дорожке несколько чёрных кривых "
        "(толстая сплошная / пунктир); \"zigzag\": true|false — сжатый лог (<4 px/ft включается сам); "
        "\"seed_x\" — x кривой в начале интервала (обязателен, если кривых в дорожке несколько).")


@mcp.tool()
def autotrace(job: str) -> str:
    """Автообводка по заданию data/jobs/<API>.json → output/<API>/. Возвращает покрытие по кривым и пути файлов."""
    from wellcurves import pipeline

    rep = pipeline.run(_p(job))
    rep["files"] = [f for f in rep["files"] if "overlay" not in f and "colored_pages" not in f]
    return json.dumps(rep, ensure_ascii=False)


@mcp.tool()
def overlay_list(api: str) -> str:
    """Список проверочных картинок output/<API>/overlay/ (обводка поверх скана, окна по 200 ft)."""
    d = ROOT / "output" / api / "overlay"
    return json.dumps(sorted(str(f) for f in d.glob("*.png")), ensure_ascii=False)


@mcp.tool()
def check_improj(project: str) -> str:
    """Автопроверка .improj перед сдачей: ошибки, пропуски, точки вне грида, масштаб."""
    from wellcurves import check, improj

    return json.dumps(check.check_project(improj.load(_p(project))), ensure_ascii=False, default=str)


@mcp.tool()
def verify(auto: str, ref: str, tol_ft: float = 1.0) -> str:
    """Сверка автообводки с эталонным .improj: медиана/p95 ошибки в % шкалы, доля совпавшей длины, худшие интервалы."""
    from wellcurves import verify as V

    return json.dumps(V.compare(_p(auto), _p(ref), 0.5, tol_ft), ensure_ascii=False)


@mcp.tool()
def publish(api: str) -> str:
    """Собрать итог в results/<API>/: auto.improj, curves.las, curves.csv, colored.png, overlay/, report.json."""
    src, dst = ROOT / "output" / api, ROOT / "results" / api
    if not src.exists():
        return json.dumps({"error": f"нет output/{api} — сначала autotrace"}, ensure_ascii=False)
    dst.mkdir(parents=True, exist_ok=True)
    skipped = []
    for name in ("auto.improj", "curves.las", "curves.csv", "colored.png", "report.json", "verify.json"):
        if (src / name).exists():
            try:
                shutil.copy2(src / name, dst / name)
            except OSError:                       # файл открыт в другой программе — пропускаем
                skipped.append(name)
    for sub in ("overlay", "colored_pages"):
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    return json.dumps({"folder": str(dst), "files": sorted(f.name for f in dst.iterdir()),
                       "skipped_open_in_other_app": skipped}, ensure_ascii=False)


@mcp.tool()
def find_doubts(project: str, image: str = "") -> str:
    """Сомнительные участки .improj (где обводка могла ошибиться): пропуски, линия по пустому месту,
    непокрытые чернила (срезанные пики), кривые вплотную, смены уровня. Пишет <имя>.doubts.json."""
    from wellcurves import doubts, improj, improj_tools

    pp = _p(project)
    pr = improj.load(pp)
    img = _p(image) if image else improj_tools.find_image(pr, pp)
    d = doubts.find_doubts(pr, img)
    pp.with_suffix(".doubts.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.dumps({"total": len(d), "serious": sum(x["severity"] >= 2 for x in d), "by_curve": doubts.summary(pr, d),
                       "first": [x for x in d if x["severity"] >= 2][:15]}, ensure_ascii=False)


@mcp.tool()
def editor_open(project: str, image: str = "", port: int = 8765) -> str:
    """Открыть редактор в браузере для .improj (скан + кривые + сомнительные места; правка мышью).
    Правки сохраняются в <имя>.edited.improj и edited/curves.las. Возвращает адрес."""
    import subprocess

    cmd = [sys.executable, "-m", "wellcurves", "edit", str(_p(project)), "--port", str(port)]
    if image:
        cmd += ["--image", str(_p(image))]
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen(cmd, cwd=str(ROOT), env=env, creationflags=flags, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return json.dumps({"url": f"http://127.0.0.1:{port}/", "note": "редактор открыт в браузере; сохранение — кнопка «Сохранить»"},
                      ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
