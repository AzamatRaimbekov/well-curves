# well-curves — автообводка каротажных логов

Цель: скан лога (TIFF/PNG) → все кривые автоматически обведены, раскрашены на белом фоне, значения точные
(LAS/CSV/`.improj` для SoloImageAnalyzer), плюс автопроверка и редактор для сомнительных мест.

## Как работать агенту (Claude Fable / Opus)

1. Прочитать `.claude/skills/well-chart-digitizer/SKILL.md` и `kb/docs/curve-math.md` (математика, уровни, типы линий).
2. Новый скан — через MCP-сервер `wellcurves` (`tools/wellcurves_mcp.py`, пример подключения `.mcp.json.example`):
   `scan_info` → `scan_crop` (посмотреть шапку и глубины) → `scan_layout` → написать `data/jobs/<имя>.json`
   по образцу `data/jobs/_example_auto.json` → `autotrace` → `overlay_list` (посмотреть обводку поверх скана) →
   `find_doubts` → `check_improj` → `verify` (если есть эталон) → `publish`.
3. Сомнительные места не выдавать за готовые: либо исправить заданием (style, seed_x, x_range, wrap), либо
   оставить человеку в редакторе (`editor_open`).
4. Агент `well-chart-analyst` — по одному на файл, когда их много.

## Код

`src/wellcurves/`: `tracer.py` (чистка сетки, Витерби, осевая линия, уровни, пики), `pipeline.py` (скан → все кривые →
colored/overlay/LAS/CSV/improj), `improj.py` (формат .improj), `verify.py` (сверка с эталоном), `doubts.py`
(сомнительные участки), `editor.py` + `editor.html` (редактор в браузере, только 127.0.0.1), `check.py`, `cli.py`.
Python: `.venv\Scripts\python`, `$env:PYTHONPATH="src"`. Тесты: `.venv\Scripts\python -m pytest -q tests`.
Точность на наборе эталонов: `tools/batch_eval.py` (нужны свои пары скан + .improj в `data/training/<имя>/`).

## Правила

- Не выдумывать данные. Всё непрочитанное — явно помечать. Сканы и эталоны не изменять.
- Сканы, эталоны и результаты в репозиторий не коммитить (`data/`, `output/`, `results/` в `.gitignore`).
- Локальные инструкции и доступы — в `CLAUDE.local.md` и `.env` (не коммитятся).
