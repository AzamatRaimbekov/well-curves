# well-curves — автообводка каротажных логов

Скан лога (TIFF/PNG) → кривые обведены автоматически → `.improj` для SoloImageAnalyzer, LAS, CSV,
цветная картинка на белом фоне и проверочные картинки «обводка поверх скана». Сомнительные места
правятся мышью в редакторе (в браузере, локально).

## Как начать (Windows)

1. Установить Python 3.11+ с python.org (галочка «Add to PATH»).
2. Скачать репозиторий (Code → Download ZIP или `git clone`), распаковать.
3. Запустить `install.bat` — один раз, ставит библиотеки в папку `.venv`.

## Как получить результат

1. Положить скан в `data\inbox\`.
2. Сделать задание `data\jobs\<имя>.json` по образцу `data\jobs\_example_auto.json`:
   путь к скану, две опорные точки глубины, края дорожек, для каждой кривой — имя и шкала.
   С Claude Code задание пишет агент сам (MCP-сервер `wellcurves`, скилл `well-chart-digitizer`).
3. `run.bat data\jobs\<имя>.json` → результат в `output\<имя>\`:
   `auto.improj`, `curves.las`, `curves.csv`, `colored.png`, `overlay\`.
4. `edit.bat output\<имя>uto.improj` → редактор в браузере: список сомнительных мест, правка мышью,
   «Сохранить» пишет `auto.edited.improj` и `edited\curves.las`.
5. Если есть эталон: `.venv\Scripts\python -m wellcurves verify <auto.improj> <эталон.improj>`.

## Что умеет и чего пока нет

- Дорожка с одной кривой: 98–99% длины совпадает с ручной оцифровкой (в пределах 3% шкалы, ±1 ft).
- Сжатые логи (2 px/ft): около 86%.
- Несколько чёрных кривых в одной дорожке, тонкий штрих-пунктир, температуры — получаются не все,
  нужна ручная правка в редакторе.
- Задание (края дорожек, шкалы) пока составляется вручную или агентом, «просто загрузить фото» ещё нет.
- Открытие `auto.improj` в SoloImageAnalyzer не проверено.

Сканы, эталоны и результаты в репозиторий не входят.

Лицензия: MIT — бесплатно, можно использовать и дорабатывать.
С Claude Code (Fable / Opus): инструкции агенту в `CLAUDE.md`, подключение MCP — `.mcp.json.example` → `.mcp.json`.

Математика кривых и алгоритм: `kb/docs/curve-math.md`.

---

## Команды разработчика

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python -m wellcurves auto   data\jobs\<имя>.json
.venv\Scripts\python -m wellcurves verify <auto.improj> <эталон.improj>
.venv\Scripts\python -m wellcurves doubts <auto.improj>
.venv\Scripts\python -m wellcurves edit   <auto.improj>
.venv\Scripts\python -m wellcurves check  <файл.improj>
.venv\Scripts\python -m pytest -q tests
```
