@echo off
rem Редактор в браузере: edit.bat output\<имя>\auto.improj [--image скан.tif]
cd /d %~dp0
set PYTHONPATH=src
.venv\Scripts\python -m wellcurves edit %*
