@echo off
rem Автообводка по заданию: run.bat data\jobs\<имя>.json  -> output\<имя>\
cd /d %~dp0
set PYTHONPATH=src
.venv\Scripts\python -m wellcurves auto %*
