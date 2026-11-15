@echo off
rem Установка: нужен Python 3.11+ (python.org). Запустить один раз.
cd /d %~dp0
python -m venv .venv || goto :err
.venv\Scripts\python -m pip install -r requirements.txt || goto :err
echo Готово. Дальше: edit.bat ^<файл.improj^> или run.bat ^<задание.json^>
exit /b 0
:err
echo Ошибка установки. Проверьте, что Python 3.11+ установлен и добавлен в PATH.
exit /b 1
