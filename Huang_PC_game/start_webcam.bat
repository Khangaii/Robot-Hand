@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "handpython=%USERPROFILE%\Documents\robot_hand\.venv\Scripts\python.exe"
if exist "%handpython%" goto run
set "handpython=python"
:run
"%handpython%" huang_live.py %*
if errorlevel 1 pause
