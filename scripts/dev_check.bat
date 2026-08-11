@echo off
setlocal
cd /d "%~dp0.."
echo === ruff format (check) ===
python -m ruff format --check .
if errorlevel 1 exit /b 1
echo === ruff check ===
python -m ruff check .
if errorlevel 1 exit /b 1
echo === pytest ===
python -m pytest
if errorlevel 1 exit /b 1
echo === demo ===
python -m frontend.cli demo --goal "描述当前屏幕" --log-level WARNING
if errorlevel 1 exit /b 1
echo All checks passed.
endlocal
