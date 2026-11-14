@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "EVALUATOR_PYTHON=python"
if exist "%USERPROFILE%\miniconda3\envs\pytorch\python.exe" set "EVALUATOR_PYTHON=%USERPROFILE%\miniconda3\envs\pytorch\python.exe"
if exist ".venv\Scripts\python.exe" set "EVALUATOR_PYTHON=%~dp0.venv\Scripts\python.exe"
"%EVALUATOR_PYTHON%" -c "import tkinter, ultralytics, yaml, PIL" 2>nul
if errorlevel 1 (
 echo 当前 Python 环境缺少运行依赖，请先按 README.md 的安装步骤准备环境。
 pause
 exit /b 1
)
"%EVALUATOR_PYTHON%" app.py
if errorlevel 1 pause
