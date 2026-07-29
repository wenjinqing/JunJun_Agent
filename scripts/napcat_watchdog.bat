@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo 启动 NapCat 看门狗（最小化窗口运行，关闭窗口即停止）...
start "NapCat看门狗" /min .venv\Scripts\python.exe scripts\napcat_watchdog.py
