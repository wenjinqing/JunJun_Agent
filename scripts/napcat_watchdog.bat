@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo 启动 NapCat 看门狗（含自动拉起 NapCat + 掉线自动重连，最小化常驻，关窗即停）...
start "NapCat看门狗" /min .venv\Scripts\python.exe scripts\napcat_watchdog.py
