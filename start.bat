@echo off
chcp 65001 >nul 2>&1
title 垃圾智能分类系统
echo ====================================================
echo   垃圾智能分类系统 — 正在启动...
echo ====================================================
echo.

cd /d "%~dp0"

echo [1/2] 启动 FastAPI 服务...
start "" "http://localhost:8080"
D:\anaconda\python.exe ui/app.py

pause
