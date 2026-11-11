@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0training\train.ps1" %*
pause
