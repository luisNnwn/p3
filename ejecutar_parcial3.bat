@echo off
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
    echo No se encontro el entorno Python 3.12 del proyecto.
    echo Ejecuta primero la instalacion de dependencias.
    pause
    exit /b 1
)
".venv312\Scripts\python.exe" "parcial3.py"
if errorlevel 1 pause
