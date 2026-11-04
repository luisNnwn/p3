@echo off
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo No se encontro el lanzador de Python.
    echo Instala Python 3.12 y marca la opcion "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv312\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -3.12 -m venv .venv312
    if errorlevel 1 (
        echo No se pudo crear el entorno virtual con Python 3.12.
        pause
        exit /b 1
    )
)

echo Instalando dependencias. Esto puede tardar varios minutos...
".venv312\Scripts\python.exe" -m pip install --upgrade pip
".venv312\Scripts\python.exe" -m pip install -r requirements-py312.txt
if errorlevel 1 (
    echo La instalacion de dependencias fallo.
    pause
    exit /b 1
)

echo Instalacion completada.
pause
