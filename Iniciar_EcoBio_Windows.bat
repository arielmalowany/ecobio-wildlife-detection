@echo off
title EcoBio - Clasificador de Fauna
color 0A

echo ============================================================
echo   EcoBio - Clasificador de Fauna Uruguaya
echo ============================================================
echo.

REM ── Ir al directorio donde esta este .bat ────────────────────
cd /d "%~dp0"

REM ── Buscar Python en el sistema ──────────────────────────────
SET PYTHON_EXE=

REM 1. Buscar Python 3.11 especificamente con el launcher py
py -3.11 --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON_EXE=py -3.11
    goto :check_version
)

REM 2. Buscar Python 3.12
py -3.12 --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON_EXE=py -3.12
    goto :check_version
)

REM 3. Buscar Python 3.10
py -3.10 --version >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON_EXE=py -3.10
    goto :check_version
)

REM 4. Buscar python estandar ignorando Anaconda
FOR /F "delims=" %%i IN ('where python 2^>nul') DO (
    IF NOT DEFINED PYTHON_EXE (
        echo %%i | findstr /i "anaconda WindowsApps" >nul
        IF %ERRORLEVEL% NEQ 0 SET PYTHON_EXE=%%i
    )
)

REM 5. Buscar en rutas comunes
IF NOT DEFINED PYTHON_EXE (
    FOR %%p IN (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
    ) DO (
        IF NOT DEFINED PYTHON_EXE IF EXIST %%p SET PYTHON_EXE=%%p
    )
)

REM 6. Ultimo recurso: cualquier python disponible
IF NOT DEFINED PYTHON_EXE (
    python --version >nul 2>&1
    IF %ERRORLEVEL% EQU 0 SET PYTHON_EXE=python
)

IF NOT DEFINED PYTHON_EXE (
    echo [ERROR] No se encontro Python en el sistema.
    echo.
    echo  Descarga Python 3.11 desde:
    echo  https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo  Durante la instalacion, tilda "Add Python to PATH".
    pause
    exit /b 1
)

REM ── Verificar version de Python ──────────────────────────────
:check_version
FOR /F "tokens=2 delims= " %%v IN ('%PYTHON_EXE% --version 2^>^&1') DO SET PY_VERSION=%%v
FOR /F "tokens=1,2 delims=." %%a IN ("%PY_VERSION%") DO (
    SET PY_MAJOR=%%a
    SET PY_MINOR=%%b
)

IF %PY_MAJOR% NEQ 3 (
    echo [ERROR] Se requiere Python 3. Tenes Python %PY_VERSION%.
    goto :python_error
)
IF %PY_MINOR% LSS 9 (
    echo [ERROR] Se requiere Python 3.9 o superior. Tenes Python %PY_VERSION%.
    goto :python_error
)
IF %PY_MINOR% GEQ 14 (
    echo [ERROR] Python %PY_VERSION% no es compatible. Se requiere Python 3.9 a 3.13.
    goto :python_error
)

echo [OK] Python %PY_VERSION% encontrado y compatible.
goto :continue_install

:python_error
echo.
echo  Descarga Python 3.11 desde:
echo  https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
echo  Durante la instalacion, tilda "Add Python to PATH".
pause
exit /b 1

:continue_install

REM ═════════════════════════════════════════════════════════════
REM  CASO A: venv ya instalado, saltar instalacion
REM ═════════════════════════════════════════════════════════════
IF EXIST "venv\installed.flag" (
    echo [OK] Entorno virtual encontrado. Saltando instalacion.
    goto :launch
)

REM ═════════════════════════════════════════════════════════════
REM  CASO B: instalar todo desde cero
REM ═════════════════════════════════════════════════════════════
echo [1/4] Creando entorno virtual...
%PYTHON_EXE% -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [2/4] Actualizando pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [3/4] Instalando dependencias base...
echo      - instalando torch CPU primero...
pip install torch==2.9.0+cpu torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la instalacion de torch.
    pause
    exit /b 1
)

echo      - fijando numpy...
pip install "numpy>=1.24,<2" --quiet

echo      - instalando resto de dependencias...
pip install -r requirements.txt --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo [4/4] Instalando megadetector y speciesnet...
IF EXIST "packages\megadetector\" (
    echo      - megadetector: instalando desde carpeta local...
    pip install packages\megadetector\ --quiet
) ELSE (
    echo      - megadetector: descargando desde GitHub...
    pip install git+https://github.com/agentmorris/MegaDetector.git --quiet
)
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la instalacion de megadetector.
    pause
    exit /b 1
)

IF EXIST "packages\cameratrapai\" (
    echo      - speciesnet: instalando desde carpeta local...
    pip install packages\cameratrapai\ --quiet
) ELSE (
    echo      - speciesnet: descargando desde GitHub...
    pip install git+https://github.com/google/cameratrapai.git --quiet
)
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la instalacion de speciesnet.
    pause
    exit /b 1
)

echo. > venv\installed.flag
echo.
echo [OK] Instalacion completada exitosamente.
echo.

REM ═════════════════════════════════════════════════════════════
:launch
REM ── Lanzar Streamlit ─────────────────────────────────────────
REM ═════════════════════════════════════════════════════════════
call venv\Scripts\activate.bat

REM Aislar PATH para evitar conflictos con Anaconda u otros entornos
SET PATH=%~dp0venv\Scripts;%SystemRoot%\system32;%SystemRoot%

echo  Iniciando aplicacion...
echo  Abriendo en el navegador: http://localhost:8501
echo  Para cerrar la app, cerra esta ventana.
echo.

start "" /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8501"
streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

pause
