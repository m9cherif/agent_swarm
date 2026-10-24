@echo off
REM =========================================================================
REM  Launch the Agent Swarm desktop console (PyQt6).
REM  Same Orchestrator as `python -m swarm run`, just with a GUI in front.
REM =========================================================================
setlocal
cd /d "%~dp0"

REM Use the python on PATH by default (no machine-specific install path).
REM Override with:  set PYTHON=C:\path\to\python.exe  before running this file.
if not defined PYTHON set "PYTHON=python"
where %PYTHON% >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH (looked for: %PYTHON%^).
    echo Install Python 3.8+ or set PYTHON to its full path.
    exit /b 1
)

"%PYTHON%" -c "import PyQt6.QtWidgets" >nul 2>nul
if errorlevel 1 (
    echo PyQt6 is not installed for %PYTHON%.
    echo Install it with:  %PYTHON% -m pip install PyQt6
    exit /b 1
)

echo Starting the Agent Swarm console...
"%PYTHON%" -m swarm gui
set ERR=%errorlevel%
if not "%ERR%"=="0" echo. & echo GUI exited with code %ERR%.
exit /b %ERR%
