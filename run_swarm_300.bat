@echo off
REM =========================================================================
REM  Launch the FULL agent swarm: 300 agents = 60 squads x 5, run sequentially
REM  Each squad has one role; a 15-AI curator council authors + votes prompts.
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

echo Starting 300-agent swarm (60 squads x 5 agents, sequential)...
"%PYTHON%" -m swarm run ^
    --squads 60 ^
    --curators 15 ^
    --agents-per-squad 5 ^
    --reviews 2 ^
    --discussants 3 ^
    --run-prefix swarm300

set ERR=%errorlevel%
if not "%ERR%"=="0" (
    echo.
    echo Swarm exited with code %ERR%. Reports written so far are in out\.
    exit /b %ERR%
)

echo.
echo Done. Open the newest file in out\swarm-dashboard-*.html in a browser.
endlocal
