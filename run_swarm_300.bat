@echo off
REM =========================================================================
REM  Launch the FULL agent swarm: 300 agents = 60 squads x 5, run sequentially
REM  Each squad has one role; a 15-AI curator council authors + votes prompts.
REM =========================================================================
setlocal
set PYTHON=C:\Users\Labo info\AppData\Local\Programs\Python\Python312\python.exe
cd /d "%~dp0"

echo Starting 300-agent swarm (60 squads x 5 agents, sequential)...
"%PYTHON%" -m swarm run ^
    --squads 60 ^
    --curators 15 ^
    --agents-per-squad 5 ^
    --concurrency 16 ^
    --reviews 2 ^
    --discussants 3 ^
    --run-prefix swarm300

echo.
echo Done. Open the newest file in out\swarm-dashboard-*.html in a browser.
endlocal
