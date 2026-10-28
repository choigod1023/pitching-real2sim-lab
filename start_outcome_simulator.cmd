@echo off
cd /d "%~dp0"
if not exist artifacts\outcome-simulator\states.json (
  "..\.venv\Scripts\python.exe" pitch_outcomes.py --source ..\trace-error-audit\artifacts\trace-error\3034_4-11520-pad1-foh-sub1.npz
  if errorlevel 1 exit /b 1
)
"..\.venv\Scripts\python.exe" pitch_outcomes.py --states artifacts/outcome-simulator/states.json --gui
if errorlevel 1 pause
