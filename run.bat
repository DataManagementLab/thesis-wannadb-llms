@echo off
REM Launch the WannaDB GUI using the project virtual environment.
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%main.py" %*
