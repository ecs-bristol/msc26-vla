@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

pushd "%ROOT%"
"%PYTHON%" -m src.vla_bench.pre_jetson_runner %*
set "STATUS=%ERRORLEVEL%"
popd

exit /b %STATUS%

