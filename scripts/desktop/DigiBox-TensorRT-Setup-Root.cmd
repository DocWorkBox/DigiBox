@echo off
setlocal EnableExtensions

set "RUNTIME_ROOT=%~dp0avtr-runtime"
set "SETUP_HELPER=%RUNTIME_ROOT%\scripts\desktop\DigiBox-TensorRT-Setup.cmd"

if not exist "%SETUP_HELPER%" (
  echo DigiBox TensorRT setup helper is missing:
  echo %SETUP_HELPER%
  exit /b 1
)

call "%SETUP_HELPER%" "%RUNTIME_ROOT%"
set "RESULT=%ERRORLEVEL%"
exit /b %RESULT%
