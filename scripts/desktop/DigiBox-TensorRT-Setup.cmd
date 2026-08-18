@echo off
setlocal EnableExtensions

set "SETUP_DIR=%~dp0"
set "RUNTIME_ROOT=%~1"
if not defined RUNTIME_ROOT set "RUNTIME_ROOT=%SETUP_DIR%..\.."

echo DigiBox TensorRT Setup Assistant
echo Runtime: %RUNTIME_ROOT%
echo.
echo [1] Standard acceleration (recommended; no C++ or CUDA Toolkit build tools)
echo [2] Full acceleration (includes Warp; requires C++ tools and CUDA Toolkit)
echo.
choice /C 12 /N /M "Select 1 or 2: "
if errorlevel 2 (
  set "BUILD_MODE=Full"
) else (
  set "BUILD_MODE=Standard"
)

echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SETUP_DIR%build_tensorrt.ps1" -RuntimeRoot "%RUNTIME_ROOT%" -Mode %BUILD_MODE%
set "RESULT=%ERRORLEVEL%"

echo.
if not "%RESULT%"=="0" echo TensorRT setup failed. Keep this window open for diagnostics.
if "%RESULT%"=="0" echo TensorRT setup completed. You can restart DigiBox now.
pause
exit /b %RESULT%
