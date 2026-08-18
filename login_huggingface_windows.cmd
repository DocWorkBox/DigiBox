@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%~dp0scripts\login_huggingface_windows.ps1"
set "result=%ERRORLEVEL%"

echo.
if "%result%"=="0" (
  echo Login succeeded. This window can now be closed.
) else (
  echo Login did not complete. Exit code: %result%
)
echo.
pause
exit /b %result%
