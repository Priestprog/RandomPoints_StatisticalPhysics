@echo off
echo Building StatPhys for Windows...

REM Создаем виртуальное окружение если его нет
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Активируем виртуальное окружение
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Устанавливаем зависимости
echo Installing dependencies...
pip install PyQt6 matplotlib numpy scipy pyinstaller

REM Собираем приложение
echo Building executable...
pyinstaller statphys_windows.spec

echo Build complete! Check the dist/ folder for StatPhys.exe
pause