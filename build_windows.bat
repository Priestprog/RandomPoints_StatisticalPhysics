@echo off
echo Building StatPhys for Windows...

REM Проверяем наличие виртуального окружения
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Активируем виртуальное окружение
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Устанавливаем/обновляем зависимости
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

REM Очищаем предыдущие сборки
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Собираем приложение
echo Building application...
pyinstaller statphys_windows.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build complete!
    echo StatPhys.exe is located at: dist\StatPhys.exe
    echo.
) else (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

pause