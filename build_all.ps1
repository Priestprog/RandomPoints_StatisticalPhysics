# PowerShell скрипт для сборки на Windows

Write-Host "================================" -ForegroundColor Cyan
Write-Host "StatPhys Multi-Platform Builder" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Обнаружена платформа: Windows" -ForegroundColor Green
Write-Host ""

# Проверяем наличие Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python найден: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python не найден. Установите Python 3.8+" -ForegroundColor Red
    exit 1
}

# Проверяем наличие pip
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✓ pip найден" -ForegroundColor Green
} catch {
    Write-Host "❌ pip не найден" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Устанавливаем зависимости
Write-Host "📦 Установка зависимостей..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Ошибка установки зависимостей" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🔨 Начинаем сборку для Windows..." -ForegroundColor Yellow
Write-Host ""

# Создаем директорию для артефактов
New-Item -ItemType Directory -Force -Path apps | Out-Null

# Сборка
Write-Host "🪟 Сборка для Windows..." -ForegroundColor Cyan
pyinstaller statphys_windows.spec --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Ошибка сборки" -ForegroundColor Red
    exit 1
}

# Проверяем результат
if (Test-Path "dist\StatPhys.exe") {
    Write-Host "✓ Исполняемый файл собран: dist\StatPhys.exe" -ForegroundColor Green

    # Создаем ZIP
    Write-Host "📦 Создание архива..." -ForegroundColor Yellow
    Compress-Archive -Path dist\StatPhys.exe -DestinationPath apps\Random_points-windows-x64.zip -Force

    Write-Host "✅ Сборка завершена!" -ForegroundColor Green
    Write-Host "   ZIP: apps\Random_points-windows-x64.zip" -ForegroundColor Cyan
} else {
    Write-Host "❌ Ошибка: исполняемый файл не найден" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Готово! Файлы находятся в папке apps/" -ForegroundColor Green
