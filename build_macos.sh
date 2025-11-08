#!/bin/bash

echo "Building StatPhys for macOS..."

# Проверяем наличие виртуального окружения
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please create it first with: python3 -m venv .venv"
    exit 1
fi

# Активируем виртуальное окружение
echo "Activating virtual environment..."
source .venv/bin/activate

# Устанавливаем/обновляем зависимости
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Очищаем предыдущие сборки
echo "Cleaning previous builds..."
rm -rf build dist

# Собираем приложение
echo "Building application..."
pyinstaller statphys.spec

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build complete!"
    echo "StatPhys.app is located at: dist/StatPhys.app"
    echo ""
    echo "To run: open dist/StatPhys.app"
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi
