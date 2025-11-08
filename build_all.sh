#!/bin/bash

# Скрипт для локальной сборки всех версий
# Автоматически определяет текущую платформу и собирает соответствующую версию

set -e  # Останавливаться при ошибках

echo "================================"
echo "StatPhys Multi-Platform Builder"
echo "================================"
echo ""

# Определяем платформу
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Обнаружена платформа: $OS"
echo ""

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8++"
    exit 1
fi

echo "✓ Python найден: $(python3 --version)"

# Проверяем наличие pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip не найден"
    exit 1
fi

echo "✓ pip найден"
echo ""

# Устанавливаем зависимости
echo "📦 Установка зависимостей..."
pip3 install -r requirements.txt

echo ""
echo "🔨 Начинаем сборку для $OS..."
echo ""

# Создаем директорию для артефактов
mkdir -p apps

# Собираем в зависимости от платформы
if [ "$OS" == "macos" ]; then
    echo "🍎 Сборка для macOS..."
    pyinstaller statphys.spec --noconfirm

    if [ -d "dist/StatPhys.app" ]; then
        echo "✓ Приложение собрано: dist/StatPhys.app"

        # Создаем ZIP (основной формат)
        echo "📦 Создание ZIP..."
        cd dist
        zip -r ../apps/StatPhys-macos-x64.zip StatPhys.app
        cd ..

        echo "✅ Сборка завершена!"
        echo "   ZIP: apps/StatPhys-macos-x64.zip"

        # Опционально: создаем DMG (может не хватить места на CI)
        if command -v hdiutil &> /dev/null; then
            echo ""
            echo "📦 Создание DMG (опционально)..."
            if hdiutil create -volname "StatPhys" -srcfolder dist/StatPhys.app -ov -format UDZO apps/StatPhys-macos-x64.dmg 2>/dev/null; then
                echo "   DMG: apps/StatPhys-macos-x64.dmg"
            else
                echo "⚠️  DMG не создан (возможно нехватка места)"
            fi
        fi
    else
        echo "❌ Ошибка сборки"
        exit 1
    fi

elif [ "$OS" == "linux" ]; then
    echo "🐧 Сборка для Linux..."
    pyinstaller statphys_linux.spec --noconfirm

    if [ -f "dist/StatPhys" ]; then
        echo "✓ Исполняемый файл собран: dist/StatPhys"

        # Создаем tar.gz
        echo "📦 Создание архива..."
        cd dist
        tar -czf ../apps/StatPhys-linux-x64.tar.gz StatPhys
        cd ..

        echo "✅ Сборка завершена!"
        echo "   Архив: apps/StatPhys-linux-x64.tar.gz"
    else
        echo "❌ Ошибка сборки"
        exit 1
    fi

elif [ "$OS" == "windows" ]; then
    echo "🪟 Сборка для Windows..."
    echo "⚠️  Для Windows используйте build_windows.bat"
    exit 1
else
    echo "❌ Неподдерживаемая платформа: $OS"
    exit 1
fi

echo ""
echo "🎉 Готово! Файлы находятся в папке apps/"
