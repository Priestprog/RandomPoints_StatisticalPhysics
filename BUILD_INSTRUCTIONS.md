# Инструкция по сборке исполняемых файлов

## Для Windows

### Автоматическая сборка:
1. Скопируйте весь проект на Windows-машину
2. Убедитесь что установлен Python 3.8+
3. Запустите `build_windows.bat` двойным кликом
4. Готовый `StatPhys.exe` будет в папке `dist/`

### Ручная сборка:
```cmd
# Создаем виртуальное окружение
python -m venv venv
venv\Scripts\activate.bat

# Устанавливаем зависимости
pip install -r requirements.txt

# Собираем приложение
pyinstaller statphys_windows.spec
```

## Для macOS

### Автоматическая сборка:
```bash
# Активируем виртуальное окружение
source .venv/bin/activate

# Собираем приложение
pyinstaller statphys.spec
```

Готовое приложение `StatPhys.app` будет в папке `dist/`

## Для Linux

### Ручная сборка:
```bash
# Создаем виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Собираем приложение (используем Windows spec без bundle)
pyinstaller --onefile --windowed tests/main.py --name StatPhys
```

## Требования

- Python 3.8 или новее
- Зависимости из `requirements.txt`
- Для Windows: Visual C++ Redistributable (обычно уже установлен)

## Размеры исполняемых файлов

- Windows: ~60-80 MB
- macOS: ~60-80 MB
- Linux: ~60-80 MB

Размер большой из-за включения всех библиотек (PyQt6, matplotlib, numpy, scipy).