# GitHub Actions - Автоматическая Сборка

Этот проект настроен для автоматической кроссплатформенной сборки с использованием GitHub Actions.

## Как это работает

### Автоматические сборки

GitHub Actions автоматически создает исполняемые файлы для всех трех платформ при:

1. **Push в ветку `main`** - сборка запускается автоматически
2. **Pull Request в `main`** - сборка для проверки
3. **Создание тега версии** (например, `v1.0.0`) - создает релиз с файлами
4. **Ручной запуск** - через вкладку Actions в GitHub

### Платформы

Сборка производится для:

- **macOS** (Intel/Apple Silicon)
  - Выходной файл: `StatPhys-macos-x64.dmg` и `.zip`
  - Формат: `.app` bundle

- **Windows** (64-bit)
  - Выходной файл: `StatPhys-windows-x64.zip`
  - Формат: `.exe` исполняемый файл

- **Linux** (x64)
  - Выходной файл: `StatPhys-linux-x64.tar.gz`
  - Формат: standalone исполняемый файл

## Где найти собранные файлы

### Артефакты (для всех коммитов)

1. Перейдите в **Actions** на GitHub
2. Выберите нужный workflow run
3. Прокрутите вниз до секции **Artifacts**
4. Скачайте нужную платформу

Артефакты хранятся **30 дней**.

### Релизы (для версионированных тегов)

Если вы создаете тег версии (например, `v1.0.0`):

```bash
git tag v1.0.0
git push origin v1.0.0
```

Тогда:
1. GitHub Actions создаст **Release**
2. Все три версии будут прикреплены к релизу
3. Релиз будет доступен в разделе **Releases**

## Ручной запуск сборки

1. Перейдите в **Actions** → **Build Cross-Platform Apps**
2. Нажмите **Run workflow**
3. Выберите ветку
4. Нажмите зеленую кнопку **Run workflow**

## Структура файлов

```
.github/
  workflows/
    build.yml          # Конфигурация CI/CD

statphys.spec          # PyInstaller spec для macOS
statphys_linux.spec    # PyInstaller spec для Linux
statphys_windows.spec  # PyInstaller spec для Windows

requirements.txt       # Python зависимости
```

## Локальная сборка

Если хотите собрать локально:

### macOS
```bash
pip install -r requirements.txt
pyinstaller statphys.spec
```

### Linux
```bash
pip install -r requirements.txt
pyinstaller statphys_linux.spec
```

### Windows
```bash
pip install -r requirements.txt
pyinstaller statphys_windows.spec
```

Результат будет в папке `dist/`.

## Создание релиза

1. Обновите версию в коде (если нужно)
2. Создайте и отправьте тег:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
3. GitHub Actions автоматически:
   - Соберет все версии
   - Создаст GitHub Release
   - Прикрепит файлы к релизу

## Troubleshooting

### Сборка падает

Проверьте логи в Actions:
1. Откройте неудавшийся workflow run
2. Нажмите на красный шаг
3. Посмотрите детали ошибки

### Недостающие файлы

Если не хватает ресурсов (картинок, и т.д.):
1. Добавьте их в соответствующий `.spec` файл в секцию `datas`
2. Коммитьте и пушьте изменения

### Проблемы с зависимостями

Обновите `requirements.txt` или добавьте библиотеки в `hiddenimports` в `.spec` файлах.

## Статус сборки

Можно добавить бейдж в README.md:

```markdown
![Build Status](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Build%20Cross-Platform%20Apps/badge.svg)
```

Замените `YOUR_USERNAME` и `YOUR_REPO` на свои значения.
