"""
Генератор визуализаций фракталов для ответов.
Оптимизированная версия с numpy для быстрой работы.

Использование:
    python generate_presets.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# ========== МЕТОДЫ ГЕНЕРАЦИИ ФРАКТАЛОВ (ОПТИМИЗИРОВАННЫЕ) ==========

def generate_sierpinski_fractal():
    """Генерация треугольника Серпинского (оптимизировано с numpy)"""
    print("Генерация треугольника Серпинского...", end=" ", flush=True)
    start = time.time()

    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    n = 500000  # еще больше точек для максимальной детализации

    # Оптимизация: векторизация вместо цикла
    points = np.zeros((n, 2))
    p = np.random.rand(2)

    for i in range(n):
        v = vertices[np.random.randint(0, 3)]
        p = (p + v) / 2
        points[i] = p

    print(f"✓ ({time.time() - start:.2f}s)")
    return points, vertices


def generate_koch_snowflake():
    """Генерация снежинки Коха (уменьшенные итерации)"""
    print("Генерация снежинки Коха...", end=" ", flush=True)
    start = time.time()

    def koch_curve(p1, p2, depth):
        if depth == 0:
            return [p1, p2]

        p1, p2 = np.array(p1), np.array(p2)
        v = (p2 - p1) / 3
        pA = p1 + v
        pB = p1 + 2*v
        angle = np.pi/3
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        pC = pA + rot @ v

        return (koch_curve(p1, pA, depth-1)[:-1] +
                koch_curve(pA, pC, depth-1)[:-1] +
                koch_curve(pC, pB, depth-1)[:-1] +
                koch_curve(pB, p2, depth-1))

    iterations = 7  # максимальная детализация

    # Начальный треугольник
    A = [0.0, 0.0]
    B = [1.0, 0.0]
    C = [0.5, np.sqrt(3)/2]

    curve = []
    curve += koch_curve(A, B, iterations)[:-1]
    curve += koch_curve(B, C, iterations)[:-1]
    curve += koch_curve(C, A, iterations)

    print(f"✓ ({time.time() - start:.2f}s)")
    return np.array(curve)


def generate_barnsley_fern():
    """Генерация папоротника Барнсли (оптимизировано)"""
    print("Генерация папоротника Барнсли...", end=" ", flush=True)
    start = time.time()

    n = 500000  # еще больше точек для максимальной детализации
    points = np.zeros((n, 2))
    x, y = 0.0, 0.0

    for i in range(n):
        r = np.random.rand()
        if r < 0.01:
            x, y = 0, 0.16*y
        elif r < 0.86:
            x, y = 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
        elif r < 0.93:
            x, y = 0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6
        else:
            x, y = -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
        points[i] = [x, y]

    # Нормализация в [0,1]^2
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())

    print(f"✓ ({time.time() - start:.2f}s)")
    return points


def generate_julia_set():
    """Генерация множества Жюлиа с escape-time раскраской"""
    print("Генерация множества Жюлиа...", end=" ", flush=True)
    start = time.time()

    c = -0.7 + 0.27015j
    max_iter = 512  # много итераций для красивой раскраски
    threshold = 2.0
    grid = 1200  # высокое разрешение

    lin = np.linspace(-1.5, 1.5, grid)
    X, Y = np.meshgrid(lin, lin)
    Z = X + 1j*Y
    C = np.full(Z.shape, c)

    # Создаем массив для хранения количества итераций
    iterations = np.zeros(Z.shape)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + C[mask]
        diverged = np.abs(Z) > threshold
        iterations[mask & diverged] = i
        mask[diverged] = False

    # Нормализация в [0,1]^2 для координат
    X_norm = (X - X.min()) / (X.max() - X.min())
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())

    print(f"✓ ({time.time() - start:.2f}s)")
    return X_norm, Y_norm, iterations, mask


def generate_pythagoras_tree():
    """Генерация дерева Пифагора (оптимизировано)"""
    print("Генерация дерева Пифагора...", end=" ", flush=True)
    start = time.time()

    def build_tree(x, y, size, angle, depth, squares):
        if depth == 0:
            return
        squares.append((x, y, size, angle))

        dx = size * np.cos(angle)
        dy = size * np.sin(angle)

        build_tree(x - dy, y + dx, size/np.sqrt(2), angle + np.pi/4, depth-1, squares)
        build_tree(x + dx - dy, y + dy + dx, size/np.sqrt(2), angle - np.pi/4, depth-1, squares)

    depth = 14  # максимальная глубина для детализации
    squares = []
    build_tree(0.5, 0.1, 0.1, 0, depth, squares)

    print(f"✓ ({time.time() - start:.2f}s, квадратов: {len(squares)})")
    return squares


# ========== ФУНКЦИИ ОТОБРАЖЕНИЯ ==========

def show_fractal(points, title, color='blue', vertices=None):
    """Универсальная функция отображения фрактала"""
    print(f"Отображение: {title}...", end=" ", flush=True)
    start = time.time()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)  # оптимальный размер для быстрой загрузки

    # Если это линия (Кох)
    if title == "Снежинка Коха":
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], s=0.1, color=color, alpha=0.95)  # минимальный размер точек

    # Для Серпинского добавляем вершины
    if vertices is not None:
        ax.scatter(vertices[:, 0], vertices[:, 1], s=150, color='red',
                  marker='o', edgecolors='black', linewidth=2, label='Вершины', zorder=10)
        triangle = np.vstack([vertices, vertices[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'r--', linewidth=2, alpha=0.6)
        ax.legend(fontsize=12)

    ax.set_title(title, fontsize=20, pad=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    print(f"✓ ({time.time() - start:.2f}s)")

    # Сохраняем как PNG в оптимальном качестве
    filename = f'{title.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')  # dpi=200 для баланса качества и скорости
    print(f"Сохранено: {filename}")

    plt.close(fig)


def show_julia_colormap(X, Y, iterations, mask, title="Множество Жюлиа"):
    """Специальная функция для отображения Жюлиа с температурной шкалой"""
    print(f"Отображение: {title}...", end=" ", flush=True)
    start = time.time()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Создаем красивую раскраску
    # Устанавливаем цвет для точек внутри множества (не ушедших на бесконечность)
    colored_iterations = np.copy(iterations)
    colored_iterations[mask] = np.nan  # NaN для точек внутри множества

    # Используем температурную цветовую схему
    im = ax.imshow(colored_iterations, cmap='hot', origin='lower', extent=[0, 1, 0, 1],
                   interpolation='bilinear')

    ax.set_title(title, fontsize=20, pad=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    print(f"✓ ({time.time() - start:.2f}s)")

    # Сохраняем как PNG
    filename = f'{title.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Сохранено: {filename}")

    plt.close(fig)


def show_pythagoras_tree(squares, title="Дерево Пифагора"):
    """Специальная функция для отображения дерева Пифагора квадратами"""
    print(f"Отображение: {title}...", end=" ", flush=True)
    start = time.time()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Рисуем квадраты
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches

    for i, (cx, cy, size, angle) in enumerate(squares):
        # Создаем квадрат с центром в (cx, cy)
        # Преобразуем в координаты левого нижнего угла
        half = size / 2
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])

        # Поворот
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = corners @ rot.T
        translated = rotated + np.array([cx, cy])

        # Рисуем квадрат как полигон
        polygon = mpatches.Polygon(translated, closed=True,
                                  edgecolor='black', facecolor='green',
                                  linewidth=0.5, alpha=0.7)
        ax.add_patch(polygon)

    # Нормализация координат для отображения
    all_x = [cx for cx, cy, s, a in squares]
    all_y = [cy for cx, cy, s, a in squares]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Масштабируем в [0, 1]
    range_x = max_x - min_x
    range_y = max_y - min_y
    ax.set_xlim(min_x - 0.05*range_x, max_x + 0.05*range_x)
    ax.set_ylim(min_y - 0.05*range_y, max_y + 0.05*range_y)

    ax.set_title(title, fontsize=20, pad=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    print(f"✓ ({time.time() - start:.2f}s)")

    # Сохраняем как PNG
    filename = f'{title.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Сохранено: {filename}")

    plt.close(fig)


def main():
    """Главная функция"""
    print("="*70)
    print("Генератор фрактальных визуализаций")
    print("="*70)
    print("\nБудет сгенерировано 5 фракталов и сохранено в PNG:")
    print("1. Треугольник Серпинского")
    print("2. Снежинка Коха")
    print("3. Папоротник Барнсли")
    print("4. Множество Жюлиа")
    print("5. Дерево Пифагора")
    print("="*70 + "\n")

    # 1. Серпинский
    points, vertices = generate_sierpinski_fractal()
    show_fractal(points, "Треугольник Серпинского", 'blue', vertices)

    # 2. Кох
    points = generate_koch_snowflake()
    show_fractal(points, "Снежинка Коха", 'blue')

    # 3. Барнсли
    points = generate_barnsley_fern()
    show_fractal(points, "Папоротник Барнсли", 'darkgreen')

    # 4. Жюлиа (с температурной шкалой)
    X, Y, iterations, mask = generate_julia_set()
    show_julia_colormap(X, Y, iterations, mask, "Множество Жюлиа")

    # 5. Пифагор (прорисованные квадраты)
    squares = generate_pythagoras_tree()
    show_pythagoras_tree(squares, "Дерево Пифагора")

    print("\n" + "="*70)
    print("✓ Генерация завершена! Все 5 фракталов сохранены в PNG.")
    print("="*70)


if __name__ == "__main__":
    main()
