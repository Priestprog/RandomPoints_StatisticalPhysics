import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class UniformStrategy:
    def generate(self, n):
        return np.random.rand(n, 2)


class RandomWalkStrategy:
    def generate(self, n):
        angles = np.random.rand(n) * 2*np.pi
        steps = np.ones(n) * 0.01
        dx = np.cos(angles) * steps
        dy = np.sin(angles) * steps
        x = np.cumsum(dx)
        y = np.cumsum(dy)
        return np.column_stack([x, y])


class SierpinskiStrategy:
    def __init__(self):
        self.vertices = np.array([[0,0], [1,0], [0.5, np.sqrt(3)/2]])

    def generate(self, n):
        p = np.random.rand(2)
        points = []
        for _ in range(n):
            v = self.vertices[np.random.randint(0, 3)]
            p = (p + v) / 2
            points.append(p.copy())
        return np.array(points)

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()
        # генерируем много точек для лучшей визуализации
        points = self.generate(10000)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, color='blue', alpha=0.8)

        # показываем исходные вершины треугольника
        ax.scatter(self.vertices[:, 0], self.vertices[:, 1], s=100, color='red',
                  marker='o', edgecolors='black', linewidth=2, label='Вершины')

        # соединяем вершины линиями
        triangle = np.vstack([self.vertices, self.vertices[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'r--', linewidth=2, alpha=0.7)

        ax.set_title('Треугольник Серпинского', fontsize=12, pad=10)
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class BoltzmannStrategy:
    """Распределение Больцмана в поле тяжести"""
    def __init__(self, temperature=0.15):
        self.temperature = temperature  # температура (влияет на распределение по высоте)

    def generate(self, n):
        points = []

        # Генерируем точки с rejection sampling
        # Плотность вероятности: ρ(y) ∝ exp(-y/T)
        batch_size = n * 5
        max_batches = 20

        for _ in range(max_batches):
            if len(points) >= n:
                break

            # Генерируем кандидатов
            candidates = np.random.rand(batch_size, 2)

            # Вероятность принятия зависит от высоты (y-координата)
            # Внизу (y=0) высокая вероятность, вверху (y=1) низкая
            acceptance_probs = np.exp(-candidates[:, 1] / self.temperature)

            # Нормализуем к [0, 1]
            acceptance_probs = acceptance_probs / np.exp(0)  # exp(0) = 1

            # Принимаем точки случайно согласно вероятности
            random_vals = np.random.rand(batch_size)
            accepted_mask = random_vals < acceptance_probs
            accepted_points = candidates[accepted_mask]

            points.extend(accepted_points)

        points = np.array(points[:n])

        # Если не набрали, добавляем случайные
        while len(points) < n:
            candidate = np.random.rand(2)
            prob = np.exp(-candidate[1] / self.temperature)
            if np.random.rand() < prob:
                points = np.vstack([points, candidate])

        np.random.shuffle(points)
        return points

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()

        # Генерируем много точек для визуализации
        n = 3000
        all_points = self.generate(n)

        # Раскрашиваем точки по высоте (градиент)
        colors = all_points[:, 1]  # y-координата
        scatter = ax.scatter(all_points[:, 0], all_points[:, 1],
                           c=colors, cmap='coolwarm', s=point_size, alpha=0.6)

        # Добавляем шкалу плотности
        from matplotlib.patches import Rectangle
        # Рисуем фон - градиент плотности
        y_values = np.linspace(0, 1, 100)
        densities = np.exp(-y_values / self.temperature)

        for i, y in enumerate(y_values[:-1]):
            alpha = densities[i] / densities.max() * 0.2
            rect = Rectangle((0, y), 1, y_values[1] - y_values[0],
                           facecolor='blue', alpha=alpha, zorder=0)
            ax.add_patch(rect)

        # Стрелка вниз (направление гравитации)
        ax.annotate('g', xy=(0.05, 0.1), xytext=(0.05, 0.3),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                   fontsize=20, fontweight='bold', ha='center')

        ax.set_title(f'Гравитация (T={self.temperature:.2f})',
                    fontsize=12, pad=10)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class CrystallizationStrategy:
    """Кристаллическая решётка с тепловыми колебаниями"""
    def __init__(self, lattice_type='hexagonal', thermal_noise=0.003):
        self.lattice_type = lattice_type  # 'hexagonal' или 'square'
        self.thermal_noise = thermal_noise  # амплитуда тепловых колебаний

    def generate(self, n):
        if self.lattice_type == 'hexagonal':
            return self._generate_hexagonal(n)
        else:
            return self._generate_square(n)

    def _generate_hexagonal(self, n):
        """Генерирует гексагональную решётку (структура льда) - векторизовано на NumPy"""
        # Размер решётки зависит от количества точек
        # Для лёгкого уровня (много точек) делаем более крупную решётку
        if n >= 1000:  # Лёгкий - крупная решётка (меньше узлов)
            grid_size = int(np.sqrt(n * 0.25)) + 1
        else:  # Средний и сложный - более плотная решётка
            grid_size = int(np.sqrt(n * 1.2)) + 1

        # Параметры гексагональной решётки
        a_x = 1.0 / grid_size  # шаг по X
        a_y = np.sqrt(3) / 2 / grid_size  # шаг по Y

        # Генерируем больше точек, чтобы гарантированно покрыть всю область [0,1]
        # Создаём сетку индексов векторизовано (с большим запасом)
        i_indices = np.arange(0, int(grid_size * 1.3))
        j_indices = np.arange(0, int(grid_size * 2))  # еще больше по Y
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')

        # Вычисляем базовые координаты
        x = i_grid * a_x
        y = j_grid * a_y

        # Смещение для гексагональной структуры (чётные/нечётные ряды)
        x = x + (j_grid % 2) * (a_x / 2)

        # Преобразуем в плоский массив точек ДО добавления шума
        points = np.stack([x.ravel(), y.ravel()], axis=1)

        # Растягиваем на всё полотно [0,1]×[0,1]
        # Нормализуем по X и Y независимо для максимального заполнения
        if len(points) > 0:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()

            # Растягиваем и по X, и по Y на весь диапазон [0,1]
            if x_max > x_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            if y_max > y_min:
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Обрезаем до нужного количества ПЕРЕД добавлением шума
        if len(points) > n:
            # Для малого количества точек используем случайную выборку
            # Это лучше показывает структуру решётки при меньшей плотности
            indices = np.random.choice(len(points), n, replace=False)
            points = points[indices]

        # Добавляем тепловые колебания
        points[:, 0] = points[:, 0] + np.random.randn(len(points)) * self.thermal_noise
        points[:, 1] = points[:, 1] + np.random.randn(len(points)) * self.thermal_noise

        # Возвращаем точки в границы (не удаляем, а обрезаем)
        points = np.clip(points, 0.0, 1.0)

        # Сохраняем точки для визуализации ответа
        self.last_generated_points = points.copy()

        # ВАЖНО: перемешиваем точки для случайного порядка появления
        np.random.shuffle(points)

        return points

    def _generate_square(self, n):
        """Генерирует квадратную решётку - векторизовано на NumPy"""
        # Размер решётки зависит от количества точек
        # Для лёгкого уровня (много точек) делаем более крупную решётку
        if n >= 1000:  # Лёгкий - крупная решётка (меньше узлов)
            grid_size = int(np.sqrt(n * 0.5)) + 1
        else:  # Средний и сложный - более плотная решётка (менее очевидная)
            grid_size = int(np.sqrt(n * 2)) + 1
        a = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0  # шаг решётки для полного покрытия [0,1]

        # Создаём сетку индексов векторизовано
        i_indices = np.arange(grid_size)
        j_indices = np.arange(grid_size)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')

        # Вычисляем координаты векторизовано
        x = i_grid * a
        y = j_grid * a

        # Преобразуем в плоский массив точек
        points = np.stack([x.ravel(), y.ravel()], axis=1)

        # Обрезаем до нужного количества ПЕРЕД добавлением шума
        if len(points) > n:
            # Для малого количества точек используем случайную выборку
            # Это лучше показывает структуру решётки при меньшей плотности
            indices = np.random.choice(len(points), n, replace=False)
            points = points[indices]

        # Добавляем тепловые колебания
        points[:, 0] = points[:, 0] + np.random.randn(len(points)) * self.thermal_noise
        points[:, 1] = points[:, 1] + np.random.randn(len(points)) * self.thermal_noise

        # Возвращаем точки в границы (не удаляем, а обрезаем)
        points = np.clip(points, 0.0, 1.0)

        # Сохраняем точки для визуализации ответа
        self.last_generated_points = points.copy()

        # Перемешиваем точки для случайного порядка появления
        np.random.shuffle(points)

        return points

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()

        # Используем ТЕ ЖЕ точки, что были сгенерированы в игре
        if not hasattr(self, 'last_generated_points') or self.last_generated_points is None:
            # Если точки не были сохранены, генерируем новые
            all_points = self.generate(800)
        else:
            all_points = self.last_generated_points

        # Рисуем связи между ближайшими соседями
        from scipy.spatial import Delaunay

        # Триангуляция Делоне для нахождения соседей
        tri = Delaunay(all_points)

        # Рисуем рёбра (только короткие - между соседями)
        # Вычисляем оптимальную длину связи на основе плотности решётки
        n = len(all_points)
        grid_size = int(np.sqrt(n * 1.2)) + 1
        max_edge_length = 1.5 / (grid_size - 1) if grid_size > 1 else 1.5  # чуть больше шага решётки

        for simplex in tri.simplices:
            for i in range(3):
                p1 = all_points[simplex[i]]
                p2 = all_points[simplex[(i + 1) % 3]]
                dist = np.linalg.norm(p1 - p2)

                if dist < max_edge_length:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           'cyan', alpha=0.4, linewidth=1.2, zorder=1)

        # Рисуем узлы решётки крупнее
        ax.scatter(all_points[:, 0], all_points[:, 1],
                  s=point_size * 3, color='darkblue', alpha=0.9,
                  edgecolors='white', linewidths=0.5, zorder=2)

        # Заголовок строго по центру
        lattice_name = 'Гексагональная' if self.lattice_type == 'hexagonal' else 'Квадратная'
        ax.set_title(f'{lattice_name} решётка (ΔT={self.thermal_noise:.3f})',
                    fontsize=12, pad=10, loc='center', ha='center')

        # Настройка границ и вида
        ax.set_aspect('equal')
        ax.set_xlim(-0.03, 1.03)  # чуть больше отступ для полного покрытия
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class RepulsionStrategy:
    """Стратегия отталкивания: точки избегают центров отталкивания"""
    def __init__(self, k=5):
        self.k = k
        self.centers = None

    def generate(self, n):
        # Генерируем центры отталкивания
        self.centers = self._generate_well_separated_centers()

        exclusion_radius = 0.08  # радиус исключённой зоны вокруг центров
        points = []

        # Векторизованный rejection sampling - генерируем батчами
        batch_size = n * 3  # генерируем больше точек за раз
        max_batches = 50

        for _ in range(max_batches):
            if len(points) >= n:
                break

            # Генерируем батч кандидатов
            candidates = np.random.rand(batch_size, 2)

            # Вычисляем расстояния до всех центров для всех кандидатов (векторизовано)
            # Shape: (batch_size, k)
            distances = np.sqrt(((candidates[:, np.newaxis, :] - self.centers[np.newaxis, :, :]) ** 2).sum(axis=2))

            # Минимальное расстояние до ближайшего центра для каждого кандидата
            min_distances = distances.min(axis=1)

            # Фильтруем точки в зоне исключения
            mask = min_distances >= exclusion_radius

            # Вычисляем вероятности принятия (векторизовано)
            acceptance_probs = np.minimum(1.0, ((min_distances - exclusion_radius) / 0.9) ** 2)

            # Генерируем случайные числа для всех кандидатов сразу
            random_vals = np.random.rand(batch_size)

            # Принимаем точки, которые прошли оба фильтра
            accepted_mask = mask & (random_vals < acceptance_probs)
            accepted_points = candidates[accepted_mask]

            points.extend(accepted_points)

        # Обрезаем до нужного количества
        points = np.array(points[:n])

        # Если не набрали достаточно точек, добавляем случайные вдали от центров
        while len(points) < n:
            candidate = np.random.rand(2)
            distances = np.sqrt(((candidate - self.centers) ** 2).sum(axis=1))
            if distances.min() >= exclusion_radius:
                points = np.vstack([points, candidate])

        # Перемешиваем точки, чтобы они появлялись в случайном порядке
        np.random.shuffle(points)
        return points

    def _generate_well_separated_centers(self):
        """Генерирует центры отталкивания с минимальным расстоянием между ними"""
        min_distance = 0.25
        max_attempts = 1000

        centers = []
        centers.append(np.random.rand(2))

        for _ in range(self.k - 1):
            best_candidate = None
            best_min_dist = 0

            for _ in range(max_attempts):
                candidate = np.random.rand(2)
                min_dist = min(np.linalg.norm(candidate - c) for c in centers)

                if min_dist >= min_distance:
                    best_candidate = candidate
                    break

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            centers.append(best_candidate)

        return np.array(centers)

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        # Генерируем меньше точек для более явного эффекта отталкивания
        n = 1000
        all_points = self.generate(n)

        # Раскрашиваем точки в зависимости от ближайшего центра отталкивания
        for i, center in enumerate(self.centers):
            color = colors[i % len(colors)]
            # Точки показываем все вместе одним цветом
            if i == 0:
                ax.scatter(all_points[:, 0], all_points[:, 1], s=point_size,
                          color='blue', alpha=0.6)

            # Показываем центры отталкивания с кругами исключённой зоны
            circle = plt.Circle(center, 0.08, color=color, fill=True,
                               linewidth=2, alpha=0.15)
            ax.add_patch(circle)
            # Внешний контур
            circle_outline = plt.Circle(center, 0.08, color=color, fill=False,
                               linewidth=2, linestyle='--', alpha=0.8)
            ax.add_patch(circle_outline)
            ax.scatter(center[0], center[1], s=150, color=color,
                      marker='x', linewidths=4)

        ax.set_aspect('equal')
        ax.set_xlim(-0.02, 1.02)  # небольшой отступ, чтобы не обрезать края
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class ClustersStrategy:
    def __init__(self, k=5):
        self.k = k
        self.centers = None

    def generate(self, n):
        # Генерируем центры с минимальным расстоянием между ними
        self.centers = self._generate_well_separated_centers()
        points = []
        for c in self.centers:
            cluster = c + 0.08*np.random.randn(n//self.k, 2)
            points.append(cluster)
        all_points = np.vstack(points)
        # Перемешиваем точки, чтобы они появлялись в случайном порядке
        np.random.shuffle(all_points)
        return all_points

    def _generate_well_separated_centers(self):
        """Генерирует центры кластеров с минимальным расстоянием между ними"""
        min_distance = 0.25  # минимальное расстояние между центрами
        max_attempts = 1000

        centers = []

        # Первый центр генерируем случайно
        centers.append(np.random.rand(2))

        # Остальные центры генерируем с учетом минимального расстояния
        for _ in range(self.k - 1):
            best_candidate = None
            best_min_dist = 0

            # Пробуем несколько кандидатов и выбираем лучший
            for _ in range(max_attempts):
                candidate = np.random.rand(2)

                # Вычисляем минимальное расстояние до существующих центров
                min_dist = min(np.linalg.norm(candidate - c) for c in centers)

                # Если найден центр с достаточным расстоянием, используем его
                if min_dist >= min_distance:
                    best_candidate = candidate
                    break

                # Запоминаем лучшего кандидата
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            centers.append(best_candidate)

        return np.array(centers)

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        # генерируем больше точек для лучшей визуализации
        n = 3000
        for i, center in enumerate(self.centers):
            cluster = center + 0.08*np.random.randn(n//self.k, 2)
            color = colors[i % len(colors)]
            ax.scatter(cluster[:, 0], cluster[:, 1], s=point_size, color=color, alpha=0.7)
            # показываем центры притяжения
            ax.scatter(center[0], center[1], s=50, color='black', marker='x')

        ax.set_aspect('equal')
        ax.set_xlim(-0.02, 1.02)  # небольшой отступ, чтобы не обрезать края
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


# === Стратегии из статистической физики ===

class IsingStrategy:
    """
    Упрощённая модель Изинга: генерируем решётку спинов +1/-1,
    выбираем точки со спином +1.
    """
    def __init__(self, grid_size=100, T=2.5, J=1.0, h=0.0, steps=None):
        self.grid_size = grid_size
        self.T = T  # температура
        self.J = J  # сила взаимодействия
        self.h = h  # внешнее магнитное поле

        # Адаптивный выбор количества шагов на основе температуры
        if steps is None:
            if T < 1.0:
                # Низкая температура - много шагов для достижения равновесия
                self.steps = grid_size * grid_size * 100
            elif T < 2.0:
                # Средняя температура - среднее количество шагов
                self.steps = grid_size * grid_size * 20
            else:
                # Высокая температура - мало шагов
                self.steps = grid_size * grid_size * 10
        else:
            self.steps = steps

        self.spins = None

    def generate(self, n, sample_fraction=1.0):
        """
        Генерирует точки по модели Изинга

        Args:
            n: максимальное количество точек (не используется напрямую, см. sample_fraction)
            sample_fraction: доля точек для отображения (0.0-1.0)
                1.0 = 100% точек (легкий уровень)
                0.7 = 70% точек (средний уровень)
                0.4 = 40% точек (сложный уровень)
        """
        N = self.grid_size

        # Умная инициализация: для низких температур начинаем с упорядоченного состояния
        if self.T < 2.0:
            # Упорядоченная конфигурация (все спины +1)
            self.spins = np.ones((N, N), dtype=int)
        else:
            # Случайная конфигурация
            self.spins = np.random.choice([-1, 1], size=(N, N))

        # Метод Метрополиса
        for _ in range(self.steps):
            i, j = np.random.randint(0, N, 2)
            s = self.spins[i, j]
            nb = self.spins[(i+1)%N,j] + self.spins[(i-1)%N,j] + self.spins[i,(j+1)%N] + self.spins[i,(j-1)%N]

            # Учет внешнего магнитного поля
            dE = 2 * self.J * s * (nb + self.h)

            if dE < 0 or np.random.rand() < np.exp(-dE/self.T):
                self.spins[i,j] *= -1

        # Берём только "спины вверх"
        coords = np.argwhere(self.spins == 1)
        # np.argwhere возвращает [row, col] = [Y, X]
        # Меняем порядок на [col, row] = [X, Y] для scatter
        # нормируем в [0,1]^2
        points = coords[:, [1, 0]] / N

        # ВАЖНО: Перемешиваем точки для случайного порядка появления
        np.random.shuffle(points)

        # Сохраняем ВСЕ точки для использования в get_correct_visualization()
        self.last_generated_points = points.copy()

        # Возвращаем sample_fraction от всех точек
        n_to_return = int(len(points) * sample_fraction)
        if n_to_return > 0:
            return points[:n_to_return]
        return points

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()

        if self.spins is not None:
            N = self.grid_size

            # 1. Показываем тепловую карту решетки спинов (фон)
            # Красный = +1 (спины вверх), Синий = -1 (спины вниз)
            im = ax.imshow(self.spins, cmap='RdBu_r', origin='lower',
                          extent=[0, 1, 0, 1], alpha=0.6, vmin=-1, vmax=1)

            # 2. Находим все точки со спинами +1 и -1
            # np.argwhere возвращает [row, col] = [Y, X]
            # Меняем порядок на [col, row] = [X, Y] для scatter
            coords_plus = np.argwhere(self.spins == 1)[:, [1, 0]] / N  # Спины +1
            coords_minus = np.argwhere(self.spins == -1)[:, [1, 0]] / N  # Спины -1

            # 3. Рисуем точки двух типов
            if len(coords_plus) > 0:
                ax.scatter(coords_plus[:, 0], coords_plus[:, 1],
                          s=point_size*2, color='darkred', alpha=0.8,
                          label=f'Спины ↑ (+1): {len(coords_plus)}',
                          edgecolors='none')

            if len(coords_minus) > 0:
                ax.scatter(coords_minus[:, 0], coords_minus[:, 1],
                          s=point_size*2, color='darkblue', alpha=0.8,
                          label=f'Спины ↓ (-1): {len(coords_minus)}',
                          edgecolors='none')


            # 5. Статистика
            total = len(coords_plus) + len(coords_minus)
            magnetization = (len(coords_plus) - len(coords_minus)) / total if total > 0 else 0

            # 6. Заголовок с подробной информацией
            title = f'Модель Изинга: T={self.T:.2f}, J={self.J:.1f}'
            ax.set_title(title, fontsize=11, pad=10)

            # 7. Легенда
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8,
                     markerscale=1.5, handlelength=1)

        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class CorrelatedFieldStrategy:
    """
    Генерация коррелированного гауссовского поля:
    создаём белый шум и фильтруем гауссовым ядром (корреляционная длина).
    """
    def __init__(self, grid_size=200, sigma=5.0):
        self.grid_size = grid_size
        self.sigma = sigma  # радиус корреляции
        self.corr_field = None

    def generate(self, n):
        field = np.random.randn(self.grid_size, self.grid_size)
        self.corr_field = gaussian_filter(field, sigma=self.sigma)
        # нормируем
        self.corr_field = (self.corr_field - self.corr_field.min()) / (self.corr_field.max()-self.corr_field.min())
        # выбираем n случайных точек с вероятностью ∝ значению поля
        flat = self.corr_field.ravel()
        flat /= flat.sum()
        idx = np.random.choice(len(flat), size=n, p=flat)
        y, x = np.unravel_index(idx, self.corr_field.shape)
        return np.column_stack([x/self.grid_size, y/self.grid_size])

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()
        if self.corr_field is not None:
            # показываем поле как тепловую карту
            im = ax.imshow(self.corr_field, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
            ax.set_title(f'Коррелированное поле (σ={self.sigma})', fontsize=12, pad=10)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class RandomWalkRepulsionStrategy:
    """
    Случайное блуждание с отталкиванием от стенок.
    Частица делает большие случайные шаги и отражается от границ графика.
    """
    def __init__(self, step_size=0.12, repulsion_strength=0.2):
        self.step_size = step_size  # размер шага (большой)
        self.repulsion_strength = repulsion_strength  # сила отталкивания от стенок
        self.trajectory = None

    def generate(self, n):
        points = [np.array([0.5, 0.5])]

        for _ in range(n-1):
            current = points[-1]

            # Случайное направление (случайное блуждание)
            angle = np.random.rand() * 2 * np.pi
            random_step = self.step_size * np.array([np.cos(angle), np.sin(angle)])

            # Пробная новая точка
            new_point = current + random_step

            # Отталкивание от краёв графика
            # Если точка выходит за границы, отражаем её
            if new_point[0] < 0:
                new_point[0] = -new_point[0] * self.repulsion_strength
            elif new_point[0] > 1:
                new_point[0] = 1 - (new_point[0] - 1) * self.repulsion_strength

            if new_point[1] < 0:
                new_point[1] = -new_point[1] * self.repulsion_strength
            elif new_point[1] > 1:
                new_point[1] = 1 - (new_point[1] - 1) * self.repulsion_strength

            # Гарантируем нахождение в границах
            new_point = np.clip(new_point, 0.0, 1.0)

            points.append(new_point)

        self.trajectory = np.array(points)
        return self.trajectory

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()
        if self.trajectory is not None:
            # Показываем границы графика более явно
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'r-', linewidth=2, alpha=0.5, label='Границы')

            # Показываем траекторию как линию (только первые 10 точек)
            ax.plot(self.trajectory[:20, 0], self.trajectory[:20, 1], 'b-', alpha=0.5, linewidth=1.5)

            # Точки с градиентом цвета по времени (только первые 10 точек)
            colors = np.linspace(0, 1, 20)
            ax.scatter(self.trajectory[:20, 0], self.trajectory[:20, 1], c=colors, cmap='plasma', s=point_size)

            # Начальная точка
            ax.scatter(self.trajectory[0, 0], self.trajectory[0, 1], c='green', s=50, marker='o', label='Старт')

            # Конечная точка
            ax.scatter(self.trajectory[19, 0], self.trajectory[19, 1], c='red', s=50, marker='s', label='Финиш')

            ax.set_title(f'Случайное блуждание (шаг={self.step_size:.2f})',
                        fontsize=12, pad=10)
            ax.legend(loc='upper right', fontsize=8)

        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class PythagorasTreeStrategy:
    """
    Дерево Пифагора: генерируем точки внутри квадратов.
    """
    def __init__(self, depth=7, jitter=True):
        self.depth = depth
        self.jitter = jitter  # добавляем случайные углы/масштабы

    def _build(self, x, y, size, angle, depth, squares):
        if depth == 0:
            return
        # сохраняем квадрат (центр, размер, угол)
        squares.append((x, y, size, angle))

        # смещение вдоль текущего направления
        dx = size * np.cos(angle)
        dy = size * np.sin(angle)

        # немного случайности
        delta = 0
        if self.jitter:
            delta = np.random.uniform(-0.05, 0.05)

        # рекурсивно строим два квадрата
        self._build(x - dy, y + dx, size/np.sqrt(2), angle + np.pi/4 + delta, depth-1, squares)
        self._build(x + dx - dy, y + dy + dx, size/np.sqrt(2), angle - np.pi/4 + delta, depth-1, squares)

    def generate(self, n):
        squares = []
        self._build(0.5, 0.1, 0.1, 0, self.depth, squares)

        points = []
        # семплируем точки внутри квадратов
        for (cx, cy, size, angle) in squares:
            for _ in range(n // len(squares) + 1):
                # точка в локальной системе ([-0.5,0.5]^2)
                px, py = np.random.rand() - 0.5, np.random.rand() - 0.5
                px *= size
                py *= size
                # поворот
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
                x, y = rot @ np.array([px, py]) + np.array([cx, cy])
                points.append([x, y])

        points = np.array(points)
        # нормализация в [0,1]^2
        points[:,0] = (points[:,0] - points[:,0].min())/(points[:,0].max()-points[:,0].min())
        points[:,1] = (points[:,1] - points[:,1].min())/(points[:,1].max()-points[:,1].min())

        # если точек слишком много
        if len(points) > n:
            idx = np.random.choice(len(points), n, replace=False)
            return points[idx]
        return points



class KochSnowflakeStrategy:
    """
    Снежинка Коха: строим линии и дискретизуем их в точки.
    """
    def __init__(self, iterations=4):
        self.iterations = iterations
        self.full_curve = None

    def _koch_curve(self, p1, p2, depth):
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
        return (self._koch_curve(p1, pA, depth-1)[:-1] +
                self._koch_curve(pA, pC, depth-1)[:-1] +
                self._koch_curve(pC, pB, depth-1)[:-1] +
                self._koch_curve(pB, p2, depth-1))

    def generate(self, n):
        # начальный треугольник
        A = [0.0, 0.0]
        B = [1.0, 0.0]
        C = [0.5, np.sqrt(3)/2]
        curve = []
        curve += self._koch_curve(A, B, self.iterations)[:-1]
        curve += self._koch_curve(B, C, self.iterations)[:-1]
        curve += self._koch_curve(C, A, self.iterations)
        self.full_curve = np.array(curve)
        if len(self.full_curve) > n:
            idx = np.random.choice(len(self.full_curve), n, replace=False)
            return self.full_curve[idx]
        return self.full_curve

    def get_correct_visualization(self, ax, point_size=2):
        ax.clear()
        if self.full_curve is not None:
            # показываем непрерывную линию
            ax.plot(self.full_curve[:, 0], self.full_curve[:, 1], 'b-', linewidth=2)
            ax.set_title(f'Снежинка Коха (итераций: {self.iterations})', fontsize=12, pad=10)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


class BarnsleyFernStrategy:
    """
    Классический папоротник Барнсли.
    """
    def generate(self, n):
        x, y = 0, 0
        points = []
        for _ in range(n):
            r = np.random.rand()
            if r < 0.01:
                x, y = 0, 0.16*y
            elif r < 0.86:
                x, y = 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
            elif r < 0.93:
                x, y = 0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6
            else:
                x, y = -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
            points.append([x, y])
        points = np.array(points)
        # нормализация в [0,1]^2
        points[:,0] = (points[:,0] - points[:,0].min()) / (points[:,0].max()-points[:,0].min())
        points[:,1] = (points[:,1] - points[:,1].min()) / (points[:,1].max()-points[:,1].min())
        # Перемешиваем точки для случайного порядка появления
        np.random.shuffle(points)
        return points


class JuliaSetStrategy:
    """
    Множество Жюлиа: z -> z^2 + c.
    Используем escape-time алгоритм, берём точки, не ушедшие за порог.

    Параметры для настройки:
    - c: комплексная константа (влияет на форму множества)
    - x_min, x_max: диапазон по оси X (по умолчанию -1.5, 1.5)
    - y_min, y_max: диапазон по оси Y (по умолчанию -1.5, 1.5)
    - max_iter: количество итераций (больше = точнее, но медленнее)
    - threshold: порог "убегания" (обычно 2.0)
    - grid: размер сетки (больше = точнее, но медленнее)
    """
    def __init__(self, c=-0.7+0.27015j,
                 x_min=-1.5, x_max=1.5,
                 y_min=-1.5, y_max=1.5,
                 max_iter=200, threshold=2.0, grid=500):
        self.c = c
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.max_iter = max_iter
        self.threshold = threshold
        self.grid = grid

    def generate(self, n):
        lin_x = np.linspace(self.x_min, self.x_max, self.grid)
        lin_y = np.linspace(self.y_min, self.y_max, self.grid)
        X, Y = np.meshgrid(lin_x, lin_y)
        Z = X + 1j*Y
        mask = np.ones(Z.shape, dtype=bool)
        C = np.full(Z.shape, self.c)
        for _ in range(self.max_iter):
            Z[mask] = Z[mask]**2 + C[mask]
            mask[np.abs(Z) > self.threshold] = False
        points = np.column_stack([X[mask], Y[mask]])

        # Проверяем, что есть точки
        if len(points) == 0:
            # Если множество пустое, возвращаем случайные точки
            return np.random.rand(n, 2)

        # Нормируем в [0,1]^2 относительно заданных границ области просмотра
        # Это гарантирует, что все точки будут видны без обрезания
        points[:,0] = (points[:,0] - self.x_min) / (self.x_max - self.x_min)
        points[:,1] = (points[:,1] - self.y_min) / (self.y_max - self.y_min)

        # Отфильтровываем точки, которые вышли за пределы [0,1]^2
        # (это может произойти, если область просмотра меньше множества)
        mask = (points[:, 0] >= 0) & (points[:, 0] <= 1) & \
               (points[:, 1] >= 0) & (points[:, 1] <= 1)
        points = points[mask]

        if len(points) == 0:
            # Если все точки вне области, возвращаем случайные
            return np.random.rand(n, 2)

        if len(points) > n:
            idx = np.random.choice(len(points), n, replace=False)
            return points[idx]
        return points

    def get_correct_visualization(self, ax, point_size=2):
        """Показывает правильную визуализацию множества Жюлиа с большим количеством точек"""
        ax.clear()

        # Генерируем множество с большим количеством точек для лучшей визуализации
        n_vis = 5000
        all_points = self.generate(n_vis)

        # Рисуем точки
        ax.scatter(all_points[:, 0], all_points[:, 1], s=point_size, color='blue', alpha=0.6)

        # Добавляем информацию о параметрах
        ax.set_title(f'Множество Жюлиа (c={self.c.real:.3f}{self.c.imag:+.3f}i)',
                    fontsize=12, pad=10)

        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
