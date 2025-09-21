import numpy as np
from scipy.ndimage import gaussian_filter


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
    def generate(self, n):
        vertices = np.array([[0,0], [1,0], [0.5, np.sqrt(3)/2]])
        p = np.random.rand(2)
        points = []
        for _ in range(n):
            v = vertices[np.random.randint(0, 3)]
            p = (p + v) / 2
            points.append(p.copy())
        return np.array(points)


class ClustersStrategy:
    def __init__(self, k=5):
        self.k = k
    def generate(self, n):
        centers = np.random.rand(self.k, 2)
        points = []
        for c in centers:
            cluster = c + 0.05*np.random.randn(n//self.k, 2)
            points.append(cluster)
        return np.vstack(points)


# === Стратегии из статистической физики ===

class IsingStrategy:
    """
    Упрощённая модель Изинга: генерируем решётку спинов +1/-1,
    выбираем точки со спином +1.
    """
    def __init__(self, grid_size=100, T=2.5, J=1.0, steps=10000):
        self.grid_size = grid_size
        self.T = T  # температура
        self.J = J  # сила взаимодействия
        self.steps = steps

    def generate(self, n):
        N = self.grid_size
        spins = np.random.choice([-1, 1], size=(N, N))
        # Метод Метрополиса
        for _ in range(self.steps):
            i, j = np.random.randint(0, N, 2)
            s = spins[i, j]
            nb = spins[(i+1)%N,j] + spins[(i-1)%N,j] + spins[i,(j+1)%N] + spins[i,(j-1)%N]
            dE = 2 * self.J * s * nb
            if dE < 0 or np.random.rand() < np.exp(-dE/self.T):
                spins[i,j] *= -1
        # Берём только "спины вверх"
        coords = np.argwhere(spins == 1)
        # нормируем в [0,1]^2
        points = coords / N
        if len(points) > n:
            idx = np.random.choice(len(points), n, replace=False)
            return points[idx]
        return points


class CorrelatedFieldStrategy:
    """
    Генерация коррелированного гауссовского поля:
    создаём белый шум и фильтруем гауссовым ядром (корреляционная длина).
    """
    def __init__(self, grid_size=200, sigma=5.0):
        self.grid_size = grid_size
        self.sigma = sigma  # радиус корреляции

    def generate(self, n):
        field = np.random.randn(self.grid_size, self.grid_size)
        corr_field = gaussian_filter(field, sigma=self.sigma)
        # нормируем
        corr_field = (corr_field - corr_field.min()) / (corr_field.max()-corr_field.min())
        # выбираем n случайных точек с вероятностью ∝ значению поля
        flat = corr_field.ravel()
        flat /= flat.sum()
        idx = np.random.choice(len(flat), size=n, p=flat)
        y, x = np.unravel_index(idx, corr_field.shape)
        return np.column_stack([x/self.grid_size, y/self.grid_size])


class LangevinStrategy:
    """
    Броуновское движение с дрейфом (уравнение Ланжевена).
    dx = v dt + sqrt(2D dt) * N(0,1)
    """
    def __init__(self, v=(0.01,0.0), D=0.005, dt=1.0):
        self.v = np.array(v)
        self.D = D
        self.dt = dt

    def generate(self, n):
        points = [np.array([0.5, 0.5])]
        for _ in range(n-1):
            drift = self.v * self.dt
            noise = np.sqrt(2*self.D*self.dt) * np.random.randn(2)
            new_point = points[-1] + drift + noise
            points.append(new_point)
        points = np.array(points)
        # нормируем в [0,1]^2
        points = np.clip(points, 0, 1)
        return points


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
        points = np.array(curve)
        if len(points) > n:
            idx = np.random.choice(len(points), n, replace=False)
            return points[idx]
        return points


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
        return points


class JuliaSetStrategy:
    """
    Множество Жюлиа: z -> z^2 + c.
    Используем escape-time алгоритм, берём точки, не ушедшие за порог.
    """
    def __init__(self, c=-0.7+0.27015j, max_iter=200, threshold=2.0, grid=500):
        self.c = c
        self.max_iter = max_iter
        self.threshold = threshold
        self.grid = grid

    def generate(self, n):
        lin = np.linspace(-1.5, 1.5, self.grid)
        X, Y = np.meshgrid(lin, lin)
        Z = X + 1j*Y
        mask = np.ones(Z.shape, dtype=bool)
        C = np.full(Z.shape, self.c)
        for _ in range(self.max_iter):
            Z[mask] = Z[mask]**2 + C[mask]
            mask[np.abs(Z) > self.threshold] = False
        points = np.column_stack([X[mask], Y[mask]])
        # нормируем в [0,1]^2
        points[:,0] = (points[:,0] - points[:,0].min())/(points[:,0].max()-points[:,0].min())
        points[:,1] = (points[:,1] - points[:,1].min())/(points[:,1].max()-points[:,1].min())
        if len(points) > n:
            idx = np.random.choice(len(points), n, replace=False)
            return points[idx]
        return points
