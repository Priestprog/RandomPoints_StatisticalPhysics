from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from strategies import (
    UniformStrategy, SierpinskiStrategy, ClustersStrategy,
    IsingStrategy, CorrelatedFieldStrategy, LangevinStrategy,
    KochSnowflakeStrategy, BarnsleyFernStrategy, JuliaSetStrategy,
    PythagorasTreeStrategy
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генерация точек")

        # Устанавливаем размер окна почти на весь экран
        screen = QApplication.primaryScreen().geometry()
        self.resize(screen.width(), screen.height() - 100)

        # Фиксируем размер окна
        self.setFixedSize(screen.width(), screen.height() - 100)

        # --- GUI: горизонтальная компоновка ---
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Левая панель с элементами управления
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # выбор уровня сложности
        self.diff_label = QLabel("Уровень сложности:")
        self.diff_combo = QComboBox()
        self.diff_combo.addItems(["Лёгкий", "Средний", "Сложный"])
        left_layout.addWidget(self.diff_label)
        left_layout.addWidget(self.diff_combo)

        # выбор стратегии
        self.strategy_label = QLabel("Стратегия:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Случайный",
            "Равномерная",
            "Серпинского",
            "Кластеризация",
            "Изинг",
            "Коррелированное поле",
            "Ланжевен",
            "Дерево Пифагора",
            "Снежинка Коха",
            "Папоротник Барнсли",
            "Множество Жюлиа"
        ])

        left_layout.addWidget(self.strategy_label)
        left_layout.addWidget(self.strategy_combo)

        # кнопка генерации
        self.gen_button = QPushButton("Сгенерировать")
        self.gen_button.clicked.connect(self.generate_points)
        left_layout.addWidget(self.gen_button)

        # Добавляем растягивающийся элемент чтобы кнопки были сверху
        left_layout.addStretch()

        # Правая панель - график matplotlib с квадратными размерами
        plot_size = 600  # фиксированный размер для всех стратегий
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(plot_size, plot_size)

        # Настройка начального вида графика - убираем оси и рамку
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Добавляем панели в основную компоновку
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.canvas)

    def generate_points(self):
        strategy_name = self.strategy_combo.currentText()
        difficulty = self.diff_combo.currentText()

        # количество точек в зависимости от сложности
        if difficulty == "Лёгкий":
            n = 1000
        elif difficulty == "Средний":
            n = 300
        elif difficulty == "Сложный":
            n = 100

        # выбор стратегии
        if strategy_name == "Равномерная":
            strat = UniformStrategy()
            points = strat.generate(n)

        elif strategy_name == "Серпинского":
            strat = SierpinskiStrategy()
            points = strat.generate(n)

        elif strategy_name == "Кластеризация":
            strat = ClustersStrategy(k=3)
            points = strat.generate(n)

        elif strategy_name == "Изинг":
            strat = IsingStrategy(grid_size=100, T=2.5, steps=5000)
            points = strat.generate(n)

        elif strategy_name == "Коррелированное поле":
            strat = CorrelatedFieldStrategy(grid_size=150, sigma=5.0)
            points = strat.generate(n)

        elif strategy_name == "Ланжевен":
            strat = LangevinStrategy(v=(0.005, 0.0), D=0.002, dt=1.0)
            points = strat.generate(n)

        elif strategy_name == "Дерево Пифагора":
            strat = PythagorasTreeStrategy(depth=7)
            points = strat.generate(n)

        elif strategy_name == "Снежинка Коха":
            strat = KochSnowflakeStrategy(iterations=4)
            points = strat.generate(n)

        elif strategy_name == "Папоротник Барнсли":
            strat = BarnsleyFernStrategy()
            points = strat.generate(n)

        elif strategy_name == "Множество Жюлиа":
            strat = JuliaSetStrategy(c=-0.7 + 0.27015j, max_iter=200)
            points = strat.generate(n)

        elif strategy_name == "Случайный":
            if np.random.rand() < 0.5:
                # простое равномерное распределение
                strat = UniformStrategy()
                points = strat.generate(n)
            else:
                # список всех остальных стратегий
                strategies = [
                    SierpinskiStrategy(),
                    ClustersStrategy(k=5),
                    IsingStrategy(grid_size=100, T=2.5, steps=3000),
                    CorrelatedFieldStrategy(grid_size=150, sigma=5.0),
                    LangevinStrategy(v=(0.005, 0.0), D=0.002),
                    PythagorasTreeStrategy(depth=7, jitter=True),
                    KochSnowflakeStrategy(iterations=4),
                    BarnsleyFernStrategy(),
                    JuliaSetStrategy(c=-0.7 + 0.27015j, max_iter=200)
                ]
                strat = np.random.choice(strategies)
                points = strat.generate(n)

        # отрисовка точек на графике
        self.ax.clear()
        self.ax.scatter(points[:, 0], points[:, 1], s=3, color='blue')
        self.ax.set_aspect('equal')

        # Убираем все элементы кроме точек
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        # Убираем рамку вокруг графика
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
