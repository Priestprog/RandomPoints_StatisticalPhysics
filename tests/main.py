from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel
)
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

        # --- GUI: панель с кнопками и комбобоксами ---
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # выбор уровня сложности
        self.diff_label = QLabel("Уровень сложности:")
        self.diff_combo = QComboBox()
        self.diff_combo.addItems(["Лёгкий", "Средний", "Сложный"])
        layout.addWidget(self.diff_label)
        layout.addWidget(self.diff_combo)

        # выбор стратегии
        self.strategy_label = QLabel("Стратегия:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
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
        layout.addWidget(self.strategy_label)
        layout.addWidget(self.strategy_combo)

        # кнопка генерации
        self.gen_button = QPushButton("Сгенерировать")
        self.gen_button.clicked.connect(self.generate_points)
        layout.addWidget(self.gen_button)

        # вставляем график matplotlib
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

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

        else:
            points = np.zeros((0, 2))

        # отрисовка
        self.ax.clear()
        self.ax.scatter(points[:, 0], points[:, 1], s=3, color='blue')
        self.ax.set_aspect('equal')
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
