from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel, QDialog, QTextEdit, QSlider)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from strategies import (
    UniformStrategy, SierpinskiStrategy, ClustersStrategy,
    IsingStrategy, CorrelatedFieldStrategy, LangevinStrategy,
    KochSnowflakeStrategy, BarnsleyFernStrategy, JuliaSetStrategy,
    PythagorasTreeStrategy
)

# Константа масштабирования размера кнопок
SCALE = 1.5

class AnswerDialog(QDialog):
    def __init__(self, strategy_name, description, scale_factor):
        super().__init__()
        self.setWindowTitle("Правильный ответ")
        self.setModal(True)
        self.resize(int(600 * scale_factor), int(400 * scale_factor))

        layout = QVBoxLayout(self)

        # Заголовок с названием стратегии
        title_text = QTextEdit()
        title_text.setHtml(f'<h2 style="color: #2e7d32; text-align: center;">{strategy_name}</h2>')
        title_text.setReadOnly(True)
        title_text.setMaximumHeight(int(80 * scale_factor))
        title_text.setStyleSheet(f"border: none; background: transparent; font-size: {int(18 * scale_factor)}px;")
        layout.addWidget(title_text)

        # Описание стратегии
        description_text = QTextEdit()
        description_text.setPlainText(description)
        description_text.setReadOnly(True)
        description_text.setStyleSheet(f"font-size: {int(14 * scale_factor)}px; padding: {int(10 * scale_factor)}px; border: 1px solid #ccc; border-radius: 5px;")
        layout.addWidget(description_text)

        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        close_button.setMinimumHeight(int(40 * scale_factor))
        close_button.setStyleSheet(f"font-size: {int(14 * scale_factor)}px; padding: {int(8 * scale_factor)}px;")
        layout.addWidget(close_button)

def get_strategy_description(strategy_name):
    descriptions = {
        "Случайные точки": "Простое равномерное распределение случайных точек по всей области. Каждая точка имеет равную вероятность появления в любом месте квадрата.",

        "Равномерная": "Равномерное распределение точек в пространстве. Все области имеют одинаковую вероятность содержать точки.",

        "Треугольник Серпинского": "Классический фрактал, созданный путем рекурсивного удаления центральных треугольников. Характеризуется самоподобием и имеет дробную размерность между 1 и 2.",

        "Кластеризация": "Алгоритм группировки точек в кластеры. Точки формируют отдельные группы (кластеры) с высокой плотностью внутри групп и низкой плотностью между ними.",

        "Изинг": "Модель Изинга из статистической физики, описывающая магнитные свойства материалов. Использует алгоритм Метрополиса для моделирования спиновых систем в состоянии равновесия.",

        "Коррелированное поле": "Пространственно коррелированное случайное поле, созданное с помощью фильтрации гауссова шума. Точки имеют пространственные корреляции и образуют плавные структуры.",

        "Ланжевен": "Динамика Ланжевена - стохастическое дифференциальное уравнение, описывающее броуновское движение частиц с учетом внешних сил и случайного шума.",

        "Дерево Пифагора": "Фрактальная структура, построенная рекурсивно из квадратов, где каждый квадрат порождает два меньших квадрата под углами. Создает древовидную структуру с самоподобными свойствами.",

        "Снежинка Коха": "Классический фрактал, созданный путем итеративного добавления треугольных выступов к сторонам треугольника. Имеет бесконечную длину периметра при конечной площади.",

        "Папоротник Барнсли": "Фрактал, созданный с помощью системы итерированных функций (IFS). Четыре аффинных преобразования создают реалистичное изображение папоротника.",

        "Множество Жюлиа": "Фрактальное множество в комплексной плоскости, определяемое итерацией комплексной функции. Характеризуется сложной границей и самоподобными структурами."
    }

    return descriptions.get(strategy_name, "Описание недоступно для данной стратегии.")

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
        self.diff_combo.setMinimumHeight(int(30 * SCALE))
        self.diff_combo.setStyleSheet(f"font-size: {int(12 * SCALE)}px; padding: {int(6 * SCALE)}px;")
        left_layout.addWidget(self.diff_label)
        left_layout.addWidget(self.diff_combo)

        # выбор стратегии
        self.strategy_label = QLabel("Стратегия:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Случайный",
            "Равномерная",
            "Треугольник Серпинского",
            "Кластеризация",
            "Изинг",
            "Коррелированное поле",
            "Ланжевен",
            "Дерево Пифагора",
            "Снежинка Коха",
            "Папоротник Барнсли",
            "Множество Жюлиа"
        ])
        self.strategy_combo.setMinimumHeight(int(30 * SCALE))
        self.strategy_combo.setStyleSheet(f"font-size: {int(12 * SCALE)}px; padding: {int(6 * SCALE)}px;")

        left_layout.addWidget(self.strategy_label)
        left_layout.addWidget(self.strategy_combo)

        # слайдер для размера точек
        self.point_size_label = QLabel("Размер точек:")
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(20)
        self.point_size_slider.setValue(3)  # начальное значение
        self.point_size_slider.setMinimumHeight(int(25 * SCALE))
        self.point_size_slider.setStyleSheet(f"QSlider::groove:horizontal {{ height: {int(8 * SCALE)}px; }} QSlider::handle:horizontal {{ width: {int(18 * SCALE)}px; height: {int(18 * SCALE)}px; }}")

        # метка со значением размера точек
        self.point_size_value_label = QLabel("3")
        self.point_size_value_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        self.point_size_slider.valueChanged.connect(lambda value: self.point_size_value_label.setText(str(value)))

        left_layout.addWidget(self.point_size_label)
        left_layout.addWidget(self.point_size_slider)
        left_layout.addWidget(self.point_size_value_label)

        # кнопка генерации
        self.gen_button = QPushButton("Сгенерировать")
        self.gen_button.clicked.connect(self.generate_points)
        self.gen_button.setMinimumHeight(int(40 * SCALE))
        self.gen_button.setStyleSheet(f"font-size: {int(14 * SCALE)}px; padding: {int(8 * SCALE)}px;")
        left_layout.addWidget(self.gen_button)

        # кнопка показа правильного ответа
        self.answer_button = QPushButton("Узнать правильный ответ")
        self.answer_button.clicked.connect(self.show_correct_answer)
        self.answer_button.setEnabled(False)  # изначально недоступна
        self.answer_button.setMinimumHeight(int(40 * SCALE))
        self.answer_button.setStyleSheet(f"font-size: {int(14 * SCALE)}px; padding: {int(8 * SCALE)}px;")
        left_layout.addWidget(self.answer_button)

        # метка для названия стратегии
        self.strategy_name_label = QLabel("")
        self.strategy_name_label.setStyleSheet("font-weight: bold; color: #333; margin-top: 10px;")
        left_layout.addWidget(self.strategy_name_label)

        # Добавляем растягивающийся элемент чтобы кнопки были сверху
        left_layout.addStretch()

        # Правая панель - график matplotlib с квадратными размерами
        plot_size = 800  # фиксированный размер для всех стратегий
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
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
            self.current_strategy_name = "Равномерная (случайные точки)"
            points = strat.generate(n)

        elif strategy_name == "Треугольник Серпинского":
            strat = SierpinskiStrategy()
            self.current_strategy_name = "Треугольник Серпинского"
            points = strat.generate(n)

        elif strategy_name == "Кластеризация":
            strat = ClustersStrategy(k=3)
            self.current_strategy_name = "Кластеризация"
            points = strat.generate(n)

        elif strategy_name == "Изинг":
            strat = IsingStrategy(grid_size=100, T=2.5, steps=5000)
            self.current_strategy_name = "Изинг"
            points = strat.generate(n)

        elif strategy_name == "Коррелированное поле":
            strat = CorrelatedFieldStrategy(grid_size=150, sigma=5.0)
            self.current_strategy_name = "Коррелированное поле"
            points = strat.generate(n)

        elif strategy_name == "Ланжевен":
            strat = LangevinStrategy(v=(0.005, 0.0), D=0.002, dt=1.0)
            self.current_strategy_name = "Ланжевен"
            points = strat.generate(n)

        elif strategy_name == "Дерево Пифагора":
            strat = PythagorasTreeStrategy(depth=10)
            self.current_strategy_name = "Дерево Пифагора"
            points = strat.generate(n)

        elif strategy_name == "Снежинка Коха":
            strat = KochSnowflakeStrategy(iterations=5)
            self.current_strategy_name = "Снежинка Коха"
            points = strat.generate(n)

        elif strategy_name == "Папоротник Барнсли":
            strat = BarnsleyFernStrategy()
            self.current_strategy_name = "Папоротник Барнсли"
            points = strat.generate(n)

        elif strategy_name == "Множество Жюлиа":
            strat = JuliaSetStrategy(c=-0.7 + 0.27015j, max_iter=200)
            self.current_strategy_name = "Множество Жюлиа"
            points = strat.generate(n)

        elif strategy_name == "Случайный":
            if np.random.rand() < 0.5:
                # простое равномерное распределение
                strat = UniformStrategy()
                self.current_strategy_name = "Случайные точки"
                points = strat.generate(n)
            else:
                # список всех остальных стратегий с их названиями
                strategies_with_names = [
                    (SierpinskiStrategy(), "Треугольник Серпинского"),
                    (ClustersStrategy(k=5), "Кластеризация"),
                    (IsingStrategy(grid_size=100, T=2.5, steps=3000), "Изинг"),
                    (CorrelatedFieldStrategy(grid_size=150, sigma=5.0), "Коррелированное поле"),
                    (LangevinStrategy(v=(0.005, 0.0), D=0.002), "Ланжевен"),
                    (PythagorasTreeStrategy(depth=7, jitter=True), "Дерево Пифагора"),
                    (KochSnowflakeStrategy(iterations=5), "Снежинка Коха"),
                    (BarnsleyFernStrategy(), "Папоротник Барнсли"),
                    (JuliaSetStrategy(c=-0.7 + 0.27015j, max_iter=200), "Множество Жюлиа")
                ]
                strat, strategy_display_name = strategies_with_names[np.random.randint(len(strategies_with_names))]
                self.current_strategy_name = strategy_display_name
                points = strat.generate(n)

        # отрисовка точек на графике
        self.ax.clear()
        point_size = self.point_size_slider.value()
        self.ax.scatter(points[:, 0], points[:, 1], s=point_size, color='blue')
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

        # активируем кнопку правильного ответа после генерации
        self.answer_button.setEnabled(True)
        self.current_strategy = strat  # сохраняем текущую стратегию

    def show_correct_answer(self):
        # отображаем реальное название сгенерированной стратегии
        self.strategy_name_label.setText(f"Стратегия: {self.current_strategy_name}")

        # получаем правильную визуализацию от стратегии
        if hasattr(self.current_strategy, 'get_correct_visualization'):
            self.current_strategy.get_correct_visualization(self.ax)
        else:
            # для равномерных стратегий просто меняем цвет на красный
            if isinstance(self.current_strategy, UniformStrategy):
                # просто перекрашиваем существующие точки в красный
                self.ax.collections[0].set_color('red')
            else:
                # для других стратегий показываем больше точек
                difficulty = self.diff_combo.currentText()
                if difficulty == "Лёгкий":
                    n = 100000  # увеличиваем количество точек
                elif difficulty == "Средний":
                    n = 100000
                else:
                    n = 100000

                points = self.current_strategy.generate(n)
                self.ax.clear()
                point_size = max(1, self.point_size_slider.value() - 2)  # немного меньше для большого количества точек
                self.ax.scatter(points[:, 0], points[:, 1], s=point_size, color='red', alpha=0.6)
            self.ax.set_aspect('equal')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            for spine in self.ax.spines.values():
                spine.set_visible(False)

        self.canvas.draw()

        # Открываем диалоговое окно с описанием стратегии
        description = get_strategy_description(self.current_strategy_name)
        dialog = AnswerDialog(self.current_strategy_name, description, SCALE)
        dialog.exec()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
