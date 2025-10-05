from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel, QDialog, QTextEdit, QSlider)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path
from strategies import (
    UniformStrategy, SierpinskiStrategy, ClustersStrategy,
    IsingStrategy, CorrelatedFieldStrategy, LangevinStrategy,
    KochSnowflakeStrategy, BarnsleyFernStrategy, JuliaSetStrategy,
    PythagorasTreeStrategy
)

# Константы масштабирования и размеров текста
SCALE = 1.7

# Размеры текста для диалогового окна с ответом
DESCRIPTION_TEXT_SIZE = 22  # Размер обычного текста
FORMULA_TEXT_SIZE = 26      # Размер формул
FORMULA_SMALL_SIZE = 20     # Размер дополнительных формул

# Параметры анимации для разных уровней сложности
# Формат: (интервал в мс, количество точек за шаг)
EASY_ANIMATION = (300, 30)    # Лёгкий: быстрая анимация
MEDIUM_ANIMATION = (500, 10)  # Средний: средняя скорость
HARD_ANIMATION = (1000, 3)     # Сложный: медленная анимация

class AnswerDialog(QDialog):
    def __init__(self, strategy_name, description, scale_factor):
        super().__init__()
        self.setWindowTitle("Правильный ответ")
        self.setModal(True)
        self.resize(int(700 * scale_factor), int(500 * scale_factor))

        layout = QVBoxLayout(self)

        # Заголовок с названием стратегии
        title_text = QTextEdit()
        title_text.setHtml(f'<h2 style="color: #2e7d32; text-align: center;">{strategy_name}</h2>')
        title_text.setReadOnly(True)
        title_text.setMaximumHeight(int(80 * scale_factor))
        title_text.setStyleSheet(f"border: none; background: transparent; font-size: {int(18 * scale_factor)}px;")
        layout.addWidget(title_text)

        # Описание стратегии с HTML-форматированием
        description_text = QTextEdit()
        description_text.setHtml(description)  # используем HTML вместо PlainText
        description_text.setReadOnly(True)
        description_text.setStyleSheet(f"font-size: {int(16 * scale_factor)}px; padding: {int(15 * scale_factor)}px; border: 1px solid #ccc; border-radius: 5px; background: white;")
        layout.addWidget(description_text)

        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        close_button.setMinimumHeight(int(40 * scale_factor))
        close_button.setStyleSheet(f"font-size: {int(14 * scale_factor)}px; padding: {int(8 * scale_factor)}px;")
        layout.addWidget(close_button)

def get_strategy_description(strategy_name):
    descriptions = {
        "Случайные точки": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Равномерное распределение точек {{x<sub>i</sub>}} ∈ [0,1]² с плотностью вероятности <i>p</i>(<i>x</i>) = const.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Каждая точка независима и имеет равную вероятность появления в любой области.</p>
        """,

        "Равномерная": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Равномерное распределение точек {{x<sub>i</sub>}} ∈ [0,1]² с плотностью вероятности <i>p</i>(<i>x</i>) = const.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Все области имеют одинаковую вероятность содержать точки.</p>
        """,

        "Треугольник Серпинского": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Классический фрактал, построенный итеративным процессом хаотической игры:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                x<sub>n+1</sub> = (x<sub>n</sub> + v<sub>k</sub>) / 2
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где v<sub>k</sub> — случайно выбранная вершина треугольника.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Фрактальная размерность:</b> <i>D</i> = log(3)/log(2) ≈ 1.585</p>
        """,

        "Кластеризация": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Процесс группировки точек {{x<sub>i</sub>}} ⊂ ℝ<sup>n</sup> в кластеры C<sub>k</sub>, минимизирующие внутрикластерное расстояние:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                min Σ<sub>k</sub> Σ<sub>x<sub>i</sub>∈C<sub>k</sub></sub> |x<sub>i</sub> - μ<sub>k</sub>|²
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где μ<sub>k</sub> — центр кластера C<sub>k</sub>.</p>
        """,

        "Изинг": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Спиновая система на решётке S<sub>i</sub> ∈ {{-1, +1}} с гамильтонианом:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                H = -J Σ<sub>⟨i,j⟩</sub> S<sub>i</sub>S<sub>j</sub> - h Σ<sub>i</sub> S<sub>i</sub>
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где <i>J</i> — взаимодействие соседей, <i>h</i> — внешнее поле.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Модель описывает фазовые переходы и магнитные свойства материалов.</p>
        """,

        "Коррелированное поле": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Случайное поле φ(<i>x</i>) с корреляционной функцией:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                C(<i>r</i>) = ⟨φ(<i>x</i>)φ(<i>x</i>+<i>r</i>)⟩
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Поле создаётся фильтрацией белого шума гауссовым ядром с заданной корреляционной длиной σ.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Точки имеют пространственные корреляции.</p>
        """,

        "Ланжевен": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Стохастическое дифференциальное уравнение:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                m(dv/dt) = -γv + ξ(t)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где γ — коэффициент трения, ξ(t) — гауссовский шум с</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_SMALL_SIZE}px; padding: 10px;">
                ⟨ξ(t)ξ(t')⟩ = 2γk<sub>B</sub>T δ(t-t')
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Описывает броуновское движение и флуктуации.</p>
        """,

        "Дерево Пифагора": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Фрактал, построенный рекурсивно: на квадрате строятся два квадрата со сторонами</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                s<sub>n</sub> = s<sub>n-1</sub>·cos(θ) &nbsp; и &nbsp; s<sub>n-1</sub>·sin(θ)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">образующие прямоугольный треугольник.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Фрактальная размерность:</b> <i>D</i> ≈ 1.5</p>
        """,

        "Снежинка Коха": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Фрактальная кривая, строящаяся заменой средней трети каждого отрезка равносторонним треугольником без основания.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Длина после <i>n</i> итераций:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                L<sub>n</sub> = L<sub>0</sub>(4/3)<sup>n</sup>
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Фрактальная размерность:</b> <i>D</i> = ln(4)/ln(3) ≈ 1.262</p>
        """,

        "Папоротник Барнсли": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Фрактал, определяемый системой итеративных аффинных преобразований (IFS):</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                f<sub>i</sub>(x,y) = (a<sub>i</sub>x + b<sub>i</sub>y + e<sub>i</sub>, c<sub>i</sub>x + d<sub>i</sub>y + f<sub>i</sub>)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где <i>i</i> = 1,...,4 с вероятностями p<sub>i</sub>.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Фрактальная размерность:</b> <i>D</i> ≈ 1.7</p>
        """,

        "Множество Жюлиа": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Множество точек <i>z</i> ∈ ℂ, для которых итерация остаётся ограниченной:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                z<sub>n+1</sub> = z<sub>n</sub>² + c
            </p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 10px;">
                J<sub>c</sub> = {{z ∈ ℂ : |z<sub>n</sub>| ↛ ∞}}
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Граница J<sub>c</sub> — фрактал с размерностью <i>D</i> ∈ [1,2).</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Цвет точек определяется скоростью расхождения (escape time).</p>
        """
    }

    return descriptions.get(strategy_name, "<p>Описание недоступно для данной стратегии.</p>")

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
        self.gen_button = QPushButton("Генерировать")
        self.gen_button.clicked.connect(self.generate_points)
        self.gen_button.setMinimumHeight(int(40 * SCALE))
        self.gen_button.setStyleSheet(f"font-size: {int(14 * SCALE)}px; padding: {int(8 * SCALE)}px;")
        left_layout.addWidget(self.gen_button)

        # кнопка показа правильного ответа
        self.answer_button = QPushButton("Ответ")
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

        # Переменные для анимации
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.current_points = None
        self.animation_index = 0
        self.animation_step = 1
        self.is_animating = False

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
            strat = IsingStrategy(grid_size=100, T=2.5, steps=3000)
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
            strat = PythagorasTreeStrategy(depth=7)
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
                    (ClustersStrategy(k=3), "Кластеризация"),
                    (IsingStrategy(grid_size=100, T=2.5, steps=3000), "Изинг"),
                    (CorrelatedFieldStrategy(grid_size=150, sigma=5.0), "Коррелированное поле"),
                    (LangevinStrategy(v=(0.005, 0.0), D=0.002), "Ланжевен"),
                    (PythagorasTreeStrategy(depth=7), "Дерево Пифагора"),
                    (KochSnowflakeStrategy(iterations=5), "Снежинка Коха"),
                    (BarnsleyFernStrategy(), "Папоротник Барнсли"),
                    (JuliaSetStrategy(c=-0.7 + 0.27015j, max_iter=200), "Множество Жюлиа")
                ]
                strat, strategy_display_name = strategies_with_names[np.random.randint(len(strategies_with_names))]
                self.current_strategy_name = strategy_display_name
                points = strat.generate(n)

        # Сохраняем стратегию
        self.current_strategy = strat

        # Настраиваем анимацию в зависимости от сложности
        if difficulty == "Лёгкий":
            animation_interval, step = EASY_ANIMATION
        elif difficulty == "Средний":
            animation_interval, step = MEDIUM_ANIMATION
        else:  # Сложный
            animation_interval, step = HARD_ANIMATION

        # Получаем размер точек из слайдера
        point_size = self.point_size_slider.value()

        # Запускаем анимацию
        self._start_animation(points, point_size, animation_interval, step)

    def _start_animation(self, points, point_size, interval, step):
        """Запускает анимацию постепенного появления точек"""
        # Останавливаем предыдущую анимацию, если она была
        if self.is_animating:
            self.animation_timer.stop()

        # Очищаем график
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Сохраняем параметры анимации
        self.current_points = points
        self.current_point_size = point_size
        self.animation_index = 0
        self.animation_step = step
        self.is_animating = True

        # Активируем кнопку ответа
        self.answer_button.setEnabled(True)

        # Запускаем таймер
        self.animation_timer.start(interval)

    def _update_animation(self):
        """Обновляет анимацию, добавляя новые точки"""
        if self.current_points is None:
            return

        # Вычисляем индекс конца текущей порции точек
        end_index = min(self.animation_index + self.animation_step, len(self.current_points))

        # Очищаем и рисуем накопленные точки
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Рисуем точки от 0 до end_index
        points_to_show = self.current_points[:end_index]
        self.ax.scatter(points_to_show[:, 0], points_to_show[:, 1],
                       s=self.current_point_size, color='blue')

        self.canvas.draw()

        # Обновляем индекс
        self.animation_index = end_index

        # Проверяем, закончилась ли анимация
        if self.animation_index >= len(self.current_points):
            self.animation_timer.stop()
            self.is_animating = False

    def show_correct_answer(self):
        # Останавливаем анимацию, если она идёт
        if self.is_animating:
            self.animation_timer.stop()
            self.is_animating = False

        # Определяем, нужно ли загружать пресет для фракталов
        preset_mapping = {
            "Треугольник Серпинского": "Треугольник_Серпинского.png",
            "Снежинка Коха": "Снежинка_Коха.png",
            "Папоротник Барнсли": "Папоротник_Барнсли.png",
            "Множество Жюлиа": "Множество_Жюлиа.png",
            "Дерево Пифагора": "Дерево_Пифагора.png"
        }

        # Проверяем, есть ли пресет для текущей стратегии
        if self.current_strategy_name in preset_mapping:
            preset_filename = preset_mapping[self.current_strategy_name]
            preset_path = Path(__file__).parent / preset_filename

            # Если файл существует, загружаем и показываем его
            if preset_path.exists():
                self.ax.clear()
                img = mpimg.imread(str(preset_path))
                self.ax.imshow(img)
                self.ax.set_aspect('equal')
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                for spine in self.ax.spines.values():
                    spine.set_visible(False)
                self.canvas.draw()
            else:
                # Если файла нет, показываем предупреждение и используем стандартную генерацию
                print(f"ПРЕДУПРЕЖДЕНИЕ: Пресет не найден: {preset_path}")
                self._show_generated_answer()
        else:
            # Для стратегий без пресетов используем стандартную логику
            self._show_generated_answer()

        # Открываем диалоговое окно с описанием стратегии
        description = get_strategy_description(self.current_strategy_name)
        dialog = AnswerDialog(self.current_strategy_name, description, SCALE)
        dialog.exec()

    def _show_generated_answer(self):
        """Вспомогательный метод для генерации ответа (для стратегий без пресетов)"""
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


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
