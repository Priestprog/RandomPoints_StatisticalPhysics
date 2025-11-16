from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel, QDialog, QTextEdit, QSlider, QStackedWidget,
    QRadioButton, QButtonGroup, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path
from strategies import (
    UniformStrategy, SierpinskiStrategy, ClustersStrategy, RepulsionStrategy,
    BoltzmannStrategy, CrystallizationStrategy,
    IsingStrategy, RandomWalkRepulsionStrategy,
    KochSnowflakeStrategy, BarnsleyFernStrategy, JuliaSetStrategy,
    PythagorasTreeStrategy
)

# Константы масштабирования и размеров текста
SCALE = 1.7

# Размеры текста для диалогового окна с ответом
DESCRIPTION_TEXT_SIZE = 22  # Размер обычного текста
FORMULA_TEXT_SIZE = 26      # Размер формул
FORMULA_SMALL_SIZE = 20     # Размер дополнительных формул

# ============================================================
# ПРЕСЕТЫ ДЛЯ МНОЖЕСТВА ЖЮЛИА
# Настройте эти параметры для изменения масштаба и параметров
# ВАЖНО: Эти параметры должны совпадать с generate_presets.py!
# ============================================================
JULIA_PRESET = {
    'c': -0.7 + 0.27015j,   # Комплексная константа (форма множества)
    'x_min': -1.5,          # Минимум по X (расширено для боковых спиралей)
    'x_max': 1.5,           # Максимум по X (расширено для боковых спиралей)
    'y_min': -1.5,          # Минимум по Y
    'y_max': 1.5,           # Максимум по Y
    'max_iter': 200,        # Количество итераций (512 в пресете, но это долго)
    'threshold': 2.0,       # Порог "убегания"
    'grid': 500             # Размер сетки (1200 в пресете, но это очень долго)
}

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
        # Ограничиваем максимальную высоту окна
        screen = QApplication.primaryScreen().geometry()
        max_height = screen.height() - 300  # Оставляем отступ от краёв экрана
        dialog_height = min(int(500 * scale_factor), max_height)
        self.resize(int(700 * scale_factor), dialog_height)

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
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Классическое равномерное распределение точек на квадрате [0,1]².</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Математическое описание:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Каждая координата точки независимо генерируется из равномерного распределения:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                x, y ~ U(0, 1)<br>
                P(x, y) = const
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Применение:</b> Базовая стратегия для сравнения, модель идеального газа.</p>
        """,

        "Равномерная": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Классическое равномерное распределение точек на квадрате [0,1]².</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Математическое описание:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Каждая координата точки независимо генерируется из равномерного распределения:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                x, y ~ U(0, 1)<br>
                P(x, y) = const
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Применение:</b> Базовая стратегия для сравнения, модель идеального газа.</p>
        """,

        "Треугольник Серпинского": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Фрактальная структура, получаемая методом "игры в хаос".</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Алгоритм:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Заданы три вершины равностороннего треугольника</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. Случайная начальная точка</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. На каждом шаге: выбрать случайную вершину и переместиться на половину расстояния к ней</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                P<sub>n+1</sub> = (P<sub>n</sub> + V<sub>i</sub>) / 2
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где V<sub>i</sub> — случайно выбранная вершина</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Фрактальная размерность:</b> D = ln(3)/ln(2) ≈ 1.585</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Самоподобие на всех масштабах. Детерминированная фрактальная структура, получаемая стохастическим методом.</p>
        """,

        "Притяжение": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Распределение Больцмана с притягивающим потенциалом.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Потенциал:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                U(r) = -Σ<sub>i</sub> ε exp(-(|r - r<sub>i</sub>|²)/(2σ²))
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Распределение Больцмана:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                P(r) ∝ exp(-U(r)/(k<sub>B</sub>T))
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Эффект:</b> Точки стремятся образовывать плотные кластеры вокруг центров притяжения.</p>
        """,

        "Отталкивание": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Распределение Больцмана с отталкивающим потенциалом.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Потенциал:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                U(r) = Σ<sub>i</sub> u(|r - r<sub>i</sub>|)<br>
                u(d) = ε(σ/d)¹²
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Леннард-Джонс, только отталкивание</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Метод генерации:</b> Rejection sampling</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Предложить случайную точку r</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. Вычислить U(r)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. Принять с вероятностью P = exp(-U(r)/(k<sub>B</sub>T))</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Эффект:</b> Точки избегают близкого расположения друг к другу, создавая более равномерное распределение.</p>
        """,

        "Гравитация": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Генерация согласно распределению Больцмана в гармоническом потенциале.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Потенциал:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                U(x, y) = k(x² + y²)/2
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Распределение:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                P(x, y) ∝ exp(-U(x,y)/(k<sub>B</sub>T))<br>
                P(x, y) ∝ exp(-k(x² + y²)/(2k<sub>B</sub>T))
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Результат:</b> Нормальное распределение с σ² = k<sub>B</sub>T/k</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Применение:</b> Моделирование термического равновесия в гармонической ловушке (оптические пинцеты, ионные ловушки).</p>
        """,

        "Кристаллизация (гексагональная)": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Моделирование роста кристаллических решеток.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Базис гексагональной решётки:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                a<sub>1</sub> = (1, 0)<br>
                a<sub>2</sub> = (1/2, √3/2)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Узлы решетки:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                r = n<sub>1</sub>a<sub>1</sub> + n<sub>2</sub>a<sub>2</sub>, n<sub>1</sub>, n<sub>2</sub> ∈ ℤ
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Применение:</b> Графен, соты, некоторые молекулярные кристаллы.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Свойства:</b> Дальний порядок, периодичность, минимум потенциальной энергии, симметрия трансляций.</p>
        """,

        "Кристаллизация (квадратная)": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Моделирование роста кристаллических решеток.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Базис квадратной решётки:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                a<sub>1</sub> = (1, 0)<br>
                a<sub>2</sub> = (0, 1)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Узлы решетки:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                r = (n<sub>1</sub>, n<sub>2</sub>), n<sub>1</sub>, n<sub>2</sub> ∈ ℤ
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Применение:</b> Простые кубические решетки, ионные кристаллы.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Свойства:</b> Дальний порядок, периодичность, минимум потенциальной энергии, симметрия трансляций.</p>
        """,

        "Изинг": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Классическая модель статистической физики для описания ферромагнетизма и фазовых переходов.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Гамильтониан:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                H = -J Σ<sub>⟨i,j⟩</sub> s<sub>i</sub>s<sub>j</sub>
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где s<sub>i</sub> = ±1 — спины на узлах решетки, J > 0 — константа обменного взаимодействия, ⟨i,j⟩ — суммирование по ближайшим соседям</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Алгоритм Метрополиса:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Случайно выбрать спин</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. Вычислить изменение энергии при переворачивании: ΔE</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. Принять с вероятностью P = min(1, exp(-ΔE/(k<sub>B</sub>T)))</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Температурные режимы:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• T ≪ T<sub>c</sub>: Упорядоченное состояние (ферромагнитная фаза)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• T ≈ T<sub>c</sub>: Критическая область (флуктуации на всех масштабах)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• T ≫ T<sub>c</sub>: Неупорядоченное состояние (парамагнитная фаза)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Критическая температура</b> (2D): T<sub>c</sub> = 2J/(k<sub>B</sub> ln(1 + √2)) ≈ 2.269 J/k<sub>B</sub></p>
        """,

        "Случайное блуждание": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Моделирование броуновского движения частицы.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Математическое описание:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Начальная позиция: (x<sub>0</sub>, y<sub>0</sub>)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Каждый шаг: смещение на случайный вектор</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                x<sub>n+1</sub> = x<sub>n</sub> + δx<br>
                y<sub>n+1</sub> = y<sub>n</sub> + δy<br>
                δx, δy ~ N(0, σ²)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Физический смысл:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• Моделирует диффузию</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• Среднеквадратичное смещение: ⟨r²⟩ ~ t</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• Демонстрирует броуновское движение</p>
        """,

        "Дерево Пифагора": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Рекурсивное построение квадратов на сторонах прямоугольных треугольников.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Алгоритм:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Начать с базового квадрата</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. На верхней стороне построить прямоугольный треугольник</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. На катетах построить два квадрата меньшего размера</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">4. Рекурсивно повторить для новых квадратов</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Углы и пропорции:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">При равнобедренном треугольнике (45°-45°-90°):</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                Масштаб = 1/√2 на каждом уровне<br>
                Угол поворота = ±45°
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Свойства:</b> Рекурсивная структура, самоподобие, напоминает ветвление деревьев.</p>
        """,

        "Снежинка Коха": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Рекурсивный фрактал, получаемый итерационным делением отрезков.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Алгоритм построения:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Начать с равностороннего треугольника</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. Каждый отрезок разделить на три части</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. Средний сегмент заменить двумя сторонами равностороннего треугольника</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">4. Повторить рекурсивно</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Математическое описание:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">После n итераций:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                L<sub>n</sub> = L<sub>0</sub>(4/3)<sup>n</sup><br>
                N<sub>n</sub> = 3·4<sup>n</sup> (число сегментов)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Фрактальная размерность:</b> D = ln(4)/ln(3) ≈ 1.262</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Периметр → ∞ при количестве итераций → ∞, площадь остается конечной.</p>
        """,

        "Папоротник Барнсли": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Система итерированных функций (IFS).</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Аффинные преобразования:</b></p>
            <p style="font-size: {FORMULA_SMALL_SIZE}px; font-family: monospace; padding: 10px;">
                f<sub>1</sub>: [x, y] → [0, 0.16y] &nbsp;&nbsp;&nbsp; (вероятность 0.01) — стебель<br>
                f<sub>2</sub>: [x, y] → [0.85x + 0.04y, -0.04x + 0.85y + 1.6] &nbsp; (0.85) — основная часть<br>
                f<sub>3</sub>: [x, y] → [0.20x - 0.26y, 0.23x + 0.22y + 1.6] &nbsp; (0.07) — левая ветвь<br>
                f<sub>4</sub>: [x, y] → [-0.15x + 0.28y, 0.26x + 0.24y + 0.44] &nbsp; (0.07) — правая ветвь
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Алгоритм:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Начальная точка (0, 0)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. Случайно выбрать преобразование согласно вероятностям</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. Применить преобразование</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">4. Повторить</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Результат:</b> Детализированная структура, напоминающая натуральный папоротник.</p>
        """,

        "Множество Жюлиа": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Принцип:</b> Динамика комплексных итераций.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Итерация:</b></p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                z<sub>n+1</sub> = z<sub>n</sub>² + c
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где c — фиксированный комплексный параметр (например, c = -0.7 + 0.27015i)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Классификация точек:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• Точка z<sub>0</sub> принадлежит множеству Жюлиа, если последовательность {{z<sub>n</sub>}} ограничена</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">• Критерий ухода: |z<sub>n</sub>| > R (обычно R = 2)</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Генерация точек:</b></p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">1. Выбрать случайную точку в комплексной плоскости</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">2. Итерировать отображение</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">3. Если последовательность не убегает на бесконечность за N итераций — сохранить точку</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;"><b>Свойства:</b> Фрактальная граница, самоподобие, хаотическая динамика.</p>
        """
    }

    return descriptions.get(strategy_name, "<p>Описание недоступно для данной стратегии.</p>")


class TitleScreen(QWidget):
    """Титульный экран с логотипами и кнопками"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 30, 50, 50)

        # Верхняя часть с логотипами
        top_layout = QHBoxLayout()

        # Левый логотип (ММП)
        logo_mmp = QLabel()
        pixmap_mmp = QPixmap(str(Path(__file__).parent / "logo-mgu.png"))
        if not pixmap_mmp.isNull():
            logo_mmp.setPixmap(pixmap_mmp.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_mmp.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        top_layout.addWidget(logo_mmp, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        top_layout.addStretch()

        # Правый логотип (копия ММП или другой)
        logo_right = QLabel()
        pixmap_right = QPixmap(str(Path(__file__).parent / "logo_mmp.png"))
        if not pixmap_right.isNull():
            logo_right.setPixmap(pixmap_right.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        top_layout.addWidget(logo_right, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        layout.addLayout(top_layout)

        layout.addSpacing(30)

        # Заголовки
        title1 = QLabel("Московский Государственный Университет\nимени М. В. Ломоносова")
        title1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title1.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        layout.addWidget(title1)

        layout.addSpacing(10)

        title2 = QLabel("Псевдослучайные структуры точек на поверхности")
        title2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title2.setFont(QFont("Arial", 36))
        layout.addWidget(title2)

        layout.addSpacing(60)

        # Кнопки с ограниченной шириной
        button_container = QHBoxLayout()
        button_container.addStretch()

        button_layout = QVBoxLayout()
        button_layout.setSpacing(30)  # Увеличили отступ между кнопками

        # Кнопка "Модель"
        self.play_button = QPushButton("Модель")
        self.play_button.setMinimumHeight(70)
        self.play_button.setMinimumWidth(400)
        self.play_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: white;
                border: 2px solid #333;
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        button_layout.addWidget(self.play_button)

        # Кнопка "Авторы"
        self.authors_button = QPushButton("Авторы")
        self.authors_button.setMinimumHeight(70)
        self.authors_button.setMinimumWidth(400)
        self.authors_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: white;
                border: 2px solid #333;
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        button_layout.addWidget(self.authors_button)

        # Кнопка "Выход"
        self.exit_button = QPushButton("Выход")
        self.exit_button.setMinimumHeight(70)
        self.exit_button.setMinimumWidth(400)
        self.exit_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: white;
                border: 2px solid #333;
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        button_layout.addWidget(self.exit_button)

        button_container.addLayout(button_layout)
        button_container.addStretch()

        layout.addLayout(button_container)
        layout.addStretch()


class AuthorsScreen(QWidget):
    """Экран с информацией об авторах"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 30, 50, 50)

        # Заголовок
        title = QLabel("Авторы проекта")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        layout.addWidget(title)

        layout.addSpacing(50)

        # Контейнер для авторов - фото в ряд
        authors_photos_layout = QHBoxLayout()
        authors_photos_layout.addStretch()

        # Автор 1: Багров Александр Михайлович
        author1_container = QVBoxLayout()
        author1_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Фото автора 1
        photo1 = QLabel()
        photo1_path = Path(__file__).parent / "bagrov.png"
        if photo1_path.exists():
            pixmap1 = QPixmap(str(photo1_path))
            if not pixmap1.isNull():
                photo1.setPixmap(pixmap1.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            photo1.setFixedSize(200, 200)
            photo1.setStyleSheet("background-color: #ddd; border: 2px solid #999; border-radius: 100px;")
        photo1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author1_container.addWidget(photo1)

        author1_container.addSpacing(20)

        # Текст автора 1
        name1 = QLabel("Багров Александр\nМихайлович")
        name1.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        name1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author1_container.addWidget(name1)

        authors_photos_layout.addLayout(author1_container)
        authors_photos_layout.addSpacing(80)

        # Автор 2: Лукьянов Артём
        author2_container = QVBoxLayout()
        author2_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Фото автора 2
        photo2 = QLabel()
        photo2_path = Path(__file__).parent / "lukanov.jpg"
        if photo2_path.exists():
            pixmap2 = QPixmap(str(photo2_path))
            if not pixmap2.isNull():
                photo2.setPixmap(pixmap2.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            photo2.setFixedSize(200, 200)
            photo2.setStyleSheet("background-color: #ddd; border: 2px solid #999; border-radius: 100px;")
        photo2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author2_container.addWidget(photo2)

        author2_container.addSpacing(20)

        # Текст автора 2
        name2 = QLabel("Лукьянов Артём\nВасильевич")
        name2.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        name2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author2_container.addWidget(name2)

        authors_photos_layout.addLayout(author2_container)
        authors_photos_layout.addSpacing(80)

        # Автор 3: Чичигина Ольга Александровна
        author3_container = QVBoxLayout()
        author3_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Фото автора 3
        photo3 = QLabel()
        photo3_path = Path(__file__).parent / "chichigina.jpg"
        if photo3_path.exists():
            pixmap3 = QPixmap(str(photo3_path))
            if not pixmap3.isNull():
                photo3.setPixmap(pixmap3.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            photo3.setFixedSize(200, 200)
            photo3.setStyleSheet("background-color: #ddd; border: 2px solid #999; border-radius: 100px;")
        photo3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author3_container.addWidget(photo3)

        author3_container.addSpacing(20)

        # Текст автора 3
        name3 = QLabel("Чичигина Ольга\nАлександровна")
        name3.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        name3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author3_container.addWidget(name3)

        authors_photos_layout.addLayout(author3_container)
        authors_photos_layout.addStretch()

        layout.addLayout(authors_photos_layout)
        layout.addStretch()

        # Кнопка "Назад"
        self.back_button = QPushButton("← Назад")
        self.back_button.setMinimumHeight(50)
        self.back_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.back_button)


class GameWindow(QWidget):
    """Игровое окно с генерацией точек"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Переменные для анимации
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.current_points = None
        self.animation_index = 0
        self.animation_step = 1
        self.is_animating = False

    def init_ui(self):
        # --- GUI: горизонтальная компоновка ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)  # уменьшаем отступы (left, top, right, bottom)
        main_layout.setSpacing(5)  # уменьшаем расстояние между элементами

        # Кнопка "Назад" вверху
        top_layout = QHBoxLayout()
        self.back_button = QPushButton("← Назад")
        self.back_button.setMaximumWidth(150)
        self.back_button.setMinimumHeight(35)  # уменьшили высоту кнопки
        self.back_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        top_layout.addWidget(self.back_button)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # Основная часть игры
        game_layout = QHBoxLayout()
        game_layout.setContentsMargins(0, 0, 0, 0)  # убираем отступы

        # Левая панель с элементами управления
        left_panel = QWidget()
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(int(5 * SCALE))  # Уменьшаем расстояние между элементами
        left_layout.setContentsMargins(int(5 * SCALE), int(5 * SCALE), int(5 * SCALE), int(5 * SCALE))  # Уменьшаем отступы

        # выбор уровня сложности
        self.diff_label = QLabel("Уровень сложности:")
        self.diff_combo = QComboBox()
        self.diff_combo.addItems(["Лёгкий", "Средний", "Сложный"])
        self.diff_combo.setMinimumHeight(int(28 * SCALE))
        self.diff_combo.setStyleSheet(f"font-size: {int(12 * SCALE)}px; padding: {int(3 * SCALE)}px;")
        left_layout.addWidget(self.diff_label)
        left_layout.addWidget(self.diff_combo)

        # выбор стратегии
        self.strategy_label = QLabel("Стратегия:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Случайная стратегия",
            "Случайные точки",
            "Треугольник Серпинского",
            "Притяжение",
            "Отталкивание",
            "Гравитация",
            "Кристаллизация (гексагон.)",
            "Кристаллизация (квадрат.)",
            "Изинг",
            "Случайное блуждание",
            "Дерево Пифагора",
            "Снежинка Коха",
            "Папоротник Барнсли",
            "Множество Жюлиа"
        ])
        self.strategy_combo.setMinimumHeight(int(28 * SCALE))
        self.strategy_combo.setStyleSheet(f"font-size: {int(12 * SCALE)}px; padding: {int(3 * SCALE)}px;")

        left_layout.addWidget(self.strategy_label)
        left_layout.addWidget(self.strategy_combo)

        # Radio кнопки для режима случайной стратегии
        self.random_mode_label = QLabel("Режим случайной стратегии:")
        self.random_mode_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        left_layout.addWidget(self.random_mode_label)

        # Группа radio кнопок
        self.random_mode_group = QButtonGroup()

        self.identify_radio = QRadioButton("Отгадай")
        self.identify_radio.setChecked(True)
        self.identify_radio.setStyleSheet(f"font-size: {int(11 * SCALE)}px; padding: {int(2 * SCALE)}px;")
        self.random_mode_group.addButton(self.identify_radio, 2)
        left_layout.addWidget(self.identify_radio)

        self.guess_radio = QRadioButton("Угадай")
        self.guess_radio.setStyleSheet(f"font-size: {int(11 * SCALE)}px; padding: {int(2 * SCALE)}px;")
        self.random_mode_group.addButton(self.guess_radio, 1)
        left_layout.addWidget(self.guess_radio)

        # Изначально скрываем radio кнопки
        self.random_mode_label.hide()
        self.guess_radio.hide()
        self.identify_radio.hide()

        # Подключаем обработчик изменения стратегии
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)

        # Проверяем начальное состояние (если уже выбрана случайная стратегия)
        self._on_strategy_changed(self.strategy_combo.currentText())

        # слайдер для скорости анимации (интервал между обновлениями в мс)
        self.speed_label = QLabel("Скорость: 100 мс")
        self.speed_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(900)
        self.speed_slider.setMinimumHeight(int(25 * SCALE))
        self.speed_slider.setStyleSheet(f"QSlider::groove:horizontal {{ height: {int(8 * SCALE)}px; }} QSlider::handle:horizontal {{ width: {int(18 * SCALE)}px; height: {int(18 * SCALE)}px; }}")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        left_layout.addWidget(self.speed_label)
        left_layout.addWidget(self.speed_slider)

        # слайдер для количества точек за шаг
        self.points_per_step_label = QLabel("Точек за шаг: 10")
        self.points_per_step_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        self.points_per_step_slider = QSlider(Qt.Orientation.Horizontal)
        self.points_per_step_slider.setMinimum(1)
        self.points_per_step_slider.setMaximum(100)
        self.points_per_step_slider.setValue(10)
        self.points_per_step_slider.setMinimumHeight(int(25 * SCALE))
        self.points_per_step_slider.setStyleSheet(f"QSlider::groove:horizontal {{ height: {int(8 * SCALE)}px; }} QSlider::handle:horizontal {{ width: {int(18 * SCALE)}px; height: {int(18 * SCALE)}px; }}")
        self.points_per_step_slider.valueChanged.connect(self._on_points_per_step_changed)

        left_layout.addWidget(self.points_per_step_label)
        left_layout.addWidget(self.points_per_step_slider)

        # слайдер для размера точек
        self.point_size_label = QLabel("Размер точек: 3")
        self.point_size_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(30)
        self.point_size_slider.setValue(3)
        self.point_size_slider.setMinimumHeight(int(25 * SCALE))
        self.point_size_slider.setStyleSheet(f"QSlider::groove:horizontal {{ height: {int(8 * SCALE)}px; }} QSlider::handle:horizontal {{ width: {int(18 * SCALE)}px; height: {int(18 * SCALE)}px; }}")
        self.point_size_slider.valueChanged.connect(self._on_point_size_changed)

        left_layout.addWidget(self.point_size_label)
        left_layout.addWidget(self.point_size_slider)

        # кнопка генерации
        self.gen_button = QPushButton("Генерировать")
        self.gen_button.clicked.connect(self.generate_points)
        self.gen_button.setMinimumHeight(int(35 * SCALE))
        self.gen_button.setStyleSheet(f"font-size: {int(14 * SCALE)}px; padding: {int(5 * SCALE)}px;")
        left_layout.addWidget(self.gen_button)

        # кнопка показа правильного ответа
        self.answer_button = QPushButton("Ответ")
        self.answer_button.clicked.connect(self.show_correct_answer)
        self.answer_button.setEnabled(False)  # изначально недоступна
        self.answer_button.setMinimumHeight(int(35 * SCALE))
        self.answer_button.setStyleSheet(f"font-size: {int(14 * SCALE)}px; padding: {int(5 * SCALE)}px;")
        left_layout.addWidget(self.answer_button)

        # метка для названия стратегии
        self.strategy_name_label = QLabel("")
        self.strategy_name_label.setStyleSheet("font-weight: bold; color: #333;")
        left_layout.addWidget(self.strategy_name_label)

        # Добавляем растягивающийся элемент чтобы кнопки были сверху
        left_layout.addStretch()

        # Правая панель - график matplotlib с квадратными размерами
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 400)  # минимальный размер вместо фиксированного

        # Делаем canvas растягивающимся
        from PyQt6.QtWidgets import QSizePolicy
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # Настраиваем отступы графика - симметрично для центрирования заголовка
        self.figure.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.06)

        # Настройка начального вида графика - убираем оси и рамку
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Добавляем панели в основную компоновку
        game_layout.addWidget(left_panel, 0)  # 0 - не растягивается
        game_layout.addWidget(self.canvas, 1)  # 1 - растягивается

        main_layout.addLayout(game_layout, 1)  # 1 - занимает все доступное пространство

    def _on_strategy_changed(self, strategy_name):
        """Показывает/скрывает radio кнопки при выборе случайной стратегии"""
        if strategy_name == "Случайная стратегия":
            self.random_mode_label.show()
            self.guess_radio.show()
            self.identify_radio.show()
        else:
            self.random_mode_label.hide()
            self.guess_radio.hide()
            self.identify_radio.hide()

    def _on_speed_changed(self, value):
        """Обработчик изменения скорости анимации"""
        # Инвертируем значение: 10->1000, 1000->10
        inverted_value = 1010 - value
        self.speed_label.setText(f"Скорость: {inverted_value} мс")
        # Если анимация идёт, изменяем интервал таймера
        if self.is_animating:
            self.animation_timer.setInterval(inverted_value)

    def _on_points_per_step_changed(self, value):
        """Обработчик изменения количества точек за шаг"""
        self.points_per_step_label.setText(f"Точек за шаг: {value}")
        # Обновляем шаг анимации во время воспроизведения
        self.animation_step = value

    def _on_point_size_changed(self, value):
        """Обработчик изменения размера точек"""
        self.point_size_label.setText(f"Размер точек: {value}")
        # Обновляем текущий размер точек
        self.current_point_size = value
        # Если анимация идёт, перерисовываем с новым размером
        if self.is_animating and self.current_points is not None:
            self._redraw_current_frame()

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
        if strategy_name == "Случайные точки":
            strat = UniformStrategy()
            self.current_strategy_name = "Случайные точки"
            points = strat.generate(n)

        elif strategy_name == "Треугольник Серпинского":
            strat = SierpinskiStrategy()
            self.current_strategy_name = "Треугольник Серпинского"
            points = strat.generate(n)

        elif strategy_name == "Притяжение":
            strat = ClustersStrategy(k=7)
            self.current_strategy_name = "Притяжение"
            points = strat.generate(n)

        elif strategy_name == "Отталкивание":
            strat = RepulsionStrategy(k=7)
            self.current_strategy_name = "Отталкивание"
            points = strat.generate(n)

        elif strategy_name == "Гравитация":
            strat = BoltzmannStrategy(temperature=0.15)
            self.current_strategy_name = "Гравитация"
            points = strat.generate(n)

        elif strategy_name == "Кристаллизация (гексагон.)":
            # Температура зависит от уровня сложности
            if n >= 1000:  # Лёгкий
                thermal_noise = 0.002
            elif n >= 300:  # Средний
                thermal_noise = 0.003
            else:  # Сложный (n = 100)
                thermal_noise = 0.004
            strat = CrystallizationStrategy(lattice_type='hexagonal', thermal_noise=thermal_noise)
            self.current_strategy_name = "Кристаллизация (гексагональная)"
            points = strat.generate(n)

        elif strategy_name == "Кристаллизация (квадрат.)":
            # Температура зависит от уровня сложности
            if n >= 1000:  # Лёгкий
                thermal_noise = 0.002
            elif n >= 300:  # Средний
                thermal_noise = 0.005
            else:  # Сложный (n = 100)
                thermal_noise = 0.009
            strat = CrystallizationStrategy(lattice_type='square', thermal_noise=thermal_noise)
            self.current_strategy_name = "Кристаллизация (квадратная)"
            points = strat.generate(n)

        elif strategy_name == "Изинг":
            # T=1.8 для заметной кластеризации (домены 10-30 спинов)
            # steps вычисляется автоматически на основе T (~200,000 для T=1.8)
            strat = IsingStrategy(grid_size=100, T=2.1, J=2.0)
            self.current_strategy_name = "Изинг"

            # Для Изинга используем проценты от всех доступных точек
            if difficulty == "Лёгкий":
                sample_fraction = 1.0  # 100% точек
            elif difficulty == "Средний":
                sample_fraction = 0.1  # 70% точек
            else:  # Сложный
                sample_fraction = 0.05  # 40% точек

            points = strat.generate(n=n, sample_fraction=sample_fraction)

        elif strategy_name == "Случайное блуждание":
            strat = RandomWalkRepulsionStrategy(step_size=0.12, repulsion_strength=0.2)
            self.current_strategy_name = "Случайное блуждание"
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
            strat = JuliaSetStrategy(**JULIA_PRESET)
            self.current_strategy_name = "Множество Жюлиа"
            points = strat.generate(n)

        elif strategy_name == "Случайная стратегия":
            # Определяем режим: "Угадай" или "Отгадай"
            is_guess_mode = self.guess_radio.isChecked()

            # Температура для гексагональной кристаллизации зависит от уровня сложности
            if n >= 1000:  # Лёгкий
                thermal_noise_hex = 0.002
            elif n >= 300:  # Средний
                thermal_noise_hex = 0.003
            else:  # Сложный (n = 100)
                thermal_noise_hex = 0.004

            # Температура для квадратной кристаллизации зависит от уровня сложности
            if n >= 1000:  # Лёгкий
                thermal_noise_square = 0.002
            elif n >= 300:  # Средний
                thermal_noise_square = 0.005
            else:  # Сложный (n = 100)
                thermal_noise_square = 0.009

            # список всех остальных стратегий (кроме случайных точек) с их названиями
            other_strategies = [
                (SierpinskiStrategy(), "Треугольник Серпинского"),
                (ClustersStrategy(k=7), "Притяжение"),
                (RepulsionStrategy(k=7), "Отталкивание"),
                (BoltzmannStrategy(temperature=0.15), "Гравитация"),
                (CrystallizationStrategy(lattice_type='hexagonal', thermal_noise=thermal_noise_hex), "Кристаллизация (гексагональная)"),
                (CrystallizationStrategy(lattice_type='square', thermal_noise=thermal_noise_square), "Кристаллизация (квадратная)"),
                (IsingStrategy(grid_size=100, T=2.1, J=2.0), "Изинг"),
                (RandomWalkRepulsionStrategy(step_size=0.12, repulsion_strength=0.2), "Случайное блуждание"),
                (PythagorasTreeStrategy(depth=7), "Дерево Пифагора"),
                (KochSnowflakeStrategy(iterations=5), "Снежинка Коха"),
                (BarnsleyFernStrategy(), "Папоротник Барнсли"),
                (JuliaSetStrategy(**JULIA_PRESET), "Множество Жюлиа")
            ]

            if is_guess_mode:
                # Режим "Угадай": 50% случайные точки, 50% другая стратегия
                if np.random.rand() < 0.5:
                    strat = UniformStrategy()
                    self.current_strategy_name = "Случайные точки"
                    points = strat.generate(n)
                else:
                    strat, strategy_display_name = other_strategies[np.random.randint(len(other_strategies))]
                    self.current_strategy_name = strategy_display_name
                    # Специальная обработка для Изинга
                    if strategy_display_name == "Изинг":
                        if difficulty == "Лёгкий":
                            sample_fraction = 1.0  # 100% точек
                        elif difficulty == "Средний":
                            sample_fraction = 0.1  # 10% точек
                        else:  # Сложный
                            sample_fraction = 0.05  # 5% точек
                        points = strat.generate(n=n, sample_fraction=sample_fraction)
                    else:
                        points = strat.generate(n)
            else:
                # Режим "Отгадай": все стратегии равновероятны (включая случайные точки)
                all_strategies = [(UniformStrategy(), "Случайные точки")] + other_strategies
                strat, strategy_display_name = all_strategies[np.random.randint(len(all_strategies))]
                self.current_strategy_name = strategy_display_name
                # Специальная обработка для Изинга
                if strategy_display_name == "Изинг":
                    if difficulty == "Лёгкий":
                        sample_fraction = 1.0  # 100% точек
                    elif difficulty == "Средний":
                        sample_fraction = 0.1  # 10% точек
                    else:  # Сложный
                        sample_fraction = 0.05  # 5% точек
                    points = strat.generate(n=n, sample_fraction=sample_fraction)
                else:
                    points = strat.generate(n)

        # Сохраняем стратегию
        self.current_strategy = strat

        # Получаем параметры анимации из слайдеров
        slider_value = self.speed_slider.value()
        animation_interval = 1010 - slider_value  # инвертируем значение
        step = self.points_per_step_slider.value()  # количество точек за шаг
        point_size = self.point_size_slider.value()  # размер точек

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

    def _redraw_current_frame(self):
        """Перерисовывает текущий кадр анимации (например, при изменении размера точек)"""
        if self.current_points is None:
            return

        # Очищаем и рисуем накопленные точки с текущим размером
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Рисуем точки от 0 до animation_index
        points_to_show = self.current_points[:self.animation_index]
        if len(points_to_show) > 0:
            self.ax.scatter(points_to_show[:, 0], points_to_show[:, 1],
                           s=self.current_point_size, color='blue')

        self.canvas.draw()

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
        point_size = self.point_size_slider.value()
        if hasattr(self.current_strategy, 'get_correct_visualization'):
            self.current_strategy.get_correct_visualization(self.ax, point_size=point_size)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генерация точек")

        # Устанавливаем начальный размер окна почти на весь экран
        screen = QApplication.primaryScreen().geometry()
        self.resize(screen.width(), screen.height() - 100)

        # Устанавливаем минимальный размер окна
        self.setMinimumSize(800, 600)

        # Создаём QStackedWidget для переключения между экранами
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Создаём экраны
        self.title_screen = TitleScreen()
        self.game_screen = GameWindow()
        self.authors_screen = AuthorsScreen()

        # Добавляем экраны в стек
        self.stacked_widget.addWidget(self.title_screen)
        self.stacked_widget.addWidget(self.game_screen)
        self.stacked_widget.addWidget(self.authors_screen)

        # Подключаем кнопки титульного экрана
        self.title_screen.play_button.clicked.connect(self.show_game)
        self.title_screen.authors_button.clicked.connect(self.show_authors)
        self.title_screen.exit_button.clicked.connect(self.close)

        # Подключаем кнопки "Назад"
        self.game_screen.back_button.clicked.connect(self.show_title)
        self.authors_screen.back_button.clicked.connect(self.show_title)

    def show_title(self):
        self.stacked_widget.setCurrentWidget(self.title_screen)

    def show_game(self):
        self.stacked_widget.setCurrentWidget(self.game_screen)

    def show_authors(self):
        self.stacked_widget.setCurrentWidget(self.authors_screen)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
