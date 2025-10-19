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

        "Притяжение": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Распределение молекул в поле притягивающих центров с потенциалом взаимодействия <i>U</i>(<i>r</i>).</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Каноническое распределение Больцмана в термодинамическом равновесии:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                ρ(<i>r</i>) ∝ exp(-<i>U</i>(<i>r</i>) / k<sub>B</sub>T)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где <i>U</i>(<i>r</i>) = -Σ<sub>k</sub> <i>ε</i> / |<i>r</i> - <i>r</i><sub>k</sub>|² — энергия взаимодействия с центрами притяжения, <i>T</i> — температура системы.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Молекулы группируются вокруг центров притяжения, образуя области повышенной плотности.</p>
        """,

        "Отталкивание": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Распределение частиц в поле отталкивающих центров с положительным потенциалом взаимодействия.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Статистическая механика систем с отталкиванием описывается распределением Больцмана:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                ρ(<i>r</i>) ∝ exp(-<i>U</i>(<i>r</i>) / k<sub>B</sub>T)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где <i>U</i>(<i>r</i>) = Σ<sub>k</sub> <i>ε</i> / |<i>r</i> - <i>r</i><sub>k</sub>|² — положительная энергия отталкивания от центров, <i>T</i> — температура.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Частицы избегают областей вблизи центров отталкивания, формируя области пониженной плотности (excluded volume effect).</p>
        """,

        "Больцмана": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Барометрическая формула — распределение частиц в поле тяжести.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Плотность газа в гравитационном поле убывает с высотой по закону:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                ρ(<i>h</i>) = ρ<sub>0</sub> exp(-<i>mgh</i> / k<sub>B</sub>T)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где <i>m</i> — масса частицы, <i>g</i> — ускорение свободного падения, <i>h</i> — высота, <i>T</i> — температура.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">При высокой температуре распределение более равномерное, при низкой — частицы концентрируются у основания.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Описывает распределение молекул в атмосфере, седиментацию коллоидных частиц.</p>
        """,

        "Кристаллизация (гексагональная)": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Упорядоченная кристаллическая структура с тепловыми колебаниями атомов около положений равновесия.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Гексагональная решётка (структура льда I<sub>h</sub>) с пространственной группой симметрии P6<sub>3</sub>/mmc.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Среднеквадратичное смещение атома от узла решётки:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                ⟨u²⟩ = (ℏ / 2mω) coth(ℏω / 2k<sub>B</sub>T)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где ω — частота колебаний, <i>T</i> — температура.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">При T → 0 колебания минимальны (квантовые нулевые колебания), при T >> T<sub>D</sub> — классический предел.</p>
        """,

        "Кристаллизация (квадратная)": f"""
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Упорядоченная кристаллическая структура с тепловыми колебаниями атомов около положений равновесия.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Простая кубическая решётка (структура типа Po, NaCl) с пространственной группой симметрии Pm3̄m.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Среднеквадратичное смещение атома от узла решётки:</p>
            <p style="text-align: center; font-family: monospace; font-size: {FORMULA_TEXT_SIZE}px; padding: 15px;">
                ⟨u²⟩ = (ℏ / 2mω) coth(ℏω / 2k<sub>B</sub>T)
            </p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">где ω — частота колебаний, <i>T</i> — температура.</p>
            <p style="font-size: {DESCRIPTION_TEXT_SIZE}px;">Квадратная решётка имеет координационное число 4 (в отличие от гексагональной с координационным числом 6).</p>
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
            "Больцмана",
            "Кристаллизация (гексагон.)",
            "Кристаллизация (квадрат.)",
            "Изинг",
            "Коррелированное поле",
            "Ланжевен",
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
        self.speed_label = QLabel("Скорость анимации: 50 мс")
        self.speed_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(500)
        self.speed_slider.setValue(50)
        self.speed_slider.setMinimumHeight(int(25 * SCALE))
        self.speed_slider.setStyleSheet(f"QSlider::groove:horizontal {{ height: {int(8 * SCALE)}px; }} QSlider::handle:horizontal {{ width: {int(18 * SCALE)}px; height: {int(18 * SCALE)}px; }}")
        self.speed_slider.valueChanged.connect(lambda value: self.speed_label.setText(f"Скорость анимации: {value} мс"))

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
        self.points_per_step_slider.valueChanged.connect(lambda value: self.points_per_step_label.setText(f"Точек за шаг: {value}"))

        left_layout.addWidget(self.points_per_step_label)
        left_layout.addWidget(self.points_per_step_slider)

        # слайдер для размера точек
        self.point_size_label = QLabel("Размер точек: 3")
        self.point_size_label.setStyleSheet(f"font-size: {int(12 * SCALE)}px; font-weight: bold;")
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(20)
        self.point_size_slider.setValue(3)
        self.point_size_slider.setMinimumHeight(int(25 * SCALE))
        self.point_size_slider.setStyleSheet(f"QSlider::groove:horizontal {{ height: {int(8 * SCALE)}px; }} QSlider::handle:horizontal {{ width: {int(18 * SCALE)}px; height: {int(18 * SCALE)}px; }}")
        self.point_size_slider.valueChanged.connect(lambda value: self.point_size_label.setText(f"Размер точек: {value}"))

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

        elif strategy_name == "Больцмана":
            strat = BoltzmannStrategy(temperature=0.15)
            self.current_strategy_name = "Больцмана"
            points = strat.generate(n)

        elif strategy_name == "Кристаллизация (гексагон.)":
            strat = CrystallizationStrategy(lattice_type='hexagonal', thermal_noise=0.003)
            self.current_strategy_name = "Кристаллизация (гексагональная)"
            points = strat.generate(n)

        elif strategy_name == "Кристаллизация (квадрат.)":
            strat = CrystallizationStrategy(lattice_type='square', thermal_noise=0.003)
            self.current_strategy_name = "Кристаллизация (квадратная)"
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

        elif strategy_name == "Случайная стратегия":
            # Определяем режим: "Угадай" или "Отгадай"
            is_guess_mode = self.guess_radio.isChecked()

            # список всех остальных стратегий (кроме случайных точек) с их названиями
            other_strategies = [
                (SierpinskiStrategy(), "Треугольник Серпинского"),
                (ClustersStrategy(k=7), "Притяжение"),
                (RepulsionStrategy(k=7), "Отталкивание"),
                (BoltzmannStrategy(temperature=0.15), "Больцмана"),
                (CrystallizationStrategy(lattice_type='hexagonal', thermal_noise=0.003), "Кристаллизация (гексагональная)"),
                (CrystallizationStrategy(lattice_type='square', thermal_noise=0.003), "Кристаллизация (квадратная)"),
                (IsingStrategy(grid_size=100, T=2.5, steps=3000), "Изинг"),
                (CorrelatedFieldStrategy(grid_size=150, sigma=5.0), "Коррелированное поле"),
                (LangevinStrategy(v=(0.005, 0.0), D=0.002), "Ланжевен"),
                (PythagorasTreeStrategy(depth=7), "Дерево Пифагора"),
                (KochSnowflakeStrategy(iterations=5), "Снежинка Коха"),
                (BarnsleyFernStrategy(), "Папоротник Барнсли"),
                (JuliaSetStrategy(c=-0.7 + 0.27015j, max_iter=200), "Множество Жюлиа")
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
                    points = strat.generate(n)
            else:
                # Режим "Отгадай": все стратегии равновероятны (включая случайные точки)
                all_strategies = [(UniformStrategy(), "Случайные точки")] + other_strategies
                strat, strategy_display_name = all_strategies[np.random.randint(len(all_strategies))]
                self.current_strategy_name = strategy_display_name
                points = strat.generate(n)

        # Сохраняем стратегию
        self.current_strategy = strat

        # Получаем параметры анимации из слайдеров
        animation_interval = self.speed_slider.value()  # интервал в мс
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
