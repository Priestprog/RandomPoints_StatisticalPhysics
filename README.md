# 🔬 StatPhys - Interactive Statistical Physics Visualizer

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org)

*A sophisticated PyQt6 application for generating and visualizing various point distributions using mathematical and physical strategies*

[🚀 Features](#-features) • [📖 Usage](#-usage) • [🧮 Algorithms](#-algorithms) • [💻 Installation](#-installation) • [🎯 Examples](#-examples)

</div>

---

## 🌟 Overview

StatPhys is an interactive educational tool that brings statistical physics and fractal geometry to life through stunning visualizations. Generate point distributions using a variety of algorithms ranging from simple uniform distributions to complex fractals and physical models like the Ising model and Langevin dynamics.

<div align="center">

*Perfect for students, researchers, and enthusiasts exploring the intersection of physics, mathematics, and computation*

</div>

## ✨ Features

### 🎨 **Visualization Strategies**
- **🎲 Basic Patterns**: Uniform, Random Walk, Sierpinski Triangle, Clusters
- **⚛️ Physics Models**: Ising Model, Correlated Fields, Langevin Dynamics
- **🌿 Fractals**: Koch Snowflake, Barnsley Fern, Julia Set, Pythagoras Tree

### 🎛️ **Interactive Controls**
- **📊 Difficulty Levels**: Easy (1000 pts), Medium (300 pts), Hard (100 pts)
- **🔄 Random Strategy Selection**: Surprise yourself with random patterns
- **👁️ Answer Reveal**: See the enhanced visualization with strategy details

### 🖼️ **Modern GUI**
- **🎯 Clean PyQt6 Interface**: Intuitive controls with matplotlib integration
- **📐 Fixed Aspect Ratio**: Consistent square visualizations
- **🎪 Real-time Generation**: Instant pattern creation and display

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyQt6
matplotlib
numpy
scipy
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/StatPhys.git
   cd StatPhys
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install PyQt6 matplotlib numpy scipy
   ```

4. **Run the application**
   ```bash
   cd tests
   python main.py
   ```

## 🎯 Usage

### Basic Operation
1. **Select Difficulty**: Choose point density (Easy/Medium/Hard)
2. **Pick Strategy**: Select from 11 different generation algorithms
3. **Generate**: Click "Сгенерировать" to create your visualization
4. **Reveal Answer**: Click "Узнать правильный ответ" for enhanced view

### Strategy Guide

| Category | Strategy | Description |
|----------|----------|-------------|
| 🎲 **Basic** | Uniform | Pure random distribution |
| 🎲 **Basic** | Sierpinski | Classic fractal triangle |
| 🎲 **Basic** | Clusters | K-means style groupings |
| ⚛️ **Physics** | Ising Model | Spin system equilibration |
| ⚛️ **Physics** | Correlated Field | Gaussian filtered noise |
| ⚛️ **Physics** | Langevin | Brownian motion with drift |
| 🌿 **Fractals** | Koch Snowflake | Recursive geometric pattern |
| 🌿 **Fractals** | Barnsley Fern | IFS-based natural form |
| 🌿 **Fractals** | Julia Set | Complex dynamics visualization |
| 🌿 **Fractals** | Pythagoras Tree | Recursive square construction |

## 🧮 Algorithms

### 🔬 Statistical Physics Models

#### **Ising Model** (`IsingStrategy`)
- **Method**: Metropolis Monte Carlo algorithm
- **Parameters**: Grid size (100×100), Temperature (2.5), Steps (5000)
- **Physics**: Simulates ferromagnetic phase transitions
```python
# Energy calculation: E = -J * Σ(si * sj)
# Acceptance probability: P = exp(-ΔE/kT)
```

#### **Correlated Field** (`CorrelatedFieldStrategy`)
- **Method**: Gaussian noise filtering with scipy
- **Parameters**: Grid (150×150), Correlation length (σ=5.0)
- **Physics**: Spatially correlated random fields
```python
# Generate white noise → Apply Gaussian filter → Sample proportionally
```

#### **Langevin Dynamics** (`LangevinStrategy`)
- **Method**: Stochastic differential equation integration
- **Parameters**: Drift velocity, Diffusion coefficient, Time step
- **Physics**: Brownian motion in potential fields
```python
# dx = v*dt + √(2D*dt) * η(t)
```

### 🌿 Fractal Generators

#### **Sierpinski Triangle** (`SierpinskiStrategy`)
- **Method**: Chaos game algorithm
- **Vertices**: Equilateral triangle corners
- **Rule**: Jump halfway to random vertex

#### **Koch Snowflake** (`KochSnowflakeStrategy`)
- **Method**: Recursive line segment subdivision
- **Iterations**: 4 levels of detail
- **Geometry**: 60° triangular protrusions

#### **Barnsley Fern** (`BarnsleyFernStrategy`)
- **Method**: Iterated Function System (IFS)
- **Transformations**: 4 affine mappings with probabilities
- **Result**: Natural fern-like structure

#### **Julia Set** (`JuliaSetStrategy`)
- **Method**: Complex dynamics escape-time algorithm
- **Function**: z → z² + c (c = -0.7 + 0.27015i)
- **Criterion**: |z| > 2 escape threshold

## 🎨 Visual Examples

<div align="center">

| Pattern | Difficulty | Points | Characteristics |
|---------|------------|--------|-----------------|
| 🔺 Sierpinski | Easy | 1000 | Self-similar fractal |
| ⚛️ Ising | Medium | 300 | Phase separation |
| 🌿 Barnsley Fern | Hard | 100 | Natural recursion |
| 🏔️ Julia Set | Easy | 1000 | Complex boundaries |

*Each pattern reveals unique mathematical beauty at different scales*

</div>

## 🛠️ Architecture

### 🏗️ Core Components

```
StatPhys/
├── tests/
│   ├── main.py           # PyQt6 GUI application
│   └── strategies.py     # Algorithm implementations
├── .venv/               # Virtual environment
├── CLAUDE.md           # Development guidelines
└── README.md           # This file
```

### 🎭 Strategy Pattern

All algorithms implement a unified interface:
```python
class Strategy:
    def generate(self, n: int) -> np.ndarray:
        """Generate n points in [0,1]² space"""
        pass

    def get_correct_visualization(self, ax) -> None:
        """Enhanced visualization (optional)"""
        pass
```

### 🎨 GUI Integration

- **Strategy Selection**: Dropdown with 11 algorithms
- **Parameter Control**: Difficulty affects point density
- **Real-time Rendering**: Matplotlib canvas integration
- **Answer System**: Enhanced visualizations on demand

## 🔧 Development

### 🏃‍♂️ Running Tests
```bash
cd tests
python main.py  # Interactive testing via GUI
```

### 🎯 Adding New Strategies

1. **Implement** the strategy class in `strategies.py`
2. **Import** in `main.py`
3. **Add** to strategy dropdown
4. **Integrate** in `generate_points()` method

Example:
```python
class NewStrategy:
    def generate(self, n):
        # Your algorithm here
        return np.random.rand(n, 2)  # Must return (n,2) array
```

## 📚 Educational Value

### 🎓 Learning Objectives
- **Statistical Mechanics**: Phase transitions, correlation functions
- **Fractal Geometry**: Self-similarity, dimension theory
- **Computational Physics**: Monte Carlo methods, numerical integration
- **Complex Systems**: Emergence, pattern formation

### 🧪 Research Applications
- **Materials Science**: Spin systems, phase diagrams
- **Mathematics**: Dynamical systems, chaos theory
- **Computer Graphics**: Procedural generation, natural patterns
- **Data Science**: Clustering algorithms, dimensionality reduction

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Report Issues**: Found a bug? Let us know!
2. **💡 Suggest Features**: New algorithms or visualizations
3. **📖 Improve Docs**: Help others understand the code
4. **🔧 Submit PRs**: Code improvements and new strategies

### 🏗️ Development Setup
```bash
git clone https://github.com/yourusername/StatPhys.git
cd StatPhys
python -m venv .venv
source .venv/bin/activate
pip install PyQt6 matplotlib numpy scipy
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **PyQt6**: Modern cross-platform GUI framework
- **Matplotlib**: Publication-quality plotting library
- **NumPy/SciPy**: Scientific computing foundations
- **Statistical Physics Community**: Inspiration for physical models

---

<div align="center">

**Made with ❤️ for science and education**

*Star ⭐ this repo if you find it useful!*

</div>