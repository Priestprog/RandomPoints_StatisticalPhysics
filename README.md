# 🔬 StatPhys - Interactive Statistical Physics Visualizer

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org)

*A sophisticated PyQt6 application for generating and visualizing pseudorandom point structures using mathematical and statistical physics strategies*

[🚀 Features](#-features) • [💻 Installation](#-installation) • [🎯 Usage](#-usage) • [🧮 Theory](#-theoretical-background) • [🔨 Building](#-building-executables)

</div>

---

## 🌟 Overview

StatPhys is an interactive educational tool that brings statistical physics, fractal geometry, and stochastic processes to life through stunning visualizations. Generate point distributions using algorithms ranging from simple uniform distributions to complex fractals and physical models like the Ising model, crystallization, and Langevin dynamics.

**Perfect for students, researchers, and enthusiasts exploring:**
- Monte Carlo methods
- Statistical mechanics & phase transitions
- Fractal geometry & chaos theory
- Stochastic processes & pattern formation

### 👥 Authors

**Alexander Bagrov** & **Alexey Lukyanov**
*Students at Lomonosov Moscow State University*
*Faculty of Computational Mathematics and Cybernetics*
*Department of Mathematical Methods of Forecasting*

---

## ✨ Features

### 🎨 **14+ Generation Strategies**

| Category | Strategies |
|----------|------------|
| **🎲 Basic** | Uniform Distribution, Random Walk, Clusters |
| **⚛️ Statistical Physics** | Ising Model, Langevin Dynamics, Boltzmann Distribution, Attraction/Repulsion |
| **🔮 Crystallization** | Hexagonal Lattice, Square Lattice |
| **🌿 Fractals** | Sierpinski Triangle, Koch Snowflake, Barnsley Fern, Julia Set, Pythagoras Tree |

### 🎛️ **Interactive Controls**
- **📊 Difficulty Levels**: Adjustable point density (Easy/Medium/Hard)
- **🎬 Animation Mode**: Step-by-step generation with dynamic parameter control
- **🔍 Answer Reveal**: Enhanced visualization with detailed theoretical descriptions
- **🎲 Random Strategy**: Automatic random strategy selection
- **⚙️ Real-time Sliders**: Control animation speed, points per step, and point size on-the-fly

### 🖼️ **Modern GUI**
- **🎯 Clean PyQt6 Interface**: Intuitive controls with matplotlib integration
- **📐 Fixed Aspect Ratio**: Consistent square visualizations
- **🎪 Title Screen**: Beautiful landing page with university logos and author information
- **📚 Theory Display**: Pop-up windows with mathematical formulations and physics explanations

---

## 💻 Installation

### 📋 Prerequisites

<div align="center">

| Dependency | Version | Purpose |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white) | 3.8+ | Core runtime |
| ![PyQt6](https://img.shields.io/badge/PyQt6-6.5.0+-41CD52?logo=qt&logoColor=white) | ≥6.5.0 | GUI framework |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.0+-11557c?logo=plotly&logoColor=white) | ≥3.7.0 | Visualization |
| ![NumPy](https://img.shields.io/badge/NumPy-1.24.0+-013243?logo=numpy&logoColor=white) | ≥1.24.0 | Numerical computing |
| ![SciPy](https://img.shields.io/badge/SciPy-1.10.0+-0C55A5?logo=scipy&logoColor=white) | ≥1.10.0 | Scientific algorithms |
| ![PyInstaller](https://img.shields.io/badge/PyInstaller-6.0.0+-0080FF?logo=python&logoColor=white) | ≥6.0.0 | Standalone builds |

</div>

### 🚀 Quick Start

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
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   cd tests
   python main.py
   ```

---

## 🎯 Usage

### Basic Operation

1. **Launch**: Run the application and click "Модель" on the title screen
2. **Select Strategy**: Choose from 14+ generation algorithms in the dropdown
3. **Set Difficulty**: Pick Easy (1000 pts), Medium (300 pts), or Hard (100 pts)
4. **Generate**: Click "Сгенерировать" to create your visualization
5. **Animate** (optional): Click "Анимация" for step-by-step generation
6. **Reveal Answer**: Click "Узнать правильный ответ" for enhanced view with theory

### Animation Controls

During animation, you can dynamically adjust:
- **Speed**: Animation interval (100-2000 ms)
- **Points per step**: How many points to add each frame (1-100)
- **Point size**: Visual size of displayed points (1-20)

### macOS Users - Removing Quarantine

If you download a pre-built `.app` bundle on macOS, you may need to remove the quarantine attribute:

```bash
# Remove quarantine from the app
xattr -d com.apple.quarantine /path/to/StatPhys.app

# Or recursively clear all attributes
xattr -cr /path/to/StatPhys.app
```

After this, you can open the app normally.

---

## 🔨 Building Executables

### Cross-Platform Builds via GitHub Actions

This project includes automated builds for all platforms via GitHub Actions:

- **macOS**: `Random_points-macos-x64.zip` (`.app` bundle)
- **Windows**: `Random_points-windows-x64.zip` (`.exe` executable)
- **Linux**: `Random_points-linux-x64.tar.gz` (standalone binary)

**Triggers:**
- Push to `main` branch
- Pull requests to `main`
- Version tags (e.g., `v1.0.0`) → Creates GitHub Release
- Manual workflow dispatch

See `.github/ACTIONS_README.md` for detailed CI/CD documentation.

### Local Builds

#### macOS
```bash
./build_all.sh
# Output: dist/StatPhys.app
# Package: apps/Random_points-macos-x64.zip
```

#### Windows
```powershell
.\build_all.ps1
# Output: dist\StatPhys.exe
# Package: apps\Random_points-windows-x64.zip
```

#### Linux
```bash
./build_all.sh
# Output: dist/StatPhys
# Package: apps/Random_points-linux-x64.tar.gz
```

All build scripts automatically:
- Create/activate virtual environment
- Install dependencies from `requirements.txt`
- Bundle images and resources via PyInstaller
- Generate platform-specific executable
- Create distribution archives

---

## 🧮 Theoretical Background

### 📚 Statistical Physics & Stochastic Processes

This application demonstrates key concepts in statistical mechanics:
- **Entropy**: Measure of disorder in point distributions
- **Phase Transitions**: Qualitative changes in system state (e.g., Ising model)
- **Correlations**: Statistical interdependence between spatial regions
- **Fluctuations**: Random deviations from mean values
- **Self-Organization**: Spontaneous emergence of ordered structures

### 🎲 Strategy Descriptions

#### **Uniform Distribution**
*Classical random point distribution on [0,1]²*

**Mathematical Description:**
```
x, y ~ U(0, 1)
P(x, y) = const
```
Each coordinate is independently sampled from a uniform distribution. Models ideal gas behavior.

---

#### **Sierpinski Triangle**
*Fractal structure via chaos game algorithm*

**Algorithm:**
1. Start with three vertices of an equilateral triangle
2. Random initial point
3. Each step: pick random vertex, move halfway toward it

**Mathematical Description:**
```
P_{n+1} = (P_n + V_i) / 2
```
where V_i is a randomly chosen vertex

**Properties:**
- Fractal dimension: D = ln(3)/ln(2) ≈ 1.585
- Self-similarity at all scales
- Deterministic fractal from stochastic process

---

#### **Ising Model**
*Classical model for ferromagnetism and phase transitions*

**Hamiltonian:**
```
H = -J Σ_{⟨i,j⟩} s_i·s_j
```
where s_i = ±1 (spins), J > 0 (exchange interaction), ⟨i,j⟩ (nearest neighbors)

**Metropolis Algorithm:**
1. Randomly select a spin
2. Calculate energy change ΔE if flipped
3. Accept with probability P = min(1, exp(-ΔE/(k_B·T)))

**Temperature Regimes:**
- **T ≪ T_c**: Ordered ferromagnetic phase
- **T ≈ T_c**: Critical region (fluctuations at all scales)
- **T ≫ T_c**: Disordered paramagnetic phase

**Critical Temperature (2D):** T_c = 2J/(k_B·ln(1 + √2)) ≈ 2.269 J/k_B

---

#### **Random Walk (Brownian Motion)**
*Models diffusion and particle trajectories*

**Mathematical Description:**
```
x_{n+1} = x_n + δx
y_{n+1} = y_n + δy
δx, δy ~ N(0, σ²)
```

**Physical Meaning:**
- Models diffusion processes
- Mean square displacement: ⟨r²⟩ ~ t
- Demonstrates Brownian motion

---

#### **Langevin Dynamics**
*Particle motion in viscous medium with thermal fluctuations*

**Equation:**
```
m(dv/dt) = -γv + F(r) + √(2γk_B·T)·ξ(t)
```
where γ (friction), F(r) (deterministic force), ξ(t) (white noise)

**Overdamped Limit:**
```
dr/dt = μF(r) + √(2D)·ξ(t)
```
where D = k_B·T/γ (diffusion coefficient), μ = 1/γ (mobility)

---

#### **Boltzmann Distribution**
*Thermal equilibrium in harmonic potential*

**Potential:**
```
U(x, y) = k(x² + y²)/2
```

**Distribution:**
```
P(x, y) ∝ exp(-U(x,y)/(k_B·T))
P(x, y) ∝ exp(-k(x² + y²)/(2k_B·T))
```

**Result:** Gaussian distribution with σ² = k_B·T/k

**Applications:** Optical tweezers, ion traps, harmonic oscillators

---

#### **Attraction / Repulsion**
*Boltzmann distribution with interacting potential*

**Repulsion Potential:**
```
U(r) = Σ_i ε(σ/|r - r_i|)¹²
```
Lennard-Jones repulsion term only

**Attraction Potential:**
```
U(r) = -Σ_i ε·exp(-(|r - r_i|²)/(2σ²))
```

**Distribution:**
```
P(r) ∝ exp(-U(r)/(k_B·T))
```

**Method:** Rejection sampling (Monte Carlo)

**Effects:**
- Repulsion: Excluded volume, uniform spacing
- Attraction: Clustering, high-density regions

---

#### **Crystallization (Hexagonal/Square Lattice)**
*Ordered crystal structures with thermal vibrations*

**Hexagonal Basis:**
```
a₁ = (1, 0)
a₂ = (1/2, √3/2)
```

**Square Basis:**
```
a₁ = (1, 0)
a₂ = (0, 1)
```

**Lattice Sites:**
```
r = n₁·a₁ + n₂·a₂,  n₁, n₂ ∈ ℤ
```

**Properties:**
- Long-range order
- Translational symmetry
- Minimum potential energy
- Applications: Graphene (hex), ionic crystals (square)

---

#### **Koch Snowflake**
*Recursive fractal via line segment subdivision*

**Construction:**
1. Start with equilateral triangle
2. Divide each segment into three parts
3. Replace middle segment with two sides of equilateral triangle
4. Repeat recursively

**After n iterations:**
```
L_n = L₀·(4/3)ⁿ
N_n = 3·4ⁿ (number of segments)
```

**Properties:**
- Fractal dimension: D = ln(4)/ln(3) ≈ 1.262
- Perimeter → ∞, finite area

---

#### **Barnsley Fern**
*Iterated Function System (IFS)*

**Affine Transformations:**
```
f₁: [x, y] → [0, 0.16y]                           (p=0.01) stem
f₂: [x, y] → [0.85x + 0.04y, -0.04x + 0.85y + 1.6]  (p=0.85) main body
f₃: [x, y] → [0.20x - 0.26y, 0.23x + 0.22y + 1.6]   (p=0.07) left branch
f₄: [x, y] → [-0.15x + 0.28y, 0.26x + 0.24y + 0.44] (p=0.07) right branch
```

**Algorithm:**
1. Start at (0, 0)
2. Randomly select transformation by probability
3. Apply transformation
4. Repeat

**Result:** Detailed fern-like natural structure

---

#### **Julia Set**
*Complex dynamics and chaotic iterations*

**Iteration:**
```
z_{n+1} = z_n² + c
```
where c is a fixed complex parameter (e.g., c = -0.7 + 0.27015i)

**Classification:**
- Point z₀ ∈ Julia set if sequence {z_n} remains bounded
- Escape criterion: |z_n| > R (typically R = 2)

**Generation:**
1. Pick random point in complex plane
2. Iterate mapping
3. If sequence doesn't escape in N iterations → save point

**Properties:**
- Fractal boundary
- Self-similarity
- Chaotic dynamics
- Dimension D ∈ [1, 2)

---

#### **Pythagoras Tree**
*Recursive construction with squares on right triangles*

**Algorithm:**
1. Start with base square
2. Construct right triangle on top edge
3. Build two smaller squares on the legs
4. Recursively repeat for new squares

**For isosceles triangle (45°-45°-90°):**
```
Scale = 1/√2 per level
Rotation angle = ±45°
```

**Properties:**
- Recursive structure
- Self-similarity
- Resembles tree branching

---

#### **Clusters**
*Spatially heterogeneous Gaussian distributions*

**Mathematical Description:**
```
Cluster centers: C_i = (x_i, y_i), i = 1..k
Points around center j:
x ~ N(x_j, σ²)
y ~ N(y_j, σ²)
```

**Parameters:**
- Number of clusters: k
- Variance: σ (controls cluster size)

**Applications:** Spatial heterogeneity, k-means-like structures

---

## 🛠️ Architecture

### 🏗️ Project Structure

```
StatPhys/
├── tests/
│   ├── main.py                      # PyQt6 GUI application
│   ├── strategies.py                # Algorithm implementations
│   ├── *.png, *.jpg                 # Logos and preset images
├── .github/
│   ├── workflows/
│   │   └── build.yml                # CI/CD pipeline
│   └── ACTIONS_README.md            # GitHub Actions documentation
├── .venv/                           # Virtual environment
├── requirements.txt                 # Python dependencies
├── statphys.spec                    # PyInstaller config (macOS)
├── statphys_linux.spec              # PyInstaller config (Linux)
├── statphys_windows.spec            # PyInstaller config (Windows)
├── build_all.sh                     # macOS/Linux build script
├── build_all.ps1                    # Windows build script
├── CLAUDE.md                        # Development guidelines
└── README.md                        # This file
```

### 🎭 Strategy Pattern

All algorithms implement a unified interface:

```python
class Strategy:
    def generate(self, n: int) -> np.ndarray:
        """Generate n points in [0,1]² space"""
        return points  # shape (n, 2)

    def get_correct_visualization(self, ax) -> None:
        """Enhanced visualization (optional)"""
        pass
```

This design allows easy addition of new strategies without modifying the GUI code.

---

## 🧪 Educational Value

### 🎓 Learning Objectives

- **Statistical Mechanics**: Phase transitions, equilibrium distributions, correlation functions
- **Fractal Geometry**: Self-similarity, fractal dimensions, recursive algorithms
- **Computational Physics**: Monte Carlo methods, numerical integration, rejection sampling
- **Stochastic Processes**: Random walks, Langevin equations, diffusion
- **Complex Systems**: Emergence, pattern formation, critical phenomena

### 📖 Research Applications

- **Materials Science**: Spin systems, magnetic materials, phase diagrams
- **Mathematics**: Dynamical systems, chaos theory, complex analysis
- **Computer Graphics**: Procedural generation, natural patterns, textures
- **Data Science**: Clustering algorithms, spatial statistics, point processes

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **🐛 Report Issues**: Found a bug? Open an issue!
2. **💡 Suggest Features**: New algorithms or visualizations
3. **📖 Improve Docs**: Help others understand the code
4. **🔧 Submit PRs**: Code improvements and new strategies

---

## 📚 References

1. Landau L.D., Lifshitz E.M. *Statistical Physics*. Nauka, 1976.
2. Newman M.E.J., Barkema G.T. *Monte Carlo Methods in Statistical Physics*. Oxford University Press, 1999.
3. Mandelbrot B.B. *The Fractal Geometry of Nature*. W.H. Freeman, 1982.
4. Gardiner C.W. *Handbook of Stochastic Methods*. Springer, 2004.
5. Peitgen H.-O., Jürgens H., Saupe D. *Chaos and Fractals*. Springer, 2004.

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgments

- **PyQt6**: Modern cross-platform GUI framework
- **Matplotlib**: Publication-quality plotting library
- **NumPy/SciPy**: Scientific computing foundations
- **Statistical Physics Community**: Inspiration for physical models

---

<div align="center">

**Developed at Lomonosov Moscow State University**
*Faculty of Computational Mathematics and Cybernetics*
*Department of Mathematical Methods of Forecasting*

**Made with ❤️ for science and education**

</div>
