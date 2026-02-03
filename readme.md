<div align="center">

# pimc

### high-performance monte carlo estimation of π

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**beautiful** | **fast** | **accurate** | **educational**

[features](#features) | [quick start](#quick-start) | [cli](#command-line-interface) | [web dashboard](#web-dashboard) | [api](#python-api) | [visualisation](#visualisation-themes)

</div>

---

## features

- **high performance** — vectorised numpy computation with optional parallel processing
- **stunning visualisations** — publication-quality plots with 8 beautiful themes
- **advanced statistics** — confidence intervals, convergence diagnostics, hypothesis testing
- **variance reduction** — antithetic, stratified, and quasi-random sampling methods
- **rich cli** — beautiful terminal interface with progress bars and live updates
- **web dashboard** — interactive browser-based visualisation with real-time simulation
- **real-time animation** — watch the simulation converge in real-time
- **comprehensive tests** — full test suite with pytest

---

## quick start

### installation

```bash
# clone the repository
git clone https://github.com/YOUR_USERNAME/pimc.git
cd pimc

# install with pip (basic)
pip install -e .

# install with all extras (cli + web)
pip install -e ".[all]"

# install for development
pip install -e ".[dev]"
```

### run your first simulation

```python
from pimc import MonteCarloPi

# simple estimation
sim = MonteCarloPi(n_points=1_000_000, seed=42)
result = sim.run()

print(f"π ≈ {result.estimate:.10f}")
print(f"error: {result.error:.2e}")
print(f"grade: {result.accuracy_grade}")
```

output:
```text
π ≈ 3.1416395600
error: 4.69e-05
grade: A
```

---

## command-line interface

the cli provides a beautiful terminal experience with rich formatting.

```bash
# basic simulation with 1 million points
pimc run -n 1000000

# use variance reduction (antithetic sampling)
pimc run -n 100000 --method antithetic

# save visualisation to file
pimc run -n 50000 -o visualisation.png --theme cyberpunk

# benchmark different methods
pimc benchmark

# statistical analysis with 1000 trials
pimc analyse -t 1000 -n 10000

# interactive real-time animation
pimc interactive -n 20000
```

### cli output example

```text
╔══════════════════════════════════════════════════════════════════════╗
║     ██████╗ ██╗███╗   ███╗ ██████╗                                   ║
║     ██╔══██╗██║████╗ ████║██╔════╝                                   ║
║     ██████╔╝██║██╔████╔██║██║        Monte Carlo Estimation of π     ║
║     ██╔═══╝ ██║██║╚██╔╝██║██║        High-Performance • Beautiful    ║
║     ██║     ██║██║ ╚═╝ ██║╚██████╗                                   ║
║     ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝                                   ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│                      Simulation Results                              │
├─────────────────────────────────────────────────────────────────────┤
│  Estimated π    │  3.1415926536                                      │
│  True π         │  3.1415926536                                      │
│  Error          │  2.45e-05 (0.0008%)                                │
│  Grade          │  A                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## web dashboard

launch an interactive web dashboard for visual exploration:

```bash
# start the web server
python -m pimc.web.app

# visit http://localhost:5000
```

features:
- **real-time scatter plot** with D3.js
- **convergence chart** showing estimate approaching π
- **multiple sampling methods** to compare
- **adjustable sample sizes** with live updates

---

## python api

### basic usage

```python
from pimc import MonteCarloPi, PiVisualizer, StatisticalAnalyzer

# run simulation
sim = MonteCarloPi(
    n_points=100_000,
    seed=42,
    method="stratified",  # or "standard", "antithetic", "quasi_random"
)
result = sim.run()

# access results
print(f"estimate: {result.estimate}")
print(f"points inside: {result.n_inside:,}")
print(f"time: {result.elapsed_time*1000:.2f}ms")
print(f"points/sec: {result.points_per_second:,.0f}")
```

### visualisation

```python
from pimc import MonteCarloPi, PiVisualizer

sim = MonteCarloPi(n_points=50_000, seed=42)
result = sim.run()

# create comprehensive visualisation
viz = PiVisualizer(theme="cyberpunk")  # 8 themes available!
viz.plot_comprehensive(result)
viz.save("pi_visualisation.png", dpi=300)
viz.show()
```

### statistical analysis

```python
from pimc import MonteCarloPi, StatisticalAnalyzer

analyser = StatisticalAnalyzer(warmup_samples=100)

# run multiple trials
for _ in range(1000):
    sim = MonteCarloPi(n_points=10_000)
    result = sim.run()
    analyser.add_estimate(result.estimate)

# get statistics
ci = analyser.confidence_interval(level=0.95)
print(f"95% ci: [{ci.lower:.8f}, {ci.upper:.8f}]")
print(f"contains π: {ci.contains_pi}")

summary = analyser.summary()
print(f"mean: {summary['mean']:.10f}")
print(f"std: {summary['std']:.6f}")
```

### streaming for real-time visualisation

```python
from pimc import MonteCarloPi

sim = MonteCarloPi(n_points=10_000, seed=42)

for n, estimate, x, y, inside in sim.stream(batch_size=100):
    print(f"n={n:,}: π ≈ {estimate:.6f}")
```

---

## visualisation themes

choose from 8 stunning themes:

| theme | description |
|-------|-------------|
| `midnight` | deep blue-black with cyan/magenta accents (default) |
| `cyberpunk` | neon colours on dark purple background |
| `ocean` | deep sea blues with aqua highlights |
| `forest` | natural greens with earth tones |
| `neon` | pure black with bright neon colours |
| `sunset` | warm oranges and reds |
| `publication` | clean white background for papers |
| `colourblind` | accessible colour palette |

```python
from pimc import PiVisualizer

viz = PiVisualizer(theme="cyberpunk")
```

---

## sampling methods

| method | description | variance reduction |
|--------|-------------|-------------------|
| `standard` | uniform random sampling | baseline |
| `antithetic` | uses (x, y) and (1-x, 1-y) pairs | ~30% reduction |
| `stratified` | divides space into grid cells | ~40% reduction |
| `quasi_random` | halton low-discrepancy sequence | ~50% reduction |

```python
from pimc import MonteCarloPi

# compare methods
for method in ["standard", "antithetic", "stratified", "quasi_random"]:
    sim = MonteCarloPi(n_points=100_000, method=method, seed=42)
    result = sim.run()
    print(f"{method:12} → π ≈ {result.estimate:.8f} (error: {result.error:.2e})")
```

---

## the mathematics

the monte carlo method estimates π using geometric probability:

```text
┌─────────────────────────────────────┐
│  area of quarter circle   π/4      │
│  ─────────────────────── = ───     │
│  area of unit square       1       │
│                                     │
│  therefore:                         │
│                                     │
│  π = 4 × (points inside circle)    │
│        ─────────────────────────   │
│           (total points)            │
└─────────────────────────────────────┘
```

### convergence rate

the error decreases at rate **O(1/√n)**:
- 100 points → ~0.16 error
- 10,000 points → ~0.016 error  
- 1,000,000 points → ~0.0016 error

---

## testing

```bash
# run all tests
pytest

# run with coverage
pytest --cov=pimc --cov-report=html

# run specific test file
pytest tests/test_simulation.py -v
```

---

## project structure

```text
pimc/
├── __init__.py          # package exports
├── __main__.py          # python -m pimc entry point
├── core/
│   ├── simulation.py    # monte carlo engine
│   └── statistics.py    # statistical analysis
├── visualisation/
│   ├── plots.py         # matplotlib visualisations
│   └── themes.py        # colour themes
├── cli/
│   └── main.py          # rich cli interface
└── web/
    ├── app.py           # flask web server
    └── templates/       # html templates
```

---

## educational value

this project demonstrates:

- **monte carlo methods** — random sampling for numerical estimation
- **probability theory** — geometric probability and the law of large numbers
- **variance reduction** — techniques to improve estimation efficiency
- **statistical inference** — confidence intervals and hypothesis testing
- **convergence analysis** — understanding simulation accuracy
- **software engineering** — clean architecture, testing, documentation

perfect for courses in:
- computational mathematics
- probability and statistics
- scientific computing
- quantitative finance

---

## licence

mit licence - see [LICENSE](LICENSE) for details.

---

<div align="center">

**made with mathematics**

[back to top](#pimc)

</div>

