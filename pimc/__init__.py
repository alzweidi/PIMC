"""
πMC - Monte Carlo Estimation of Pi
===================================

A high-performance, beautifully visualized Monte Carlo simulation
for estimating π with advanced statistical analysis.

Features:
- Vectorized NumPy computation with optional parallel processing
- Interactive web dashboard with real-time visualization
- Publication-quality static and animated plots
- Rich terminal CLI with progress bars and live updates
- Advanced statistical methods with confidence intervals
- Variance reduction techniques (antithetic, stratified sampling)

Example:
    >>> from pimc import MonteCarloPi
    >>> sim = MonteCarloPi(n_points=1_000_000)
    >>> result = sim.run()
    >>> print(f"π ≈ {result.estimate:.10f}")
"""

__version__ = "2.0.0"
__author__ = "Monte Carlo Team"

from pimc.core.simulation import MonteCarloPi, SimulationResult
from pimc.core.statistics import StatisticalAnalyzer
from pimc.visualization.plots import PiVisualizer

__all__ = [
    "MonteCarloPi",
    "SimulationResult",
    "StatisticalAnalyzer",
    "PiVisualizer",
    "__version__",
]
