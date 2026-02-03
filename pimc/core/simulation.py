"""
High-performance Monte Carlo simulation engine for π estimation.

Supports:
- Vectorized NumPy computation
- Parallel processing with multiprocessing
- Multiple sampling methods (standard, antithetic, stratified)
- Real-time progress callbacks
- Comprehensive result tracking
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from numpy.random import Generator, default_rng


class SamplingMethod(Enum):
    """Available sampling methods for variance reduction."""
    
    STANDARD = "standard"
    ANTITHETIC = "antithetic"
    STRATIFIED = "stratified"
    QUASI_RANDOM = "quasi_random"


@dataclass
class SimulationResult:
    """Complete results from a Monte Carlo π simulation."""
    
    estimate: float
    n_points: int
    n_inside: int
    error: float
    error_percent: float
    
    x_coords: np.ndarray
    y_coords: np.ndarray
    inside_mask: np.ndarray
    
    elapsed_time: float
    points_per_second: float
    
    convergence_data: Optional[ConvergenceData] = None
    sampling_method: SamplingMethod = SamplingMethod.STANDARD
    seed: Optional[int] = None
    
    @property
    def ratio(self) -> float:
        """Ratio of points inside to total points."""
        return self.n_inside / self.n_points
    
    @property
    def theoretical_ratio(self) -> float:
        """Theoretical ratio (π/4)."""
        return np.pi / 4
    
    @property
    def accuracy_grade(self) -> str:
        """Letter grade for accuracy."""
        if self.error < 0.001:
            return "S"  # Superior
        elif self.error < 0.01:
            return "A"
        elif self.error < 0.05:
            return "B"
        elif self.error < 0.1:
            return "C"
        elif self.error < 0.2:
            return "D"
        return "F"
    
    def __repr__(self) -> str:
        return (
            f"SimulationResult(π≈{self.estimate:.10f}, "
            f"error={self.error:.2e}, grade={self.accuracy_grade}, "
            f"n={self.n_points:,})"
        )


@dataclass
class ConvergenceData:
    """Tracks convergence of π estimate over sample sizes."""
    
    sample_sizes: np.ndarray
    estimates: np.ndarray
    errors: np.ndarray
    cumulative_inside: np.ndarray
    
    @property
    def final_estimate(self) -> float:
        return float(self.estimates[-1])
    
    @property
    def final_error(self) -> float:
        return float(self.errors[-1])


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    
    n_points: int = 100_000
    seed: Optional[int] = None
    method: SamplingMethod = SamplingMethod.STANDARD
    n_workers: int = 1
    chunk_size: int = 100_000
    track_convergence: bool = True
    n_convergence_checkpoints: int = 500
    store_points: bool = True
    
    def __post_init__(self):
        if self.n_points < 1:
            raise ValueError("n_points must be positive")
        if self.n_workers < 1:
            raise ValueError("n_workers must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


class MonteCarloPi:
    """
    High-performance Monte Carlo π estimator.
    
    Features:
    - Vectorized computation with NumPy
    - Optional parallel processing
    - Multiple variance reduction techniques
    - Real-time progress tracking
    - Comprehensive statistical output
    
    Example:
        >>> sim = MonteCarloPi(n_points=1_000_000)
        >>> result = sim.run()
        >>> print(f"π ≈ {result.estimate:.10f} (error: {result.error:.2e})")
        
        # With parallel processing
        >>> sim = MonteCarloPi(n_points=10_000_000, n_workers=4)
        >>> result = sim.run()
        
        # With variance reduction
        >>> sim = MonteCarloPi(n_points=100_000, method="antithetic")
        >>> result = sim.run()
    """
    
    def __init__(
        self,
        n_points: int = 100_000,
        seed: Optional[int] = None,
        method: str | SamplingMethod = SamplingMethod.STANDARD,
        n_workers: int = 1,
        track_convergence: bool = True,
        store_points: bool = True,
    ):
        if isinstance(method, str):
            method = SamplingMethod(method)
        
        self.config = SimulationConfig(
            n_points=n_points,
            seed=seed,
            method=method,
            n_workers=n_workers,
            track_convergence=track_convergence,
            store_points=store_points,
        )
        
        self._rng: Optional[Generator] = None
        self._progress_callback: Optional[Callable[[int, int, float], None]] = None
    
    def on_progress(self, callback: Callable[[int, int, float], None]) -> MonteCarloPi:
        """Register a progress callback: (current, total, estimate) -> None."""
        self._progress_callback = callback
        return self
    
    def run(self) -> SimulationResult:
        """Execute the Monte Carlo simulation."""
        start_time = time.perf_counter()
        
        self._rng = default_rng(self.config.seed)
        
        if self.config.n_workers > 1 and self.config.n_points > 100_000:
            x, y, inside = self._run_parallel()
        else:
            x, y, inside = self._run_single()
        
        elapsed = time.perf_counter() - start_time
        
        n_inside = int(np.sum(inside))
        estimate = 4.0 * n_inside / self.config.n_points
        error = abs(estimate - np.pi)
        
        convergence = None
        if self.config.track_convergence:
            convergence = self._compute_convergence(inside)
        
        if not self.config.store_points:
            x = np.array([])
            y = np.array([])
            inside = np.array([])
        
        return SimulationResult(
            estimate=estimate,
            n_points=self.config.n_points,
            n_inside=n_inside,
            error=error,
            error_percent=(error / np.pi) * 100,
            x_coords=x,
            y_coords=y,
            inside_mask=inside,
            elapsed_time=elapsed,
            points_per_second=self.config.n_points / elapsed,
            convergence_data=convergence,
            sampling_method=self.config.method,
            seed=self.config.seed,
        )
    
    def _run_single(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run simulation in a single thread."""
        method = self.config.method
        
        if method == SamplingMethod.STANDARD:
            return self._sample_standard(self.config.n_points)
        elif method == SamplingMethod.ANTITHETIC:
            return self._sample_antithetic(self.config.n_points)
        elif method == SamplingMethod.STRATIFIED:
            return self._sample_stratified(self.config.n_points)
        elif method == SamplingMethod.QUASI_RANDOM:
            return self._sample_quasi_random(self.config.n_points)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _run_parallel(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run simulation with parallel workers."""
        n = self.config.n_points
        n_workers = self.config.n_workers
        chunk_size = n // n_workers
        
        seeds = self._rng.integers(0, 2**31, size=n_workers)
        chunks = [chunk_size] * n_workers
        chunks[-1] += n % n_workers  # Handle remainder
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_worker_simulate, size, int(seed))
                for size, seed in zip(chunks, seeds)
            ]
            
            completed = 0
            for future in as_completed(futures):
                x_chunk, y_chunk, inside_chunk = future.result()
                results.append((x_chunk, y_chunk, inside_chunk))
                completed += 1
                
                if self._progress_callback:
                    current = sum(len(r[0]) for r in results)
                    est = 4.0 * sum(np.sum(r[2]) for r in results) / current
                    self._progress_callback(current, n, est)
        
        x = np.concatenate([r[0] for r in results])
        y = np.concatenate([r[1] for r in results])
        inside = np.concatenate([r[2] for r in results])
        
        return x, y, inside
    
    def _sample_standard(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard uniform random sampling."""
        x = self._rng.uniform(0, 1, n)
        y = self._rng.uniform(0, 1, n)
        inside = (x**2 + y**2) <= 1.0
        return x, y, inside
    
    def _sample_antithetic(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Antithetic sampling for variance reduction.
        
        For each point (x, y), also uses (1-x, 1-y) to reduce variance.
        """
        half_n = n // 2
        
        x1 = self._rng.uniform(0, 1, half_n)
        y1 = self._rng.uniform(0, 1, half_n)
        
        x2 = 1 - x1
        y2 = 1 - y1
        
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        
        if n % 2 == 1:
            x = np.append(x, self._rng.uniform(0, 1))
            y = np.append(y, self._rng.uniform(0, 1))
        
        inside = (x**2 + y**2) <= 1.0
        return x, y, inside
    
    def _sample_stratified(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratified sampling for variance reduction.
        
        Divides the unit square into a grid and samples uniformly within each cell.
        """
        grid_size = int(np.sqrt(n))
        actual_n = grid_size * grid_size
        
        cell_size = 1.0 / grid_size
        
        i_indices = np.repeat(np.arange(grid_size), grid_size)
        j_indices = np.tile(np.arange(grid_size), grid_size)
        
        x = (i_indices + self._rng.uniform(0, 1, actual_n)) * cell_size
        y = (j_indices + self._rng.uniform(0, 1, actual_n)) * cell_size
        
        extra = n - actual_n
        if extra > 0:
            x_extra = self._rng.uniform(0, 1, extra)
            y_extra = self._rng.uniform(0, 1, extra)
            x = np.concatenate([x, x_extra])
            y = np.concatenate([y, y_extra])
        
        inside = (x**2 + y**2) <= 1.0
        return x, y, inside
    
    def _sample_quasi_random(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quasi-random (low-discrepancy) sampling using Halton sequence.
        
        Provides more uniform coverage than pseudo-random sampling.
        """
        x = self._halton_sequence(n, base=2)
        y = self._halton_sequence(n, base=3)
        inside = (x**2 + y**2) <= 1.0
        return x, y, inside
    
    def _halton_sequence(self, n: int, base: int) -> np.ndarray:
        """Generate Halton sequence for quasi-random sampling."""
        result = np.zeros(n)
        for i in range(n):
            f = 1.0
            r = 0.0
            idx = i + 1
            while idx > 0:
                f = f / base
                r = r + f * (idx % base)
                idx = idx // base
            result[i] = r
        return result
    
    def _compute_convergence(self, inside: np.ndarray) -> ConvergenceData:
        """Compute convergence data at logarithmically spaced checkpoints."""
        n = len(inside)
        n_checkpoints = min(self.config.n_convergence_checkpoints, n)
        
        checkpoints = np.unique(
            np.logspace(1, np.log10(n), n_checkpoints).astype(int)
        )
        checkpoints = checkpoints[checkpoints <= n]
        
        cumsum = np.cumsum(inside)
        cumulative_inside = cumsum[checkpoints - 1]
        estimates = 4.0 * cumulative_inside / checkpoints
        errors = np.abs(estimates - np.pi)
        
        return ConvergenceData(
            sample_sizes=checkpoints,
            estimates=estimates,
            errors=errors,
            cumulative_inside=cumulative_inside,
        )
    
    def stream(self, batch_size: int = 1000) -> Iterator[Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Stream simulation results in batches for real-time visualization.
        
        Yields:
            (current_n, current_estimate, x_batch, y_batch, inside_batch)
        """
        self._rng = default_rng(self.config.seed)
        
        total = 0
        n_inside = 0
        
        while total < self.config.n_points:
            batch = min(batch_size, self.config.n_points - total)
            
            x = self._rng.uniform(0, 1, batch)
            y = self._rng.uniform(0, 1, batch)
            inside = (x**2 + y**2) <= 1.0
            
            total += batch
            n_inside += np.sum(inside)
            estimate = 4.0 * n_inside / total
            
            yield total, estimate, x, y, inside


def _worker_simulate(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker function for parallel simulation."""
    rng = default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    inside = (x**2 + y**2) <= 1.0
    return x, y, inside


def quick_estimate(n_points: int = 100_000, seed: Optional[int] = None) -> float:
    """Quick π estimation without storing points."""
    rng = default_rng(seed)
    x = rng.uniform(0, 1, n_points)
    y = rng.uniform(0, 1, n_points)
    return 4.0 * np.sum((x**2 + y**2) <= 1.0) / n_points


def benchmark(
    sample_sizes: Optional[List[int]] = None,
    methods: Optional[List[SamplingMethod]] = None,
    seed: int = 42,
) -> dict:
    """
    Benchmark different sample sizes and methods.
    
    Returns dict with results for each configuration.
    """
    if sample_sizes is None:
        sample_sizes = [1_000, 10_000, 100_000, 1_000_000]
    
    if methods is None:
        methods = list(SamplingMethod)
    
    results = {}
    
    for method in methods:
        results[method.value] = {}
        for n in sample_sizes:
            sim = MonteCarloPi(n_points=n, seed=seed, method=method, store_points=False)
            result = sim.run()
            results[method.value][n] = {
                "estimate": result.estimate,
                "error": result.error,
                "time": result.elapsed_time,
                "points_per_sec": result.points_per_second,
            }
    
    return results
