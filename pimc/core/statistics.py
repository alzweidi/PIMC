"""
Advanced statistical analysis for Monte Carlo π estimation.

Includes:
- Running statistics with online algorithms
- Confidence intervals and hypothesis testing
- Convergence diagnostics
- Outlier detection and filtering
- Error analysis and variance estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Confidence interval for π estimate."""
    
    lower: float
    upper: float
    level: float  # e.g., 0.95 for 95%
    center: float
    margin: float
    
    @property
    def width(self) -> float:
        return self.upper - self.lower
    
    @property
    def contains_pi(self) -> bool:
        return self.lower <= np.pi <= self.upper
    
    def __repr__(self) -> str:
        return f"CI({self.level*100:.0f}%): [{self.lower:.6f}, {self.upper:.6f}]"


@dataclass
class ConvergenceDiagnostics:
    """Results from convergence analysis."""
    
    has_converged: bool
    estimated_pi: float
    standard_error: float
    effective_sample_size: float
    autocorrelation: float
    geweke_statistic: float
    geweke_pvalue: float
    burn_in_suggested: int
    
    @property
    def is_stationary(self) -> bool:
        """Check if chain appears stationary (Geweke test p > 0.05)."""
        return self.geweke_pvalue > 0.05


@dataclass
class RunningStats:
    """Online algorithm for computing running statistics."""
    
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared deviations
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, value: float) -> None:
        """Update statistics with a new value using Welford's algorithm."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def update_batch(self, values: np.ndarray) -> None:
        """Update with a batch of values."""
        for v in values:
            self.update(float(v))
    
    @property
    def variance(self) -> float:
        """Population variance."""
        return self.m2 / self.n if self.n > 0 else 0.0
    
    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel's correction)."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0
    
    @property
    def std(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)
    
    @property
    def sample_std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.sample_variance)
    
    @property
    def standard_error(self) -> float:
        """Standard error of the mean."""
        return self.sample_std / math.sqrt(self.n) if self.n > 0 else 0.0
    
    @property
    def range(self) -> float:
        """Range of values."""
        return self.max_val - self.min_val


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for Monte Carlo simulations.
    
    Features:
    - Multiple confidence interval methods
    - Convergence diagnostics
    - Outlier detection with z-score filtering
    - Variance estimation for error bounds
    - Batch statistical analysis
    
    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> for trial in range(1000):
        ...     pi_est = run_simulation()
        ...     analyzer.add_estimate(pi_est)
        >>> ci = analyzer.confidence_interval()
        >>> print(f"π ∈ {ci}")
    """
    
    def __init__(self, warmup_samples: int = 100):
        self.warmup_samples = warmup_samples
        self.estimates: List[float] = []
        self.filtered_estimates: List[float] = []
        self.running_stats = RunningStats()
        self.filtered_stats = RunningStats()
        
        self._running_means: List[float] = []
        self._upper_bounds: List[float] = []
        self._lower_bounds: List[float] = []
    
    def add_estimate(self, estimate: float, z_threshold: float = 2.5) -> bool:
        """
        Add a π estimate, optionally filtering outliers.
        
        Args:
            estimate: The π estimate to add
            z_threshold: Z-score threshold for outlier rejection
            
        Returns:
            True if estimate was accepted, False if filtered out
        """
        self.estimates.append(estimate)
        self.running_stats.update(estimate)
        
        if len(self.estimates) <= self.warmup_samples:
            self.filtered_estimates.append(estimate)
            self.filtered_stats.update(estimate)
            self._update_bounds()
            return True
        
        if self.filtered_stats.std > 0:
            z_score = abs(estimate - self.filtered_stats.mean) / self.filtered_stats.std
            if z_score > z_threshold:
                return False
        
        self.filtered_estimates.append(estimate)
        self.filtered_stats.update(estimate)
        self._update_bounds()
        return True
    
    def add_estimates(self, estimates: List[float], z_threshold: float = 2.5) -> int:
        """Add multiple estimates. Returns count of accepted estimates."""
        accepted = 0
        for est in estimates:
            if self.add_estimate(est, z_threshold):
                accepted += 1
        return accepted
    
    def _update_bounds(self) -> None:
        """Update running confidence bounds."""
        n = len(self.filtered_estimates)
        mean = self.filtered_stats.mean
        se = self.filtered_stats.standard_error
        
        self._running_means.append(mean)
        self._upper_bounds.append(mean + 1.96 * se)
        self._lower_bounds.append(mean - 1.96 * se)
    
    def confidence_interval(
        self,
        level: float = 0.95,
        method: str = "t",
        use_filtered: bool = True,
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for π estimate.
        
        Args:
            level: Confidence level (default 0.95 for 95%)
            method: "t" for t-distribution, "normal" for normal
            use_filtered: Use filtered estimates if True
            
        Returns:
            ConfidenceInterval object
        """
        data = self.filtered_estimates if use_filtered else self.estimates
        n = len(data)
        
        if n < 2:
            mean = data[0] if n == 1 else 0.0
            return ConfidenceInterval(
                lower=mean, upper=mean, level=level,
                center=mean, margin=0.0
            )
        
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        
        if method == "t":
            alpha = 1 - level
            t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_crit * se
        else:  # normal
            alpha = 1 - level
            z_crit = stats.norm.ppf(1 - alpha / 2)
            margin = z_crit * se
        
        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            level=level,
            center=mean,
            margin=margin,
        )
    
    def bootstrap_ci(
        self,
        level: float = 0.95,
        n_bootstrap: int = 10000,
        use_filtered: bool = True,
    ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval (non-parametric).
        
        More robust when normality assumption is questionable.
        """
        data = np.array(self.filtered_estimates if use_filtered else self.estimates)
        n = len(data)
        
        if n < 2:
            mean = float(data[0]) if n == 1 else 0.0
            return ConfidenceInterval(
                lower=mean, upper=mean, level=level,
                center=mean, margin=0.0
            )
        
        rng = np.random.default_rng(42)
        bootstrap_means = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)
        
        alpha = 1 - level
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
        center = float(np.mean(data))
        
        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            level=level,
            center=center,
            margin=(upper - lower) / 2,
        )
    
    def geweke_test(
        self,
        first_fraction: float = 0.1,
        last_fraction: float = 0.5,
        use_filtered: bool = True,
    ) -> Tuple[float, float]:
        """
        Geweke convergence diagnostic.
        
        Compares means of first and last portions of the chain.
        Returns (z-statistic, p-value).
        
        If p-value > 0.05, the chain has likely converged.
        """
        data = np.array(self.filtered_estimates if use_filtered else self.estimates)
        n = len(data)
        
        if n < 20:
            return 0.0, 1.0
        
        n_first = int(n * first_fraction)
        n_last = int(n * last_fraction)
        
        first = data[:n_first]
        last = data[-n_last:]
        
        mean_first = np.mean(first)
        mean_last = np.mean(last)
        var_first = np.var(first, ddof=1)
        var_last = np.var(last, ddof=1)
        
        se = np.sqrt(var_first / n_first + var_last / n_last)
        
        if se < 1e-10:
            return 0.0, 1.0
        
        z = (mean_first - mean_last) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return float(z), float(p)
    
    def effective_sample_size(self, use_filtered: bool = True) -> float:
        """
        Compute effective sample size accounting for autocorrelation.
        
        ESS < n suggests correlated samples; ESS ≈ n suggests independence.
        """
        data = np.array(self.filtered_estimates if use_filtered else self.estimates)
        n = len(data)
        
        if n < 10:
            return float(n)
        
        max_lag = min(n // 2, 100)
        data_centered = data - np.mean(data)
        
        acf = np.correlate(data_centered, data_centered, mode='full')
        acf = acf[n - 1:n + max_lag]
        acf = acf / acf[0]
        
        tau = 1.0
        for k in range(1, max_lag):
            if acf[k] < 0.05:
                break
            tau += 2 * acf[k]
        
        ess = n / tau
        return max(1.0, ess)
    
    def convergence_diagnostics(self, use_filtered: bool = True) -> ConvergenceDiagnostics:
        """Comprehensive convergence analysis."""
        data = np.array(self.filtered_estimates if use_filtered else self.estimates)
        n = len(data)
        
        if n < 10:
            return ConvergenceDiagnostics(
                has_converged=False,
                estimated_pi=np.mean(data) if n > 0 else 0.0,
                standard_error=0.0,
                effective_sample_size=float(n),
                autocorrelation=0.0,
                geweke_statistic=0.0,
                geweke_pvalue=1.0,
                burn_in_suggested=0,
            )
        
        ess = self.effective_sample_size(use_filtered)
        z_stat, p_val = self.geweke_test(use_filtered=use_filtered)
        
        data_centered = data - np.mean(data)
        if np.std(data) > 1e-10:
            autocorr = float(np.corrcoef(data[:-1], data[1:])[0, 1])
        else:
            autocorr = 0.0
        
        burn_in = max(100, int(n * 0.1))
        
        has_converged = (
            p_val > 0.05 and
            ess > n * 0.1 and
            abs(np.mean(data) - np.pi) < 0.1
        )
        
        se = np.std(data, ddof=1) / np.sqrt(ess)
        
        return ConvergenceDiagnostics(
            has_converged=has_converged,
            estimated_pi=float(np.mean(data)),
            standard_error=float(se),
            effective_sample_size=ess,
            autocorrelation=autocorr if not np.isnan(autocorr) else 0.0,
            geweke_statistic=z_stat,
            geweke_pvalue=p_val,
            burn_in_suggested=burn_in,
        )
    
    def hypothesis_test(
        self,
        null_value: float = np.pi,
        alternative: str = "two-sided",
        use_filtered: bool = True,
    ) -> Tuple[float, float, bool]:
        """
        Test H0: mean = null_value vs H1: mean ≠ null_value.
        
        Args:
            null_value: Value under null hypothesis (default: π)
            alternative: "two-sided", "less", or "greater"
            use_filtered: Use filtered estimates
            
        Returns:
            (t-statistic, p-value, reject_null at α=0.05)
        """
        data = np.array(self.filtered_estimates if use_filtered else self.estimates)
        n = len(data)
        
        if n < 2:
            return 0.0, 1.0, False
        
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        
        if se < 1e-10:
            return 0.0, 1.0, False
        
        t_stat = (mean - null_value) / se
        
        if alternative == "two-sided":
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        elif alternative == "less":
            p_val = stats.t.cdf(t_stat, df=n - 1)
        else:  # greater
            p_val = 1 - stats.t.cdf(t_stat, df=n - 1)
        
        return float(t_stat), float(p_val), p_val < 0.05
    
    def summary(self, use_filtered: bool = True) -> dict:
        """Generate comprehensive summary statistics."""
        stats_obj = self.filtered_stats if use_filtered else self.running_stats
        ci = self.confidence_interval(use_filtered=use_filtered)
        diagnostics = self.convergence_diagnostics(use_filtered=use_filtered)
        
        return {
            "n_total": len(self.estimates),
            "n_filtered": len(self.filtered_estimates),
            "acceptance_rate": len(self.filtered_estimates) / max(1, len(self.estimates)),
            "mean": stats_obj.mean,
            "std": stats_obj.sample_std,
            "standard_error": stats_obj.standard_error,
            "min": stats_obj.min_val,
            "max": stats_obj.max_val,
            "range": stats_obj.range,
            "ci_95_lower": ci.lower,
            "ci_95_upper": ci.upper,
            "ci_contains_pi": ci.contains_pi,
            "error_from_pi": abs(stats_obj.mean - np.pi),
            "error_percent": abs(stats_obj.mean - np.pi) / np.pi * 100,
            "has_converged": diagnostics.has_converged,
            "effective_sample_size": diagnostics.effective_sample_size,
        }
    
    @property
    def running_means(self) -> np.ndarray:
        """Get running mean values for plotting."""
        return np.array(self._running_means)
    
    @property
    def confidence_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get upper and lower confidence bounds for plotting."""
        return np.array(self._lower_bounds), np.array(self._upper_bounds)
    
    @property
    def best_estimate(self) -> float:
        """Best estimate of π (mean of filtered estimates)."""
        return self.filtered_stats.mean
    
    def __repr__(self) -> str:
        return (
            f"StatisticalAnalyzer(n={len(self.estimates)}, "
            f"filtered={len(self.filtered_estimates)}, "
            f"π̂={self.best_estimate:.8f})"
        )


def theoretical_variance(n_points: int) -> float:
    """
    Theoretical variance of Monte Carlo π estimator.
    
    Var(π̂) = 16 * p * (1-p) / n, where p = π/4
    """
    p = np.pi / 4
    return 16 * p * (1 - p) / n_points


def theoretical_standard_error(n_points: int) -> float:
    """Theoretical standard error of Monte Carlo π estimator."""
    return np.sqrt(theoretical_variance(n_points))


def required_samples_for_precision(target_error: float, confidence: float = 0.95) -> int:
    """
    Calculate samples needed to achieve target precision.
    
    Args:
        target_error: Desired half-width of confidence interval
        confidence: Confidence level (default 0.95)
        
    Returns:
        Number of samples needed
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = np.pi / 4
    variance = 16 * p * (1 - p)
    n = (z**2 * variance) / (target_error**2)
    return int(np.ceil(n))
