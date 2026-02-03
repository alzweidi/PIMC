"""
Test suite for statistical analysis module.

Tests cover:
- Running statistics
- Confidence intervals
- Convergence diagnostics
- Outlier filtering
- Hypothesis testing
"""

import numpy as np
import pytest

from pimc.core.statistics import (
    StatisticalAnalyzer,
    RunningStats,
    ConfidenceInterval,
    ConvergenceDiagnostics,
    theoretical_variance,
    theoretical_standard_error,
    required_samples_for_precision,
)
from pimc.core.simulation import MonteCarloPi


class TestRunningStats:
    """Tests for RunningStats online algorithm."""
    
    def test_single_value(self):
        """Single value produces correct mean."""
        stats = RunningStats()
        stats.update(5.0)
        
        assert stats.n == 1
        assert stats.mean == 5.0
        assert stats.min_val == 5.0
        assert stats.max_val == 5.0
    
    def test_multiple_values(self):
        """Multiple values produce correct statistics."""
        stats = RunningStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for v in values:
            stats.update(v)
        
        assert stats.n == 5
        assert stats.mean == 3.0
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0
        assert abs(stats.variance - 2.0) < 1e-10  # Population variance
    
    def test_batch_update(self):
        """Batch update produces same results as individual updates."""
        stats1 = RunningStats()
        stats2 = RunningStats()
        
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        for v in values:
            stats1.update(v)
        
        stats2.update_batch(values)
        
        assert stats1.n == stats2.n
        assert abs(stats1.mean - stats2.mean) < 1e-10
        assert abs(stats1.variance - stats2.variance) < 1e-10
    
    def test_standard_error(self):
        """Standard error is computed correctly."""
        stats = RunningStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for v in values:
            stats.update(v)
        
        expected_se = stats.sample_std / np.sqrt(stats.n)
        assert abs(stats.standard_error - expected_se) < 1e-10


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""
    
    def test_add_estimates(self):
        """Adding estimates updates statistics correctly."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(100):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            result = sim.run()
            analyzer.add_estimate(result.estimate)
        
        assert len(analyzer.estimates) == 100
        assert 2.5 < analyzer.best_estimate < 3.8
    
    def test_outlier_filtering(self):
        """Outliers are filtered correctly."""
        analyzer = StatisticalAnalyzer(warmup_samples=50)
        
        # Add normal estimates
        for _ in range(100):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        # Try to add extreme outlier
        rejected = not analyzer.add_estimate(10.0, z_threshold=2.5)
        
        # Outlier should be rejected after warmup
        assert rejected or len(analyzer.estimates) <= analyzer.warmup_samples + 1
    
    def test_confidence_interval_t(self):
        """T-distribution confidence interval is computed correctly."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        np.random.seed(42)
        for _ in range(100):
            analyzer.add_estimate(np.random.normal(np.pi, 0.1))
        
        ci = analyzer.confidence_interval(level=0.95, method="t")
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.level == 0.95
        assert ci.lower < ci.center < ci.upper
        assert ci.width > 0
    
    def test_confidence_interval_contains_pi(self):
        """95% CI should contain true π most of the time."""
        contains_count = 0
        n_trials = 50
        
        for seed in range(n_trials):
            analyzer = StatisticalAnalyzer(warmup_samples=10)
            
            np.random.seed(seed)
            for _ in range(100):
                sim = MonteCarloPi(n_points=10000, store_points=False)
                analyzer.add_estimate(sim.run().estimate)
            
            ci = analyzer.confidence_interval(level=0.95)
            if ci.contains_pi:
                contains_count += 1
        
        # Should contain π at least 80% of the time (allowing for variation)
        assert contains_count / n_trials >= 0.80
    
    def test_bootstrap_ci(self):
        """Bootstrap confidence interval works correctly."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(100):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        ci = analyzer.bootstrap_ci(level=0.95, n_bootstrap=1000)
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower < ci.upper
    
    def test_geweke_test(self):
        """Geweke convergence test returns valid statistics."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(200):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        z_stat, p_val = analyzer.geweke_test()
        
        assert isinstance(z_stat, float)
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1
    
    def test_effective_sample_size(self):
        """ESS is computed and is reasonable."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(100):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        ess = analyzer.effective_sample_size()
        
        assert ess > 0
        assert ess <= len(analyzer.filtered_estimates)
    
    def test_convergence_diagnostics(self):
        """Convergence diagnostics are computed correctly."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(200):
            sim = MonteCarloPi(n_points=10000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        diagnostics = analyzer.convergence_diagnostics()
        
        assert isinstance(diagnostics, ConvergenceDiagnostics)
        assert diagnostics.has_converged in (True, False)  # Works with numpy.bool_
        assert diagnostics.effective_sample_size > 0
    
    def test_hypothesis_test(self):
        """Hypothesis test returns valid statistics."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(100):
            sim = MonteCarloPi(n_points=10000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        t_stat, p_val, reject = analyzer.hypothesis_test(null_value=np.pi)
        
        assert isinstance(t_stat, (float, np.floating))
        assert isinstance(p_val, (float, np.floating))
        assert reject in (True, False)  # Works with numpy.bool_
        assert 0 <= p_val <= 1
    
    def test_summary(self):
        """Summary returns all expected keys."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(100):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        summary = analyzer.summary()
        
        expected_keys = [
            "n_total", "n_filtered", "acceptance_rate",
            "mean", "std", "standard_error",
            "min", "max", "range",
            "ci_95_lower", "ci_95_upper", "ci_contains_pi",
            "error_from_pi", "error_percent",
            "has_converged", "effective_sample_size",
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_running_means_and_bounds(self):
        """Running means and confidence bounds are tracked."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(50):
            sim = MonteCarloPi(n_points=1000, store_points=False)
            analyzer.add_estimate(sim.run().estimate)
        
        means = analyzer.running_means
        lower, upper = analyzer.confidence_bands
        
        assert len(means) == len(analyzer.filtered_estimates)
        assert len(lower) == len(upper) == len(means)
        assert np.all(lower <= means)
        assert np.all(upper >= means)


class TestTheoreticalFunctions:
    """Tests for theoretical calculation functions."""
    
    def test_theoretical_variance(self):
        """Theoretical variance formula is correct."""
        n = 10000
        var = theoretical_variance(n)
        
        # Var(π̂) = 16 * p * (1-p) / n where p = π/4
        p = np.pi / 4
        expected = 16 * p * (1 - p) / n
        
        assert abs(var - expected) < 1e-10
    
    def test_theoretical_standard_error(self):
        """Theoretical SE is sqrt of variance."""
        n = 10000
        se = theoretical_standard_error(n)
        var = theoretical_variance(n)
        
        assert abs(se - np.sqrt(var)) < 1e-10
    
    def test_required_samples_for_precision(self):
        """Sample size calculation is reasonable."""
        n = required_samples_for_precision(target_error=0.01, confidence=0.95)
        
        assert n > 10000  # Should need many samples for this precision
        assert isinstance(n, int)
    
    def test_required_samples_decreases_with_error(self):
        """Larger error tolerance requires fewer samples."""
        n_tight = required_samples_for_precision(target_error=0.001)
        n_loose = required_samples_for_precision(target_error=0.1)
        
        assert n_tight > n_loose


class TestEdgeCases:
    """Tests for edge cases in statistics module."""
    
    def test_empty_analyzer(self):
        """Empty analyzer handles gracefully."""
        analyzer = StatisticalAnalyzer()
        
        assert analyzer.best_estimate == 0.0
        assert len(analyzer.estimates) == 0
    
    def test_single_estimate(self):
        """Single estimate produces valid (limited) statistics."""
        analyzer = StatisticalAnalyzer(warmup_samples=0)
        analyzer.add_estimate(3.14)
        
        ci = analyzer.confidence_interval()
        
        assert ci.center == 3.14
        assert ci.width == 0  # No variance with single point
    
    def test_identical_estimates(self):
        """Identical estimates produce zero variance."""
        analyzer = StatisticalAnalyzer(warmup_samples=10)
        
        for _ in range(100):
            analyzer.add_estimate(3.14159)
        
        assert analyzer.running_stats.variance < 1e-10
