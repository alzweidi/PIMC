"""
Comprehensive test suite for the Monte Carlo simulation engine.

Tests cover:
- Core simulation functionality
- Sampling methods
- Convergence properties
- Parallel processing
- Edge cases
"""

import numpy as np
import pytest

from pimc.core.simulation import (
    MonteCarloPi,
    SimulationResult,
    ConvergenceData,
    SamplingMethod,
    quick_estimate,
    benchmark,
)


class TestMonteCarloPi:
    """Tests for the MonteCarloPi simulation class."""
    
    def test_basic_simulation(self):
        """Basic simulation returns valid result."""
        sim = MonteCarloPi(n_points=1000, seed=42)
        result = sim.run()
        
        assert isinstance(result, SimulationResult)
        assert result.n_points == 1000
        assert 2.5 < result.estimate < 3.8  # Reasonable range
        assert result.error >= 0
        assert result.elapsed_time > 0
    
    def test_reproducibility_with_seed(self):
        """Same seed produces identical results."""
        sim1 = MonteCarloPi(n_points=1000, seed=123)
        sim2 = MonteCarloPi(n_points=1000, seed=123)
        
        result1 = sim1.run()
        result2 = sim2.run()
        
        assert result1.estimate == result2.estimate
        assert result1.n_inside == result2.n_inside
        np.testing.assert_array_equal(result1.x_coords, result2.x_coords)
        np.testing.assert_array_equal(result1.y_coords, result2.y_coords)
    
    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        sim1 = MonteCarloPi(n_points=1000, seed=123)
        sim2 = MonteCarloPi(n_points=1000, seed=456)
        
        result1 = sim1.run()
        result2 = sim2.run()
        
        assert result1.estimate != result2.estimate
    
    def test_points_in_unit_square(self):
        """All generated points are within [0, 1] x [0, 1]."""
        sim = MonteCarloPi(n_points=10000, seed=42)
        result = sim.run()
        
        assert np.all(result.x_coords >= 0)
        assert np.all(result.x_coords <= 1)
        assert np.all(result.y_coords >= 0)
        assert np.all(result.y_coords <= 1)
    
    def test_inside_classification(self):
        """Points are correctly classified as inside/outside quarter circle."""
        sim = MonteCarloPi(n_points=10000, seed=42)
        result = sim.run()
        
        distance_sq = result.x_coords**2 + result.y_coords**2
        expected_inside = distance_sq <= 1.0
        
        np.testing.assert_array_equal(result.inside_mask, expected_inside)
    
    def test_pi_formula(self):
        """π estimate follows the formula: 4 * n_inside / n_total."""
        sim = MonteCarloPi(n_points=10000, seed=42)
        result = sim.run()
        
        expected = 4.0 * result.n_inside / result.n_points
        assert abs(result.estimate - expected) < 1e-10
    
    def test_large_sample_accuracy(self):
        """Large sample produces accurate estimate."""
        sim = MonteCarloPi(n_points=1_000_000, seed=42)
        result = sim.run()
        
        assert abs(result.estimate - np.pi) < 0.01
        assert result.accuracy_grade in ("S", "A")
    
    def test_convergence_data_included(self):
        """Convergence data is computed when requested."""
        sim = MonteCarloPi(n_points=10000, seed=42, track_convergence=True)
        result = sim.run()
        
        assert result.convergence_data is not None
        assert isinstance(result.convergence_data, ConvergenceData)
        assert len(result.convergence_data.sample_sizes) > 0
        assert len(result.convergence_data.estimates) == len(result.convergence_data.sample_sizes)
    
    def test_convergence_data_disabled(self):
        """Convergence data is None when disabled."""
        sim = MonteCarloPi(n_points=1000, seed=42, track_convergence=False)
        result = sim.run()
        
        assert result.convergence_data is None
    
    def test_store_points_disabled(self):
        """Points not stored when disabled."""
        sim = MonteCarloPi(n_points=1000, seed=42, store_points=False)
        result = sim.run()
        
        assert len(result.x_coords) == 0
        assert len(result.y_coords) == 0
        assert len(result.inside_mask) == 0
    
    def test_accuracy_grades(self):
        """Accuracy grades are assigned correctly."""
        sim = MonteCarloPi(n_points=10_000_000, seed=42, store_points=False)
        result = sim.run()
        
        if result.error < 0.001:
            assert result.accuracy_grade == "S"
        elif result.error < 0.01:
            assert result.accuracy_grade == "A"
    
    def test_stream_method(self):
        """Stream method yields correct batches."""
        sim = MonteCarloPi(n_points=500, seed=42)
        
        batches = list(sim.stream(batch_size=100))
        
        assert len(batches) == 5
        
        # Last batch should have final count
        final_n, final_est, _, _, _ = batches[-1]
        assert final_n == 500
        assert 2.5 < final_est < 3.8


class TestSamplingMethods:
    """Tests for different sampling methods."""
    
    @pytest.mark.parametrize("method", list(SamplingMethod))
    def test_all_methods_work(self, method):
        """All sampling methods produce valid results."""
        sim = MonteCarloPi(n_points=1000, seed=42, method=method)
        result = sim.run()
        
        assert isinstance(result, SimulationResult)
        assert 2.0 < result.estimate < 4.0
    
    def test_antithetic_reduces_variance(self):
        """Antithetic sampling should reduce variance over many runs."""
        n_trials = 100
        n_points = 1000
        
        standard_estimates = []
        antithetic_estimates = []
        
        for i in range(n_trials):
            sim_std = MonteCarloPi(n_points=n_points, seed=i, method=SamplingMethod.STANDARD)
            sim_anti = MonteCarloPi(n_points=n_points, seed=i, method=SamplingMethod.ANTITHETIC)
            
            standard_estimates.append(sim_std.run().estimate)
            antithetic_estimates.append(sim_anti.run().estimate)
        
        std_variance = np.var(standard_estimates)
        anti_variance = np.var(antithetic_estimates)
        
        # Antithetic should have lower or similar variance
        # Allow some tolerance as it's stochastic
        assert anti_variance <= std_variance * 1.5
    
    def test_stratified_coverage(self):
        """Stratified sampling provides uniform coverage."""
        sim = MonteCarloPi(n_points=10000, seed=42, method=SamplingMethod.STRATIFIED)
        result = sim.run()
        
        # Check coverage in quadrants
        q1 = np.sum((result.x_coords < 0.5) & (result.y_coords < 0.5))
        q2 = np.sum((result.x_coords >= 0.5) & (result.y_coords < 0.5))
        q3 = np.sum((result.x_coords < 0.5) & (result.y_coords >= 0.5))
        q4 = np.sum((result.x_coords >= 0.5) & (result.y_coords >= 0.5))
        
        # Each quadrant should have roughly 25% of points
        expected = result.n_points / 4
        for q in [q1, q2, q3, q4]:
            assert abs(q - expected) / expected < 0.1  # Within 10%
    
    def test_quasi_random_low_discrepancy(self):
        """Quasi-random sampling has low discrepancy."""
        sim = MonteCarloPi(n_points=1000, seed=42, method=SamplingMethod.QUASI_RANDOM)
        result = sim.run()
        
        # Points should be well-distributed
        assert len(result.x_coords) == 1000
        
        # Check no large gaps (basic check)
        x_sorted = np.sort(result.x_coords)
        gaps = np.diff(x_sorted)
        max_gap = np.max(gaps)
        
        # Max gap should be relatively small for low-discrepancy sequence
        assert max_gap < 0.1


class TestConvergenceProperties:
    """Tests for convergence behavior."""
    
    def test_error_decreases_with_samples(self):
        """Error generally decreases as sample size increases."""
        errors = []
        sample_sizes = [100, 1000, 10000, 100000]
        
        for n in sample_sizes:
            sim = MonteCarloPi(n_points=n, seed=42, store_points=False)
            result = sim.run()
            errors.append(result.error)
        
        # Each step should generally have lower error
        # Allow for some stochastic variation
        assert errors[-1] < errors[0]
    
    def test_convergence_data_monotonic_sample_sizes(self):
        """Convergence checkpoints are in increasing order."""
        sim = MonteCarloPi(n_points=10000, seed=42)
        result = sim.run()
        
        conv = result.convergence_data
        assert np.all(np.diff(conv.sample_sizes) > 0)
    
    def test_convergence_approaches_pi(self):
        """Final convergence estimate matches simulation estimate."""
        sim = MonteCarloPi(n_points=50000, seed=42)
        result = sim.run()
        
        conv = result.convergence_data
        assert abs(conv.final_estimate - result.estimate) < 0.001


class TestQuickEstimate:
    """Tests for the quick_estimate function."""
    
    def test_quick_estimate_basic(self):
        """Quick estimate produces reasonable result."""
        estimate = quick_estimate(10000, seed=42)
        
        assert 2.5 < estimate < 3.8
    
    def test_quick_estimate_reproducible(self):
        """Same seed produces same quick estimate."""
        est1 = quick_estimate(1000, seed=123)
        est2 = quick_estimate(1000, seed=123)
        
        assert est1 == est2


class TestBenchmark:
    """Tests for the benchmark function."""
    
    def test_benchmark_returns_dict(self):
        """Benchmark returns properly structured dictionary."""
        results = benchmark(
            sample_sizes=[100, 1000],
            methods=[SamplingMethod.STANDARD],
            seed=42
        )
        
        assert "standard" in results
        assert 100 in results["standard"]
        assert 1000 in results["standard"]
        
        for n in [100, 1000]:
            assert "estimate" in results["standard"][n]
            assert "error" in results["standard"][n]
            assert "time" in results["standard"][n]


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_result_properties(self):
        """Result properties compute correctly."""
        sim = MonteCarloPi(n_points=10000, seed=42)
        result = sim.run()
        
        assert result.ratio == result.n_inside / result.n_points
        assert abs(result.theoretical_ratio - np.pi / 4) < 1e-10
        assert result.points_per_second > 0
    
    def test_result_repr(self):
        """Result has informative string representation."""
        sim = MonteCarloPi(n_points=1000, seed=42)
        result = sim.run()
        
        repr_str = repr(result)
        assert "SimulationResult" in repr_str
        assert "π≈" in repr_str


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_minimum_points(self):
        """Simulation works with minimum points."""
        sim = MonteCarloPi(n_points=1, seed=42)
        result = sim.run()
        
        assert result.n_points == 1
        assert result.estimate in (0.0, 4.0)  # Either inside or outside
    
    def test_very_large_sample(self):
        """Large samples work correctly."""
        sim = MonteCarloPi(n_points=1_000_000, seed=42, store_points=False)
        result = sim.run()
        
        assert result.n_points == 1_000_000
        assert abs(result.estimate - np.pi) < 0.005
    
    def test_invalid_n_points_raises(self):
        """Invalid n_points raises ValueError."""
        with pytest.raises(ValueError):
            MonteCarloPi(n_points=0)
        
        with pytest.raises(ValueError):
            MonteCarloPi(n_points=-100)
    
    def test_method_string_conversion(self):
        """String method names are converted correctly."""
        sim = MonteCarloPi(n_points=100, method="antithetic")
        assert sim.config.method == SamplingMethod.ANTITHETIC
