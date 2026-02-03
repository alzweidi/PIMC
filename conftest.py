"""Pytest configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset numpy random state before each test for reproducibility."""
    np.random.seed(None)
    yield


@pytest.fixture
def sample_result():
    """Create a sample simulation result for testing."""
    from pimc.core.simulation import MonteCarloPi
    
    sim = MonteCarloPi(n_points=1000, seed=42)
    return sim.run()


@pytest.fixture
def large_result():
    """Create a larger simulation result for testing."""
    from pimc.core.simulation import MonteCarloPi
    
    sim = MonteCarloPi(n_points=100_000, seed=42)
    return sim.run()
