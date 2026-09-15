import numpy as np

from src.monitoring.drift import drift_status, population_stability_index


def test_identical_distributions_are_stable():
    sample = np.linspace(1, 100, 100)
    psi = population_stability_index(sample, sample.copy())
    assert psi < 1e-9
    assert drift_status(psi) == "stable"


def test_shifted_distribution_increases_psi():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 500)
    current = rng.normal(3, 1, 500)
    assert population_stability_index(reference, current) > 0.25
