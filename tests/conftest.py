"""Shared seeded fixtures; test_inference.py and test_figures.py read saved results."""

import pytest

from bayesfreq.bayesian_risk import train_amortized
from bayesfreq.config import FULL_CONFIG


@pytest.fixture(scope="session")
def full_neural():
    """Train a real Path A predictor on a separate reproducible stream."""
    return train_amortized(config=FULL_CONFIG, n_train=12000, epochs=45, seed=46192)
