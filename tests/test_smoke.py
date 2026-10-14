"""Minimal package and joint-simulator integration check."""

import numpy as np

import bayesfreq
from bayesfreq.data_simulator import sample_joint


def test_import_and_joint_batch() -> None:
    theta, x = sample_joint(16, rng=np.random.default_rng(42))
    assert bayesfreq.__version__ == "0.1.0"
    assert theta.shape == x.shape == (16, 3)
    assert theta.dtype == x.dtype == np.float64
    assert np.all(np.isfinite(theta)) and np.all(np.isfinite(x))
    assert np.all(x[:, :2] >= 0) and np.all(x[:, :2] == np.floor(x[:, :2]))
    assert np.all(x[:, 2] > 0)
