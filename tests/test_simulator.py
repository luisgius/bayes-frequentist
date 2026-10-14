"""Independent DGP checks with fixed seeds and four Monte Carlo SE tolerances."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm, poisson

from bayesfreq.config import CORE_CONFIG, FULL_CONFIG
from bayesfreq.data_simulator import (
    analysis_priors,
    log_likelihood_core,
    log_likelihood_full,
    poisson_pmf,
    poisson_support,
    sample_joint,
    simulate,
    simulate_counts,
    theta_grid,
)


@pytest.mark.parametrize("mu,b,eps", [(0, 0, 1), (0, 0.5, 0.82), (10, 10, 1.22), (2, 3, 1)])
def test_conditional_poisson_moments(mu, b, eps):
    count = 200000
    x = simulate_counts(mu, b, eps, size=count, rng=np.random.default_rng(910))
    rates = np.array([eps * mu + b, 5 * b])
    np.testing.assert_array_less(abs(x.mean(axis=0) - rates), 4 * np.sqrt(rates / count) + 1e-12)
    # Exact variance of the unbiased sample variance: mu4=lambda+3*lambda**2.
    variance_se = np.sqrt((rates + (2 + 2 / (count - 1)) * rates**2) / count)
    np.testing.assert_array_less(abs(x.var(axis=0, ddof=1) - rates), 4 * variance_se + 1e-12)
    covariance = np.cov(x.T, ddof=1)[0, 1]
    assert abs(covariance) <= 4 * np.sqrt(np.prod(rates) / count) + 1e-12


def test_auxiliary_measurement_and_prior_are_distinct_draws():
    count = 100000
    theta, x = sample_joint(count, rng=np.random.default_rng(991))
    residual = np.log(x[:, 2]) - np.log(theta[:, 2])
    sigma = FULL_CONFIG.problem.sigma_eps
    assert abs(residual.mean()) <= 4 * sigma / np.sqrt(count)
    assert abs(residual.var(ddof=1) - sigma**2) <= 4 * sigma**2 * np.sqrt(2 / (count - 1))
    assert abs(np.cov(residual, np.log(theta[:, 2]))[0, 1]) <= 4 * sigma**2 / np.sqrt(count)


@pytest.mark.parametrize("name", ["reality", "flat", "gamma_diffuse", "gamma_shifted"])
def test_prior_sampler_matches_mixed_density(name):
    prior = analysis_priors()[name]
    count = 150000
    theta = prior.sample(count, rng=np.random.default_rng(719))
    atom = prior.mu_density(0.0)
    assert abs(np.mean(theta[:, 0] == 0) - atom) <= 4 * np.sqrt(atom * (1 - atom) / count)
    edges = np.geomspace(prior.mu_min if name == "reality" else 0.001, 10, 9)
    edges = np.insert(edges, 0, 0.0)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        probability = quad(
            lambda mu: float(prior.mu_density(mu)),
            lower,
            upper,
            points=[prior.mu_min] if lower < prior.mu_min < upper else None,
        )[0]
        observed = np.mean((theta[:, 0] > lower) & (theta[:, 0] <= upper))
        assert abs(observed - probability) <= 4 * np.sqrt(probability * (1 - probability) / count)
    assert np.all(np.isfinite(prior.log_prob(theta)))
    for column, cdf in [(1, lambda b: (b - 0.5) / 9.5), (2, lambda e: lognorm.cdf(e, 0.1))]:
        probabilities = cdf(theta[:, column])
        frequencies = np.histogram(probabilities, np.linspace(0, 1, 11))[0] / count
        np.testing.assert_array_less(abs(frequencies - 0.1), 4 * np.sqrt(0.1 * 0.9 / count))


def test_likelihoods_match_scipy_and_broadcast_scalar_loop():
    x = np.array([[0.0, 0.0, 1.0], [3.0, 17.0, 0.91], [1000.0, 5000.0, 1.2]])
    mu, b, eps = np.array([0.0, 0.5, 10.0]), np.array([0.0, 3.0, 10.0]), np.array([1.0, 0.82, 1.22])
    expected = (
        poisson.logpmf(x[:, 0, None], eps * mu + b)
        + poisson.logpmf(x[:, 1, None], 5 * b)
        + lognorm.logpdf(x[:, 2, None], 0.1, scale=eps)
    )
    actual = log_likelihood_full(x[:, None, :], mu, b, eps)
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    loop = [
        [log_likelihood_full(row, m, bg, e) for m, bg, e in zip(mu, b, eps, strict=True)]
        for row in x
    ]
    np.testing.assert_array_equal(actual, loop)
    np.testing.assert_allclose(
        log_likelihood_core(x[:, 0, None], mu, b=b), poisson.logpmf(x[:, 0, None], mu + b)
    )
    assert poisson_pmf(0, 0) == 1
    assert poisson_pmf(-1, 3) == poisson_pmf(0.5, 3) == 0


def test_vectorized_simulation_matches_scalar_draw_order():
    # Preserve component-major RNG order; row-wise calls consume a different stream order.
    mu = np.array([0.0, 1.0, 10.0, 2.0])
    b = np.array([0.0, 0.5, 10.0, 3.0])
    eps = np.array([1.0, 0.82, 1.22, 1.0])
    rng = np.random.default_rng(192)
    expected = np.column_stack(
        (
            [rng.poisson(e * m + bg) for m, bg, e in zip(mu, b, eps, strict=True)],
            [rng.poisson(5 * bg) for bg in b],
            [rng.lognormal(np.log(e), 0.1) for e in eps],
        )
    )
    actual = simulate(mu, b, eps, rng=np.random.default_rng(192))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("config", [CORE_CONFIG, FULL_CONFIG])
def test_rng_reproducibility_empty_batch_and_no_global_state(config):
    state = np.random.get_state()  # noqa: NPY002 — audit legacy state without drawing
    a = sample_joint(100, rng=np.random.default_rng(17), config=config)
    b = sample_joint(100, rng=np.random.default_rng(17), config=config)
    for first, second in zip(a, b, strict=True):
        np.testing.assert_array_equal(first, second)
    now = np.random.get_state()  # noqa: NPY002 — audit legacy state without drawing
    np.testing.assert_array_equal(state[1], now[1])
    assert state[2:] == now[2:]
    empty_theta, empty_x = sample_joint(0, rng=np.random.default_rng(1), config=config)
    assert empty_theta.shape == empty_x.shape == (0, 3)
    assert np.all(np.isfinite(a[0])) and np.all(np.isfinite(a[1]))


def test_grid_includes_nondivisible_upper_bound_and_poisson_tail():
    config = replace(FULL_CONFIG, grid=replace(FULL_CONFIG.grid, mu_step=0.7))
    grid = theta_grid(config)
    assert grid[0] == 0 and grid[-1] == 10 and np.all(np.diff(grid) > 0)
    support = poisson_support(30)
    assert poisson.sf(support[-1], 30) <= 1e-12
    assert poisson.sf(support[-1] - 1, 30) > 1e-12


@pytest.mark.parametrize("mu,b,eps", [(-1, 1, 1), (1, -1, 1), (1, 1, 0), (np.nan, 1, 1)])
def test_invalid_simulator_inputs_raise(mu, b, eps):
    with pytest.raises(ValueError):
        simulate(mu, b, eps, rng=np.random.default_rng(1))
