"""Fast numerical, decision-theory, and cross-path posterior-contract checks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import gammainc, logsumexp, roots_hermitenorm

from bayesfreq.bayesian_risk import (
    AmortizedPosterior,
    GammaPluginPosterior,
    clipped_mle,
    estimate_bayes_risk,
    estimate_frequentist_risk,
    exact_posterior,
    gamma_plugin_moments,
    laplace_moments,
    simulation_based_calibration,
    time_inference,
    train_amortized,
)
from bayesfreq.config import CORE_CONFIG, FULL_CONFIG
from bayesfreq.data_simulator import Prior, log_likelihood_full, reality_prior, sample_joint


@pytest.fixture(scope="module")
def neural():
    return train_amortized(config=CORE_CONFIG, n_train=2500, epochs=15, seed=123)


def test_gamma_prior_closed_form_without_background():
    config = replace(CORE_CONFIG, problem=replace(CORE_CONFIG.problem, known_b=0, mu_max=80))
    n = np.array([0, 1, 5, 15], dtype=float)
    x = np.column_stack((n, np.zeros(len(n)), np.ones(len(n))))
    a, beta = 2.0, 0.7
    moments = gamma_plugin_moments(x, shape=a, rate=beta, config=config)
    np.testing.assert_allclose(moments.mean, (a + n) / (1 + beta), atol=1e-10)
    np.testing.assert_allclose(moments.variance, (a + n) / (1 + beta) ** 2, atol=1e-10)


def test_truncated_gamma_mixture_matches_grid_known_background():
    config = replace(CORE_CONFIG, grid=replace(CORE_CONFIG.grid, mu_step=0.001))
    x = np.array([[0, 15, 1], [3, 15, 1], [12, 15, 1]], dtype=float)
    prior = Prior(config=config, family="gamma", shape=1.0, rate=0.1)
    exact = exact_posterior(x, prior=prior, config=config)
    mixture = gamma_plugin_moments(x, config=config)
    np.testing.assert_allclose(exact.mean, mixture.mean, atol=1e-6)
    np.testing.assert_allclose(exact.variance, mixture.variance, atol=1e-6)


def test_flat_limit_matches_gamma_mixture():
    config = replace(CORE_CONFIG, grid=replace(CORE_CONFIG.grid, mu_step=0.002))
    x = np.array([[0, 15, 1], [4, 15, 1], [10, 15, 1]], dtype=float)
    grid = exact_posterior(x, prior=Prior(config=config, family="flat"), config=config)
    flatish = gamma_plugin_moments(x, shape=1, rate=1e-8, config=config)
    np.testing.assert_allclose(grid.mean, flatish.mean, atol=1e-6)


def test_grid_normalization_atom_and_credible_sets():
    _, x = sample_joint(30, rng=np.random.default_rng(123), config=CORE_CONFIG)
    posterior = exact_posterior(x, config=CORE_CONFIG)
    np.testing.assert_allclose(posterior.weights.sum(axis=1), 1, atol=1e-12)
    assert np.all(posterior.weights >= 0)
    assert np.all(posterior.weights[:, 0] > 0)
    interval = posterior.intervals(alpha=0.1)
    hpd = posterior.hpd_intervals(alpha=0.1)
    assert np.all(interval.lower <= interval.upper)
    assert np.all(hpd.lower <= hpd.upper)
    assert np.all(np.sum(posterior.weights * hpd.accepted, axis=1) >= 0.9 - 1e-12)
    assert np.all(hpd.accepted[:, 0])
    draws = posterior.sample(100, rng=np.random.default_rng(55))
    assert draws.shape == (30, 100)
    assert np.any(draws == 0)


def test_full_nuisance_quadrature_against_direct_likelihood():
    config = replace(FULL_CONFIG, grid=replace(FULL_CONFIG.grid, mu_step=0.1))
    x = np.array([[5, 17, 1.13]], dtype=float)
    posterior = exact_posterior(x, config=config)
    p = config.problem
    b_nodes, b_weights = np.polynomial.legendre.leggauss(81)
    b = p.b_min + (b_nodes + 1) * (p.b_max - p.b_min) / 2
    z, e_weights = roots_hermitenorm(81)
    eps = np.exp(np.log(p.eps0) + p.sigma_eps * z)
    log_likelihood = log_likelihood_full(
        x[:, None, None, None, :],
        posterior.mu_grid[None, :, None, None],
        b[None, None, :, None],
        eps[None, None, None, :],
        config=config,
    )
    log_integrated = logsumexp(
        log_likelihood
        + np.log(b_weights / 2)[None, None, :, None]
        + np.log(e_weights / np.sqrt(2 * np.pi))[None, None, None, :],
        axis=(2, 3),
    )[0]
    prior = reality_prior(config)
    masses = prior.mu_density(posterior.mu_grid) * posterior.cell_widths
    masses[0] = prior.zero_mass
    with np.errstate(divide="ignore"):
        log_posterior = log_integrated + np.log(masses)
    weights = np.exp(log_posterior - logsumexp(log_posterior))
    np.testing.assert_allclose(posterior.weights[0], weights, atol=1e-10, rtol=1e-8)


def test_reference_mean_has_smallest_squared_bayes_risk(neural):
    count = 6000
    estimators = [
        lambda x: gamma_plugin_moments(x, config=CORE_CONFIG).mean,
        lambda x: laplace_moments(x, config=CORE_CONFIG).mean,
        lambda x: clipped_mle(x, config=CORE_CONFIG),
        neural,
    ]
    exact = estimate_bayes_risk(
        lambda x: exact_posterior(x, config=CORE_CONFIG).mean,
        rng=np.random.default_rng(7),
        n_samples=count,
        config=CORE_CONFIG,
    )
    for estimator in estimators:
        result = estimate_bayes_risk(
            estimator, rng=np.random.default_rng(7), n_samples=count, config=CORE_CONFIG
        )
        combined_se = np.hypot(exact.mc_se, result.mc_se)
        assert exact.value <= result.value + 3 * combined_se


def test_mle_agrees_with_direct_full_likelihood_optimization():
    x = np.array([[2, 15, 1.2], [9, 15, 0.9], [40, 5, 0.8]], dtype=float)
    estimates = clipped_mle(x)
    for row, estimate in zip(x, estimates, strict=True):

        def objective(parameters, observation=row):
            mu, log_b, log_eps = parameters
            return -float(log_likelihood_full(observation, mu, np.exp(log_b), np.exp(log_eps)))

        fit = minimize(
            objective,
            [float(estimate), np.log(max(row[1] / 5, 0.01)), np.log(row[2])],
            method="L-BFGS-B",
            bounds=[(0, 10), (-15, 6), (-5, 5)],
        )
        assert fit.success
        assert abs(fit.x[0] - estimate) < 2e-5


def test_estimator_bounds_and_batch_contract(neural):
    _, x = sample_joint(25, rng=np.random.default_rng(88), config=CORE_CONFIG)
    candidates = [
        exact_posterior(x, config=CORE_CONFIG),
        gamma_plugin_moments(x, config=CORE_CONFIG),
        laplace_moments(x, config=CORE_CONFIG),
        neural.moments(x),
    ]
    for posterior in candidates:
        assert posterior.mean.shape == posterior.variance.shape == (len(x),)
        assert posterior.mean.dtype == posterior.variance.dtype == np.float64
        assert np.all((posterior.mean >= 0) & (posterior.mean <= 10))
        assert np.all(np.isfinite(posterior.variance))
        assert np.all(posterior.variance >= 0)
    mle = clipped_mle(x, config=CORE_CONFIG)
    assert np.all((mle >= 0) & (mle <= 10))


def test_seeded_neural_reproducible_and_saved(tmp_path, neural):
    first = train_amortized(config=CORE_CONFIG, n_train=600, epochs=3, seed=91)
    second = train_amortized(config=CORE_CONFIG, n_train=600, epochs=3, seed=91)
    _, x = sample_joint(12, rng=np.random.default_rng(99), config=CORE_CONFIG)
    np.testing.assert_array_equal(first(x), second(x))
    restored = AmortizedPosterior.load(neural.save(tmp_path / "model.pt"))
    np.testing.assert_array_equal(neural(x), restored(x))
    np.testing.assert_array_equal(neural.moments(x).variance, restored.moments(x).variance)
    assert restored.config == CORE_CONFIG


def test_neural_draw_distribution_matches_moments(neural):
    x = np.array([[0, 15, 1], [7, 15, 1]], dtype=float)
    draws = neural.sample(x, 30000, rng=np.random.default_rng(4))
    moments = neural.moments(x)
    assert np.all((draws >= 0) & (draws <= 10))
    assert np.any(draws == 0)
    np.testing.assert_allclose(draws.mean(axis=1), moments.mean, atol=0.01, rtol=0.04)
    np.testing.assert_allclose(draws.var(axis=1), moments.variance, atol=0.03, rtol=0.05)
    intervals = neural.intervals(x)
    assert np.all((intervals.lower >= 0) & (intervals.upper <= 10))


def test_risk_mc_error_and_seeded_results():
    count, seed = 400, 4
    theta, x = sample_joint(count, rng=np.random.default_rng(seed), config=CORE_CONFIG)
    losses = (clipped_mle(x, config=CORE_CONFIG) - theta[:, 0]) ** 2
    risk = estimate_bayes_risk(
        lambda x: clipped_mle(x, config=CORE_CONFIG),
        rng=np.random.default_rng(seed),
        n_samples=count,
        config=CORE_CONFIG,
    )
    assert risk.value == pytest.approx(losses.mean())
    assert risk.mc_se == pytest.approx(losses.std(ddof=1) / np.sqrt(count))
    absolute = estimate_bayes_risk(
        lambda x: clipped_mle(x, config=CORE_CONFIG),
        rng=np.random.default_rng(seed),
        n_samples=count,
        config=CORE_CONFIG,
        loss="absolute",
    )
    assert absolute.value == pytest.approx(np.sqrt(losses).mean())
    points = np.array([[0, 3, 1], [10, 3, 1]], dtype=float)
    results = estimate_frequentist_risk(
        lambda x: clipped_mle(x, config=CORE_CONFIG),
        points,
        rng=np.random.default_rng(seed),
        n_samples=100,
        config=CORE_CONFIG,
    )
    assert len(results) == 2
    assert all(result.mc_se >= 0 and result.n_samples == 100 for result in results)
    assert time_inference(lambda x: clipped_mle(x, config=CORE_CONFIG), x, repeats=3) > 0


def test_sbc_rank_bounds_and_tie_randomization():
    class ZeroSampler:
        def sample(self, x, n_draws, *, rng):
            return np.zeros((len(x), n_draws))

    prior = reality_prior(CORE_CONFIG, zero_mass=1)
    ranks = simulation_based_calibration(
        ZeroSampler(),
        rng=np.random.default_rng(22),
        n_samples=1000,
        n_draws=10,
        prior=prior,
        config=CORE_CONFIG,
    )
    assert ranks.dtype == np.int64
    assert set(ranks) == set(range(11))
    assert abs(ranks.mean() - 5) < 0.4
    repeated = simulation_based_calibration(
        ZeroSampler(),
        rng=np.random.default_rng(22),
        n_samples=1000,
        n_draws=10,
        prior=prior,
        config=CORE_CONFIG,
    )
    np.testing.assert_array_equal(ranks, repeated)


def test_gamma_sampler_matches_truncated_moments():
    config = replace(CORE_CONFIG, problem=replace(CORE_CONFIG.problem, known_b=0))
    x = np.array([[0, 0, 1], [8, 0, 1]], dtype=float)
    sampler = GammaPluginPosterior(config)
    draws = sampler.sample(x, 20000, rng=np.random.default_rng(3))
    moments = gamma_plugin_moments(x, config=config)
    np.testing.assert_allclose(draws.mean(axis=1), moments.mean, atol=0.03)
    expected = 1 / 1.1 * gammainc(2, 11) / gammainc(1, 11)
    assert moments.mean[0] == pytest.approx(expected)


def test_validation_and_missing_model(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        exact_posterior(np.ones(3))
    with pytest.raises(ValueError, match="same config"):
        exact_posterior(np.array([[1, 15, 1.0]]), prior=reality_prior(CORE_CONFIG))
    with pytest.raises(ValueError, match="at least two"):
        estimate_bayes_risk(clipped_mle, rng=np.random.default_rng(2), n_samples=1)
    with pytest.raises(FileNotFoundError, match="Run Path A first"):
        AmortizedPosterior.load(tmp_path / "missing.pt")
