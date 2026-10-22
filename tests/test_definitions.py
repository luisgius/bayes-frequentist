"""Path A estimators and interval rules checked against their definitions, plus a quick run.

Each definition test uses an independent closed form or reference, so a wrong formula
fails even when bounds, finiteness and coverage-style checks would still pass.
"""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesfreq import plotting
from bayesfreq.bayesian_risk import (
    AmortizedPosterior,
    _laplace_grid,
    exact_posterior,
    gamma_plugin_moments,
)
from bayesfreq.config import CORE_CONFIG, FULL_CONFIG
from bayesfreq.data_simulator import Prior, sample_joint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("n_on", "eps_hat"), [(0, 0.8), (5, 1.25), (15, 0.9)])
def test_gamma_plugin_full_mode_closed_form_with_efficiency(n_on, eps_hat):
    # n_off = 0 gives b_hat = 0; Gamma(a, beta) x Poisson(n | eps mu) is Gamma(a + n, beta + eps).
    # A large mu_max makes the truncation negligible, so the closed form applies.
    config = replace(FULL_CONFIG, problem=replace(FULL_CONFIG.problem, mu_max=200.0))
    shape, rate = 2.0, 0.7
    moments = gamma_plugin_moments(
        np.array([[n_on, 0.0, eps_hat]]), shape=shape, rate=rate, config=config
    )
    np.testing.assert_allclose(moments.mean, (shape + n_on) / (rate + eps_hat), rtol=1e-10)
    np.testing.assert_allclose(moments.variance, (shape + n_on) / (rate + eps_hat) ** 2, rtol=1e-10)


def test_laplace_quadratic_uses_minus_second_derivative_of_log_posterior():
    posterior = _laplace_grid(np.array([[12.0, 15.0, 1.0]]), None, FULL_CONFIG)
    # On the slab the Laplace log density is a quadratic in mu; recover its curvature.
    curvature = np.polyfit(
        posterior.mu_grid[1:], np.log(posterior.weights[0, 1:] / posterior.cell_widths[1:]), 2
    )[0]

    def log_posterior(mu):
        # Plug-in log posterior: b_hat = 15 / 5 = 3, eps = 1, log-uniform slab density 1 / mu.
        return 12 * np.log(mu + 3) - mu - np.log(mu)

    nodes = np.linspace(0.1, 10, 201)
    mode = nodes[np.argmax(log_posterior(nodes))]
    step = 1e-4
    below, at, above = log_posterior(mode + np.array([-step, 0.0, step]))
    precision = -(above - 2 * at + below) / step**2
    assert -2 * curvature == pytest.approx(precision, rel=1e-4)


def test_flat_prior_grid_credible_intervals_have_nominal_prior_averaged_coverage():
    # Without an atom, data drawn from the analysis prior make the equal-tailed set nominal.
    prior = Prior(config=CORE_CONFIG, family="flat")
    theta, x = sample_joint(20000, rng=np.random.default_rng(5), config=CORE_CONFIG, prior=prior)
    intervals = exact_posterior(x, prior=prior, config=CORE_CONFIG).intervals(alpha=0.1)
    covered = (theta[:, 0] >= intervals.lower) & (theta[:, 0] <= intervals.upper)
    se = np.sqrt(0.9 * 0.1 / len(covered))
    # The 0.005 allowance covers endpoints that sit on grid nodes.
    assert abs(covered.mean() - 0.9) < 4 * se + 0.005


def test_neural_quantiles_match_its_own_sampler():
    model = AmortizedPosterior.load(ROOT / "results/path_a_amortized.pt")
    x = np.array([[9.0, 15.0, 1.0], [14.0, 10.0, 0.95], [18.0, 5.0, 1.1]])
    draws = model.sample(x, 1_000_000, rng=np.random.default_rng(8))
    intervals = model.intervals(x, alpha=0.1)
    np.testing.assert_allclose(intervals.lower, np.quantile(draws, 0.05, axis=1), atol=0.03)
    np.testing.assert_allclose(intervals.upper, np.quantile(draws, 0.95, axis=1), atol=0.03)


def test_large_count_log_space_branch_matches_polynomial_branch():
    # The branch is chosen per batch, so one n_on >= 128 row moves every row to log space.
    row = np.array([[127.0, 30.0, 1.05]])
    polynomial = exact_posterior(row)
    log_space = exact_posterior(np.vstack((row, [[128.0, 30.0, 1.0]])))
    np.testing.assert_allclose(log_space.weights[0], polynomial.weights[0], rtol=1e-8, atol=1e-14)


def _check_metrics(value):
    """Assert every metric has a nonnegative SE and every float in the record is finite."""
    if isinstance(value, dict):
        if {"value", "mc_se", "n_samples", "method"} <= value.keys():
            assert value["mc_se"] >= 0 and value["n_samples"] > 0
        for child in value.values():
            _check_metrics(child)
    elif isinstance(value, list):
        for child in value:
            _check_metrics(child)
    elif isinstance(value, float):
        assert np.isfinite(value)


def _repository_hashes():
    """Hash the saved results and figures, which a test run must leave unchanged."""
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("results", "figures")
        for path in (ROOT / folder).rglob("*")
        if path.is_file()
    }


def test_quick_path_a_run_writes_complete_record_to_tmp_path(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("path_a_runner", ROOT / "scripts/run_path_a.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    before = _repository_hashes()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(plotting, "FIGURES_DIR", tmp_path / "figures")
    result = runner.run(quick=True)
    assert _repository_hashes() == before
    saved = json.loads((tmp_path / "results/path_a.json").read_text())
    assert saved == json.loads(json.dumps(result))
    assert (tmp_path / "results/path_a_amortized.pt").stat().st_size > 1000
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    assert saved["diagnostics"]["quick"] is True and saved["schema_version"] == 1
    _check_metrics(saved)
    methods = saved["methods"]
    known_b = saved["config"]["problem"]["known_b"]
    for name, entry in methods.items():
        assert entry["max_grid_risk_common_slice"]["theta"][1] == known_b
        if name == "exact_grid":
            assert "paired_vs_exact_grid" not in entry
            assert entry["max_grid_risk_common_slice"] == entry["max_grid_risk"]
            continue
        paired = entry["paired_vs_exact_grid"]
        gap = entry["bayes_risk"]["squared"]["value"]
        gap -= methods["exact_grid"]["bayes_risk"]["squared"]["value"]
        assert paired["squared_loss_difference"]["value"] == pytest.approx(gap, abs=1e-9)
        assert paired["excess_squared_risk"]["value"] >= 0
    # Each interval holds at least 1 - alpha of its own posterior, and at most all of it.
    for name in ("exact_grid", "amortized"):
        content = methods[name]["interval_summary"]["posterior_content"]["value"]
        assert 0.9 - 1e-9 <= content <= 1 + 1e-9
        assert all("posterior_content" in row for row in methods[name]["coverage"])
    rows = methods["amortized"]["paired_coverage_vs_exact_grid"]
    assert [row["theta"] for row in rows] == [
        row["theta"] for row in methods["exact_grid"]["coverage"]
    ]
    sensitivity = saved["diagnostics"]["prior_sensitivity"]
    assert "vs_reality" not in sensitivity["reality"]["paired"]
    flat = sensitivity["flat"]["paired"]["vs_clipped_mle"]["squared_loss_difference"]["value"]
    flat_gap = sensitivity["flat"]["bayes_risk"]["squared"]["value"]
    flat_gap -= methods["clipped_mle"]["bayes_risk"]["squared"]["value"]
    assert flat == pytest.approx(flat_gap, abs=1e-9)
    comparison = saved["diagnostics"]["neural_vs_exact_posterior"]
    assert comparison["kl_exact_to_neural"]["value"] >= 0
    assert comparison["n_classes"] == 101
    assert {"platform", "machine"} <= saved["environment"].keys()


def test_paired_comparison_se_uses_per_draw_differences():
    spec = importlib.util.spec_from_file_location("path_a_runner", ROOT / "scripts/run_path_a.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 10, 5000)
    reference = truth + rng.normal(0, 2, truth.size)
    same = runner.paired_comparison(truth, reference, reference, excess=True)
    assert same["squared_loss_difference"]["value"] == 0
    assert same["squared_loss_difference"]["mc_se"] == 0
    # A paired SE is the spread of the per-draw differences, not of the two risks.
    shifted = runner.paired_comparison(truth, reference + 0.1, reference)
    difference = (reference + 0.1 - truth) ** 2 - (reference - truth) ** 2
    expected = np.std(difference, ddof=1) / np.sqrt(difference.size)
    assert shifted["squared_loss_difference"]["mc_se"] == pytest.approx(expected, rel=1e-12)
