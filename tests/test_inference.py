"""Cross-path checks: different optimality criteria, raw coverage, and edge inputs."""

import gzip
import importlib.util
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import xlogy

from bayesfreq.bayesian_risk import (
    GammaPluginPosterior,
    clipped_mle,
    exact_posterior,
    gamma_plugin_moments,
    laplace_moments,
    train_amortized,
)
from bayesfreq.config import CORE_CONFIG, FULL_CONFIG
from bayesfreq.data_simulator import Prior, sample_joint, simulate, theta_grid
from bayesfreq.frequentist_bounds import (
    CalibratedBelt,
    calibrate_belt,
    exact_coverage,
    exact_waldo_belt,
    feldman_cousins_belt,
    flat_prior_interval,
    profile_likelihood_statistic,
    refine_belt,
    waldo_statistic,
)


@pytest.fixture(scope="module")
def core_neural():
    return train_amortized(config=CORE_CONFIG, n_train=6000, epochs=30, seed=9311)


def test_reconciliation_bayes_risk_and_worst_case_coverage(core_neural, record_property):
    """Compare explicit midpoint actions under squared loss, then interval coverage."""
    count = 12000
    theta, x = sample_joint(count, rng=np.random.default_rng(774), config=CORE_CONFIG)
    reference = exact_posterior(x, config=CORE_CONFIG)
    fc = feldman_cousins_belt(config=CORE_CONFIG)
    waldo = exact_waldo_belt(posterior=core_neural, config=CORE_CONFIG)
    bayes_loss = (reference.mean - theta[:, 0]) ** 2
    for name, procedure in [("fc", fc), ("waldo", waldo)]:
        intervals = procedure(x)
        midpoint = (intervals.lower + intervals.upper) / 2
        assert np.all(np.isfinite(midpoint))
        difference = (midpoint - theta[:, 0]) ** 2 - bayes_loss
        difference_se = difference.std(ddof=1) / np.sqrt(count)
        assert difference.mean() > 4 * difference_se
        record_property(name + "_excess_midpoint_bayes_risk", float(difference.mean()))
        record_property(name + "_paired_mc_se", float(difference_se))
        coverage = exact_coverage(procedure, theta_grid(CORE_CONFIG), config=CORE_CONFIG)
        assert coverage.value.min() >= 0.9 - CORE_CONFIG.mc.tol_cov_exact
    # Report credible coverage. It is deliberately not asserted to be nominal.
    grid = theta_grid(CORE_CONFIG)
    credible = exact_coverage(
        lambda rows: exact_posterior(rows, config=CORE_CONFIG).intervals(),
        grid,
        config=CORE_CONFIG,
    )
    record_property("bayes_credible_min_coverage", float(credible.value.min()))
    assert np.all(np.isfinite(credible.value))
    # A real counterexample in the interior: any equal-tailed rule has coverage 0
    # near mu_max (the last node carries no mass), so the edge proves nothing.
    # In core mode the interior minimum is about 0.746, at mu = 3.35.
    interior = (grid >= 1) & (grid <= 9)
    record_property("bayes_credible_interior_min_coverage", float(credible.value[interior].min()))
    assert credible.value[interior].min() < 0.85
    for procedure in (fc, waldo):
        coverage = exact_coverage(procedure, theta_grid(CORE_CONFIG), config=CORE_CONFIG)
        assert coverage.value.min() > credible.value.min()


def test_path_b_intervals_overlap_path_a_typical_region(full_neural):
    fc = feldman_cousins_belt(config=CORE_CONFIG, n_max=100)
    # Compare on a known-background slice; no full-case FC claim is made here.
    # Counts 0..20 hold all but about 1e-7 of Pois(mu + b = 5); the off count 15
    # gives the background estimate 3.
    counts = np.arange(21.0)
    rows = np.column_stack((counts, np.full(counts.size, 15.0), np.ones(counts.size)))
    mean = exact_posterior(rows, config=CORE_CONFIG).mean
    intervals = fc(rows)
    # The posterior mean lies strictly inside the FC interval, not at an endpoint
    # (the smallest margin is about 0.24, at n_on = 0).
    assert np.min(np.minimum(mean - intervals.lower, intervals.upper - mean)) > 0.2
    # The full-case neural posterior tracks the exact full posterior on typical toys.
    x = simulate(2.0, 3.0, 1.0, size=2000, rng=np.random.default_rng(819), config=CORE_CONFIG)
    gap = abs(full_neural.moments(x).mean - exact_posterior(x).mean)
    assert gap.shape == (2000,) and gap.mean() < 0.15


def test_flat_prior_interval_matches_path_a_flat_grid_posterior():
    # Path B's local flat-prior posterior and Path A's grid posterior under a flat
    # prior are separate implementations of the same full-case posterior.
    x = np.array([[2.0, 15.0, 0.92], [8.0, 15.0, 1.1], [20.0, 40.0, 1.0]])
    path_b = flat_prior_interval(x)
    path_a = exact_posterior(x, prior=Prior(config=FULL_CONFIG, family="flat")).intervals()
    np.testing.assert_allclose(path_b.lower, path_a.lower, atol=0.02)
    np.testing.assert_allclose(path_b.upper, path_a.upper, atol=0.02)


@pytest.mark.slow
def test_real_path_a_waldo_raw_coverage_on_nuisance_box(full_neural, record_property):
    statistic = partial(waldo_statistic, posterior=full_neural)
    nuisance = np.array(
        [[b, np.exp(k * 0.1)] for b in FULL_CONFIG.grid.b_cov_grid for k in (-2, -1, 0, 1, 2)]
    )
    coarse = calibrate_belt(
        statistic,
        rng=np.random.default_rng(51401),
        mu_grid=np.linspace(0, 10, 51),
        nuisance_grid=nuisance,
        n_toys=3000,
    )
    check_mu = np.unique(np.r_[0.0, 0.01, 0.02, 0.05, np.arange(0.5, 10.1, 0.5)])
    grid = np.unique(np.r_[np.linspace(0, 10, 501), check_mu])
    belt = refine_belt(coarse, grid)
    coverages = []
    for mu in check_mu:
        index = np.searchsorted(grid, mu)
        for b, eps in nuisance:
            x = simulate(
                mu,
                b,
                eps,
                rng=np.random.default_rng(
                    np.random.SeedSequence([81591, index, int(b * 10), int(eps * 10000)])
                ),
                size=3000,
            )
            accepted = statistic(x, mu) <= belt.critical_values[index]
            coverages.append(float(accepted.mean()))
            assert accepted.mean() >= 1 - FULL_CONFIG.problem.alpha - FULL_CONFIG.mc.tol_cov_mc
    # Verify that the direct true-mu check is the membership returned by inversion.
    x = simulate(5.0, 3.0, 1.0, rng=np.random.default_rng(665), size=64)
    intervals = belt(x)
    index = np.searchsorted(grid, 5.0)
    np.testing.assert_array_equal(
        intervals.accepted[:, index], statistic(x, 5.0) <= belt.critical_values[index]
    )
    record_property("minimum_raw_coverage", min(coverages))
    record_property("grid_points", len(coverages))


def test_refinement_is_continuous_upper_envelope_without_node_holes():
    # A constant statistic reveals holes introduced only by critical-value interpolation.
    def statistic(x, mu):
        return np.full(np.broadcast_shapes(x.shape[:-1], mu.shape), 2.0)

    coarse = CalibratedBelt(
        np.array([0.0, 1.0, 2.0]), np.array([3.0, 1.0, 3.0]), statistic, np.array([[3.0, 1.0]]), 0.1
    )
    grid = np.linspace(0, 2, 201)
    fine = refine_belt(coarse, grid)
    assert np.all(fine.critical_values >= 3.0)
    assert fine(np.array([[1.0, 15.0, 1.0]])).accepted.all()
    around = refine_belt(coarse, np.array([0.0, 0.999999, 1.0, 1.000001, 2.0]))
    np.testing.assert_allclose(around.critical_values[1:4], 3.0, atol=1e-6)


@pytest.mark.parametrize("counts", [(0, 0), (1000, 0), (1000, 5000)])
def test_large_count_and_zero_background_inputs_are_finite(counts, full_neural):
    x = np.array([[*counts, 1.0]], dtype=float)
    config = replace(FULL_CONFIG, grid=replace(FULL_CONFIG.grid, mu_step=0.05, eps_points=15))
    for moments in (
        exact_posterior(x, config=config),
        gamma_plugin_moments(x, config=config),
        laplace_moments(x, config=config),
        full_neural.moments(x),
    ):
        assert np.all(np.isfinite(moments.mean)) and np.all(np.isfinite(moments.variance))
        assert np.all((moments.mean >= 0) & (moments.mean <= 10 + 1e-10))
        assert np.all(moments.variance >= 0)
    interval = flat_prior_interval(x, config=config)
    assert np.all(np.isfinite(interval.lower)) and np.all(np.isfinite(interval.upper))
    if counts == (1000, 0):
        assert interval.lower[0] > 9  # Catch the old silent [0,0] numerical result.
    assert np.all(np.isfinite(profile_likelihood_statistic(x[:, None, :], np.array([[0.0, 10.0]]))))


def test_large_count_gamma_matches_independent_scaled_quadrature():
    x = np.array([[1000.0, 0.0, 1.0]])

    # Scale at the physical endpoint to avoid the same incomplete-Gamma implementation.
    def density(mu):
        return np.exp(1000 * np.log(mu / 10) - 1.1 * (mu - 10))

    mass = quad(density, 0.0, 10.0, epsabs=1e-12)[0]
    mean = quad(lambda mu: mu * density(mu), 0.0, 10.0, epsabs=1e-12)[0] / mass
    variance = quad(lambda mu: (mu - mean) ** 2 * density(mu), 0.0, 10.0, epsabs=1e-12)[0] / mass
    moments = gamma_plugin_moments(x)
    np.testing.assert_allclose(moments.mean, mean, atol=1e-9)
    np.testing.assert_allclose(moments.variance, variance, atol=1e-9)
    draws = GammaPluginPosterior().sample(x, 20000, rng=np.random.default_rng(691))
    assert np.all(np.isfinite(draws)) and np.all((draws > 9) & (draws <= 10))
    assert abs(draws.mean() - mean) <= 4 * np.sqrt(variance / draws.size)


def test_deterministic_batch_inference_matches_scalar_loop(full_neural):
    x = np.array([[0.0, 0.0, 1.0], [4.0, 17.0, 0.92], [28.0, 3.0, 1.12]])
    for estimator in [
        lambda rows: exact_posterior(rows).mean,
        lambda rows: gamma_plugin_moments(rows).mean,
        lambda rows: laplace_moments(rows).mean,
        clipped_mle,
        full_neural,
    ]:
        np.testing.assert_allclose(
            estimator(x), np.concatenate([estimator(row[None]) for row in x]), atol=2e-6, rtol=2e-6
        )
    mu = np.array([0.0, 1.0, 10.0])
    batch = profile_likelihood_statistic(x[:, None, :], mu[None])
    loop = np.array([[profile_likelihood_statistic(row, m) for m in mu] for row in x])
    np.testing.assert_allclose(batch, loop, atol=1e-10)


def test_laplace_outlier_is_a_real_boundary_mode_failure():
    x = np.array([[20.0, 50.0, 1.0]])
    reference = exact_posterior(x).mean[0]
    approximation = laplace_moments(x).mean[0]
    assert reference > 5 and approximation < 0.2
    mu = np.linspace(0.1, 10, 201)
    log_density = xlogy(20, mu + 10) - mu - 10 - np.log(mu)
    assert mu[np.argmax(log_density)] == 0.1


def test_final_runner_requires_real_path_a_model(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_path_b.py"
    spec = importlib.util.spec_from_file_location("path_b_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = tmp_path
    with pytest.raises(FileNotFoundError, match="Path A"):
        runner.load_posterior(FULL_CONFIG)


def test_result_json_envelopes_and_finite_metrics():
    root = Path(__file__).resolve().parents[1]
    required = {
        "schema_version",
        "path",
        "generated_at",
        "seed",
        "config",
        "runtime_seconds",
        "environment",
        "methods",
        "diagnostics",
        "artifacts",
        "notes",
    }

    def check(value):
        if isinstance(value, dict):
            if {"value", "mc_se", "n_samples", "method"} <= value.keys():
                assert value["mc_se"] >= 0 and value["n_samples"] >= 0
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, float):
            assert np.isfinite(value)

    for name in ("a", "b"):
        result = json.loads((root / f"results/path_{name}.json").read_text())
        assert required <= result.keys() and result["schema_version"] == (1 if name == "a" else 2)
        assert result["path"] == name
        check(result)
        if name == "b":
            with gzip.open(root / result["artifacts"]["details"], "rt", encoding="utf-8") as stream:
                details = json.load(stream)
            assert required <= details.keys() and details["schema_version"] == 1
            assert details["path"] == name
            check(details)


def test_path_b_summary_and_lossless_serialization(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("path_b_runner", root / "scripts/run_path_b.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    summary_path = root / "results/path_b.json"
    summary = json.loads(summary_path.read_text())
    details = runner.load_details(root / summary["artifacts"]["details"])
    assert summary_path.stat().st_size < 300_000
    assert summary["format"] == "summary"
    for key in ("config", "seed", "runtime_seconds", "environment", "generated_at"):
        assert summary[key] == details[key]
    for check in details["diagnostics"]["coverage_check"]:
        name = check["method"]
        actual = summary["methods"][name]
        assert "coverage" not in actual
        assert actual["interval_summary"] == details["methods"][name]["interval_summary"]
        assert actual["bayes_risk"] == details["methods"][name]["bayes_risk"]
        verdict = actual["coverage_check"]
        assert verdict["minimum_coverage"] == {
            key: check["minimum_coverage"][key] for key in ("theta", "coverage")
        }
        assert verdict["flagged_count_before_reruns"] == len(check["flagged_points"])
        assert verdict["flagged_count_after_reruns"] == len(check["remaining_flagged_points"])
        assert verdict["passed"] == check["passed"]
        rows = details["diagnostics"]["set_diagnostics"].get(name, {}).get("initial_toys", [])
        disconnected = [row for row in rows if "disconnected_rate" in row]
        if disconnected:
            maximum = max(disconnected, key=lambda row: row["disconnected_rate"]["value"])
            assert actual["maximum_disconnected_rate"] == {
                key: maximum[key] for key in ("theta", "disconnected_rate")
            }
        else:
            assert actual["maximum_disconnected_rate"] is None
    original = json.dumps(details)
    destination = tmp_path / "results/path_b.json"
    regenerated = runner.write_results(details, destination)
    assert destination.stat().st_size < 300_000
    assert runner.load_details(destination.with_name("path_b_details.json.gz")) == details
    assert json.dumps(details) == original, "Serialization must not mutate the statistical payload"
    assert regenerated["methods"] == summary["methods"]
    assert regenerated["diagnostics"] == summary["diagnostics"]
