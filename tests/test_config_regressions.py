"""Regression checks for configured confidence levels and local execution state."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
import torch
from scipy.stats import binom, chi2

from bayesfreq import bayesian_risk
from bayesfreq.config import CORE_CONFIG, FULL_CONFIG
from bayesfreq.frequentist_bounds import (
    CalibratedBelt,
    LocalPosteriorStandIn,
    central_neyman_interval,
    exact_coverage,
    exact_waldo_belt,
    feldman_cousins_belt,
    flat_prior_interval,
    naive_wald_interval,
    upper_limit_interval,
)
from bayesfreq.interfaces import IntervalBatch


@pytest.mark.parametrize("name", ["fc", "wald", "central", "upper", "flat", "waldo"])
def test_exact_core_coverage_accepts_fractional_background(name):
    config = replace(CORE_CONFIG, problem=replace(CORE_CONFIG.problem, known_b=0.5))
    grid = np.linspace(0, config.problem.mu_max, 51)

    def procedure(cfg):
        if name == "fc":
            return feldman_cousins_belt(b=0.5, mu_grid=grid, config=cfg)
        if name == "waldo":
            return exact_waldo_belt(posterior=LocalPosteriorStandIn(cfg), mu_grid=grid, config=cfg)
        function = {
            "wald": naive_wald_interval,
            "central": central_neyman_interval,
            "upper": upper_limit_interval,
            "flat": flat_prior_interval,
        }[name]
        kwargs = {"b": 0.5} if name in ("central", "upper") else {}
        return partial(function, config=cfg, **kwargs)

    result = exact_coverage(procedure(config), grid, b=0.5, config=config)
    integer_off_config = replace(config, problem=replace(config.problem, tau=2.0))
    reference = exact_coverage(
        procedure(integer_off_config), grid, b=0.5, config=integer_off_config
    )
    for attribute in ("value", "expected_length", "empty_rate"):
        np.testing.assert_array_equal(getattr(result, attribute), getattr(reference, attribute))
    assert result.exact and np.all(result.mc_se == 0)
    if name in ("fc", "waldo"):
        assert np.min(result.value) >= 1 - config.problem.alpha - 1e-10


def test_fc_default_alpha_uses_config_and_explicit_alpha_overrides_it():
    config = replace(CORE_CONFIG, problem=replace(CORE_CONFIG.problem, alpha=0.05))
    grid = np.linspace(0, config.problem.mu_max, 101)
    belt = feldman_cousins_belt(mu_grid=grid, config=config)
    explicit = feldman_cousins_belt(alpha=0.05, mu_grid=grid, config=config)
    override = feldman_cousins_belt(alpha=0.1, mu_grid=grid, config=config)
    np.testing.assert_array_equal(belt.acceptance, explicit.acceptance)
    assert np.any(belt.acceptance != override.acceptance)
    coverage = exact_coverage(belt, grid, config=config)
    assert coverage.value.min() >= 0.95 - 1e-10


@pytest.mark.parametrize("function", [central_neyman_interval, upper_limit_interval])
def test_exact_poisson_limits_use_configured_alpha(function):
    config = replace(CORE_CONFIG, problem=replace(CORE_CONFIG.problem, alpha=0.05))
    counts = np.array([3.0, 5.0, 8.0])
    x = np.column_stack((counts, np.zeros_like(counts), np.ones_like(counts)))
    result = function(x, config=config)
    tail = 0.025 if function is central_neyman_interval else 0.05
    upper = np.minimum(chi2.ppf(1 - tail, 2 * (counts + 1)) / 2 - 3, 10)
    lower = (
        np.maximum(chi2.ppf(tail, 2 * counts) / 2 - 3, 0)
        if function is central_neyman_interval
        else np.zeros_like(counts)
    )
    np.testing.assert_array_equal(result.lower, lower)
    np.testing.assert_array_equal(result.upper, upper)
    override = function(x, alpha=0.1, config=config)
    reference = function(x, config=CORE_CONFIG)
    np.testing.assert_array_equal(override.lower, reference.lower)
    np.testing.assert_array_equal(override.upper, reference.upper)


@pytest.mark.parametrize("fail", [False, True])
def test_training_restores_torch_threads_on_success_and_failure(monkeypatch, fail):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        if fail:

            def fail_simulation(*args, **kwargs):
                assert torch.get_num_threads() == 1
                raise RuntimeError("simulation failed")

            monkeypatch.setattr(bayesian_risk, "sample_joint", fail_simulation)
            with pytest.raises(RuntimeError, match="simulation failed"):
                bayesian_risk.train_amortized(config=CORE_CONFIG, n_train=20, epochs=1)
        else:
            bayesian_risk.train_amortized(config=CORE_CONFIG, n_train=20, epochs=1)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous_threads)


def test_path_b_metadata_efficiencies_and_seed_positions_follow_config(monkeypatch):
    namespace = run_path(str(Path(__file__).parents[1] / "scripts/run_path_b.py"))
    main = namespace["main"]
    config = replace(
        FULL_CONFIG,
        problem=replace(FULL_CONFIG.problem, alpha=0.2, eps0=1.7),
        grid=replace(
            FULL_CONFIG.grid,
            mu_step=1.0,
            mu_step_calib=2.0,
            mu_cov_grid=(0.0, 1.0),
            b_cov_grid=(3.0,),
            eps_points=5,
        ),
        mc=replace(
            FULL_CONFIG.mc,
            n_toys_cov=20,
            n_toys_calib=20,
            m_bayes_risk=20,
            flagged_multiplier=1,
        ),
    )
    captured = {}

    def capture(output, destination):
        captured.update(output)

    monkeypatch.setitem(main.__globals__, "FULL_CONFIG", config)
    monkeypatch.setitem(
        main.__globals__,
        "load_posterior",
        lambda cfg: (LocalPosteriorStandIn(cfg), {"path": "test_model"}),
    )
    monkeypatch.setitem(main.__globals__, "reference_checks", lambda cfg: {})
    monkeypatch.setitem(main.__globals__, "draw_figures", lambda *args: [])
    monkeypatch.setitem(main.__globals__, "write_results", capture)
    monkeypatch.setattr("sys.argv", ["run_path_b.py"])
    main()

    diagnostics = captured["diagnostics"]
    assert diagnostics["calibration"]["quantile_order"] == 17
    full_checks = [c for c in diagnostics["coverage_check"] if c["method"].endswith("full")]
    assert all(c["domain"]["eps"] == [1.7] for c in full_checks)
    off_grid = np.array(diagnostics["off_grid_coverage_check"][0]["domain"]["theta"])
    np.testing.assert_array_equal(
        np.unique(off_grid[:, 2]), 1.7 * np.exp(np.array([-1.5, 0, 1.5]) * 0.1)
    )
    streams = diagnostics["rng_streams"]
    assert "numerical_checks" not in streams
    reference = np.random.SeedSequence(config.seed).spawn(13)
    positions = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    for stream, index in zip(streams.values(), positions, strict=True):
        assert stream["spawn_key"] == list(reference[index].spawn_key)
        reconstructed = np.random.SeedSequence(stream["entropy"], spawn_key=stream["spawn_key"])
        np.testing.assert_array_equal(
            reconstructed.generate_state(8), reference[index].generate_state(8)
        )
    # The main grid is mu in {0, 1, 10} (mu_max is always added) at b = 3 and eps0:
    # all are calibration nuisance rows; mu = 0 and 10 are on the calibration
    # signal grid (step 2), mu = 1 is not.
    overlap = diagnostics["calibration_node_overlap"]
    assert overlap["coverage"] == {
        "n_points": 3,
        "nuisance_on_node": 3,
        "mu_and_nuisance_on_node": 2,
    }
    assert overlap["random_box"]["n_points"] == 300
    assert overlap["random_box"]["nuisance_on_node"] == 0
    box = diagnostics["random_box_check"]
    assert [check["method"] for check in box["checks"]] == [
        "profile_lr_full",
        "waldo_full",
        "profile_lr_full_eps0_only",
        "waldo_full_eps0_only",
    ]
    theta = np.array(box["theta"])
    assert np.all((theta[:, 0] >= 0) & (theta[:, 0] <= 10))
    assert np.all(abs(np.log(theta[:, 2] / 1.7)) <= 0.2 + 1e-12)
    for name in ("profile_lr_full", "waldo_full"):
        reestimated = diagnostics["reestimated_minima"][name]
        assert {
            point["reestimated"]["n_samples"] for point in reestimated["coverage"]["points"]
        } == {20}
    fc_check = next(c for c in diagnostics["coverage_check"] if c["method"] == "fc_core")
    assert len(fc_check["domain"]["mu"]) == 10001 and "on_node_minimum_coverage" in fc_check
    assert diagnostics["core_off_node_scan"]["fc_core"]["n_points"] == 10001
    assert {"platform", "machine"} <= captured["environment"].keys()


def test_monte_carlo_coverage_check_requires_rerun_inputs():
    runner = run_path(str(Path(__file__).parents[1] / "scripts/run_path_b.py"))
    rows = [
        {
            "theta": [1.0, 3.0, 1.0],
            "coverage": {"value": 0.5, "mc_se": 0.01, "n_samples": 100, "method": "mc"},
        }
    ]
    with pytest.raises(ValueError, match="rng and procedure"):
        runner["coverage_check"]("m", rows, FULL_CONFIG, domain={}, exact=False)
    exact = runner["coverage_check"]("m", rows, CORE_CONFIG, domain={}, exact=True)
    assert not exact["passed"]


def test_pass_rule_operating_characteristic_is_exact():
    runner = run_path(str(Path(__file__).parents[1] / "scripts/run_path_b.py"))
    result = runner["pass_rule_characteristic"](FULL_CONFIG, {"coverage": 728})
    # 3,540 of 4,000 (and 35,400 of 40,000) is the smallest count that passes 0.885.
    assert result["smallest_passing_count"] == {"initial": 3540, "rerun": 35400}
    rows = {row["true_coverage"]: row for row in result["by_true_coverage"]}
    # Independent binomial calculation of the same two-stage rule.
    flag = binom.cdf(3539, 4000, 0.89)
    assert rows[0.89]["initial_flag_probability"] == pytest.approx(flag, rel=1e-12)
    assert rows[0.89]["final_pass_probability"] == pytest.approx(
        1 - flag + flag * binom.sf(35399, 40000, 0.89), rel=1e-12
    )
    # Values reported in an earlier, separate analysis of the rule.
    assert rows[0.90]["initial_flag_probability"] == pytest.approx(8.79e-4, rel=2e-3)
    assert rows[0.885]["final_pass_probability"] == pytest.approx(0.7581, abs=1e-4)
    assert rows[0.88]["final_pass_probability"] == pytest.approx(0.1723, abs=1e-4)
    assert rows[0.875]["final_pass_probability"] == pytest.approx(0.0284, abs=1e-4)
    assert result["any_initial_flag_at_090"]["coverage"]["probability"] == pytest.approx(
        0.473, abs=1e-3
    )


def test_monte_carlo_rerun_decides_whether_a_flagged_point_passes():
    runner = run_path(str(Path(__file__).parents[1] / "scripts/run_path_b.py"))

    def always_covers(x):
        return IntervalBatch(np.zeros(len(x)), np.full(len(x), 10.0))

    rows = [
        {
            "theta": [1.0, 3.0, 1.0],
            "coverage": {"value": 0.5, "mc_se": 0.01, "n_samples": 100, "method": "mc"},
        }
    ]
    check = runner["coverage_check"](
        "m",
        rows,
        FULL_CONFIG,
        domain={},
        exact=False,
        rng=np.random.default_rng(0),
        procedure=always_covers,
    )
    assert check["flagged_points"] == [[1.0, 3.0, 1.0]]
    assert check["remaining_flagged_points"] == [] and check["passed"]


def test_random_box_check_reruns_flagged_points():
    runner = run_path(str(Path(__file__).parents[1] / "scripts/run_path_b.py"))
    calls = {"n": 0}

    def statistic(x, mu):
        # The first (initial) call misses every toy; the reruns cover every toy.
        calls["n"] += 1
        return np.full(x.shape[:-1], 1.0 if calls["n"] == 1 else 0.0)

    belt = CalibratedBelt(
        np.array([0.0, 10.0]), np.array([0.5, 0.5]), statistic, np.array([[3.0, 1.0]]), 0.1
    )
    _, checks = runner["random_box_check"](
        {"stub": belt},
        np.array([[5.0, 3.0, 1.0]]),
        FULL_CONFIG,
        np.random.default_rng(1),
        np.random.default_rng(2),
    )
    assert checks[0]["flagged_points"] and not checks[0]["remaining_flagged_points"]
    assert checks[0]["passed"]


def test_random_box_points_fill_the_nuisance_box():
    runner = run_path(str(Path(__file__).parents[1] / "scripts/run_path_b.py"))
    points = runner["random_box_points"](FULL_CONFIG, np.random.default_rng(3))
    problem = FULL_CONFIG.problem
    span = np.abs(np.log(points[:, 2] / problem.eps0)) / problem.sigma_eps
    assert 1.9 < span.max() <= 2 + 1e-12
    assert points[:, 0].max() > 9.5
    b_range = np.log(FULL_CONFIG.grid.b_cov_grid[-1] / FULL_CONFIG.grid.b_cov_grid[0])
    assert np.log(points[:, 1].max() / points[:, 1].min()) > 0.95 * b_range
