"""Fast numerical and published-reference checks for Path B."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gamma, norm

from bayesfreq.config import CORE_CONFIG, FULL_CONFIG
from bayesfreq.data_simulator import poisson_support, simulate, theta_grid
from bayesfreq.frequentist_bounds import (
    _fc_acceptance,
    _fc_merge_walk,
    central_neyman_interval,
    exact_coverage,
    feldman_cousins_belt,
    flat_prior_interval,
    naive_wald_interval,
    profile_background,
    upper_limit_interval,
)

# Feldman and Cousins (1998), Tables IV–V, 90% CL (physics/9711021): all 420 published
# intervals, as rows [n0, b, lower, upper]; see tests/data/fc1998_90cl.json.
FC_TABLE = json.loads((Path(__file__).parent / "data/fc1998_90cl.json").read_text())["entries"]


@pytest.fixture(scope="module")
def corrected_intervals():
    """FC's corrected intervals on the validation grid, built once per background."""
    grid = theta_grid(validation=True)
    cache = {}

    def intervals(b):
        if b not in cache:
            belt = feldman_cousins_belt(
                b=b, mu_grid=grid, n_max=FULL_CONFIG.grid.n_fc_max, enforce_b_monotone=True
            )
            cache[b] = belt(np.column_stack((np.arange(21), np.zeros(21), np.ones(21))))
        return cache[b]

    return intervals


def check_published(rows, intervals):
    for n, b, lower, upper in rows:
        found = intervals(b)
        assert found.lower[n] == pytest.approx(lower, abs=0.0100001), (n, b)
        assert found.upper[n] == pytest.approx(upper, abs=0.0100001), (n, b)


@pytest.mark.slow
def test_fc_published_tables_all_entries(corrected_intervals):
    assert len(FC_TABLE) == 420
    check_published(FC_TABLE, corrected_intervals)


def test_fc_published_b15_column(corrected_intervals):
    # Fast subset: the correction changes the b = 15 entries for n0 = 1..10, for
    # example (2, 15) and (10, 15); it needs only the backgrounds b' >= 15.
    rows = [row for row in FC_TABLE if row[1] == 15.0]
    assert len(rows) == 21
    check_published(rows, corrected_intervals)


@pytest.fixture(scope="module")
def reference_belts():
    grid = theta_grid(validation=True)
    return {3.0: feldman_cousins_belt(b=3.0, mu_grid=grid, n_max=150)}


def test_fc_table_one_ordering_and_crossing(reference_belts):
    belt = reference_belts[3.0]
    row = np.argmin(abs(belt.mu_grid - 0.5))
    np.testing.assert_array_equal(np.flatnonzero(belt.acceptance[row]), np.arange(7))


def test_merge_walk_reproduces_sorted_fc_construction():
    grid = np.linspace(0, 8, 161)
    counts = poisson_support(grid[-1] + 25)
    backgrounds = np.array([0.0, 0.37, 3.0, 12.5, 25.0])
    lowest, highest = _fc_merge_walk(grid, counts.size, backgrounds, 0.1)
    for k, b in enumerate(backgrounds):
        accepted = _fc_acceptance(grid, counts, float(b), 0.1)
        np.testing.assert_array_equal(lowest[k], np.argmax(accepted, axis=1))
        np.testing.assert_array_equal(highest[k], counts.size - 1 - np.argmax(accepted[:, ::-1], 1))


def test_b_monotone_correction_matches_its_definition():
    # Brute force of FC's rule on a coarse grid: the corrected upper endpoint is the
    # largest plain endpoint over b' in {b} and the 0.01 grid on [b, 25]; the count is
    # then accepted from its plain lower endpoint up to that endpoint.
    grid = np.linspace(0, 5, 51)
    b = 2.005
    counts = poisson_support(grid[-1] + 25)

    def plain_upper(accepted):
        last = grid[grid.size - 1 - np.argmax(accepted[::-1], axis=0)]
        return np.where(accepted.any(axis=0), last, -np.inf)

    plain = _fc_acceptance(grid, counts, b, 0.1)
    upper = plain_upper(plain)
    for aux in np.arange(201, 2501) / 100:
        upper = np.maximum(upper, plain_upper(_fc_acceptance(grid, counts, float(aux), 0.1)))
    nodes = grid[:, None]
    lengthen = upper > plain_upper(plain)
    expected = plain | (lengthen & (nodes >= grid[np.argmax(plain, axis=0)]) & (nodes <= upper))
    belt = feldman_cousins_belt(
        b=b, mu_grid=grid, n_max=int(counts[-1]), enforce_b_monotone=True, config=CORE_CONFIG
    )
    np.testing.assert_array_equal(belt.acceptance, expected)
    assert np.any(belt.acceptance & ~plain), "the correction should change this belt"
    with pytest.raises(ValueError, match="b<=25"):
        feldman_cousins_belt(b=25.5, mu_grid=grid, enforce_b_monotone=True)


def test_corrected_upper_endpoints_do_not_increase_with_background():
    x = np.column_stack((np.arange(8), np.zeros(8), np.ones(8)))
    uppers = np.array(
        [
            feldman_cousins_belt(b=b, config=CORE_CONFIG, enforce_b_monotone=True)(x).upper
            for b in np.arange(0, 6.01, 0.25)
        ]
    )
    assert np.all(np.diff(uppers, axis=0) <= 1e-12)


@pytest.mark.parametrize("correct", [False, True])
def test_bounded_fc_nonempty_monotone_and_exact_coverage(correct):
    belt = feldman_cousins_belt(config=CORE_CONFIG, enforce_b_monotone=correct)
    x = np.column_stack((belt.counts, np.zeros(belt.counts.size), np.ones(belt.counts.size)))
    intervals = belt(x)
    assert np.all(np.isfinite(intervals.lower))
    assert np.all(intervals.lower >= 0) and np.all(intervals.upper <= 10)
    assert np.all(np.diff(intervals.lower) >= 0)
    assert np.all(np.diff(intervals.upper) >= 0)
    coverage = exact_coverage(belt, belt.mu_grid, config=CORE_CONFIG)
    assert np.min(coverage.value) >= 0.9 - 1e-12
    assert coverage.exact and coverage.n_samples == 0
    assert np.all(coverage.mc_se == 0)
    with pytest.raises(ValueError, match="exceeds belt support"):
        belt(np.array([[10000.0, 0.0, 1.0]]))


def test_wald_boundary_undercoverage_and_empty_rate():
    def procedure(x):
        return naive_wald_interval(x, config=CORE_CONFIG)

    result = exact_coverage(procedure, np.array([0.0, 0.01, 0.05, 0.1, 1.0]), config=CORE_CONFIG)
    assert result.value[0] < 0.85
    assert result.empty_rate[0] > 0.04
    assert np.all(result.expected_length >= 0)


def test_central_and_upper_limits_keep_empty_intersections():
    x = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
    central = central_neyman_interval(x, b=3)
    upper = upper_limit_interval(x, b=3)
    assert np.isnan(central.lower[0]) and np.isnan(central.upper[0])
    assert np.isnan(upper.lower[0]) and np.isnan(upper.upper[0])
    assert np.isnan(central.lower[2])
    assert upper.upper[2] == 10


def test_full_wald_variance_matches_sampling_variance_of_mu_hat():
    # Row (6, 5, 1) is at the mean counts of (mu, b, eps) = (5, 1, 1), and the interval
    # lies inside Theta, so its half-width is z * sqrt(V) for the delta-method V.
    interval = naive_wald_interval(np.array([[6.0, 5.0, 1.0]]))
    variance = ((interval.upper - interval.lower) / (2 * norm.ppf(0.95))) ** 2
    x = simulate(5.0, 1.0, 1.0, rng=np.random.default_rng(3), size=400000)
    mu_hat = (x[:, 0] - x[:, 1] / FULL_CONFIG.problem.tau) / x[:, 2]
    assert variance[0] == pytest.approx(mu_hat.var(), rel=0.05)


def test_profile_background_closed_form_and_zero_boundary():
    n, m, mu = (
        np.array([2.0, 8.0, 0.0, 4.0]),
        np.array([15.0, 15.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 7.0, 8.0]),
    )
    a = n + m - 6 * mu
    expected = (a + np.sqrt(a**2 + 24 * m * mu)) / 12
    np.testing.assert_allclose(profile_background(n, m, mu), expected, atol=1e-12)
    assert profile_background(np.array(0.0), np.array(0.0), np.array(1e10)) == 0


def test_flat_core_posterior_matches_truncated_gamma():
    config = replace(CORE_CONFIG, problem=replace(CORE_CONFIG.problem, known_b=0.0))
    x = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
    intervals = flat_prior_interval(x, config=config)
    for i, n in enumerate(x[:, 0]):
        mass = gamma.cdf(10, n + 1)
        expected = gamma.ppf(np.array([0.05, 0.95]) * mass, n + 1)
        np.testing.assert_allclose([intervals.lower[i], intervals.upper[i]], expected, atol=0.011)


def test_full_flat_posterior_quadrature_converges():
    x = np.array([[2.0, 15.0, 0.92], [8.0, 15.0, 1.1]])
    coarse = flat_prior_interval(x)
    finer = replace(FULL_CONFIG, grid=replace(FULL_CONFIG.grid, mu_step=0.005, eps_points=61))
    fine = flat_prior_interval(x, config=finer)
    np.testing.assert_allclose(coarse.lower, fine.lower, atol=0.015)
    np.testing.assert_allclose(coarse.upper, fine.upper, atol=0.015)
