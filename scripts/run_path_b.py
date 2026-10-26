"""Run the Path B study: exact core intervals and calibrated full-case intervals.

The core case (known b, eps = 1) uses exact Poisson sums for the FC, Wald,
central, upper-limit, flat-prior, and WALDO intervals. The full case calibrates
profile likelihood ratio and WALDO belts over a bounded (b, eps) grid and checks
their coverage with independent toys; WALDO uses the saved Path A posterior
(results/path_a_amortized.pt). Off-node checks test the calibrated sets away
from their calibration nodes, and the exact core belts are also checked between
their grid nodes. The script writes results/path_b.json (summary),
results/path_b_details.json.gz (complete payload), and four figures. Use --quick
for smaller budgets. Both experiment modes overwrite results/ and figures/.
Use --figures-only to redraw saved figures without simulation or results writes.
"""

import argparse
import gzip
import hashlib
import json
import platform
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, poisson

from bayesfreq.bayesian_risk import AmortizedPosterior, clipped_mle
from bayesfreq.config import FULL_CONFIG, StudyConfig
from bayesfreq.data_simulator import poisson_support, sample_joint, simulate, theta_grid
from bayesfreq.frequentist_bounds import (
    CalibratedBelt,
    LocalPosteriorStandIn,
    NeymanBelt,
    calibrate_belt,
    central_neyman_interval,
    estimate_coverage,
    exact_coverage,
    exact_waldo_belt,
    feldman_cousins_belt,
    flat_prior_interval,
    naive_wald_interval,
    profile_likelihood_statistic,
    refine_belt,
    upper_limit_interval,
    waldo_statistic,
)
from bayesfreq.interfaces import (
    CoverageResult,
    IntervalBatch,
    IntervalProcedure,
    PosteriorPredictor,
)
from bayesfreq.plotting import METHOD_COLORS, METHOD_LABELS, save_figure, saved_config, set_style

ROOT = Path(__file__).resolve().parents[1]
# Feldman and Cousins (1998), Tables IV and V, 90% CL: all 420 published intervals,
# as [n_on, b, lower limit, upper limit] rows (path relative to ROOT).
FC_TABLE = Path("tests/data/fc1998_90cl.json")
# Exact core coverage is also scanned between grid nodes, at this signal step.
SCAN_STEP = 0.001
# Random points for the off-node check inside the calibration box.
N_BOX_POINTS = 300
# Lowest-coverage points re-estimated with fresh toys, per check grid.
N_REESTIMATED = 5


def metric(value: float, se: float, n: int, exact: bool = False) -> dict[str, Any]:
    """Return a Metric record for a measured probability or expectation.

    Parameters
    ----------
    value : float
        Measured value.
    se : float
        Monte Carlo standard error (0 for exact sums).
    n : int
        Number of toys (0 for exact sums).
    exact : bool, default False
        If True, ``method`` is ``"exact"``; otherwise ``"mc"``.

    Returns
    -------
    dict
        ``{"value", "mc_se", "n_samples", "method"}`` as in docs/DESIGN.md.
    """
    return {
        "value": float(value),
        "mc_se": float(se),
        "n_samples": int(n),
        "method": "exact" if exact else "mc",
    }


def coverage_rows(result: CoverageResult) -> list[dict[str, Any]]:
    """Convert a CoverageResult into one JSON record per theta row.

    Parameters
    ----------
    result : CoverageResult
        Exact or Monte Carlo coverage result.

    Returns
    -------
    list of dict
        Records ``{"theta", "coverage", "expected_length", "empty_rate"}``;
        every value except ``theta`` is a Metric from ``metric``.
    """
    rows = []
    for i, theta in enumerate(result.theta):
        rows.append(
            {
                "theta": theta.tolist(),
                "coverage": metric(
                    result.value[i], result.mc_se[i], result.n_samples, result.exact
                ),
                "expected_length": metric(
                    result.expected_length[i],
                    result.length_mc_se[i],
                    result.n_samples,
                    result.exact,
                ),
                "empty_rate": metric(
                    result.empty_rate[i], result.empty_mc_se[i], result.n_samples, result.exact
                ),
            }
        )
    return rows


def coverage_check(
    name: str,
    rows: list[dict[str, Any]],
    config: StudyConfig,
    *,
    domain: dict[str, Any],
    exact: bool,
    rng: np.random.Generator | None = None,
    procedure: IntervalProcedure | None = None,
) -> dict[str, Any]:
    """Flag low hull coverage, rerun flagged points (MC only), and give a verdict.

    Parameters
    ----------
    name : str
        Method identifier.
    rows : list of dict
        Records from ``coverage_rows``. Changed in place: each flagged row is
        replaced by its rerun record.
    config : StudyConfig
        Supplies alpha, tolerances, and toy budgets.
    domain : dict
        Description of the checked grid, stored unchanged.
    exact : bool
        True for exact sums (tolerance ``tol_cov_exact``, no reruns); False
        for Monte Carlo (tolerance ``tol_cov_mc``).
    rng : numpy.random.Generator, optional
        Stream for reruns. Required when ``exact`` is False.
    procedure : IntervalProcedure, optional
        Procedure to rerun. Required when ``exact`` is False.

    Returns
    -------
    dict
        Check record (see "Results JSON schema" in docs/DESIGN.md) with
        ``coverage_target="hull"``, the minimum before reruns
        (``initial_minimum``) and after them (``minimum_coverage``), and
        ``passed`` True when no point stays below the threshold.

    Raises
    ------
    ValueError
        If ``exact`` is False and ``rng`` or ``procedure`` is missing.

    Notes
    -----
    A point is flagged when its coverage is below ``1 - alpha - tolerance``.
    Reruns use ``n_toys_cov * flagged_multiplier`` fresh toys per point.
    """
    if not exact and (rng is None or procedure is None):
        raise ValueError("a Monte Carlo coverage check needs rng and procedure for reruns")
    tolerance = config.mc.tol_cov_exact if exact else config.mc.tol_cov_mc
    target = 1 - config.problem.alpha
    initial = [row for row in rows if row["coverage"]["value"] < target - tolerance]
    rerun_n = config.mc.n_toys_cov * config.mc.flagged_multiplier
    reruns = []
    if initial and not exact:
        parameters = np.array([row["theta"] for row in initial])
        checked = estimate_coverage(
            procedure, parameters, rng=rng, n_samples=rerun_n, config=config
        )
        reruns = coverage_rows(checked)
        by_theta = {tuple(row["theta"]): row for row in reruns}
        rows[:] = [by_theta.get(tuple(row["theta"]), row) for row in rows]
    minimum = min(rows, key=lambda row: row["coverage"]["value"])
    remaining = [row["theta"] for row in rows if row["coverage"]["value"] < target - tolerance]
    return {
        "method": name,
        "alpha": config.problem.alpha,
        "tolerance": tolerance,
        "domain": domain,
        "coverage_target": "hull",
        "minimum_coverage": minimum,
        "initial_minimum": min(row["coverage"]["value"] for row in initial)
        if initial
        else minimum["coverage"]["value"],
        "flagged_points": [row["theta"] for row in initial],
        "rerun_n_samples": 0 if exact else rerun_n,
        "reruns": reruns,
        "remaining_flagged_points": remaining,
        "passed": not remaining,
    }


class DiagnosticsProcedure:
    """Wrap an interval procedure and record set diagnostics for each toy batch.

    ``estimate_coverage`` calls the procedure once per theta row, in order, and
    the wrapper relies on this: call k is matched to ``theta[k]``. Each call
    records the degenerate rate (finite hulls with ``lower == upper``). When
    raw grid sets are available, it also records the disconnected rate (raw
    sets with more than one run of accepted grid points) and, if the true mu
    is a grid node, the raw-set coverage. Rates carry binomial errors.
    """

    def __init__(self, procedure: IntervalProcedure, theta: NDArray[np.float64]) -> None:
        """Store the procedure and the theta rows in call order.

        Parameters
        ----------
        procedure : IntervalProcedure
            Procedure to wrap.
        theta : ndarray of float64, shape (K, 3)
            Rows ``(mu, b, eps)``, in the order of the expected calls.
        """
        self.procedure = procedure
        self.theta = theta
        self.rows = []

    def __call__(self, x: NDArray[np.float64]) -> IntervalBatch:
        """Run the wrapped procedure on one toy batch and record its diagnostics.

        Parameters
        ----------
        x : ndarray of float64, shape (N, 3)
            Toys simulated at the next unused row of ``theta``.

        Returns
        -------
        IntervalBatch
            The wrapped procedure's result, unchanged.
        """
        result = self.procedure(x)
        mu = self.theta[len(self.rows), 0]
        count = x.shape[0]
        degenerate = np.mean(np.isfinite(result.lower) & (result.lower == result.upper))
        row = {
            "theta": self.theta[len(self.rows)].tolist(),
            "degenerate_rate": metric(
                degenerate, np.sqrt(degenerate * (1 - degenerate) / count), count
            ),
        }
        if result.accepted is not None:
            starts = np.sum(result.accepted[:, 1:] & ~result.accepted[:, :-1], axis=1)
            starts += result.accepted[:, 0]
            disconnected = np.mean(starts > 1)
            row["disconnected_rate"] = metric(
                disconnected, np.sqrt(disconnected * (1 - disconnected) / count), count
            )
            index = np.argmin(abs(result.mu_grid - mu))
            if abs(result.mu_grid[index] - mu) < 1e-10:
                raw = np.mean(result.accepted[:, index])
                row["raw_set_coverage"] = metric(raw, np.sqrt(raw * (1 - raw) / count), count)
        self.rows.append(row)
        return result


def prior_summary(
    procedure: IntervalProcedure, config: StudyConfig, rng: np.random.Generator
) -> dict[str, Any]:
    """Average interval metrics and the midpoint Bayes risk under the reality prior.

    Parameters
    ----------
    procedure : IntervalProcedure
        Interval procedure to evaluate.
    config : StudyConfig
        Core or full model; ``config.mc.m_bayes_risk`` gives the number of
        draws M.
    rng : numpy.random.Generator
        Stream for the (theta, x) draws.

    Returns
    -------
    dict
        Metrics ``coverage``, ``expected_length``, ``empty_rate``, and
        ``bayes_risk`` (``squared`` and ``absolute`` loss), plus a
        ``point_summary`` description.

    Notes
    -----
    M independent pairs come from ``sample_joint`` (the reality prior). The
    point summary is the hull midpoint, or the clipped MLE when the hull is
    empty. Errors are ``sqrt(p (1 - p) / M)`` for rates and
    ``std(ddof=1) / sqrt(M)`` for means.
    """
    count = config.mc.m_bayes_risk
    theta, x = sample_joint(count, rng=rng, config=config)
    batch = procedure(x)
    empty = ~np.isfinite(batch.lower) | ~np.isfinite(batch.upper)
    length = np.where(empty, 0, batch.upper - batch.lower)
    covered = (theta[:, 0] >= batch.lower - 1e-12) & (theta[:, 0] <= batch.upper + 1e-12)
    p, e = np.mean(covered), np.mean(empty)
    midpoint = np.where(empty, clipped_mle(x, config=config), (batch.lower + batch.upper) / 2)
    residual = midpoint - theta[:, 0]
    return {
        "coverage": metric(p, np.sqrt(p * (1 - p) / count), count),
        "expected_length": metric(np.mean(length), np.std(length, ddof=1) / np.sqrt(count), count),
        "empty_rate": metric(e, np.sqrt(e * (1 - e) / count), count),
        "bayes_risk": {
            loss: metric(np.mean(values), np.std(values, ddof=1) / np.sqrt(count), count)
            for loss, values in [("squared", residual**2), ("absolute", abs(residual))]
        },
        "point_summary": "inclusive-hull midpoint; clipped-MLE fallback for empty hulls",
    }


def reference_checks(config: StudyConfig) -> dict[str, Any]:
    """Compare FC intervals with all 420 published Feldman-Cousins table entries.

    Parameters
    ----------
    config : StudyConfig
        Supplies the validation grid (``mu_grid_max_fc``, ``mu_step_fc``),
        the count support ``n_fc_max``, and the tolerance
        ``config.grid.fc_table_tol``.

    Returns
    -------
    dict
        Per-entry comparisons, an overall ``passed`` flag, the correction
        settings, and a b-monotonicity diagnostic of the plain and corrected
        upper endpoints.

    Notes
    -----
    Every entry of FC Tables IV-V (90% CL; n_on = 0..20, b = 0..15, read from
    ``FC_TABLE``) is recomputed with the corrected FC belt
    (``enforce_b_monotone=True``), which is FC's own construction, on the
    validation grid with counts 0..``n_fc_max``. The plain belt is recorded
    alongside. An entry passes when both endpoints are within the tolerance.
    The diagnostic counts every step b -> b + 0.1 (b from 0 to 15) where an
    upper endpoint for n_on = 0..20 increases, for both belts.
    """
    entries = json.loads((ROOT / FC_TABLE).read_text(encoding="utf-8"))["entries"]
    validation = theta_grid(config, validation=True)
    n_max = config.grid.n_fc_max
    largest = max(int(entry[0]) for entry in entries)
    x = np.column_stack((np.arange(largest + 1), np.zeros(largest + 1), np.ones(largest + 1)))

    def endpoints(b: float, corrected: bool) -> IntervalBatch:
        """Return the plain or corrected FC intervals for n_on = 0..largest at b."""
        belt = feldman_cousins_belt(
            b=b, mu_grid=validation, n_max=n_max, enforce_b_monotone=corrected
        )
        return belt(x)

    backgrounds = sorted({float(entry[1]) for entry in entries})
    plain = {b: endpoints(b, False) for b in backgrounds}
    ours = {b: endpoints(b, True) for b in backgrounds}
    rows = []
    for n, b, lo, hi in entries:
        n, b = int(n), float(b)
        found = [float(ours[b].lower[n]), float(ours[b].upper[n])]
        base = [float(plain[b].lower[n]), float(plain[b].upper[n])]
        error = max(abs(found[0] - lo), abs(found[1] - hi))
        plain_error = max(abs(base[0] - lo), abs(base[1] - hi))
        rows.append(
            {
                "n_on": n,
                "b": b,
                "published": [lo, hi],
                "ours": found,
                "plain": base,
                "correction_changes_row": found != base,
                "max_error": error,
                "plain_max_error": plain_error,
                "passed": error <= config.grid.fc_table_tol + 1e-12,
            }
        )
    # Upper endpoints on a b grid of step 0.1: the plain belt can increase with
    # b; the corrected one must not.
    steps = np.arange(151) / 10
    uppers = {
        name: np.array([endpoints(float(b), corrected).upper for b in steps])
        for name, corrected in (("plain", False), ("corrected", True))
    }
    violations = {
        name: [
            {
                "n_on": int(j),
                "b_from": float(steps[i]),
                "b_to": float(steps[i + 1]),
                "upper_from": float(table[i, j]),
                "upper_to": float(table[i + 1, j]),
            }
            for i, j in np.argwhere(np.diff(table, axis=0) > 1e-12)
        ]
        for name, table in uppers.items()
    }
    return {
        "source": "https://arxiv.org/pdf/physics/9711021",
        "tables": "IV and V",
        "table_file": str(FC_TABLE),
        "mu_grid": {
            "max": float(validation[-1]),
            "step": config.grid.mu_step_fc,
            "n_points": int(validation.size),
        },
        "n_max": n_max,
        "tolerance": config.grid.fc_table_tol,
        "n_entries": len(rows),
        "n_passed": sum(row["passed"] for row in rows),
        "maximum_error": max(row["max_error"] for row in rows),
        "plain_n_passed": sum(
            row["plain_max_error"] <= config.grid.fc_table_tol + 1e-12 for row in rows
        ),
        "n_changed_by_correction": sum(row["correction_changes_row"] for row in rows),
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "b_monotone_correction": {
            "scope": "every count (FC 1998, Sec. IV A)",
            "b_min": 0.0,
            "b_max": 25.0,
            "b_step": 0.01,
        },
        "b_monotonicity": {
            "scope": "n_on=0..20; b=0..15 step .1",
            "plain_violation_count": len(violations["plain"]),
            "corrected_violation_count": len(violations["corrected"]),
            "plain_violations": violations["plain"],
            "corrected_violations": violations["corrected"],
        },
    }


def load_posterior(config: StudyConfig) -> tuple[PosteriorPredictor, dict[str, Any]]:
    """Load the Path A amortized posterior and record its provenance.

    Parameters
    ----------
    config : StudyConfig
        The artifact's ``problem`` settings must equal ``config.problem``.

    Returns
    -------
    posterior : PosteriorPredictor
        Loaded with ``AmortizedPosterior.load`` and tested once on two rows.
    provenance : dict
        ``kind``, repository-relative ``path``, and ``sha256`` of the file.

    Raises
    ------
    FileNotFoundError
        If results/path_a_amortized.pt is missing (raised by the loader).
    ValueError
        If the artifact format is incompatible or it uses a different
        physical model.

    Notes
    -----
    There is no fallback to a stand-in posterior: final results need the real
    Path A model.
    """
    path = ROOT / "results/path_a_amortized.pt"
    posterior = AmortizedPosterior.load(path)
    if posterior.config.problem != config.problem:
        raise ValueError("Path A artifact uses a different physical observation model")
    posterior.moments(np.array([[0.0, 15.0, 1.0], [5.0, 15.0, 1.0]]))
    return posterior, {
        "kind": "path_a_amortized",
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def raw_coverage_check(
    name: str,
    belt: CalibratedBelt,
    parameters: NDArray[np.float64],
    config: StudyConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check raw-set coverage of a calibrated belt at the true signal.

    Parameters
    ----------
    name : str
        Method identifier.
    belt : CalibratedBelt
        Belt to check; its statistic and critical values are used directly.
    parameters : ndarray of float64, shape (K, 3)
        Rows ``(mu, b, eps)`` to check.
    config : StudyConfig
        Supplies alpha, the tolerance ``tol_cov_mc``, and toy budgets.
    rng : numpy.random.Generator
        Stream for the toys; must be independent of the calibration stream.

    Returns
    -------
    rows : list of dict
        ``{"theta", "coverage"}`` per row, with reruns replacing flagged rows.
    check : dict
        Check record with ``coverage_target="raw_set"``.

    Notes
    -----
    Coverage at a row is ``P(T(x; mu) <= C(mu))``, where ``C(mu)`` linearly
    interpolates ``belt.critical_values`` at the true ``mu``. At grid nodes
    this equals the belt inversion, but no full inversion is done, so no hull
    lengths are reported. Rows below ``1 - alpha - tol_cov_mc`` are rerun with
    ``n_toys_cov * flagged_multiplier`` fresh toys.
    """

    def rows_for(theta: NDArray[np.float64], count: int) -> list[dict[str, Any]]:
        """Return raw-set coverage records, simulating ``count`` toys per theta row.

        Parameters
        ----------
        theta : ndarray of float64, shape (K, 3)
            Rows ``(mu, b, eps)``.
        count : int
            Toys per row.

        Returns
        -------
        list of dict
            ``{"theta", "coverage"}`` records with binomial errors.
        """
        rows = []
        for mu, b, eps in theta:
            x = simulate(mu, b, eps, rng=rng, size=count, config=config)
            cutoff = np.interp(mu, belt.mu_grid, belt.critical_values)
            p = np.mean(belt.statistic(x, np.asarray(mu)) <= cutoff)
            rows.append(
                {
                    "theta": [float(mu), float(b), float(eps)],
                    "coverage": metric(p, np.sqrt(p * (1 - p) / count), count),
                }
            )
        return rows

    target = 1 - config.problem.alpha - config.mc.tol_cov_mc
    rows = rows_for(parameters, config.mc.n_toys_cov)
    initial = [r for r in rows if r["coverage"]["value"] < target]
    rerun_n = config.mc.n_toys_cov * config.mc.flagged_multiplier
    reruns = rows_for(np.array([r["theta"] for r in initial]).reshape(-1, 3), rerun_n)
    by_theta = {tuple(r["theta"]): r for r in reruns}
    final = [by_theta.get(tuple(r["theta"]), r) for r in rows]
    remaining = [r["theta"] for r in final if r["coverage"]["value"] < target]
    return final, {
        "method": name,
        "alpha": config.problem.alpha,
        "tolerance": config.mc.tol_cov_mc,
        "domain": {"theta": parameters.tolist(), "nuisance": "explicit bounded b/eps box"},
        "coverage_target": "raw_set",
        "minimum_coverage": min(final, key=lambda r: r["coverage"]["value"]),
        "flagged_points": [r["theta"] for r in initial],
        "rerun_n_samples": rerun_n,
        "reruns": reruns,
        "remaining_flagged_points": remaining,
        "passed": not remaining,
    }


def raw_set_rows(
    belt: CalibratedBelt,
    parameters: NDArray[np.float64],
    config: StudyConfig,
    rng: np.random.Generator,
    count: int,
) -> list[dict[str, Any]]:
    """Return raw-set coverage records ``P(T(x; mu) <= C(mu))``, with fresh toys.

    Parameters
    ----------
    belt : CalibratedBelt
        Belt whose statistic and interpolated critical values are used.
    parameters : ndarray of float64, shape (K, 3)
        Rows ``(mu, b, eps)``.
    config : StudyConfig
        Simulator settings.
    rng : numpy.random.Generator
        Stream for the toys.
    count : int
        Toys per row.

    Returns
    -------
    list of dict
        ``{"theta", "coverage"}`` records with binomial errors, as in
        ``raw_coverage_check``.
    """
    rows = []
    for mu, b, eps in parameters:
        x = simulate(mu, b, eps, rng=rng, size=count, config=config)
        cutoff = np.interp(mu, belt.mu_grid, belt.critical_values)
        p = np.mean(belt.statistic(x, np.asarray(mu)) <= cutoff)
        rows.append(
            {
                "theta": [float(mu), float(b), float(eps)],
                "coverage": metric(p, np.sqrt(p * (1 - p) / count), count),
            }
        )
    return rows


def random_box_points(config: StudyConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw ``N_BOX_POINTS`` (mu, b, eps) points uniformly inside the calibration box.

    Parameters
    ----------
    config : StudyConfig
        Supplies ``mu_max``, the background range of ``b_cov_grid``, and
        ``eps0`` and ``sigma_eps``.
    rng : numpy.random.Generator
        Stream for the points.

    Returns
    -------
    ndarray of float64, shape (N_BOX_POINTS, 3)
        mu uniform on [0, mu_max]; log b uniform between the smallest and
        largest calibrated background; log eps uniform on
        ``log(eps0) +/- 2 sigma_eps``, the calibrated efficiency range.
    """
    p = config.problem
    u = rng.random((N_BOX_POINTS, 3))
    low, high = np.log(min(config.grid.b_cov_grid)), np.log(max(config.grid.b_cov_grid))
    return np.column_stack(
        (
            u[:, 0] * p.mu_max,
            np.exp(low + u[:, 1] * (high - low)),
            p.eps0 * np.exp((2 * u[:, 2] - 1) * 2 * p.sigma_eps),
        )
    )


def random_box_check(
    belts: dict[str, CalibratedBelt],
    parameters: NDArray[np.float64],
    config: StudyConfig,
    rng: np.random.Generator,
    rerun_rng: np.random.Generator,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Check raw-set coverage of several belts on shared toys at off-node points.

    Parameters
    ----------
    belts : dict of str to CalibratedBelt
        Belts to check, by name. All of them see the same toys.
    parameters : ndarray of float64, shape (K, 3)
        Rows ``(mu, b, eps)``, normally from ``random_box_points``.
    config : StudyConfig
        Supplies alpha, the tolerance ``tol_cov_mc``, and toy budgets.
    rng : numpy.random.Generator
        Stream for the shared initial toys.
    rerun_rng : numpy.random.Generator
        Separate stream for the reruns of flagged points.

    Returns
    -------
    rows : dict of str to list of dict
        Final ``{"theta", "coverage"}`` records per belt, with reruns
        replacing flagged rows.
    checks : list of dict
        One check record per belt, with the schema of ``raw_coverage_check``.

    Notes
    -----
    Each point gets ``n_toys_cov`` toys, shared by all belts, so the belts are
    compared on common random numbers. The pass rule is the two-stage rule of
    the other checks: a point below ``1 - alpha - tol_cov_mc`` is rerun with
    ``n_toys_cov * flagged_multiplier`` fresh toys for that belt, and the
    rerun decides.
    """
    count = config.mc.n_toys_cov
    rerun_n = count * config.mc.flagged_multiplier
    target = 1 - config.problem.alpha - config.mc.tol_cov_mc
    initial: dict[str, list[dict[str, Any]]] = {name: [] for name in belts}
    for mu, b, eps in parameters:
        x = simulate(mu, b, eps, rng=rng, size=count, config=config)
        for name, belt in belts.items():
            cutoff = np.interp(mu, belt.mu_grid, belt.critical_values)
            p = np.mean(belt.statistic(x, np.asarray(mu)) <= cutoff)
            initial[name].append(
                {
                    "theta": [float(mu), float(b), float(eps)],
                    "coverage": metric(p, np.sqrt(p * (1 - p) / count), count),
                }
            )
    rows, checks = {}, []
    for name, belt in belts.items():
        flagged = [r for r in initial[name] if r["coverage"]["value"] < target]
        theta = np.array([r["theta"] for r in flagged]).reshape(-1, 3)
        reruns = raw_set_rows(belt, theta, config, rerun_rng, rerun_n)
        by_theta = {tuple(r["theta"]): r for r in reruns}
        final = [by_theta.get(tuple(r["theta"]), r) for r in initial[name]]
        remaining = [r["theta"] for r in final if r["coverage"]["value"] < target]
        rows[name] = final
        checks.append(
            {
                "method": name,
                "alpha": config.problem.alpha,
                "tolerance": config.mc.tol_cov_mc,
                "domain": {
                    "theta": parameters.tolist(),
                    "nuisance": "uniform random points in the calibration box "
                    "(mu; log b; log eps), off the calibration nodes",
                },
                "coverage_target": "raw_set",
                "minimum_coverage": min(final, key=lambda r: r["coverage"]["value"]),
                "initial_minimum": min(r["coverage"]["value"] for r in initial[name]),
                "flagged_points": [r["theta"] for r in flagged],
                "rerun_n_samples": rerun_n,
                "reruns": reruns,
                "remaining_flagged_points": remaining,
                "passed": not remaining,
            }
        )
    return rows, checks


def node_overlap(
    parameters: NDArray[np.float64],
    mu_nodes: NDArray[np.float64],
    nuisance_nodes: NDArray[np.float64],
) -> dict[str, int]:
    """Count check points that coincide with calibration nodes.

    Parameters
    ----------
    parameters : ndarray of float64, shape (K, 3)
        Check rows ``(mu, b, eps)``.
    mu_nodes : ndarray of float64
        Calibration signal grid.
    nuisance_nodes : ndarray of float64, shape (H, 2)
        Calibration ``(b, eps)`` rows.

    Returns
    -------
    dict
        ``n_points``; ``nuisance_on_node``, the points whose (b, eps) is a
        calibration row; and ``mu_and_nuisance_on_node``, the points that are
        also on the calibration signal grid. Values match within 1e-10.
    """
    on_mu = np.any(abs(parameters[:, :1] - mu_nodes[None, :]) < 1e-10, axis=1)
    on_nuisance = np.any(
        np.all(abs(parameters[:, None, 1:] - nuisance_nodes[None, :, :]) < 1e-10, axis=2), axis=1
    )
    return {
        "n_points": int(parameters.shape[0]),
        "nuisance_on_node": int(on_nuisance.sum()),
        "mu_and_nuisance_on_node": int((on_mu & on_nuisance).sum()),
    }


def reestimate_lowest(
    procedure: CalibratedBelt,
    main_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
    config: StudyConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Re-estimate the lowest-coverage points of two check grids with fresh toys.

    Parameters
    ----------
    procedure : CalibratedBelt
        Calibrated belt whose checks are re-estimated.
    main_rows : list of dict
        Hull-coverage records of the main (mu, b) check grid.
    efficiency_rows : list of dict
        Raw-set records of the efficiency-variation grid.
    config : StudyConfig
        Supplies the toy budgets.
    rng : numpy.random.Generator
        Stream for the new toys, independent of the original checks.

    Returns
    -------
    dict
        For ``"coverage"`` (hull target) and ``"efficiency_variations"``
        (raw-set target): the ``N_REESTIMATED`` lowest points with their
        original and re-estimated coverage, and the lowest re-estimate.

    Notes
    -----
    The minimum of many noisy estimates is biased low, and the point that
    attains it is partly chosen by noise. Fresh toys at the selected points
    give unbiased estimates there. The original minima are kept unchanged.
    Each point gets ``n_toys_cov * flagged_multiplier`` toys.
    """
    count = config.mc.n_toys_cov * config.mc.flagged_multiplier
    output = {}
    for key, rows, target in (
        ("coverage", main_rows, "hull"),
        ("efficiency_variations", efficiency_rows, "raw_set"),
    ):
        lowest = sorted(rows, key=lambda row: row["coverage"]["value"])[:N_REESTIMATED]
        theta = np.array([row["theta"] for row in lowest])
        if target == "hull":
            fresh = coverage_rows(
                estimate_coverage(procedure, theta, rng=rng, n_samples=count, config=config)
            )
        else:
            fresh = raw_set_rows(procedure, theta, config, rng, count)
        points = [
            {"theta": row["theta"], "original": row["coverage"], "reestimated": new["coverage"]}
            for row, new in zip(lowest, fresh, strict=True)
        ]
        best = min(points, key=lambda point: point["reestimated"]["value"])
        output[key] = {
            "coverage_target": target,
            "n_samples": count,
            "points": points,
            "minimum_reestimated": {"theta": best["theta"], "coverage": best["reestimated"]},
        }
    return output


def pass_rule_characteristic(config: StudyConfig, sizes: dict[str, int]) -> dict[str, Any]:
    """Return the exact operating characteristic of the two-stage pass rule.

    Parameters
    ----------
    config : StudyConfig
        Supplies alpha, ``tol_cov_mc``, ``n_toys_cov`` and
        ``flagged_multiplier``.
    sizes : dict of str to int
        Check-grid sizes K for the any-flag probabilities.

    Returns
    -------
    dict
        For each true coverage C in 0.875..0.90, the probability of an initial
        flag and of a final pass, and, for each K, the probability of at least
        one initial flag when every point has C = 0.90.

    Notes
    -----
    The rule flags a point when ``X / N < 1 - alpha - tol_cov_mc`` with
    ``X ~ Binomial(N, C)``, ``N = n_toys_cov``, compared in floating point as
    in the checks. A flagged point passes if its rerun, with
    ``M = N * flagged_multiplier`` fresh toys, is not below the same
    threshold. So ``P(pass) = P(X >= k_N) + P(X < k_N) P(Y >= k_M)``, where
    ``k_n`` is the smallest passing count for n toys. Binomial tail sums are
    exact; no simulation is used.
    """
    threshold = 1 - config.problem.alpha - config.mc.tol_cov_mc
    first = config.mc.n_toys_cov
    second = first * config.mc.flagged_multiplier
    # Smallest count that is not flagged, using the same float comparison.
    need = {n: int(np.count_nonzero(np.arange(n + 1) / n < threshold)) for n in (first, second)}
    rows = []
    for coverage in (0.875, 0.88, 0.885, 0.89, 0.895, 0.90):
        flag = float(binom.cdf(need[first] - 1, first, coverage))
        rerun_pass = float(binom.sf(need[second] - 1, second, coverage))
        rows.append(
            {
                "true_coverage": coverage,
                "initial_flag_probability": flag,
                "final_pass_probability": (1 - flag) + flag * rerun_pass,
            }
        )
    flag_90 = float(binom.cdf(need[first] - 1, first, 0.90))
    return {
        "threshold": threshold,
        "n_toys": first,
        "rerun_n_toys": second,
        "smallest_passing_count": {"initial": need[first], "rerun": need[second]},
        "by_true_coverage": rows,
        "any_initial_flag_at_090": {
            name: {
                "n_points": size,
                "probability": float(-np.expm1(size * np.log1p(-flag_90))),
                "expected_flags": size * flag_90,
            }
            for name, size in sizes.items()
        },
        "method": "exact binomial tail sums",
    }


def load_details(path: Path) -> dict[str, Any]:
    """Read the complete Path B payload from its gzip JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the gzip payload, normally results/path_b_details.json.gz.

    Returns
    -------
    dict
        The decoded version-1 payload.
    """
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def summarize_check(check: dict[str, Any]) -> dict[str, Any]:
    """Return a compact copy of one coverage check for the summary file.

    Parameters
    ----------
    check : dict
        Check record from ``coverage_check`` or ``raw_coverage_check``.

    Returns
    -------
    dict
        The scalar fields, the ``domain`` reduced to ``n_points``, mu/b/eps
        ``bounds``, and the nuisance description, the ``minimum_coverage``
        point, the flagged counts before and after reruns, and, when present,
        ``initial_minimum`` and ``on_node_minimum_coverage``. Per-point lists
        are dropped.

    Notes
    -----
    If the domain has no explicit ``theta`` list, its points are the product
    of its ``mu``, ``b``, and ``eps`` lists.
    """
    domain = check["domain"]
    if "theta" in domain:
        theta = np.asarray(domain["theta"])
    else:
        theta = np.array(
            [[mu, b, eps] for mu in domain["mu"] for b in domain["b"] for eps in domain["eps"]]
        )
    minimum = check["minimum_coverage"]
    summary = {
        key: check[key]
        for key in ("method", "alpha", "tolerance", "coverage_target", "rerun_n_samples", "passed")
    }
    summary.update(
        domain={
            "n_points": len(theta),
            "bounds": {
                key: [float(theta[:, i].min()), float(theta[:, i].max())]
                for i, key in enumerate(("mu", "b", "eps"))
            },
            "nuisance": domain["nuisance"],
        },
        minimum_coverage={"theta": minimum["theta"], "coverage": minimum["coverage"]},
        flagged_count_before_reruns=len(check["flagged_points"]),
        flagged_count_after_reruns=len(check["remaining_flagged_points"]),
    )
    if "initial_minimum" in check:
        summary["initial_minimum"] = check["initial_minimum"]
    if "on_node_minimum_coverage" in check:
        summary["on_node_minimum_coverage"] = check["on_node_minimum_coverage"]
    return summary


def summarize_results(output: dict[str, Any], details_path: Path) -> dict[str, Any]:
    """Build the compact version-2 summary of a complete Path B payload.

    Parameters
    ----------
    output : dict
        Complete version-1 payload built by ``main``.
    details_path : pathlib.Path
        Location of the gzip payload, stored as ``artifacts.details``
        (relative to the repository root when possible).

    Returns
    -------
    dict
        Summary with the same envelope keys, ``schema_version=2`` and
        ``format="summary"`` (see "Results JSON schema" in docs/DESIGN.md).

    Notes
    -----
    Per-point arrays are dropped and recorded scalar metrics are kept. Each
    method gains a compact ``coverage_check``, the ``maximum_empty_rate`` over
    its coverage grid, and ``maximum_disconnected_rate`` (None when not
    recorded). Set diagnostics keep only the maximum degenerate rate and the
    minimum raw-set coverage per method. The FC reference check keeps its
    per-entry comparisons and drops the per-step monotonicity violations.
    Records that older payloads lack are summarized only when present.
    """
    diagnostics = output["diagnostics"]
    checks = {
        key: [summarize_check(check) for check in diagnostics[key]]
        for key in ("coverage_check", "efficiency_coverage_check", "off_grid_coverage_check")
    }
    methods = {}
    set_summary = {}
    for name, method in output["methods"].items():
        rows = diagnostics["set_diagnostics"].get(name, {}).get("initial_toys", [])
        disconnected = [row for row in rows if "disconnected_rate" in row]
        maximum_empty = max(method["coverage"], key=lambda row: row["empty_rate"]["value"])
        methods[name] = {
            key: value
            for key, value in method.items()
            if key not in {"coverage", "efficiency_variations", "off_grid_coverage"}
        }
        methods[name].update(
            coverage_check=next(
                check for check in checks["coverage_check"] if check["method"] == name
            ),
            maximum_empty_rate={
                "theta": maximum_empty["theta"],
                "empty_rate": maximum_empty["empty_rate"],
            },
            maximum_disconnected_rate=None,
        )
        if disconnected:
            maximum = max(disconnected, key=lambda row: row["disconnected_rate"]["value"])
            methods[name]["maximum_disconnected_rate"] = {
                "theta": maximum["theta"],
                "disconnected_rate": maximum["disconnected_rate"],
            }
        for field in ("degenerate_rate", "raw_set_coverage"):
            candidates = [row for row in rows if field in row]
            if candidates:
                key = "minimum" if field == "raw_set_coverage" else "maximum"
                choose = min if key == "minimum" else max
                row = choose(candidates, key=lambda row, field=field: row[field]["value"])
                set_summary.setdefault(name, {})[f"{key}_{field}"] = {
                    "theta": row["theta"],
                    field: row[field],
                }
        exact = diagnostics["set_diagnostics"].get(name, {})
        if "mu" in exact:
            i = int(np.argmax(exact["degenerate_rate"]))
            set_summary[name] = {
                "maximum_degenerate_rate": {
                    "theta": [exact["mu"][i], output["config"]["problem"]["known_b"], 1.0],
                    "degenerate_rate": metric(exact["degenerate_rate"][i], 0, 0, exact=True),
                }
            }
    # Keep the per-entry comparisons; drop the per-step monotonicity violations.
    references = {
        key: {k: v for k, v in value.items() if not k.endswith("violations")}
        if isinstance(value, dict)
        else value
        for key, value in diagnostics["fc_reference_checks"].items()
    }
    # Off-node and pass-rule records, summarized when the payload has them.
    extras = {
        key: diagnostics[key]
        for key in (
            "calibration_node_overlap",
            "reestimated_minima",
            "pass_rule_characteristic",
            "core_off_node_scan",
        )
        if key in diagnostics
    }
    if "random_box_check" in diagnostics:
        extras["random_box_check"] = [
            summarize_check(check) for check in diagnostics["random_box_check"]["checks"]
        ]
    calibration = diagnostics["calibration"]
    ablation = diagnostics["eps0_ablation"]
    off_grid = diagnostics["fc_off_grid"]
    summary = {key: value for key, value in output.items() if key not in {"methods", "diagnostics"}}
    summary.update(
        schema_version=2,
        format="summary",
        methods=methods,
        diagnostics={
            **checks,
            **extras,
            "eps0_ablation": {
                "n_nuisance_points": len(ablation["nuisance_grid"]),
                "coverage_check": [summarize_check(check) for check in ablation["coverage_check"]],
                "note": ablation["note"],
            },
            "fc_off_grid": {
                "n_points": len(off_grid["theta"]),
                "minimum_coverage": off_grid["minimum_coverage"],
                "tolerance": off_grid["tolerance"],
            },
            "fc_reference_checks": references,
            "calibration": {
                **{
                    key: value
                    for key, value in calibration.items()
                    if key
                    not in {
                        "mu_grid",
                        "nuisance_grid",
                        "profile_critical_values",
                        "waldo_critical_values",
                    }
                },
                "n_mu_points": len(calibration["mu_grid"]),
                "n_nuisance_points": len(calibration["nuisance_grid"]),
            },
            "set_diagnostics": set_summary,
            "quadrature_convergence": {
                **{
                    key: value
                    for key, value in diagnostics["quadrature_convergence"].items()
                    if key != "observations"
                },
                "n_observations": len(diagnostics["quadrature_convergence"]["observations"]),
            },
            **{
                key: diagnostics[key]
                for key in (
                    "waldo_model",
                    "rng_streams",
                    "variance_floor",
                    "profile_optimizer",
                    "count_truncation",
                    "timing",
                )
            },
        },
        artifacts={
            **output["artifacts"],
            "details": str(details_path.relative_to(ROOT))
            if details_path.is_relative_to(ROOT)
            else str(details_path),
        },
        notes=[
            *output["notes"],
            "The gzip artifact preserves the complete version-1 payload, "
            "including per-point arrays.",
            "interval_summary reports reality-prior averages; maximum rates report grid extrema.",
            "Disconnected rates use initial full-case toys where recorded; "
            "null means not recorded.",
            "Bounds summarize the checked domain; exact grid nodes and reruns are in the details.",
        ],
    )
    return summary


def write_results(output: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Write the indented summary JSON and the complete compact gzip payload.

    Parameters
    ----------
    output : dict
        Complete version-1 payload.
    destination : pathlib.Path
        Summary path, normally results/path_b.json. The payload is written to
        ``path_b_details.json.gz`` in the same directory.

    Returns
    -------
    dict
        The summary that was written.

    Raises
    ------
    ValueError
        If either JSON document contains NaN or infinity (``allow_nan=False``).

    Notes
    -----
    The payload uses compact separators and a fixed gzip time stamp
    (``mtime=0``), so equal payloads give equal bytes. The summary is
    indented by 2 spaces.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    details = destination.with_name("path_b_details.json.gz")
    payload = json.dumps(output, separators=(",", ":"), allow_nan=False).encode("utf-8")
    details.write_bytes(gzip.compress(payload, mtime=0))
    summary = summarize_results(output, details)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def draw_figures(methods: dict[str, Any] | Path, fc: NeymanBelt, config: StudyConfig) -> list[str]:
    """Draw and save the four Path B figures with the shared plot style.

    Parameters
    ----------
    methods : dict or pathlib.Path
        The ``methods`` mapping of the complete payload, or the path of the
        gzip payload to read it from.
    fc : NeymanBelt
        Core FC belt (known b) for the belt figure.
    config : StudyConfig
        Supplies alpha and the coverage grids.

    Returns
    -------
    list of str
        Repository-relative paths of the saved figures.

    Notes
    -----
    - path_b_fc_belt.png: FC acceptance table for counts 0..28.
    - path_b_coverage.png, path_b_length.png: hull coverage and expected hull
      length versus mu. Left: core methods (exact sums). Right: full methods
      at the single slice b = 3, eps = 1, with +/- 2 standard-error bands.
      Points rerun by ``coverage_check`` show their rerun values.
    - path_b_nuisance_coverage.png: hull coverage of ``profile_lr_full`` and
      ``waldo_full`` over the mu x b check grid at eps = 1 only, from the same
      rows. The colour scale runs from 0.85 to 1; lower values get the lowest
      colour. Efficiency variations are not shown.
    """
    if isinstance(methods, Path):
        methods = load_details(methods)["methods"]
    set_style()
    paths = []
    fig, ax = plt.subplots(figsize=(7.5, 5))
    selected = fc.counts <= 28
    ax.pcolormesh(
        fc.counts[selected],
        fc.mu_grid,
        fc.acceptance[:, selected],
        shading="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set(
        xlabel="Observed on-region count",
        ylabel="Signal μ",
        title=f"Exact {1 - config.problem.alpha:.0%} Feldman–Cousins belt: "
        f"known b={config.problem.known_b:g}, Θ=[0,{config.problem.mu_max:g}]",
    )
    paths.append(str(save_figure(fig, "path_b_fc_belt.png").relative_to(ROOT, walk_up=True)))
    plt.close(fig)
    for field, name, ylabel in [
        ("coverage", "path_b_coverage.png", "Coverage C(μ)"),
        ("expected_length", "path_b_length.png", "Expected hull length"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
        for key, method in methods.items():
            ax = axes[int(method["mode"] == "full")]
            efficiency = 1.0 if method["mode"] == "core" else config.problem.eps0
            rows = [
                row
                for row in method["coverage"]
                if row["theta"][1:] == [config.problem.known_b, efficiency]
            ]
            mu = [row["theta"][0] for row in rows]
            value = np.array([row[field]["value"] for row in rows])
            se = np.array([row[field]["mc_se"] for row in rows])
            ax.plot(mu, value, label=METHOD_LABELS[key], color=METHOD_COLORS[key], linewidth=1.6)
            if np.any(se):
                ax.fill_between(
                    mu, value - 2 * se, value + 2 * se, alpha=0.12, color=METHOD_COLORS[key]
                )
        for ax, title in zip(
            axes,
            [
                "Known background b = 3 (exact)",
                "Unknown background and efficiency\n(b = 3, ε = 1, simulated)",
            ],
            strict=True,
        ):
            ax.set(xlabel="Signal μ", ylabel=ylabel, title=title)
            if field == "coverage":
                ax.axhline(
                    1 - config.problem.alpha,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label=f"Nominal {1 - config.problem.alpha:.0%}",
                )
                ax.set_ylim(0, 1.02)
            ax.legend(fontsize=11, loc="best")
        paths.append(str(save_figure(fig, name).relative_to(ROOT, walk_up=True)))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    mus = theta_grid(config, coverage=True)
    backgrounds = np.asarray(config.grid.b_cov_grid)
    for ax, name in zip(axes, ["profile_lr_full", "waldo_full"], strict=True):
        rows = methods[name]["coverage"]
        table = np.array(
            [
                [
                    next(
                        row["coverage"]["value"]
                        for row in rows
                        if np.allclose(row["theta"], [mu, b, config.problem.eps0])
                    )
                    for mu in mus
                ]
                for b in backgrounds
            ]
        )
        mesh = ax.pcolormesh(
            mus,
            np.arange(backgrounds.size),
            table,
            shading="nearest",
            cmap="viridis",
            vmin=0.85,
            vmax=1,
        )
        ax.set(xlabel="Signal μ", title=METHOD_LABELS[name])
        ax.set_yticks(np.arange(backgrounds.size), [str(b) for b in backgrounds])
        ax.set_ylabel("True background b")
    fig.colorbar(
        mesh, ax=axes, label=f"Coverage (target {1 - config.problem.alpha:.2f})", shrink=0.9
    )
    paths.append(
        str(save_figure(fig, "path_b_nuisance_coverage.png").relative_to(ROOT, walk_up=True))
    )
    plt.close(fig)
    return paths


def redraw_saved() -> list[str]:
    """Redraw saved rows and the deterministic FC table, without toy draws.

    The FC acceptance table is not serialized; reconstruct it using the saved
    configuration, with the same b-monotone correction as the study. This
    exact enumeration uses no RNG or calibration.
    """
    summary = json.loads((ROOT / "results/path_b.json").read_text())
    config = saved_config(summary)
    core = replace(config, problem=replace(config.problem, mode="core", eps0=1.0, sigma_eps=0.0))
    fc = feldman_cousins_belt(b=config.problem.known_b, config=core, enforce_b_monotone=True)
    return draw_figures(ROOT / "results/path_b_details.json.gz", fc, config)


def main() -> None:
    """Run the Path B study and write its results and figures.

    The stages are: 1. exact core coverage (on the coverage grid and on a
    ``SCAN_STEP`` signal scan) and reality-prior summaries for six methods;
    2. loading the Path A posterior; 3. Monte Carlo calibration of the
    full-case profile LR and WALDO belts over the (b, eps) grid, refined to
    ``theta_grid(config)``; 4. hull coverage on the (mu, b) grid at eps = eps0,
    with reruns of flagged points; 5. efficiency-variation and off-grid checks;
    6. an eps0-only calibration ablation; 7. a raw-set check at random points
    of the calibration box, off the calibration nodes; 8. fresh-toy
    re-estimates of the lowest checked points; 9. FC reference, off-grid and
    quadrature checks, and the exact operating characteristic of the pass
    rule; and 10. the figures. Each task draws from its own child of
    ``SeedSequence(config.seed)``; tasks added later use new children, so the
    earlier streams are unchanged. ``--quick`` reduces toy counts and grids.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller calibration/coverage budgets")
    parser.add_argument("--figures-only", action="store_true", help="redraw saved results only")
    args = parser.parse_args()
    if args.figures_only:
        redraw_saved()
        return
    started = perf_counter()
    config = FULL_CONFIG
    if args.quick:
        config = replace(
            config,
            mc=replace(config.mc, n_toys_cov=500, n_toys_calib=800, m_bayes_risk=2000),
            grid=replace(
                config.grid,
                mu_cov_grid=(0.0, 0.01, 0.02, 0.05, 0.1) + tuple(np.arange(0.5, 10.1, 0.5)),
                b_cov_grid=(0.5, 3.0, 10.0),
                mu_step_calib=0.2,
            ),
        )
    # Core model: known b, eps = 1, no efficiency measurement noise.
    core = replace(config, problem=replace(config.problem, mode="core", eps0=1.0, sigma_eps=0.0))
    # Skip unused child stream 7 to keep the seeds of the later streams unchanged.
    seed_names = {
        0: "profile_calibration",
        1: "waldo_calibration",
        2: "coverage",
        3: "prior_summary",
        4: "flagged_reruns",
        5: "eps_variations",
        6: "off_grid",
        8: "ablation_calibration",
        9: "ablation_coverage",
        10: "random_box",
        11: "minimum_reestimation",
        12: "random_box_reruns",
    }
    seeds = {index: np.random.SeedSequence(config.seed, spawn_key=(index,)) for index in seed_names}
    rngs = {index: np.random.default_rng(seed) for index, seed in seeds.items()}
    seed_info = {
        seed_names[index]: {"entropy": int(seed.entropy), "spawn_key": list(seed.spawn_key)}
        for index, seed in seeds.items()
    }
    grid = theta_grid(config)
    mu = theta_grid(config, coverage=True)
    known_b = config.problem.known_b
    # Stage 1: exact core coverage and reality-prior summaries.
    # FC with its published non-increasing-in-b correction, as in the FC tables.
    fc = feldman_cousins_belt(b=known_b, config=core, enforce_b_monotone=True)
    core_posterior = LocalPosteriorStandIn(core)
    core_waldo = exact_waldo_belt(posterior=core_posterior, config=core)
    core_procedures = {
        "fc_core": (
            fc,
            "FC",
            "Exact restricted likelihood-ratio ordered Poisson belt with the FC "
            "non-increasing-in-b correction; outward-rounded hull",
        ),
        "wald_core": (
            partial(naive_wald_interval, config=core),
            "Naive Wald",
            "Gaussian interval intersected with Θ; variance=n_on",
        ),
        "central_core": (
            partial(central_neyman_interval, b=known_b, config=core),
            "Central Neyman",
            "Exact known-b equal-tail Garwood interval intersected with Θ",
        ),
        "upper_core": (
            partial(upper_limit_interval, b=known_b, config=core),
            "Upper limit",
            "Prechosen exact known-b one-sided upper limit",
        ),
        "flat_prior_core": (
            partial(flat_prior_interval, config=core),
            "Flat credible",
            "Equal-tail credible interval under flat μ on Θ",
        ),
        "waldo_core": (
            core_waldo,
            "WALDO (exact posterior)",
            "WALDO with local exact flat-grid posterior and exact Poisson calibration; "
            "outward-rounded hull",
        ),
    }
    # Exact sums on a fine signal scan, between and on the construction nodes.
    steps = int(round(config.problem.mu_max / SCAN_STEP))
    scan = np.round(np.arange(steps + 1) * SCAN_STEP, 9)
    construction = theta_grid(core)
    nearest = np.clip(np.searchsorted(construction, scan), 1, construction.size - 1)
    on_node = (
        np.minimum(abs(construction[nearest] - scan), abs(construction[nearest - 1] - scan)) < 1e-9
    )
    # The exact belts are built on grid nodes, where coverage holds by
    # construction, so their check uses the scan.
    exact_belts = {"fc_core", "waldo_core"}
    methods, checks, raw_diagnostics, core_scans = {}, [], {}, {}
    for name, (procedure, label, description) in core_procedures.items():
        result = exact_coverage(procedure, mu, b=known_b, config=core)
        rows = coverage_rows(result)
        scanned = exact_coverage(procedure, scan, b=known_b, config=core)
        scan_rows = coverage_rows(scanned)
        between = np.flatnonzero(~on_node)
        lowest = between[np.argmin(scanned.value[between])]
        core_scans[name] = {
            "mu_step": SCAN_STEP,
            "n_points": int(scan.size),
            "minimum_coverage": {
                key: scan_rows[int(np.argmin(scanned.value))][key] for key in ("theta", "coverage")
            },
            "minimum_between_nodes": {key: scan_rows[lowest][key] for key in ("theta", "coverage")},
            "fraction_below_nominal": float(np.mean(scanned.value < 1 - config.problem.alpha)),
            "construction_grid_step": float(config.grid.mu_step),
        }
        summary = prior_summary(procedure, core, rngs[3])
        methods[name] = {
            "mode": "core",
            "label": label,
            "description": description,
            "coverage": rows,
            "interval_summary": {
                k: summary[k] for k in ("coverage", "expected_length", "empty_rate")
            },
            "bayes_risk": summary["bayes_risk"],
            "point_summary": summary["point_summary"],
        }
        if name in exact_belts:
            check = coverage_check(
                name,
                scan_rows,
                core,
                domain={
                    "mu": scan.tolist(),
                    "b": [known_b],
                    "eps": [1.0],
                    "nuisance": "known b, eps=1",
                    "mu_scan_step": SCAN_STEP,
                },
                exact=True,
            )
            # Secondary value: the coverage grid, whose points are construction nodes.
            node_minimum = min(rows, key=lambda row: row["coverage"]["value"])
            check["on_node_minimum_coverage"] = {
                key: node_minimum[key] for key in ("theta", "coverage")
            }
        else:
            check = coverage_check(
                name,
                rows,
                core,
                domain={
                    "mu": mu.tolist(),
                    "b": [known_b],
                    "eps": [1.0],
                    "nuisance": "known b, eps=1",
                },
                exact=True,
            )
        checks.append(check)
        if name == "flat_prior_core":
            methods[name]["analysis_prior"] = {
                "family": "flat",
                "zero_mass": 0.0,
                "mu_min": 0.0,
                "mu_max": config.problem.mu_max,
            }
        if name == "wald_core":
            support = poisson_support(mu[-1] + known_b, tail_prob=config.grid.poisson_tail)
            x = np.column_stack((support, np.zeros(support.size), np.ones(support.size)))
            ci = procedure(x)
            degenerate = np.isfinite(ci.lower) & (ci.lower == ci.upper)
            raw_diagnostics[name] = {
                "degenerate_rate": (
                    poisson.pmf(support[None, :], mu[:, None] + known_b) @ degenerate
                ).tolist(),
                "mu": mu.tolist(),
                "method": "exact",
            }
        print(
            f"{name}: min coverage={result.value.min():.5f}; scan min={scanned.value.min():.5f}",
            flush=True,
        )
    # Stage 2: load the saved neural posterior.
    posterior, provenance = load_posterior(config)
    profile_statistic = partial(profile_likelihood_statistic, config=config)
    waldo = partial(waldo_statistic, posterior=posterior, variance_floor=1e-8)
    print(
        "Calibrating profile LR and WALDO over the bounded background/efficiency box...", flush=True
    )
    # Stage 3: calibrate and refine the full-case belts.
    profile_coarse = calibrate_belt(profile_statistic, rng=rngs[0], config=config)
    waldo_coarse = calibrate_belt(waldo, rng=rngs[1], config=config)
    profile = refine_belt(profile_coarse, grid)
    calibrated_waldo = refine_belt(waldo_coarse, grid)
    full_procedures = {
        "profile_lr_full": (profile, "Profile LR", "Profile b>=0 and eps>0; max-box toy cutoffs"),
        "waldo_full": (calibrated_waldo, "WALDO", "Real Path A posterior; max-box toy cutoffs"),
        "wald_full": (
            partial(naive_wald_interval, config=config),
            "Naive Wald",
            "Off-count and efficiency plug-in Wald baseline with delta-method variance",
        ),
    }
    # Stage 4: hull coverage with flagged-point reruns.
    # Main full-case check grid: every (mu, b) pair at eps = eps0.
    parameters = np.array(
        [[signal, b, config.problem.eps0] for b in config.grid.b_cov_grid for signal in mu]
    )
    # Efficiency variations: eps = eps0 * exp(k * sigma_eps) for k = -2, -1, 1, 2.
    eps_parameters = np.array(
        [
            [signal, b, config.problem.eps0 * np.exp(sign * config.problem.sigma_eps)]
            for sign in (-2, -1, 1, 2)
            for b in config.grid.b_cov_grid
            for signal in mu
        ]
    )
    # Points off the calibration grid; eps = eps0 * exp(k * sigma_eps), k = -1.5, 0, 1.5.
    off_parameters = np.array(
        [
            [signal, b, config.problem.eps0 * np.exp(k * config.problem.sigma_eps)]
            for b in (0.75, 4.0, 9.0)
            for k in (-1.5, 0, 1.5)
            for signal in (0.015, 0.075, 1.05, 5.05, 9.95)
        ]
    )
    eps_checks, off_checks = [], []
    for name, (procedure, label, description) in full_procedures.items():
        print(f"Checking {name}: {len(parameters)} nuisance-grid points...", flush=True)
        observed = DiagnosticsProcedure(procedure, parameters)
        result = estimate_coverage(observed, parameters, rng=rngs[2], config=config)
        rows = coverage_rows(result)
        check = coverage_check(
            name,
            rows,
            config,
            domain={
                "mu": mu.tolist(),
                "b": list(config.grid.b_cov_grid),
                "eps": [config.problem.eps0],
                "nuisance": "max critical quantile over finite b/eps box grid",
            },
            exact=False,
            rng=rngs[4],
            procedure=procedure,
        )
        checks.append(check)
        summary = prior_summary(procedure, config, rngs[3])
        methods[name] = {
            "mode": "full",
            "label": label,
            "description": description,
            "coverage": rows,
            "interval_summary": {
                k: summary[k] for k in ("coverage", "expected_length", "empty_rate")
            },
            "bayes_risk": summary["bayes_risk"],
            "point_summary": summary["point_summary"],
        }
        raw_diagnostics[name] = {
            "initial_toys": observed.rows,
            "note": "Raw-set diagnostics use initial independent toys; hull reruns are separate",
        }
        # Stage 5: efficiency variations and off-grid checks for this method.
        if isinstance(procedure, CalibratedBelt):
            eps_rows, eps_check = raw_coverage_check(
                name, procedure, eps_parameters, config, rngs[5]
            )
        else:
            eps_result = estimate_coverage(procedure, eps_parameters, rng=rngs[5], config=config)
            eps_rows = coverage_rows(eps_result)
            eps_check = coverage_check(
                name,
                eps_rows,
                config,
                domain={
                    "theta": eps_parameters.tolist(),
                    "nuisance": "bounded efficiency-box sensitivity grid",
                },
                exact=False,
                rng=rngs[4],
                procedure=procedure,
            )
        eps_checks.append(eps_check)
        methods[name]["efficiency_variations"] = eps_rows
        off_result = estimate_coverage(procedure, off_parameters, rng=rngs[6], config=config)
        off_rows = coverage_rows(off_result)
        off_checks.append(
            coverage_check(
                name,
                off_rows,
                config,
                domain={
                    "theta": off_parameters.tolist(),
                    "nuisance": "off-calibration-grid signal checks",
                },
                exact=False,
                rng=rngs[4],
                procedure=procedure,
            )
        )
        methods[name]["off_grid_coverage"] = off_rows
        print(
            f"{name}: min coverage={check['minimum_coverage']['coverage']['value']:.5f}; "
            f"initial flags={len(check['flagged_points'])}; "
            f"remaining={len(check['remaining_flagged_points'])}",
            flush=True,
        )
    # Stage 6: calibration at ε = 1 only.
    # Ablation: calibrate over b only, with eps fixed at eps0.
    eps0_nuisance = np.array([[b, config.problem.eps0] for b in config.grid.b_cov_grid])
    ablation_checks, ablation_rows, ablation_cutoffs, ablation_belts = [], {}, {}, {}
    print("Checking the eps0-only calibration ablation with fresh toys...", flush=True)
    for name, statistic in [("profile_lr_full", profile_statistic), ("waldo_full", waldo)]:
        coarse = calibrate_belt(statistic, rng=rngs[8], nuisance_grid=eps0_nuisance, config=config)
        belt = refine_belt(coarse, grid)
        rows, check = raw_coverage_check(name, belt, eps_parameters, config, rngs[9])
        ablation_rows[name] = rows
        ablation_checks.append(check)
        ablation_cutoffs[name] = coarse.critical_values.tolist()
        ablation_belts[name] = belt
    # Stage 7: raw-set coverage at random points of the calibration box, off the
    # calibration nodes, for both calibrated belts and their eps0-only ablations.
    print(f"Checking {N_BOX_POINTS} random off-node points in the box...", flush=True)
    box_parameters = random_box_points(config, rngs[10])
    box_belts = {
        "profile_lr_full": profile,
        "waldo_full": calibrated_waldo,
        "profile_lr_full_eps0_only": ablation_belts["profile_lr_full"],
        "waldo_full_eps0_only": ablation_belts["waldo_full"],
    }
    box_rows, box_checks = random_box_check(box_belts, box_parameters, config, rngs[10], rngs[12])
    for check in box_checks:
        print(
            f"{check['method']} random box: "
            f"min={check['minimum_coverage']['coverage']['value']:.5f}; "
            f"initial flags={len(check['flagged_points'])}; "
            f"remaining={len(check['remaining_flagged_points'])}",
            flush=True,
        )
    # How many checked points are calibration nodes, where coverage holds by
    # construction up to calibration error.
    overlap = {
        name: node_overlap(points, profile_coarse.mu_grid, profile_coarse.nuisance_grid)
        for name, points in (
            ("coverage", parameters),
            ("efficiency_variations", eps_parameters),
            ("off_grid_coverage", off_parameters),
            ("random_box", box_parameters),
        )
    }
    # Stage 8: fresh-toy re-estimates of the lowest points of the two main grids.
    print("Re-estimating the lowest checked points with fresh toys...", flush=True)
    reestimated = {
        name: reestimate_lowest(
            full_procedures[name][0],
            methods[name]["coverage"],
            methods[name]["efficiency_variations"],
            config,
            rngs[11],
        )
        for name in ("profile_lr_full", "waldo_full")
    }
    print("Checking published FC references and saving figures...", flush=True)
    # Stage 9: FC reference, off-grid and quadrature checks; pass-rule characteristic.
    references = reference_checks(config)
    characteristic = pass_rule_characteristic(
        config,
        {
            "coverage": len(parameters),
            "efficiency_variations": len(eps_parameters),
            "random_box": len(box_parameters),
        },
    )
    off_mu = np.unique(np.append(np.arange(0.005, 10, 0.1), [0.015, 0.025, 0.055]))
    fc_off = exact_coverage(fc, off_mu, b=known_b, config=core)
    convergence_x = np.array(
        [[0.0, 15.0, 1.0], [2.0, 15.0, 0.92], [8.0, 15.0, 1.1], [20.0, 40.0, 1.0]]
    )
    coarse_flat = flat_prior_interval(convergence_x, config=config)
    fine_config = replace(config, grid=replace(config.grid, mu_step=0.005, eps_points=61))
    fine_flat = flat_prior_interval(convergence_x, config=fine_config)
    # Stage 10: save figures and result records.
    figures = draw_figures(methods, fc, config)
    notes = [
        "Coverage refers to reported hulls: inclusive for calibrated belts, outward-rounded "
        "(one grid node beyond the accepted nodes, within Θ) for the exact core FC and WALDO "
        "belts; raw disconnected-set diagnostics are separate.",
        "Exact core FC and WALDO coverage checks use a mu scan of step 0.001, on and between "
        "the construction nodes; on_node_minimum_coverage is the coverage-grid minimum.",
        "Finite calibration/nuisance grids and Monte Carlo checks do not establish "
        "continuous-domain coverage.",
        "Core central and upper-limit procedures assume known b; selecting one after seeing data "
        "can break coverage.",
        "Flat equal-tail credible intervals have zero endpoint coverage in the enumerated core "
        "check, up to the reported count-tail error.",
        "Naive Wald is a deliberately failing small-count baseline; "
        "its coverage-check failure is expected.",
        "fc_core uses FC's non-increasing-in-b correction for every count "
        "(auxiliary b grid 0..25, step 0.01), as in the published FC tables.",
        "Core WALDO uses a local exact posterior; full WALDO provenance is recorded separately.",
        "Default calibration checks b in [0.5,10] and eps in eps0*exp([-2,-1,0,1,2]*sigma_eps); "
        "this bounded coverage box does not truncate the unbounded reality prior.",
        "The eps0-only calibration ablation ignores an efficiency nuisance; "
        "its independent box coverage failures are retained.",
        "Calibration quantile error is separate from the reported coverage toy error.",
        "Most points of the main and efficiency check grids are calibration nodes "
        "(calibration_node_overlap); random_box_check tests random points off the nodes.",
        "Reported minima are minima of many noisy estimates and are biased low; "
        "reestimated_minima gives fresh-toy estimates at the lowest points.",
        "The pass rule certifies coverage of about 0.885 or more at the checked points; "
        "pass_rule_characteristic gives its exact operating characteristic.",
    ]
    if args.quick:
        notes.append("Quick run: reduced toy and grid budgets; actual overrides are in config.")
    output = {
        "schema_version": 1,
        "path": "b",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "config": asdict(config),
        "runtime_seconds": perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": version("torch"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "methods": methods,
        "diagnostics": {
            "coverage_check": checks,
            "efficiency_coverage_check": eps_checks,
            "eps0_ablation": {
                "nuisance_grid": eps0_nuisance.tolist(),
                "coverage_check": ablation_checks,
                "coverage": ablation_rows,
                "critical_values": ablation_cutoffs,
                "note": "Same continuous upper-envelope interpolation; "
                "only the nuisance grid differs",
            },
            "off_grid_coverage_check": off_checks,
            "random_box_check": {
                "theta": box_parameters.tolist(),
                "coverage": box_rows,
                "checks": box_checks,
                "note": "Raw-set coverage on toys shared by all belts; "
                "reruns use a separate stream",
            },
            "calibration_node_overlap": overlap,
            "reestimated_minima": reestimated,
            "pass_rule_characteristic": characteristic,
            "core_off_node_scan": core_scans,
            "fc_off_grid": {
                "theta": fc_off.theta.tolist(),
                "coverage": fc_off.value.tolist(),
                "minimum_coverage": float(fc_off.value.min()),
                "tolerance": config.mc.tol_cov_exact,
            },
            "fc_reference_checks": references,
            "waldo_model": provenance,
            "rng_streams": seed_info,
            "variance_floor": 1e-8,
            "calibration": {
                "n_toys_per_point": config.mc.n_toys_calib,
                "quantile_order": int(
                    np.ceil((config.mc.n_toys_calib + 1) * (1 - config.problem.alpha))
                ),
                "mu_grid": profile_coarse.mu_grid.tolist(),
                "nuisance_grid": profile_coarse.nuisance_grid.tolist(),
                "profile_critical_values": profile_coarse.critical_values.tolist(),
                "waldo_critical_values": waldo_coarse.critical_values.tolist(),
                "interpolation": "linear interpolation of three-node maximum upper envelope",
                "nuisance_box": {
                    "b": [config.problem.b_min, config.problem.b_max],
                    "eps": (
                        config.problem.eps0 * np.exp(np.array([-2, 2]) * config.problem.sigma_eps)
                    ).tolist(),
                },
                "inversion_mu_step": config.grid.mu_step,
            },
            "profile_optimizer": {
                "method": "analytic b root; vectorized Newton in log eps",
                "max_iterations": 30,
                "log_eps_step_tolerance": 1e-10,
            },
            "count_truncation": {
                "n_max_core": int(fc.counts[-1]),
                "rate_envelope": float(config.problem.mu_max + known_b),
                "maximum_omitted_mass": float(poisson.sf(fc.counts[-1], mu[-1] + known_b)),
                "simulation_counts": "untruncated",
            },
            "quadrature_convergence": {
                "observations": convergence_x.tolist(),
                "method": "local full flat posterior; analytic b integration",
                "coarse_mu_step": config.grid.mu_step,
                "fine_mu_step": 0.005,
                "coarse_eps_points": config.grid.eps_points,
                "fine_eps_points": 61,
                "max_endpoint_change": float(
                    max(
                        np.max(abs(coarse_flat.lower - fine_flat.lower)),
                        np.max(abs(coarse_flat.upper - fine_flat.upper)),
                    )
                ),
            },
            "set_diagnostics": raw_diagnostics,
            "timing": "perf_counter from script main through figure saving; "
            "JSON serialization excluded",
        },
        "artifacts": {"figures": figures, "model": provenance["path"]},
        "notes": notes,
    }
    destination = ROOT / "results/path_b.json"
    write_results(output, destination)
    print(
        f"Wrote {destination.relative_to(ROOT)} and {len(figures)} figures in "
        f"{output['runtime_seconds']:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
