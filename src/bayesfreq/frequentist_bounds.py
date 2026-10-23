"""Path B: confidence belts, baseline intervals, and coverage checks.

This module builds Neyman confidence sets for the signal mu in Theta = [0, mu_max]
of the on/off Poisson experiment. It provides the exact Feldman-Cousins (FC) belt
for a known background, simple baseline intervals, and exact or Monte Carlo
coverage estimates. FC and exact coverage use finite Poisson sums. Exact belts
report outward-rounded hulls, which remove the between-node coverage dips of the
inclusive hull.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.special import gammainc, gammaincc, gammaln, logsumexp, xlogy
from scipy.stats import chi2, norm, poisson

from bayesfreq.config import FULL_CONFIG, StudyConfig
from bayesfreq.data_simulator import poisson_support, simulate, theta_grid
from bayesfreq.interfaces import (
    CoverageResult,
    IntervalBatch,
    IntervalProcedure,
)

__all__ = [
    "NeymanBelt",
    "feldman_cousins_belt",
    "profile_background",
    "naive_wald_interval",
    "central_neyman_interval",
    "upper_limit_interval",
    "flat_prior_interval",
    "exact_coverage",
    "estimate_coverage",
]


def _observations(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return x as float64 after checking shape (..., 3) and valid counts and eps_hat."""
    obs = np.asarray(x, dtype=np.float64)
    if obs.ndim < 1 or obs.shape[-1] != 3:
        raise ValueError("x must have final dimension 3: n_on, n_off, eps_hat")
    if (
        np.any(~np.isfinite(obs))
        or np.any(obs[..., :2] < 0)
        or np.any(obs[..., :2] != np.floor(obs[..., :2]))
        or np.any(obs[..., 2] <= 0)
    ):
        raise ValueError("observations require nonnegative integer counts and positive eps_hat")
    return obs


def _grid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a checked signal grid: 1-D, finite, strictly increasing, starting at 0."""
    grid = np.asarray(values, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size < 2
        or np.any(~np.isfinite(grid))
        or grid[0] != 0
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("mu_grid must be increasing, finite, and start at zero")
    return grid


def _hull(
    accepted: NDArray[np.bool_], grid: NDArray[np.float64], *, outward: bool = False
) -> IntervalBatch:
    """Return the hull of the accepted grid points per row (NaN if none), keeping the mask.

    The inclusive hull is [first, last] accepted point. With ``outward=True``
    each end moves one node outward (clipped to the first and last node).
    """
    nonempty = accepted.any(axis=-1)
    first = np.argmax(accepted, axis=-1)
    last = grid.size - 1 - np.argmax(accepted[..., ::-1], axis=-1)
    if outward:
        first = np.maximum(first - 1, 0)
        last = np.minimum(last + 1, grid.size - 1)
    lower = np.where(nonempty, grid[first], np.nan)
    upper = np.where(nonempty, grid[last], np.nan)
    return IntervalBatch(lower, upper, accepted, grid)


def _intersection(
    lower: NDArray[np.float64], upper: NDArray[np.float64], high: float
) -> IntervalBatch:
    """Intersect [lower, upper] with [0, high]; empty results get NaN endpoints."""
    lo = np.maximum(0.0, lower)
    hi = np.minimum(high, upper)
    empty = lo > hi
    return IntervalBatch(np.where(empty, np.nan, lo), np.where(empty, np.nan, hi))


@dataclass(frozen=True)
class NeymanBelt:
    """Exact Neyman belt for the on-region count, stored as an acceptance table.

    Parameters
    ----------
    mu_grid : ndarray of float64, shape (G,)
        Signal grid, increasing and starting at 0.
    counts : ndarray of int64, shape (J,)
        Count support 0..n_max; these are the table columns.
    acceptance : ndarray of bool, shape (G, J)
        ``acceptance[i, j]`` is True when count ``counts[j]`` is in the
        acceptance region of ``mu_grid[i]``.

    Notes
    -----
    Inversion gives ``CI(n) = {mu_i : acceptance[i, n]}``. The reported interval
    is the outward-rounded hull of this set: its lower end is the grid node just
    below the lowest accepted node (0 if that node is 0), and its upper end is
    the node just above the highest accepted node (``mu_grid[-1]`` if that node
    is the last one). Coverage at the grid nodes holds by construction; the
    inclusive hull can undercover between two nodes, because a count is then
    covered only if it is accepted at both of them. With outward rounding,
    every count accepted at either node is covered between them. This is not
    a proof of coverage off the nodes; the study checks it with exact sums on
    a fine signal scan. FC (1998) also round their table endpoints outward.
    The raw grid membership is kept unchanged.
    """

    mu_grid: NDArray[np.float64]
    counts: NDArray[np.int64]
    acceptance: NDArray[np.bool_]

    def __call__(self, x: NDArray[np.float64]) -> IntervalBatch:
        """Invert the belt for observed on-region counts.

        Parameters
        ----------
        x : ndarray of float64, shape (N, 3)
            Observations ``(n_on, n_off, eps_hat)``. Only ``n_on`` is used; the
            other columns are validated and then ignored.

        Returns
        -------
        IntervalBatch
            Outward-rounded hulls ``lower``/``upper`` of shape (N,), NaN for
            empty sets, and the raw membership ``accepted`` of shape (N, G).

        Raises
        ------
        ValueError
            If ``x`` is invalid or an observed count is not in ``counts``
            (for example above ``n_max``; rebuild the belt with a larger ``n_max``).
        """
        obs = _observations(x)
        counts = obs[..., 0].astype(np.int64)
        indices = np.searchsorted(self.counts, counts)
        if np.any(indices >= self.counts.size):
            raise ValueError("observed count exceeds belt support; rebuild with larger n_max")
        if np.any(self.counts[indices] != counts):
            raise ValueError("observed count is absent from belt support")
        return _hull(self.acceptance[:, indices].T, self.mu_grid, outward=True)


def _fc_acceptance(
    grid: NDArray[np.float64], counts: NDArray[np.int64], b: float, alpha: float
) -> NDArray[np.bool_]:
    """Return the plain FC acceptance table (G, J) for known b, by likelihood-ratio rank."""
    rates = grid[:, None] + b
    logp = xlogy(counts[None, :], rates) - rates - gammaln(counts[None, :] + 1)
    best_rate = np.clip(counts - b, 0, grid[-1]) + b
    logbest = xlogy(counts, best_rate) - best_rate - gammaln(counts + 1)
    order = np.argsort(-(logp - logbest), axis=1, kind="stable")
    cumulative = np.cumsum(np.take_along_axis(np.exp(logp), order, axis=1), axis=1)
    if np.any(cumulative[:, -1] < 1 - alpha):
        raise ValueError("n_max is too small to reach the requested confidence level")
    crossing = np.argmax(cumulative >= 1 - alpha, axis=1)
    sorted_accept = np.arange(counts.size)[None, :] <= crossing[:, None]
    accepted = np.zeros_like(sorted_accept)
    np.put_along_axis(accepted, order, sorted_accept, axis=1)
    return accepted


def feldman_cousins_belt(
    *,
    b: float = 3.0,
    alpha: float | None = None,
    mu_grid: NDArray[np.float64] | None = None,
    n_max: int | None = None,
    enforce_b_monotone: bool = False,
    config: StudyConfig = FULL_CONFIG,
) -> NeymanBelt:
    """Build the exact Feldman-Cousins (FC) belt for a known background.

    Parameters
    ----------
    b : float, default 3.0 (``FULL_CONFIG.problem.known_b``)
        Known background mean, ``b >= 0``.
    alpha : float, optional
        Miss probability; each grid row has coverage ``>= 1 - alpha`` on the
        count support. Default: ``config.problem.alpha`` (0.1 by default).
    mu_grid : ndarray of float64, optional
        Signal grid, increasing from 0. Default: ``theta_grid(config)``.
    n_max : int, optional
        Largest tabulated count. Default: from ``poisson_support`` at rate
        ``mu_grid[-1] + b`` with tail ``config.grid.poisson_tail``.
    enforce_b_monotone : bool, default False
        If True, apply FC's correction that makes every upper endpoint
        non-increasing in b (see Notes). The published FC tables use it.
    config : StudyConfig
        Supplies the default alpha, grid and Poisson tail target.

    Returns
    -------
    NeymanBelt
        Acceptance table of shape (G, n_max + 1).

    Raises
    ------
    ValueError
        For ``b < 0``, ``alpha`` outside (0, 1), an invalid grid or ``n_max``,
        a count support with total mass below ``1 - alpha``, or ``b > 25``
        with ``enforce_b_monotone=True``.

    Notes
    -----
    For each mu, counts are ranked by ``R(n; mu) = P(n | mu + b) / P(n | mu_best + b)``
    with ``mu_best = clip(n - b, 0, mu_grid[-1])``. Ties keep ascending count
    order. Counts are accepted in rank order up to and including the first
    count where the summed probability reaches ``1 - alpha``.

    For fixed n, the plain upper endpoint U(n, b) is not always a decreasing
    function of b. FC (1998, Sec. IV A) "force the function to be
    non-increasing, by lengthening selected confidence intervals as
    necessary", over a fine grid of b in [0, 25]. The correction does the same
    for every count n: the corrected endpoint is the largest plain upper
    endpoint U(n, b') over ``b'`` in ``{b} U {0, 0.01, ..., 25}`` with
    ``b' >= b``, on this belt's signal grid. Count n is then accepted at every
    grid point from its plain lower endpoint up to the corrected endpoint, so
    the correction only adds acceptance and every row keeps coverage
    ``>= 1 - alpha``. The plain endpoints on the auxiliary background grid are
    computed once per (alpha, grid, count support) and cached.
    """
    alpha = config.problem.alpha if alpha is None else alpha
    if not np.isfinite(b) or b < 0 or not 0 < alpha < 1:
        raise ValueError("require b>=0 and 0<alpha<1")
    grid = _grid(theta_grid(config) if mu_grid is None else mu_grid)
    if n_max is None:
        counts = poisson_support(grid[-1] + b, tail_prob=config.grid.poisson_tail)
    else:
        if n_max < 0 or int(n_max) != n_max:
            raise ValueError("n_max must be a nonnegative integer")
        counts = np.arange(n_max + 1, dtype=np.int64)
    accepted = _fc_acceptance(grid, counts, b, alpha)
    if enforce_b_monotone:
        if b > 25:
            raise ValueError("the b-monotone correction is defined only for b<=25")
        # The auxiliary belts need enough counts for rates up to mu_max + 25.
        support = poisson_support(grid[-1] + 25, tail_prob=config.grid.poisson_tail).size
        corrected = _fc_corrected_uppers(float(alpha), grid, max(counts.size, support), b)
        corrected = corrected[: counts.size]
        found = accepted.any(axis=0)
        low = np.argmax(accepted, axis=0)
        high = grid.size - 1 - np.argmax(accepted[::-1], axis=0)
        # Corrected endpoints are nodes of this grid (or -inf if never accepted).
        top = np.searchsorted(grid, corrected)
        lengthen = found & (corrected > grid[high])
        nodes = np.arange(grid.size)[:, None]
        accepted |= lengthen & (nodes >= low) & (nodes <= top)
    return NeymanBelt(grid, counts, accepted)


def profile_background(
    n_on: NDArray[np.float64],
    n_off: NDArray[np.float64],
    mu: NDArray[np.float64],
    *,
    eps: float | NDArray[np.float64] = 1.0,
    tau: float = 5.0,
) -> NDArray[np.float64]:
    """Return the profile MLE of the background, b_hat >= 0, in closed form.

    Parameters
    ----------
    n_on, n_off : ndarray of float64
        On-region count n and off-region count m (nonnegative).
    mu : ndarray of float64
        Signal values (nonnegative).
    eps : float or ndarray of float64, default 1.0
        Efficiency; the on-region signal is ``s = eps * mu``.
    tau : float, default 5.0 (``FULL_CONFIG.problem.tau``)
        Off/on exposure ratio, ``tau > 0``.

    Returns
    -------
    ndarray of float64
        ``b_hat`` with the broadcast shape of all inputs.

    Raises
    ------
    ValueError
        If ``tau <= 0`` or any count or ``eps * mu`` is negative.

    Notes
    -----
    Setting the b-derivative of ``log[Pois(n | s + b) Pois(m | tau b)]`` to zero
    gives ``(1 + tau) b^2 - a b - m s = 0`` with ``a = n + m - (1 + tau) s``.
    The nonnegative root is ``(a + sqrt(a^2 + 4 (1 + tau) m s)) / (2 (1 + tau))``.
    For ``a < 0`` the equal form ``2 m s / (sqrt(...) - a)`` avoids cancellation.
    The domain is ``b >= 0``, not the prior range [b_min, b_max].
    """
    n, m, signal = np.broadcast_arrays(n_on, n_off, np.asarray(mu) * eps)
    if tau <= 0 or np.any(n < 0) or np.any(m < 0) or np.any(signal < 0):
        raise ValueError("counts/signal must be nonnegative and tau positive")
    a = n + m - (1 + tau) * signal
    discriminant = np.sqrt(a * a + 4 * (1 + tau) * m * signal)
    positive = (a + discriminant) / (2 * (1 + tau))
    negative = np.divide(
        2 * m * signal,
        discriminant - a,
        out=np.zeros_like(a, dtype=np.float64),
        where=discriminant != a,
    )
    return np.where(a >= 0, positive, negative)


def naive_wald_interval(
    x: NDArray[np.float64],
    *,
    config: StudyConfig = FULL_CONFIG,
) -> IntervalBatch:
    """Return the naive Wald interval, intersected with Theta = [0, mu_max].

    Parameters
    ----------
    x : ndarray of float64, shape (..., 3)
        Observations ``(n_on, n_off, eps_hat)``.
    config : StudyConfig
        Mode, ``alpha``, and model constants.

    Returns
    -------
    IntervalBatch
        Endpoints only; both NaN when the interval misses Theta.

    Raises
    ------
    ValueError
        If ``x`` is invalid.

    Notes
    -----
    The interval is ``mu_hat +/- z_{1 - alpha/2} sqrt(V)``, where

    - core: ``mu_hat = n_on - known_b`` and ``V = n_on``;
    - full: ``mu_hat = (n_on - n_off / tau) / eps_hat`` and
      ``V = (n_on + n_off / tau^2) / eps_hat^2 + mu_hat^2 sigma_eps^2``
      (delta method).

    ``mu_hat`` is not clipped. The raw interval is intersected with Theta, so
    the result can be empty (for example ``n_on = 0`` in core mode, where V = 0).
    """
    obs = _observations(x)
    p = config.problem
    if p.mode == "core":
        mean = obs[..., 0] - p.known_b
        variance = obs[..., 0]
    else:
        mean = (obs[..., 0] - obs[..., 1] / p.tau) / obs[..., 2]
        variance = (obs[..., 0] + obs[..., 1] / p.tau**2) / obs[
            ..., 2
        ] ** 2 + mean**2 * p.sigma_eps**2
    width = norm.ppf(1 - p.alpha / 2) * np.sqrt(variance)
    return _intersection(mean - width, mean + width, p.mu_max)


def central_neyman_interval(
    x: NDArray[np.float64],
    *,
    b: float = 3.0,
    alpha: float | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> IntervalBatch:
    """Return the central (Garwood) Neyman interval for mu with known background.

    Parameters
    ----------
    x : ndarray of float64, shape (..., 3)
        Observations; only ``n_on`` is used.
    b : float, default 3.0 (``FULL_CONFIG.problem.known_b``)
        Known background.
    alpha : float, optional
        Total miss probability, split equally between the two tails.
        Default: ``config.problem.alpha`` (0.1 by default).
    config : StudyConfig
        Supplies ``mu_max`` and the default alpha.

    Returns
    -------
    IntervalBatch
        Endpoints only; both NaN when the interval misses Theta.

    Raises
    ------
    ValueError
        If ``x`` is invalid.

    Notes
    -----
    Exact equal-tail limits for the Poisson rate are
    ``L = chi2.ppf(alpha / 2, 2 n) / 2`` (``L = 0`` for ``n = 0``) and
    ``U = chi2.ppf(1 - alpha / 2, 2 (n + 1)) / 2``. The signal interval is
    ``[L - b, U - b]`` intersected with [0, mu_max].
    """
    alpha = config.problem.alpha if alpha is None else alpha
    n = _observations(x)[..., 0]
    lower = np.zeros_like(n)
    positive = n > 0
    lower[positive] = chi2.ppf(alpha / 2, 2 * n[positive]) / 2
    upper = chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2
    return _intersection(lower - b, upper - b, config.problem.mu_max)


def upper_limit_interval(
    x: NDArray[np.float64],
    *,
    b: float = 3.0,
    alpha: float | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> IntervalBatch:
    """Return the exact one-sided interval [0, U] for mu with known background.

    Parameters
    ----------
    x : ndarray of float64, shape (..., 3)
        Observations; only ``n_on`` is used.
    b : float, default 3.0 (``FULL_CONFIG.problem.known_b``)
        Known background.
    alpha : float, optional
        Miss probability of the upper limit.
        Default: ``config.problem.alpha`` (0.1 by default).
    config : StudyConfig
        Supplies ``mu_max`` and the default alpha.

    Returns
    -------
    IntervalBatch
        Endpoints only; both NaN when ``U < 0`` (empty set).

    Raises
    ------
    ValueError
        If ``x`` is invalid.

    Notes
    -----
    ``U = chi2.ppf(1 - alpha, 2 (n + 1)) / 2 - b``, capped at ``mu_max``. The
    one-sided form is chosen before seeing the data; choosing between one- and
    two-sided intervals after seeing the data can break coverage.
    """
    alpha = config.problem.alpha if alpha is None else alpha
    n = _observations(x)[..., 0]
    upper = chi2.ppf(1 - alpha, 2 * (n + 1)) / 2 - b
    return _intersection(np.zeros_like(n), upper, config.problem.mu_max)


def _flat_weights(
    x: NDArray[np.float64], config: StudyConfig, grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return posterior masses (N, G) on ``grid`` under a flat prior for mu.

    This is computed locally, without the Path A model. Core mode uses known b.
    Full mode integrates the uniform background prior exactly, as a finite
    mixture of incomplete Gamma integrals. The lognormal efficiency prior times
    the lognormal auxiliary likelihood is Normal in log eps; it is integrated
    by Gauss-Hermite quadrature. Masses use trapezoidal weights in mu, and
    constants that do not depend on mu cancel.
    """
    obs = _observations(x).reshape(-1, 3)
    p = config.problem
    widths = np.empty(grid.size)
    widths[0] = (grid[1] - grid[0]) / 2
    widths[-1] = (grid[-1] - grid[-2]) / 2
    widths[1:-1] = (grid[2:] - grid[:-2]) / 2
    if p.mode == "core":
        rate = grid[None, :] + p.known_b
        logweight = xlogy(obs[:, 0, None], rate) - rate + np.log(widths)
        return np.exp(logweight - logsumexp(logweight, axis=1, keepdims=True))
    nodes, quadrature = hermgauss(config.grid.eps_points)
    quadrature /= np.sqrt(np.pi)
    posterior_log_sd = p.sigma_eps / np.sqrt(2)
    output = np.empty((obs.shape[0], grid.size))
    for i, (n_raw, m, eps_hat) in enumerate(obs):
        n = int(n_raw)
        if max(n, m) >= 128:
            # Rare large counts: integrate b by Gauss-Legendre quadrature in log
            # space. This avoids incomplete-Gamma underflow and polynomial overflow.
            b_nodes, b_weights = np.polynomial.legendre.leggauss(config.grid.b_points)
            b = p.b_min + (b_nodes + 1) * (p.b_max - p.b_min) / 2
            log_center = (
                np.log(eps_hat) if p.sigma_eps == 0 else (np.log(p.eps0) + np.log(eps_hat)) / 2
            )
            eps = np.exp(log_center + np.sqrt(2) * posterior_log_sd * nodes)
            s = grid[:, None, None] * eps[None, None, :]
            log_density = (
                xlogy(n, s + b[None, :, None])
                - s
                - (1 + p.tau) * b[None, :, None]
                + xlogy(m, b[None, :, None])
            )
            logs = logsumexp(
                log_density
                + np.log(b_weights / 2)[None, :, None]
                + np.log(quadrature)[None, None, :],
                axis=(1, 2),
            )
            logs += np.log(widths)
            output[i] = np.exp(logs - logsumexp(logs))
            continue
        # Expand (s + b)^n binomially; the b integral of each term over
        # [b_min, b_max] is a Gamma integral, giving a polynomial in s.
        k = np.arange(n + 1)
        shape = n - k + m + 1
        c = 1 + p.tau
        # Mass = P(c*b_max) - P(c*b_min), P = regularized lower incomplete Gamma.
        # If P(c*b_min) >= 0.5, use Q = 1 - P instead to avoid cancellation near 1.
        lower = gammainc(shape, c * p.b_min)
        upper = gammainc(shape, c * p.b_max)
        mass = np.where(
            lower < 0.5,
            upper - lower,
            gammaincc(shape, c * p.b_min) - gammaincc(shape, c * p.b_max),
        )
        logcoef = (
            gammaln(n + 1)
            - gammaln(k + 1)
            - gammaln(n - k + 1)
            + gammaln(shape)
            - shape * np.log(c)
            + np.log(mass)
        )
        coefficients = np.exp(logcoef - np.max(logcoef))
        if p.sigma_eps == 0:
            eps = np.array([eps_hat])
            eps_weights = np.ones(1)
        else:
            log_center = (np.log(p.eps0) + np.log(eps_hat)) / 2
            eps = np.exp(log_center + np.sqrt(2) * posterior_log_sd * nodes)
            eps_weights = quadrature
        s = eps[:, None] * grid[None, :]
        # Coefficients and s are nonnegative, so Horner evaluation has no
        # cancellation at these count sizes.
        density = np.exp(-s) * np.polynomial.polynomial.polyval(s, coefficients)
        weight = (eps_weights @ density) * widths
        output[i] = weight / weight.sum()
    return output


def flat_prior_interval(
    x: NDArray[np.float64],
    *,
    config: StudyConfig = FULL_CONFIG,
) -> IntervalBatch:
    """Return the equal-tail credible interval for mu under a flat prior on Theta.

    Parameters
    ----------
    x : ndarray of float64, shape (..., 3)
        Observations ``(n_on, n_off, eps_hat)``.
    config : StudyConfig
        Mode, ``alpha``, signal grid, and nuisance priors.

    Returns
    -------
    IntervalBatch
        Endpoints of shape (N,), where N is the number of observation rows.
        The interval is never empty.

    Raises
    ------
    ValueError
        If ``x`` is invalid.

    Notes
    -----
    The posterior is computed on ``theta_grid(config)``. In full mode the
    background (uniform on [b_min, b_max]) and the efficiency (lognormal prior
    plus the ``eps_hat`` measurement) are integrated out. The endpoints are the
    first grid points where the posterior CDF reaches ``alpha / 2`` and
    ``1 - alpha / 2``. This is a Bayesian interval; its frequentist coverage is
    measured separately.
    """
    obs = _observations(x)
    grid = theta_grid(config)
    weights = _flat_weights(obs, config, grid)
    cumulative = weights.cumsum(axis=1)
    lower = grid[np.argmax(cumulative >= config.problem.alpha / 2, axis=1)]
    upper = grid[np.argmax(cumulative >= 1 - config.problem.alpha / 2, axis=1)]
    return IntervalBatch(lower, upper)


def exact_coverage(
    procedure: IntervalProcedure,
    mu_grid: NDArray[np.float64],
    *,
    b: float = 3.0,
    n_max: int | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> CoverageResult:
    """Compute exact hull coverage, expected length, and empty rate for known b.

    Parameters
    ----------
    procedure : IntervalProcedure
        Maps observations (J, 3) to intervals; must depend only on ``n_on``.
    mu_grid : ndarray of float64, shape (K,)
        Signal values at which to evaluate.
    b : float, default 3.0 (``FULL_CONFIG.problem.known_b``)
        Known background. The unused ``n_off`` column is set to the integer
        placeholder 0 and ``eps_hat`` to 1; ``tau * b`` need not be an integer.
    n_max : int, optional
        Largest count in the sums. Default: from ``poisson_support`` at rate
        ``max(mu_grid) + b`` with tail ``config.grid.poisson_tail``.
    config : StudyConfig
        Supplies the count-tail target.

    Returns
    -------
    CoverageResult
        ``theta`` rows ``(mu, b, 1)``, zero Monte Carlo errors,
        ``n_samples=0``, and ``exact=True``.

    Notes
    -----
    With ``P(n | mu) = Pois(n | mu + b)`` and reported hull ``[L(n), U(n)]``:
    ``C(mu) = sum_n P(n | mu) 1{L(n) <= mu <= U(n)}`` (tolerance 1e-12);
    expected length ``sum_n P(n | mu) (U(n) - L(n))``, with length 0 for
    empty sets; empty rate ``sum_n P(n | mu) 1{empty}``. The sums stop at
    n_max, so the omitted tail mass is a separate numerical error.
    """
    grid = np.asarray(mu_grid, dtype=np.float64)
    counts = (
        poisson_support(float(np.max(grid)) + b, tail_prob=config.grid.poisson_tail)
        if n_max is None
        else np.arange(n_max + 1)
    )
    x = np.column_stack((counts, np.zeros(counts.size), np.ones(counts.size))).astype(np.float64)
    intervals = procedure(x)
    empty = ~np.isfinite(intervals.lower) | ~np.isfinite(intervals.upper)
    length = np.where(empty, 0, intervals.upper - intervals.lower)
    covered = (grid[:, None] >= intervals.lower - 1e-12) & (
        grid[:, None] <= intervals.upper + 1e-12
    )
    probabilities = poisson.pmf(counts[None, :], grid[:, None] + b)
    zeros = np.zeros(grid.size)
    theta = np.column_stack((grid, np.full(grid.size, b), np.ones(grid.size)))
    return CoverageResult(
        theta,
        np.sum(probabilities * covered, axis=1),
        zeros.copy(),
        probabilities @ length,
        zeros.copy(),
        probabilities @ empty,
        zeros.copy(),
        0,
        True,
    )


def estimate_coverage(
    procedure: IntervalProcedure,
    theta: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    n_samples: int | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> CoverageResult:
    """Estimate hull coverage, expected length, and empty rate by Monte Carlo.

    Parameters
    ----------
    procedure : IntervalProcedure
        Maps observations (N, 3) to intervals. It is called once per row of
        ``theta``, in row order.
    theta : ndarray of float64, shape (K, 3)
        Rows ``(mu, b, eps)``.
    rng : numpy.random.Generator
        Stream for the toys.
    n_samples : int, optional
        Toys per row, an integer >= 2. Default: ``config.mc.n_toys_cov``.
    config : StudyConfig
        Simulator settings.

    Returns
    -------
    CoverageResult
        Per-row estimates and Monte Carlo errors, with ``exact=False``.

    Raises
    ------
    ValueError
        If ``theta`` is not (K, 3) or ``n_samples`` is not an integer >= 2.

    Notes
    -----
    Each row uses N fresh toys, including new ``eps_hat`` draws. Coverage is
    ``p = mean(1{L - 1e-12 <= mu <= U + 1e-12})`` for the reported hull, with
    error ``sqrt(p (1 - p) / N)``; the empty rate uses the same error formula.
    The length error is ``std(length, ddof=1) / sqrt(N)``, with length 0 for
    empty sets. Rerunning flagged rows is left to the caller.
    """
    parameters = np.asarray(theta, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("theta must have shape (K,3)")
    count = config.mc.n_toys_cov if n_samples is None else n_samples
    if count < 2 or int(count) != count:
        raise ValueError("n_samples must be an integer >=2")
    metrics = np.empty((parameters.shape[0], 6))
    for i, (mu, b, eps) in enumerate(parameters):
        x = simulate(mu, b, eps, rng=rng, config=config, size=count)
        intervals = procedure(x)
        empty = ~np.isfinite(intervals.lower) | ~np.isfinite(intervals.upper)
        length = np.where(empty, 0, intervals.upper - intervals.lower)
        cover = np.mean((mu >= intervals.lower - 1e-12) & (mu <= intervals.upper + 1e-12))
        empty_rate = np.mean(empty)
        metrics[i] = (
            cover,
            np.sqrt(cover * (1 - cover) / count),
            np.mean(length),
            np.std(length, ddof=1) / np.sqrt(count),
            empty_rate,
            np.sqrt(empty_rate * (1 - empty_rate) / count),
        )
    return CoverageResult(parameters, *[metrics[:, j] for j in range(6)], count, False)


def _fc_merge_walk(
    grid: NDArray[np.float64], n_counts: int, backgrounds: NDArray[np.float64], alpha: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the lowest and highest accepted count of plain FC belts, shape (B, G).

    For fixed (mu, b), log R(n) is concave in n: it is linear for n <= b, equal
    to ``n log(rate / n) + n - rate`` up to ``b + mu_max`` and linear above,
    with matching slopes. The stable rank order of ``_fc_acceptance`` is
    therefore the merge of the two monotone sides of the mode. The walk starts
    at the mode (count 0 at mu = 0, where every n <= b has R = 1) and adds the
    better-ranked neighbour, the lower count on ties, until the summed
    probability reaches 1 - alpha. It visits counts and adds probabilities in
    the same order as the sorted construction, so it gives the same table
    without sorting. Cells are walked in chunks of similar rate.
    """
    high = grid[-1]
    top = n_counts - 1
    counts = np.arange(n_counts)
    log_factorial = gammaln(counts + 1.0)
    # Padded tables: column p holds count p - 1. The end columns are sentinels
    # with log_best = +inf, so a missing neighbour never ranks first.
    width = n_counts + 2
    b = backgrounds[:, None]
    best_rate = np.clip(counts[None, :] - b, 0, high) + b
    log_best = np.full((backgrounds.size, width), np.inf)
    log_best[:, 1:-1] = xlogy(counts[None, :], best_rate) - best_rate - log_factorial[None, :]
    log_best = log_best.ravel()
    log_fact = np.zeros(width)
    log_fact[1:-1] = log_factorial
    rate = (grid[None, :] + b).ravel()
    offset = np.repeat(np.arange(backgrounds.size) * width, grid.size)
    at_zero = np.tile(grid == 0, backgrounds.size)
    # The last node has a flat right tail (R = 1 above b + mu_max): walk it last.
    order = np.lexsort((rate, np.tile(grid == high, backgrounds.size)))
    lowest = np.empty(rate.size, dtype=np.int64)
    highest = np.empty(rate.size, dtype=np.int64)
    target = 1 - alpha
    chunk = 1 << 17
    for start in range(0, order.size, chunk):
        cells = order[start : start + chunk]
        lam = rate[cells]
        log_lam = np.log(np.where(lam > 0, lam, 1.0))
        base = offset[cells]
        # The mode of a concave sequence peaking at n = rate is floor(rate) or
        # the next count; ties go to the lower count.
        lo = np.minimum(np.floor(lam).astype(np.int64), top)
        hi = np.minimum(lo + 1, top)
        logp_lo = lo * log_lam - lam - log_fact[lo + 1]
        logp_hi = hi * log_lam - lam - log_fact[hi + 1]
        pick = logp_hi - log_best[base + hi + 1] > logp_lo - log_best[base + lo + 1]
        a = np.where(pick, hi, lo)
        logp = np.where(pick, logp_hi, logp_lo)
        zero = at_zero[cells]
        a[zero] = 0
        logp[zero] = -lam[zero]
        c = a.copy()
        mass = np.exp(logp)
        walking = np.flatnonzero(mass < target)
        while walking.size:
            # Walk the remaining cells densely; compact once most have stopped.
            aa, cc, mm = a[walking], c[walking], mass[walking]
            ll, lg, bb = lam[walking], log_lam[walking], base[walking]
            active = np.ones(walking.size, dtype=np.bool_)
            while active.mean() > 0.25:
                if np.any(active & (aa == 0) & (cc == top)):
                    raise ValueError("n_max is too small to reach the requested confidence level")
                logp_left = (aa - 1) * lg - ll - log_fact[aa]
                logp_right = (cc + 1) * lg - ll - log_fact[cc + 2]
                left = logp_left - log_best[bb + aa] >= logp_right - log_best[bb + cc + 2]
                step = np.exp(np.where(left, logp_left, logp_right))
                mm = np.where(active, mm + step, mm)
                aa -= active & left
                cc += active & ~left
                active = mm < target
            a[walking], c[walking], mass[walking] = aa, cc, mm
            walking = walking[active]
        lowest[cells] = a
        highest[cells] = c
    shape = (backgrounds.size, grid.size)
    return lowest.reshape(shape), highest.reshape(shape)


def _fc_upper_endpoints(
    grid: NDArray[np.float64],
    lowest: NDArray[np.int64],
    highest: NDArray[np.int64],
    n_counts: int,
) -> NDArray[np.float64]:
    """Return the largest grid point that accepts each count, shape (B, n_counts).

    Each row of ``lowest``/``highest`` (shape (B, G)) gives the contiguous
    accepted counts at every node. Counts that are never accepted get -inf.
    """
    counts = np.arange(n_counts)
    uppers = np.full((lowest.shape[0], n_counts), -np.inf)
    for k in range(lowest.shape[0]):
        # The last node whose lowest accepted count is <= n; suffix minima are
        # sorted, so a binary search finds it.
        suffix = np.minimum.accumulate(lowest[k, ::-1])[::-1]
        node = np.searchsorted(suffix, counts, side="right") - 1
        hit = (node >= 0) & (highest[k, np.maximum(node, 0)] >= counts)
        uppers[k, hit] = grid[node[hit]]
        for n in np.flatnonzero(~hit):
            nodes = np.flatnonzero((lowest[k] <= n) & (highest[k] >= n))
            if nodes.size:
                uppers[k, n] = grid[nodes[-1]]
    return uppers


# Auxiliary background grid of the b-monotone FC correction, cached in blocks.
_AUX_BACKGROUNDS = np.arange(2501) / 100
_AUX_BLOCK = 40


@lru_cache(maxsize=256)
def _fc_plain_upper_block(
    alpha: float, grid_bytes: bytes, n_counts: int, block: int
) -> NDArray[np.float64]:
    """Return plain-FC upper endpoints for one block of auxiliary backgrounds.

    The rows are backgrounds ``_AUX_BACKGROUNDS[block * _AUX_BLOCK :]`` (at most
    ``_AUX_BLOCK`` of them) and the columns are counts 0..n_counts-1, on the
    float64 grid given by ``grid_bytes``. The belts are built exactly by
    ``_fc_merge_walk``.
    """
    grid = np.frombuffer(grid_bytes, dtype=np.float64)
    backgrounds = _AUX_BACKGROUNDS[block * _AUX_BLOCK : (block + 1) * _AUX_BLOCK]
    lowest, highest = _fc_merge_walk(grid, n_counts, backgrounds, alpha)
    return _fc_upper_endpoints(grid, lowest, highest, n_counts)


def _fc_corrected_uppers(
    alpha: float, grid: NDArray[np.float64], n_counts: int, b: float
) -> NDArray[np.float64]:
    """Return max over ``b' >= b`` on the auxiliary grid of the plain upper endpoints.

    Only the blocks with ``b' >= b`` are built, on a few threads; finished
    blocks are cached, and the result does not depend on the thread count.
    """
    start = int(np.searchsorted(_AUX_BACKGROUNDS, b, side="left"))
    key = grid.tobytes()
    blocks = range(start // _AUX_BLOCK, -(-_AUX_BACKGROUNDS.size // _AUX_BLOCK))
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        tables = list(pool.map(lambda j: _fc_plain_upper_block(alpha, key, n_counts, j), blocks))
    tables[0] = tables[0][start % _AUX_BLOCK :]
    return np.max(np.vstack(tables), axis=0)
