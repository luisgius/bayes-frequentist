"""Path A: Bayesian inference for the signal mu and Monte Carlo risk diagnostics.

This module holds the Path A estimators of mu on Theta = [0, mu_max]: the exact
grid posterior, a Gamma plug-in posterior, a Laplace-type approximation and
the clipped MLE. It also estimates Bayes risk,
frequentist risk and simulation-based calibration (SBC) ranks, and times inference.
Every public predictor takes raw float64 observations x of shape (N, 3) with
columns (n_on, n_off, eps_hat). The exact posterior uses quadrature: a zero atom
plus midpoint cells for mu, Gauss-Legendre nodes for the background b, and
Gauss-Hermite nodes for log eps given eps_hat.
"""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammainc, gammaincinv, gammaln, logsumexp, roots_hermitenorm, xlogy

from bayesfreq.config import (
    FULL_CONFIG,
    StudyConfig,
)
from bayesfreq.data_simulator import Prior, reality_prior, sample_joint, simulate
from bayesfreq.interfaces import (
    IntervalBatch,
    PointEstimator,
    PosteriorMoments,
    PosteriorSampler,
    RiskResult,
)

FloatArray = NDArray[np.float64]
__all__ = [
    "GridPosterior",
    "GammaPluginPosterior",
    "LaplacePosterior",
    "exact_posterior",
    "gamma_plugin_moments",
    "laplace_moments",
    "clipped_mle",
    "estimate_bayes_risk",
    "estimate_frequentist_risk",
    "simulation_based_calibration",
    "time_inference",
]


def _observations(x: FloatArray) -> FloatArray:
    """Validate raw observations and return them as a float64 (N, 3) array."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("x must have shape (N,3)")
    if (
        not np.all(np.isfinite(values))
        or np.any(values[:, :2] < 0)
        or np.any(values[:, :2] != np.floor(values[:, :2]))
        or np.any(values[:, 2] <= 0)
    ):
        raise ValueError("observations require nonnegative integer counts and positive eps_hat")
    return values


def _prior(prior: Prior | None, config: StudyConfig) -> Prior:
    """Return ``prior`` (the reality prior if None) after checking it uses ``config``."""
    selected = reality_prior(config) if prior is None else prior
    if selected.config != config:
        raise ValueError("prior and inference must use the same config")
    return selected


def _alpha(alpha: float) -> None:
    """Raise ValueError unless 0 < alpha < 1."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")


def _draw_indices(weights: FloatArray, n_draws: int, rng: np.random.Generator) -> NDArray:
    """Draw ``n_draws`` category indices per row from row-normalized weights (N, K)."""
    if n_draws < 0:
        raise ValueError("n_draws must be nonnegative")
    # Inverse-CDF sampling with a vectorized binary search; no Python loop over rows.
    cdf = np.cumsum(weights, axis=1)
    # Set the last CDF value to exactly 1 so rounding cannot leave a uniform above it.
    cdf[:, -1] = 1.0
    uniforms = rng.random((len(weights), n_draws))
    low = np.zeros(uniforms.shape, dtype=np.int64)
    high = np.full(uniforms.shape, weights.shape[1] - 1, dtype=np.int64)
    rows = np.arange(len(weights))[:, None]
    while np.any(low < high):
        mid = (low + high) // 2
        left = uniforms <= cdf[rows, mid]
        high = np.where(left, mid, high)
        low = np.where(left, low, mid + 1)
    return low


@dataclass(frozen=True)
class GridPosterior:
    """Discrete posterior of mu on a sorted signal grid, for a batch of observations.

    Parameters
    ----------
    mu_grid : ndarray, shape (G,)
        Sorted signal nodes. ``exact_posterior`` puts the zero atom at index 0.
    weights : ndarray, shape (N, G)
        Posterior probability mass at each node; each row sums to one.
    mean, variance : ndarray, shape (N,)
        Posterior mean and variance of mu, computed from ``weights``.
    cell_widths : ndarray, shape (G,), optional
        Quadrature width of each node's continuous cell. A width of zero marks
        a node with no continuous cell, such as the zero atom.

    Notes
    -----
    The posterior mean is the Bayes rule for squared loss: it minimizes the
    posterior expected loss ``E[(mu - d)**2 | x]`` over the estimate d.
    """

    mu_grid: FloatArray
    weights: FloatArray
    mean: FloatArray
    variance: FloatArray
    cell_widths: FloatArray | None = None

    def intervals(self, *, alpha: float = 0.1) -> IntervalBatch:
        """Return equal-tail credible intervals with total tail probability alpha.

        Parameters
        ----------
        alpha : float, default 0.1 (``FULL_CONFIG.problem.alpha``)
            Total tail probability, strictly between 0 and 1.

        Returns
        -------
        IntervalBatch
            Inclusive endpoints on grid nodes, each of shape (N,). ``lower`` is
            the first node with CDF >= alpha/2 and ``upper`` is the first node
            with CDF >= 1 - alpha/2.

        Raises
        ------
        ValueError
            If alpha is not strictly between 0 and 1.
        """
        _alpha(alpha)
        cdf = np.cumsum(self.weights, axis=1)
        lower = np.argmax(cdf >= alpha / 2, axis=1)
        upper = np.argmax(cdf >= 1 - alpha / 2, axis=1)
        return IntervalBatch(self.mu_grid[lower], self.mu_grid[upper])

    def hpd_intervals(self, *, alpha: float = 0.1) -> IntervalBatch:
        """Return a smallest-volume grid credible set and its inclusive hull.

        Nodes are ranked by mass per unit signal length (weight / cell width).
        A zero atom with positive mass ranks first: its length is zero, so its
        density is infinite. Nodes are accepted in this order until the
        accepted mass reaches at least ``1 - alpha``.

        Parameters
        ----------
        alpha : float, default 0.1 (``FULL_CONFIG.problem.alpha``)
            Maximum excluded probability, strictly between 0 and 1.

        Returns
        -------
        IntervalBatch
            ``lower`` and ``upper`` are the smallest and largest accepted nodes
            (the hull), each of shape (N,). ``accepted`` is the (N, G) bool mask
            and ``mu_grid`` is the grid. The mask can have gaps, so the hull
            itself need not be an HPD set.

        Raises
        ------
        ValueError
            If alpha is not strictly between 0 and 1.

        Notes
        -----
        If ``cell_widths`` is None (a user-built grid), the widths come from
        ``np.gradient(mu_grid)``, the local spacing between neighbouring nodes.
        """
        _alpha(alpha)
        widths = self.cell_widths
        if widths is None:
            widths = np.gradient(self.mu_grid)
        density = np.divide(self.weights, widths, out=np.zeros_like(self.weights), where=widths > 0)
        atom = (self.mu_grid == 0) & (widths == 0)
        density[:, atom] = np.where(self.weights[:, atom] > 0, np.inf, 0)
        order = np.argsort(-density, axis=1, kind="stable")
        ordered = np.take_along_axis(self.weights, order, axis=1)
        keep = (np.cumsum(ordered, axis=1) - ordered) < 1 - alpha
        accepted = np.zeros(self.weights.shape, dtype=bool)
        np.put_along_axis(accepted, order, keep, axis=1)
        lower = self.mu_grid[np.argmax(accepted, axis=1)]
        upper = self.mu_grid[len(self.mu_grid) - 1 - np.argmax(accepted[:, ::-1], axis=1)]
        return IntervalBatch(lower, upper, accepted, self.mu_grid)

    def sample(self, n_draws: int, *, rng: np.random.Generator) -> FloatArray:
        """Draw signal values from each row's discrete posterior.

        Parameters
        ----------
        n_draws : int
            Number of draws per observation; must be nonnegative.
        rng : numpy.random.Generator
            Explicit random source.

        Returns
        -------
        ndarray, shape (N, n_draws)
            Grid nodes drawn with probabilities ``weights``. There is no
            jitter inside cells, so the zero atom gives exact zeros.

        Raises
        ------
        ValueError
            If n_draws is negative.
        """
        return self.mu_grid[_draw_indices(self.weights, n_draws, rng)]


def _signal_rule(prior: Prior, config: StudyConfig) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return nodes [0, cell midpoints, mu_max], their widths and their prior masses."""
    low = prior.mu_min if prior.family == "reality" else 0.0
    high = config.problem.mu_max
    cells = max(2, int(np.ceil((high - low) / config.grid.mu_step)))
    edges = np.linspace(low, high, cells + 1)
    widths = np.diff(edges)
    mu = np.concatenate(([0.0], (edges[:-1] + edges[1:]) / 2, [high]))
    integration_widths = np.concatenate(([0.0], widths, [0.0]))
    masses = prior.mu_density(mu) * integration_widths
    masses[0] = prior.zero_mass if prior.family == "reality" else 0.0
    return mu, integration_widths, masses


@lru_cache(maxsize=32)
def _background_integrals(config: StudyConfig, max_power: int) -> FloatArray:
    """Return log E_b[b**q exp(-(1+tau)b)], q = 0..max_power, by Gauss-Legendre quadrature."""
    p = config.problem
    nodes, weights = np.polynomial.legendre.leggauss(config.grid.b_points)
    b = p.b_min + (nodes + 1) * (p.b_max - p.b_min) / 2
    with np.errstate(divide="ignore"):
        return logsumexp(
            np.arange(max_power + 1)[:, None] * np.log(b) - (1 + p.tau) * b + np.log(weights / 2),
            axis=1,
        )


def _log_marginal_likelihood(x: FloatArray, mu: FloatArray, config: StudyConfig) -> FloatArray:
    """Return log p(x | mu) with b and eps integrated out, up to a per-row constant, (N, G)."""
    p = config.problem
    if p.mode == "core":
        rate = mu[None, :] + p.known_b
        return xlogy(x[:, 0, None], rate) - rate
    if not len(x):
        return np.empty((0, len(mu)))
    n = x[:, 0].astype(int)
    m = x[:, 1].astype(int)
    k = np.arange(int(n.max()) + 1)
    remainder = np.maximum(n[:, None] - k, 0)
    log_integrals = _background_integrals(config, int((n + m).max()))
    # logs[i, k] = log[C(n_i, k) I(m_i + n_i - k)]: the coefficient of t**k, t = eps*mu.
    logs = (
        gammaln(n[:, None] + 1)
        - gammaln(k + 1)
        - gammaln(remainder + 1)
        + log_integrals[m[:, None] + remainder]
    )
    logs[k[None, :] > n[:, None]] = -np.inf
    coefficients = np.exp(logs - logs.max(axis=1, keepdims=True))
    if p.sigma_eps == 0:
        z, z_weights = np.zeros(1), np.ones(1)
        log_center, log_sd = np.log(x[:, 2]), 0.0
    else:
        z, z_weights = roots_hermitenorm(config.grid.eps_points)
        z_weights /= np.sqrt(2 * np.pi)
        # The eps prior and the calibration share the log-scale s.d. sigma_eps, so
        # log eps | eps_hat ~ Normal((log eps0 + log eps_hat) / 2, sigma_eps**2 / 2).
        log_center = (np.log(p.eps0) + np.log(x[:, 2])) / 2
        log_sd = p.sigma_eps / np.sqrt(2)
    answer = np.empty((len(x), len(mu)))
    # Process 64 rows at a time to bound memory. The inner Python loop is Horner's
    # rule over the polynomial degree in t; nodes and eps points are vectorized.
    for start in range(0, len(x), 64):
        stop = min(start + 64, len(x))
        eps = np.exp(log_center[start:stop, None] + log_sd * z[None, :])
        t = mu[None, :, None] * eps[:, None, :]
        if n.max() >= 128:
            # Rare large counts (n_on >= 128 in the batch): Horner in log space avoids overflow.
            with np.errstate(divide="ignore"):
                log_t = np.log(t)
            log_poly = np.full(t.shape, -np.inf)
            for degree in range(len(k) - 1, -1, -1):
                log_poly = np.logaddexp(log_poly + log_t, logs[start:stop, degree, None, None])
            log_values = log_poly - t + np.log(z_weights)
        else:
            poly = np.zeros(t.shape)
            for degree in range(len(k) - 1, -1, -1):
                poly *= t
                poly += coefficients[start:stop, degree, None, None]
            with np.errstate(divide="ignore"):
                log_values = np.log(poly) - t + np.log(z_weights)
        answer[start:stop] = logsumexp(log_values, axis=2)
    return answer


def _grid_posterior(mu: FloatArray, logs: FloatArray, widths: FloatArray) -> GridPosterior:
    """Normalize unnormalized log masses (N, G) row-wise and return the GridPosterior."""
    normalizers = logsumexp(logs, axis=1, keepdims=True)
    if not np.all(np.isfinite(normalizers)):
        raise ValueError("posterior normalization failed; check numerical support and observations")
    weights = np.exp(logs - normalizers)
    mean = np.sum(weights * mu, axis=1)
    variance = np.maximum(np.sum(weights * mu**2, axis=1) - mean**2, 0.0)
    return GridPosterior(mu, weights, mean, variance, widths)


def exact_posterior(
    x: FloatArray,
    *,
    prior: Prior | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> GridPosterior:
    """Compute the posterior of mu on a signal grid by numerical quadrature.

    Parameters
    ----------
    x : ndarray, shape (N, 3)
        Raw observations (n_on, n_off, eps_hat).
    prior : Prior, optional
        Analysis prior for mu; it must use ``config``. Default: the reality prior.
    config : StudyConfig, default FULL_CONFIG
        Model constants and quadrature sizes (``grid.mu_step``,
        ``grid.b_points``, ``grid.eps_points``).

    Returns
    -------
    GridPosterior
        Masses, mean and variance on the nodes [0, cell midpoints, mu_max].

    Raises
    ------
    ValueError
        If x is invalid, the prior uses another config, or normalization fails.

    Notes
    -----
    The mass at node mu_j is proportional to ``pi_j L(mu_j)``. ``pi_j`` is the
    atom mass at 0 (reality prior only) or the prior density times the cell
    width. Cells cover [mu_min, mu_max] for the reality prior and [0, mu_max]
    otherwise; the mu_max node has zero width and so zero mass. ``L`` is the
    likelihood with b ~ Uniform[b_min, b_max] and eps integrated out::

        L(mu) = sum_k C(n_on, k) I(n_off + n_on - k) E[(eps*mu)**k exp(-eps*mu) | eps_hat]
        I(q) = E_b[b**q exp(-(1 + tau) b)]

    The sum is the exact binomial expansion of ``(eps*mu + b)**n_on``. ``I(q)``
    uses Gauss-Legendre nodes. The eps average uses Gauss-Hermite nodes for
    ``log eps | eps_hat ~ Normal((log eps0 + log eps_hat)/2, sigma_eps**2/2)``,
    which is the eps prior times the calibration likelihood, renormalized.
    Factors free of mu cancel in normalization. In core mode,
    ``L(mu) = Pois(n_on | mu + known_b)``.
    """
    values = _observations(x)
    selected = _prior(prior, config)
    mu, widths, masses = _signal_rule(selected, config)
    with np.errstate(divide="ignore"):
        logs = _log_marginal_likelihood(values, mu, config) + np.log(masses)
    return _grid_posterior(mu, logs, widths)


def _log_gamma_cdf(shape: FloatArray, upper: FloatArray) -> FloatArray:
    """Return log P(a, z), the regularized lower incomplete Gamma, also where it underflows.

    Where scipy returns 0 for z > 0, use the convergent positive power series
    P(a,z) = exp(a*log(z) - z - lgamma(a+1)) * sum_k z**k / (a+1)_k, where
    (a+1)_k = (a+1)(a+2)...(a+k) is the rising factorial.
    """
    a, z = np.broadcast_arrays(shape, upper)
    probability = gammainc(a, z)
    with np.errstate(divide="ignore"):
        answer = np.log(probability)
    underflow = (probability == 0) & (z > 0)
    if np.any(underflow):
        aa, zz = a[underflow], z[underflow]
        term = np.ones(aa.shape)
        total = term.copy()
        for k in range(1, 10000):
            term *= zz / (aa + k)
            total += term
            if np.all(term <= total * np.finfo(float).eps):
                break
        else:
            raise ValueError("Gamma tail series failed to converge")
        answer[underflow] = aa * np.log(zz) - zz - gammaln(aa + 1) + np.log(total)
    return answer


def _gamma_components(
    x: FloatArray,
    shape: float,
    rate: float,
    config: StudyConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return plug-in mixture weights, shapes a_k, rate c and log P(a_k, c*mu_max)."""
    values = _observations(x)
    if not np.isfinite(shape + rate) or shape <= 0 or rate <= 0:
        raise ValueError("Gamma shape and rate must be positive and finite")
    p = config.problem
    n = values[:, 0, None].astype(int)
    k = np.arange(int(n.max(initial=0)) + 1)[None, :]
    remainder = np.maximum(n - k, 0)
    b = np.full((len(values), 1), p.known_b) if p.mode == "core" else values[:, 1, None] / p.tau
    eps = np.ones((len(values), 1)) if p.mode == "core" else values[:, 2, None]
    c = eps + rate
    shapes = np.broadcast_to(float(shape) + k, (len(values), k.shape[1]))
    log_truncation = _log_gamma_cdf(shapes, c * p.mu_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = (
            gammaln(n + 1)
            - gammaln(k + 1)
            - gammaln(remainder + 1)
            + xlogy(remainder, b)
            + k * np.log(eps)
            + gammaln(shapes)
            - shapes * np.log(c)
            + log_truncation
        )
    logs = np.where(k <= n, logs, -np.inf)
    weights = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
    return weights, shapes, c, log_truncation


def gamma_plugin_moments(
    x: FloatArray,
    *,
    shape: float = 1.0,
    rate: float = 0.1,
    config: StudyConfig = FULL_CONFIG,
) -> PosteriorMoments:
    """Return posterior moments of mu for a truncated Gamma prior and plug-in nuisances.

    Parameters
    ----------
    x : ndarray, shape (N, 3)
        Raw observations (n_on, n_off, eps_hat).
    shape, rate : float, default 1.0 and 0.1
        Parameters of the Gamma(shape, rate) prior on mu, truncated to
        [0, mu_max]. The defaults match the ``gamma_diffuse`` analysis prior.
    config : StudyConfig, default FULL_CONFIG
        Model constants.

    Returns
    -------
    PosteriorMoments
        Posterior mean and variance, each of shape (N,).

    Raises
    ------
    ValueError
        If x is invalid or shape or rate is not positive and finite.

    Notes
    -----
    The nuisances are fixed at b_hat = n_off/tau and eps = eps_hat (core
    mode: known_b and 1), so their uncertainty is ignored. The prior has no
    zero atom. The binomial expansion of ``(eps*mu + b_hat)**n_on`` makes the
    posterior a finite mixture of Gamma(a_k, c) laws truncated to [0, mu_max],
    with ``a_k = shape + k``, ``c = rate + eps`` and k = 0..n_on. This is exact
    for the plug-in likelihood, not a conjugacy claim for background-subtracted
    counts. Component k has weight proportional to
    ``C(n_on, k) eps**k b_hat**(n_on - k) Gamma(a_k) c**(-a_k) P(a_k, c*mu_max)``,
    where P is the regularized lower incomplete Gamma function. Each component
    has ``E[mu] = (a/c) P(a+1, c*mu_max) / P(a, c*mu_max)`` and
    ``E[mu**2] = a(a+1)/c**2 P(a+2, c*mu_max) / P(a, c*mu_max)``.
    If b_hat = 0, only the component k = n_on remains.
    """
    weights, shapes, c, log_probability = _gamma_components(x, shape, rate, config)
    upper = config.problem.mu_max * c
    first = shapes / c * np.exp(_log_gamma_cdf(shapes + 1, upper) - log_probability)
    second = (
        shapes * (shapes + 1) / c**2 * np.exp(_log_gamma_cdf(shapes + 2, upper) - log_probability)
    )
    mean = np.sum(weights * first, axis=1)
    return PosteriorMoments(mean, np.maximum(np.sum(weights * second, axis=1) - mean**2, 0))


@dataclass(frozen=True)
class GammaPluginPosterior:
    """Posterior sampler for the truncated Gamma mixture of ``gamma_plugin_moments``.

    Parameters
    ----------
    config : StudyConfig, default FULL_CONFIG
        Model constants.
    shape, rate : float, default 1.0 and 0.1
        Gamma prior parameters for mu, truncated to [0, mu_max].
    """

    config: StudyConfig = FULL_CONFIG
    shape: float = 1.0
    rate: float = 0.1

    def sample(self, x: FloatArray, n_draws: int, *, rng: np.random.Generator) -> FloatArray:
        """Draw mu from the plug-in Gamma-mixture posterior of each observation.

        Parameters
        ----------
        x : ndarray, shape (N, 3)
            Raw observations (n_on, n_off, eps_hat).
        n_draws : int
            Number of draws per observation.
        rng : numpy.random.Generator
            Explicit random source.

        Returns
        -------
        ndarray, shape (N, n_draws)
            Draws in [0, mu_max]. The law is continuous, with no atom at zero.

        Raises
        ------
        ValueError
            If x, shape, rate or n_draws is invalid.

        Notes
        -----
        Each draw picks component k with probability equal to its weight.
        It then inverts the truncated CDF: it solves
        ``P(a_k, y) = u P(a_k, c*mu_max)`` for y, with u ~ Uniform(0, 1),
        and returns mu = y / c. Where the log target is below -700 (so
        ``exp`` would underflow), 55 bisection steps on [0, c*mu_max] in log
        space replace ``gammaincinv``.
        """
        weights, shapes, c, log_probability = _gamma_components(
            x, self.shape, self.rate, self.config
        )
        indices = _draw_indices(weights, n_draws, rng)
        rows = np.arange(len(weights))[:, None]
        a = shapes[rows, indices]
        with np.errstate(divide="ignore"):
            log_u = np.log(rng.random(indices.shape)) + log_probability[rows, indices]
        values = gammaincinv(a, np.exp(log_u))
        underflow = log_u < -700
        if np.any(underflow):
            low = np.zeros(np.sum(underflow))
            high = np.broadcast_to(self.config.problem.mu_max * c, a.shape)[underflow].copy()
            for _ in range(55):
                midpoint = (low + high) / 2
                below = _log_gamma_cdf(a[underflow], midpoint) < log_u[underflow]
                low = np.where(below, midpoint, low)
                high = np.where(below, high, midpoint)
            values[underflow] = (low + high) / 2
        return values / c


def _laplace_grid(x: FloatArray, prior: Prior | None, config: StudyConfig) -> GridPosterior:
    """Return the atom-plus-local-quadratic approximation on a 201-node signal grid."""
    values = _observations(x)
    selected = _prior(prior, config)
    p = config.problem
    low = selected.mu_min if selected.family == "reality" else 0.0
    mu = np.linspace(max(low, 1e-6), p.mu_max, 201)
    step = mu[1] - mu[0]
    b = p.known_b if p.mode == "core" else values[:, 1, None] / p.tau
    eps = 1.0 if p.mode == "core" else values[:, 2, None]
    n = values[:, 0, None]
    lam = eps * mu + b
    logs = xlogy(n, lam) - lam + np.log(selected.mu_density(mu))
    mode_index = np.argmax(logs, axis=1)
    mode = mu[mode_index][:, None]
    lam_mode = eps * mode + b
    # Slope and precision (minus the second derivative) of the log posterior at the mode.
    gradient = eps * (n / lam_mode - 1)
    curvature = n * eps**2 / lam_mode**2
    if selected.family == "reality":
        gradient -= 1 / mode
        curvature -= 1 / mode**2
    elif selected.family == "gamma":
        gradient += (selected.shape - 1) / mode - selected.rate
        curvature += (selected.shape - 1) / mode**2
    # Negative precision has no real s.d.; keep only the linear tangent there.
    curvature = np.maximum(curvature, 0)
    at_mode = logs[np.arange(len(values)), mode_index, None]
    approximation = at_mode + gradient * (mu - mode) - curvature * (mu - mode) ** 2 / 2
    # Trapezoid weights on the slab nodes.
    widths = np.full(len(mu), step)
    widths[[0, -1]] /= 2
    slab = approximation + np.log(widths)
    atom_mass = selected.zero_mass if selected.family == "reality" else 0.0
    with np.errstate(divide="ignore"):
        atom = xlogy(n, b) - b + np.log(atom_mass)
    all_mu = np.concatenate(([0.0], mu))
    return _grid_posterior(
        all_mu, np.concatenate((atom, slab), axis=1), np.concatenate(([0.0], widths))
    )


def laplace_moments(
    x: FloatArray,
    *,
    prior: Prior | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> PosteriorMoments:
    """Return posterior moments of mu from a bounded Laplace-type approximation.

    Parameters
    ----------
    x : ndarray, shape (N, 3)
        Raw observations (n_on, n_off, eps_hat).
    prior : Prior, optional
        Analysis prior for mu; it must use ``config``. Default: the reality prior.
    config : StudyConfig, default FULL_CONFIG
        Model constants.

    Returns
    -------
    PosteriorMoments
        Posterior mean and variance, each of shape (N,).

    Raises
    ------
    ValueError
        If x is invalid, the prior uses another config, or normalization fails.

    Notes
    -----
    The nuisances are fixed at b_hat = n_off/tau and eps = eps_hat (core
    mode: known_b and 1). Let l(mu) be the plug-in log likelihood plus the
    log prior density. The mode m is the argmax of l on 201 equally spaced
    nodes over the continuous prior support (lower end at least 1e-6). With
    slope g = l'(m) and precision h = max(-l''(m), 0)::

        l(mu) ~ l(m) + g (mu - m) - h (mu - m)**2 / 2

    This curve is integrated by the trapezoid rule on the same nodes. The
    slope g is kept, so a mode on a boundary is handled, and h = 0 gives a
    linear tangent. The zero atom (reality prior only) keeps its exact
    plug-in weight: atom mass times the likelihood at mu = 0. This is not
    exact nuisance marginalization, and SBC can reject it.
    """
    posterior = _laplace_grid(x, prior, config)
    return PosteriorMoments(posterior.mean, posterior.variance)


@dataclass(frozen=True)
class LaplacePosterior:
    """Posterior sampler for the approximation used by ``laplace_moments``.

    Parameters
    ----------
    config : StudyConfig, default FULL_CONFIG
        Model constants.
    prior : Prior, optional
        Analysis prior for mu. Default: the reality prior for ``config``.
    """

    config: StudyConfig = FULL_CONFIG
    prior: Prior | None = None

    def sample(self, x: FloatArray, n_draws: int, *, rng: np.random.Generator) -> FloatArray:
        """Draw mu from the bounded atom-plus-local-quadratic approximation.

        Parameters
        ----------
        x : ndarray, shape (N, 3)
            Raw observations (n_on, n_off, eps_hat).
        n_draws : int
            Number of draws per observation.
        rng : numpy.random.Generator
            Explicit random source.

        Returns
        -------
        ndarray, shape (N, n_draws)
            Grid-node draws: 0 for the atom, or one of the 201 slab nodes.
        """
        return _laplace_grid(x, self.prior, self.config).sample(n_draws, rng=rng)


def clipped_mle(x: FloatArray, *, config: StudyConfig = FULL_CONFIG) -> FloatArray:
    """Return the maximum likelihood estimate of mu, clipped to Theta = [0, mu_max].

    Parameters
    ----------
    x : ndarray, shape (N, 3)
        Raw observations (n_on, n_off, eps_hat).
    config : StudyConfig, default FULL_CONFIG
        Model constants (tau and mu_max; known_b in core mode).

    Returns
    -------
    ndarray, shape (N,)
        ``clip((n_on - n_off/tau) / eps_hat, 0, mu_max)`` in full mode and
        ``clip(n_on - known_b, 0, mu_max)`` in core mode, as float64.

    Raises
    ------
    ValueError
        If x is invalid.

    Notes
    -----
    Without the bound on mu, the joint likelihood is maximized at
    eps = eps_hat, b = n_off/tau and eps*mu + b = n_on. If that mu lies
    outside Theta, the constrained optimum for mu is the nearest boundary.
    Profiling b and eps at that boundary changes their estimates but not mu,
    so the clipped value is also the profile-likelihood MLE. No prior is
    used, and the prior range [b_min, b_max] of b is not imposed.
    """
    values = _observations(x)
    p = config.problem
    estimate = (
        values[:, 0] - p.known_b
        if p.mode == "core"
        else (values[:, 0] - values[:, 1] / p.tau) / values[:, 2]
    )
    return np.clip(estimate, 0, p.mu_max).astype(np.float64)


def _loss_result(truth: FloatArray, estimate: FloatArray, loss: str) -> RiskResult:
    """Return the mean loss and its MC SE, sd(loss, ddof=1) / sqrt(M), as a RiskResult."""
    if loss not in ("squared", "absolute"):
        raise ValueError("loss must be squared or absolute")
    predicted = np.asarray(estimate, dtype=np.float64)
    if predicted.shape != truth.shape or not np.all(np.isfinite(predicted)):
        raise ValueError("estimator must return one finite estimate per observation")
    losses = (predicted - truth) ** 2 if loss == "squared" else np.abs(predicted - truth)
    return RiskResult(
        float(losses.mean()), float(losses.std(ddof=1) / np.sqrt(len(losses))), len(losses), loss
    )


def estimate_bayes_risk(
    estimator: PointEstimator,
    *,
    rng: np.random.Generator,
    n_samples: int | None = None,
    prior: Prior | None = None,
    config: StudyConfig = FULL_CONFIG,
    loss: Literal["squared", "absolute"] = "squared",
) -> RiskResult:
    """Estimate the Bayes risk of a point estimator by Monte Carlo.

    Parameters
    ----------
    estimator : PointEstimator
        Callable from raw observations (M, 3) to estimates of mu, shape (M,).
    rng : numpy.random.Generator
        Explicit random source for the joint draws.
    n_samples : int, optional
        Number M of joint draws, at least 2. Default: ``config.mc.m_bayes_risk``.
    prior : Prior, optional
        Prior pi that generates theta. Default: the reality prior for ``config``.
    config : StudyConfig, default FULL_CONFIG
        Model constants.
    loss : {"squared", "absolute"}, default "squared"
        Loss ``(d - mu)**2`` or ``|d - mu|`` for an estimate d.

    Returns
    -------
    RiskResult
        Risk estimate, its Monte Carlo standard error, M and the loss name.

    Raises
    ------
    ValueError
        If n_samples < 2, the prior uses another config, the loss name is
        unknown, or the estimator does not return M finite values.

    Notes
    -----
    The target is ``r(pi, delta) = E_{theta~pi} E_{x|theta} L(mu, delta(x))``.
    The estimate is the mean loss over M independent draws (theta_i, x_i)
    from the prior and the simulator. Its standard error is
    ``sd(loss, ddof=1) / sqrt(M)``; it measures simulation noise only, not
    posterior uncertainty. The estimator is called once on the whole batch.
    """
    count = config.mc.m_bayes_risk if n_samples is None else n_samples
    if count < 2:
        raise ValueError("n_samples must be at least two for a Monte Carlo standard error")
    theta, x = sample_joint(count, rng=rng, config=config, prior=prior)
    return _loss_result(theta[:, 0], estimator(x), loss)


def estimate_frequentist_risk(
    estimator: PointEstimator,
    theta: FloatArray,
    *,
    rng: np.random.Generator,
    n_samples: int | None = None,
    config: StudyConfig = FULL_CONFIG,
    loss: Literal["squared", "absolute"] = "squared",
) -> list[RiskResult]:
    """Estimate the frequentist risk of a point estimator at fixed parameter values.

    Parameters
    ----------
    estimator : PointEstimator
        Callable from raw observations (M, 3) to estimates of mu, shape (M,).
    theta : array_like, shape (K, 3)
        Parameter rows (mu, b, eps).
    rng : numpy.random.Generator
        Explicit random source for the toys.
    n_samples : int, optional
        Number M of toys per row, at least 2. Default: ``config.mc.n_toys_cov``.
    config : StudyConfig, default FULL_CONFIG
        Model constants.
    loss : {"squared", "absolute"}, default "squared"
        Loss ``(d - mu)**2`` or ``|d - mu|`` for an estimate d.

    Returns
    -------
    list of RiskResult
        One result per row of theta, in the same order.

    Raises
    ------
    ValueError
        If theta is not (K, 3), n_samples < 2, the loss name is unknown, or the
        estimator returns invalid values. In core mode the simulator also
        rejects rows with b != known_b or eps != 1.

    Notes
    -----
    For each row the target is ``R(theta, delta) = E_{x|theta} L(mu, delta(x))``.
    Each row gets its own M new toys from ``rng``, so the row estimates are
    independent. Each standard error is ``sd(loss, ddof=1) / sqrt(M)``.
    """
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("theta must have shape (K,3)")
    count = config.mc.n_toys_cov if n_samples is None else n_samples
    if count < 2:
        raise ValueError("n_samples must be at least two")
    results = []
    for mu, b, eps in values:
        x = simulate(mu, b, eps, rng=rng, config=config, size=count)
        results.append(_loss_result(np.full(count, mu), estimator(x), loss))
    return results


def time_inference(estimator: PointEstimator, x: FloatArray, *, repeats: int = 7) -> float:
    """Return the median batched inference time per observation, in microseconds.

    Parameters
    ----------
    estimator : PointEstimator
        Callable to time; it receives the whole batch in each call.
    x : ndarray, shape (N, 3)
        Nonempty batch of raw observations.
    repeats : int, default 7
        Number of timed calls, at least 3.

    Returns
    -------
    float
        Median over the timed calls of (wall time / N), in microseconds.

    Raises
    ------
    ValueError
        If x is invalid or empty, or repeats < 3.

    Notes
    -----
    One untimed warm-up call comes first. The timed calls include all work
    inside ``estimator``, such as feature transforms or quadrature. They
    exclude simulation and model training.
    """
    values = _observations(x)
    if not len(values) or repeats < 3:
        raise ValueError("timing requires nonempty data and at least three repetitions")
    estimator(values)
    times = []
    for _ in range(repeats):
        started = perf_counter()
        estimator(values)
        times.append((perf_counter() - started) * 1e6 / len(values))
    return float(np.median(times))


def simulation_based_calibration(
    posterior: PosteriorSampler,
    *,
    rng: np.random.Generator,
    n_samples: int | None = None,
    n_draws: int | None = None,
    prior: Prior | None = None,
    config: StudyConfig = FULL_CONFIG,
) -> NDArray[np.int64]:
    """Return simulation-based calibration (SBC) ranks with random tie breaking.

    Parameters
    ----------
    posterior : PosteriorSampler
        Object whose ``sample(x, n_draws, rng=rng)`` returns (M, n_draws) draws.
    rng : numpy.random.Generator
        Explicit random source for joint draws, posterior draws and tie breaking.
    n_samples : int, optional
        Number M of prior draws, at least 1. Default: ``config.mc.n_sbc``.
    n_draws : int, optional
        Number D of posterior draws per observation, at least 1. Default:
        ``config.mc.n_posterior_draws``.
    prior : Prior, optional
        Prior that generates theta. Default: the reality prior for ``config``.
    config : StudyConfig, default FULL_CONFIG
        Model constants.

    Returns
    -------
    ndarray of int64, shape (M,)
        Ranks in 0..D.

    Raises
    ------
    ValueError
        If M or D is below 1, the prior uses another config, or the sampler
        returns draws with the wrong shape or non-finite values.

    Notes
    -----
    For each draw (mu_i, x_i) from the prior and simulator, the rank is
    ``#{draws < mu_i} + U``, where U is a uniform integer in
    0..#{draws == mu_i}. If the sampler draws from the exact posterior for
    this prior, the rank is uniform on 0..D. Random tie breaking keeps this
    true at the zero atom, where draws can equal mu_i = 0 exactly. SBC checks
    calibration averaged over the prior, not coverage at a fixed theta.
    """
    count = config.mc.n_sbc if n_samples is None else n_samples
    draws_count = config.mc.n_posterior_draws if n_draws is None else n_draws
    if count < 1 or draws_count < 1:
        raise ValueError("SBC requires positive sample and posterior-draw counts")
    theta, x = sample_joint(count, rng=rng, config=config, prior=prior)
    draws = np.asarray(posterior.sample(x, draws_count, rng=rng))
    if draws.shape != (count, draws_count) or not np.all(np.isfinite(draws)):
        raise ValueError("posterior sampler returned invalid draws")
    truth = theta[:, 0, None]
    below = np.sum(draws < truth, axis=1)
    ties = np.sum(draws == truth, axis=1)
    return (below + rng.integers(0, ties + 1)).astype(np.int64)
