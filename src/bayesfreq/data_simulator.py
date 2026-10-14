"""Priors, the on/off simulator, and exact likelihoods shared by both paths.

This module defines one data-generating process (DGP). Path A and Path B
both use it. Rows of ``theta`` are parameters (mu, b, eps): signal rate,
background rate, and relative detector efficiency. Rows of ``x`` are
observations (n_on, n_off, eps_hat): the on-region count, the off-region
count, and an auxiliary efficiency calibration measurement.

Notes
-----
Given theta, the three observations are independent::

    n_on          ~ Poisson(eps * mu + b)
    n_off         ~ Poisson(tau * b)
    log(eps_hat)  ~ Normal(log(eps), sigma_eps**2)

The two counts are integer-valued, but ``x`` is float64 so that it can also
hold the positive calibration measurement in its third column.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc, gammaincinv, gammaln, xlogy
from scipy.stats import poisson

from bayesfreq.config import FULL_CONFIG, StudyConfig


@dataclass(frozen=True)
class Prior:
    """Independent signal, background, and efficiency prior.

    Parameters
    ----------
    config : StudyConfig
        Defines support and nuisance distributions. Core nuisance values are fixed.
    family : {"reality", "flat", "gamma"}
        Signal distribution. Gamma is truncated and normalized on [0, mu_max].
    zero_mass, mu_min : float
        Reality mixture weight at zero and positive log-uniform lower cutoff.
    shape, rate : float
        Gamma shape and rate, used only for the gamma family.

    Notes
    -----
    ``mu_density`` is a density with respect to delta_0 + Lebesgue measure:
    its value at exactly zero is the atom probability, never a continuous
    density limit. Numerical integration must give the zero atom weight 1
    and use quadrature weights only for positive mu. In the core case, b and
    eps use point measures at known_b and 1; their density there is 1.
    """

    config: StudyConfig = FULL_CONFIG
    family: Literal["reality", "flat", "gamma"] = "reality"
    zero_mass: float = 0.3
    mu_min: float = 0.1
    shape: float = 1.0
    rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate the family name, mixture weight, cutoff, and Gamma parameters."""
        if self.family not in ("reality", "flat", "gamma"):
            raise ValueError("unknown prior family")
        if not 0 <= self.zero_mass <= 1 or not 0 < self.mu_min < self.config.problem.mu_max:
            raise ValueError("invalid reality mixture weight or cutoff")
        if not np.isfinite(self.shape) or not np.isfinite(self.rate):
            raise ValueError("gamma shape and rate must be finite")
        if self.shape <= 0 or self.rate <= 0:
            raise ValueError("gamma shape and rate must be positive")

    def sample(self, n: int, *, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw n independent theta rows, returning float64 shape (n, 3).

        Parameters
        ----------
        n : int
            Nonnegative batch size.
        rng : numpy.random.Generator
            Explicit source of randomness; no global random state is used.

        Returns
        -------
        numpy.ndarray
            Signal, background, and efficiency in that column order.
        """
        if n < 0:
            raise ValueError("n must be nonnegative")
        p = self.config.problem
        if self.family == "reality":
            mu = np.exp(rng.uniform(np.log(self.mu_min), np.log(p.mu_max), n))
            mu[rng.random(n) < self.zero_mass] = 0.0
        elif self.family == "flat":
            mu = rng.uniform(0.0, p.mu_max, n)
        else:
            normalizer = gammainc(self.shape, self.rate * p.mu_max)
            mu = gammaincinv(self.shape, rng.random(n) * normalizer) / self.rate
        if p.mode == "core":
            b = np.full(n, p.known_b)
            eps = np.ones(n)
        else:
            b = rng.uniform(p.b_min, p.b_max, n)
            eps = rng.lognormal(np.log(p.eps0), p.sigma_eps, n)
        return np.column_stack((mu, b, eps))

    def mu_density(self, mu: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the signal prior density with respect to delta_0 + Lebesgue.

        Parameters
        ----------
        mu : array_like
            Signal values of any shape.

        Returns
        -------
        numpy.ndarray
            Float64 array with the shape of ``mu``. At ``mu == 0`` the value is
            the atom probability: ``zero_mass`` for the reality family and 0
            for the flat and Gamma families. For ``0 < mu <= mu_max`` it is the
            normalized continuous density. Outside the support, including
            negative or nonfinite inputs, it is 0.

        Notes
        -----
        The densities on ``(0, mu_max]`` are::

            reality: (1 - zero_mass) / (mu * log(mu_max / mu_min)),  mu >= mu_min
            flat:    1 / mu_max
            gamma:   rate**shape * mu**(shape - 1) * exp(-rate * mu)
                     / (Gamma(shape) * P(shape, rate * mu_max))

        where P is the regularized lower incomplete gamma function.
        """
        values = np.asarray(mu, dtype=np.float64)
        p = self.config.problem
        positive = (values > 0) & (values <= p.mu_max)
        safe = np.where(positive, values, 1.0)
        if self.family == "reality":
            density = (1 - self.zero_mass) / (safe * np.log(p.mu_max / self.mu_min))
            result = np.where(positive & (values >= self.mu_min), density, 0.0)
            return np.where(values == 0, self.zero_mass, result)
        if self.family == "flat":
            return np.where(positive, 1.0 / p.mu_max, 0.0)
        log_density = (
            self.shape * np.log(self.rate)
            - gammaln(self.shape)
            + (self.shape - 1) * np.log(safe)
            - self.rate * safe
            - np.log(gammainc(self.shape, self.rate * p.mu_max))
        )
        return np.where(positive, np.exp(log_density), 0.0)

    def log_prob(self, theta: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the joint prior log density of (mu, b, eps).

        Parameters
        ----------
        theta : array_like, shape (..., 3)
            Parameter rows (mu, b, eps).

        Returns
        -------
        numpy.ndarray
            Float64 array of shape (...,). It is -inf outside the support.

        Notes
        -----
        The signal part uses the measure delta_0 + Lebesgue (see
        ``mu_density``). In the full case, b has the uniform density
        ``1 / (b_max - b_min)`` and eps has a log-normal density with median
        eps0 and log-scale s.d. sigma_eps. In the core case, b and eps are
        point masses at ``known_b`` and 1. This is a prior density, not a
        likelihood.
        """
        values = _three_columns(theta, "theta")
        mu, b, eps = np.moveaxis(values, -1, 0)
        density = self.mu_density(mu)
        with np.errstate(divide="ignore"):
            log_density = np.log(density)
        p = self.config.problem
        if p.mode == "core":
            return np.where((b == p.known_b) & (eps == 1.0), log_density, -np.inf)
        in_b = (b >= p.b_min) & (b <= p.b_max)
        log_eps_density = _lognormal_logpdf(eps, p.eps0, p.sigma_eps)
        return np.where(in_b, log_density - np.log(p.b_max - p.b_min) + log_eps_density, -np.inf)


def reality_prior(
    config: StudyConfig = FULL_CONFIG, *, zero_mass: float = 0.3, mu_min: float = 0.1
) -> Prior:
    """Build the reality prior used by the simulator.

    Parameters
    ----------
    config : StudyConfig, default FULL_CONFIG
        Supplies mu_max, the background range, and the efficiency prior.
    zero_mass : float, default 0.3
        Probability of "no source" (mu = 0).
    mu_min : float, default 0.1
        Lower cutoff of the log-uniform component.

    Returns
    -------
    Prior
        The reality-family prior.

    Notes
    -----
    In the full case::

        mu        ~ zero_mass * delta_0 + (1 - zero_mass) * LogUniform[mu_min, mu_max]
        b         ~ Uniform[b_min, b_max]
        log(eps)  ~ Normal(log(eps0), sigma_eps**2)

    In the core case, b and eps are point masses at ``known_b`` and 1. The
    keyword arguments allow the prior-sensitivity runs to vary the weight and
    the cutoff.
    """
    return Prior(config=config, zero_mass=zero_mass, mu_min=mu_min)


def analysis_priors(config: StudyConfig = FULL_CONFIG) -> dict[str, Prior]:
    """Return the four analysis priors used in the prior-sensitivity study.

    Parameters
    ----------
    config : StudyConfig, default FULL_CONFIG
        Shared problem configuration.

    Returns
    -------
    dict of str to Prior
        Keys ``"reality"``, ``"flat"`` (uniform on [0, mu_max]),
        ``"gamma_diffuse"`` (Gamma with shape 1 and rate 0.1) and
        ``"gamma_shifted"`` (shape 5, rate 0.5, mean 10 before truncation).
        The keys are stable experiment identifiers.

    Notes
    -----
    All priors share the configured nuisance distributions and the bounded
    signal support [0, mu_max]. Gamma priors are truncated to this interval
    and then normalized.
    """
    return {
        "reality": reality_prior(config),
        "flat": Prior(config=config, family="flat"),
        "gamma_diffuse": Prior(config=config, family="gamma", shape=1.0, rate=0.1),
        "gamma_shifted": Prior(config=config, family="gamma", shape=5.0, rate=0.5),
    }


def simulate_counts(
    mu: ArrayLike,
    b: ArrayLike,
    eps: ArrayLike = 1.0,
    *,
    rng: np.random.Generator,
    config: StudyConfig = FULL_CONFIG,
    size: int | None = None,
) -> NDArray[np.int64]:
    """Draw independent Pois(eps*mu+b) and Pois(tau*b) counts.

    Parameters
    ----------
    mu, b, eps : array_like
        Broadcastable finite parameters with mu,b >= 0 and eps > 0. Simulation
        permits mu beyond the study bound for separate FC validation.
    rng : numpy.random.Generator
        Source of randomness.
    config : StudyConfig
        Supplies tau. This count-only helper does not fix core nuisances.
    size : int, optional
        Broadcast scalar parameters to this batch size. For arrays, supply
        repeated/broadcasted parameters directly and leave size as None.

    Returns
    -------
    numpy.ndarray
        Int64 array of shape broadcast(mu,b,eps).shape + (2,).
    """
    mu_arr, b_arr, eps_arr = _parameters(mu, b, eps, size)
    n_on = rng.poisson(eps_arr * mu_arr + b_arr)
    n_off = rng.poisson(config.problem.tau * b_arr)
    return np.stack((n_on, n_off), axis=-1).astype(np.int64, copy=False)


def simulate(
    mu: ArrayLike,
    b: ArrayLike,
    eps: ArrayLike = 1.0,
    *,
    rng: np.random.Generator,
    config: StudyConfig = FULL_CONFIG,
    size: int | None = None,
) -> NDArray[np.float64]:
    """Draw full observations (n_on, n_off, eps_hat).

    Parameters
    ----------
    mu, b, eps : array_like
        Broadcastable finite parameters with mu, b >= 0 and eps > 0.
    rng : numpy.random.Generator
        Source of randomness.
    config : StudyConfig, default FULL_CONFIG
        Supplies tau and sigma_eps. In the core case, b must equal
        ``known_b`` and eps must equal 1.
    size : int, optional
        Broadcast scalar parameters to this batch size.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``broadcast(mu, b, eps).shape + (3,)``. Scalar
        inputs without ``size`` return one row of shape (3,), not a one-row
        batch.

    Raises
    ------
    ValueError
        If the core case receives another background or efficiency, or if a
        parameter is invalid.

    Notes
    -----
    Given theta, the three draws are independent, with
    ``log(eps_hat) ~ Normal(log(eps), sigma_eps**2)``. When sigma_eps is 0
    (core case), eps_hat equals eps exactly.
    """
    mu_arr, b_arr, eps_arr = _parameters(mu, b, eps, size)
    p = config.problem
    if p.mode == "core" and (np.any(b_arr != p.known_b) or np.any(eps_arr != 1.0)):
        raise ValueError("core simulation requires known_b and eps=1")
    counts = simulate_counts(mu_arr, b_arr, eps_arr, rng=rng, config=config)
    eps_hat = eps_arr if p.sigma_eps == 0 else rng.lognormal(np.log(eps_arr), p.sigma_eps)
    return np.concatenate((counts, np.asarray(eps_hat)[..., None]), axis=-1).astype(np.float64)


def sample_joint(
    n: int,
    *,
    rng: np.random.Generator,
    config: StudyConfig = FULL_CONFIG,
    prior: Prior | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw parameter and observation batches from the joint model.

    Parameters
    ----------
    n : int
        Number of draws.
    rng : numpy.random.Generator
        Source of randomness.
    config : StudyConfig, default FULL_CONFIG
        Problem configuration used by both the prior and the simulator.
    prior : Prior, optional
        Prior for theta. The default is the reality prior.

    Returns
    -------
    theta, x : tuple of numpy.ndarray
        Two float64 arrays of shape (n, 3): parameters (mu, b, eps) and
        observations (n_on, n_off, eps_hat).

    Raises
    ------
    ValueError
        If the prior uses a different config. This prevents a silent model
        mismatch.

    Notes
    -----
    This is the joint draw ``theta ~ pi``, ``x | theta ~ p(x | theta)`` used
    for neural training, Bayes risk, and simulation-based calibration.
    """
    selected = reality_prior(config) if prior is None else prior
    if selected.config != config:
        raise ValueError("prior and simulator must use the same config")
    theta = selected.sample(n, rng=rng)
    x = simulate(theta[:, 0], theta[:, 1], theta[:, 2], rng=rng, config=config)
    return theta, x


def poisson_pmf(n: ArrayLike, rate: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the exact Poisson probability mass, with NumPy broadcasting.

    Parameters
    ----------
    n : array_like
        Counts. Negative or noninteger counts have probability 0.
    rate : array_like
        Finite, nonnegative Poisson means.

    Returns
    -------
    numpy.ndarray
        ``Pois(n | rate) = rate**n * exp(-rate) / n!``, with
        ``Pois(0 | 0) = 1``.

    Raises
    ------
    ValueError
        If a rate is negative or not finite.
    """
    return np.exp(_poisson_logpmf(n, rate))


def log_likelihood_core(
    n_on: ArrayLike,
    mu: ArrayLike,
    *,
    b: ArrayLike = 3.0,
) -> NDArray[np.float64]:
    """Return the core-case log likelihood ``log Pois(n_on | mu + b)``.

    Parameters
    ----------
    n_on : array_like
        On-region counts.
    mu : array_like
        Finite, nonnegative signal values.
    b : array_like, default 3.0
        Known, finite, nonnegative background.

    Returns
    -------
    numpy.ndarray
        Broadcast log likelihood. Pass ``n_on[:, None]`` and ``mu[None, :]``
        to obtain an (N, G) table.

    Notes
    -----
    The core case has known b and eps = 1. The off count and the calibration
    measurement then carry no information about mu, so they are omitted.
    """
    mu_arr, b_arr, _ = _parameters(mu, b, 1.0, None)
    return _poisson_logpmf(n_on, mu_arr + b_arr)


def log_likelihood_full(
    x: ArrayLike,
    mu: ArrayLike,
    b: ArrayLike,
    eps: ArrayLike,
    *,
    config: StudyConfig = FULL_CONFIG,
) -> NDArray[np.float64]:
    """Return the full joint log likelihood, including auxiliary calibration.

    Parameters
    ----------
    x : array_like, shape (..., 3)
        Observations (n_on, n_off, eps_hat). Invalid counts or eps_hat have
        log likelihood -inf. Insert singleton axes to compare batches/grids.
    mu, b, eps : array_like
        Broadcastable parameter arrays; must be finite and physically valid.
    config : StudyConfig
        Supplies tau and calibration log-scale noise. At sigma_eps=0 the
        auxiliary observation is a point measure, not a Gaussian density.

    Returns
    -------
    numpy.ndarray
        Broadcast joint log likelihood including factorials and the eps_hat
        Jacobian. No prior term is included.

    Notes
    -----
    The log likelihood is::

        log Pois(n_on | eps*mu + b) + log Pois(n_off | tau*b)
            + log LogNormal(eps_hat | median=eps, sigma=sigma_eps)

    Path A uses the last term as part of the posterior for eps. Path B
    treats eps as a nuisance parameter and profiles it.
    """
    obs = _three_columns(x, "x")
    mu_arr, b_arr, eps_arr = _parameters(mu, b, eps, None)
    p = config.problem
    return (
        _poisson_logpmf(obs[..., 0], eps_arr * mu_arr + b_arr)
        + _poisson_logpmf(obs[..., 1], p.tau * b_arr)
        + _lognormal_logpdf(obs[..., 2], eps_arr, p.sigma_eps)
    )


def theta_grid(
    config: StudyConfig = FULL_CONFIG,
    *,
    validation: bool = False,
    coverage: bool = False,
) -> NDArray[np.float64]:
    """Return a one-dimensional signal grid for mu (not a nuisance mesh).

    Parameters
    ----------
    config : StudyConfig, default FULL_CONFIG
        Supplies the grid steps and bounds.
    validation : bool, default False
        Return the Feldman-Cousins validation grid [0, 50] at step 0.005.
    coverage : bool, default False
        Return the coverage-check grid 0, 0.01, 0.02, 0.05, 0.1, ..., 10,
        which has extra points near the boundary mu = 0.

    Returns
    -------
    numpy.ndarray
        Sorted float64 grid that includes both endpoints. The default is
        [0, mu_max] at step ``mu_step``.

    Raises
    ------
    ValueError
        If both ``validation`` and ``coverage`` are True.
    """
    if validation and coverage:
        raise ValueError("choose either validation or coverage")
    if coverage:
        values = np.asarray(config.grid.mu_cov_grid)
        return np.unique(np.append(values[values <= config.problem.mu_max], config.problem.mu_max))
    high = config.grid.mu_grid_max_fc if validation else config.problem.mu_max
    step = config.grid.mu_step_fc if validation else config.grid.mu_step
    values = np.arange(int(np.floor(high / step)) + 1, dtype=np.float64) * step
    if np.isclose(values[-1], high, rtol=0, atol=1e-12):
        values[-1] = high
    else:
        values = np.append(values, high)
    return values


def poisson_support(rate_max: float, *, tail_prob: float = 1e-12) -> NDArray[np.int64]:
    """Return the counts 0, 1, ..., n_max needed for a finite Poisson sum.

    Parameters
    ----------
    rate_max : float
        Largest Poisson mean in the sum (an explicit numerical envelope).
    tail_prob : float, default 1e-12
        Allowed omitted tail mass.

    Returns
    -------
    numpy.ndarray
        Int64 counts 0..n_max, where n_max is the smallest count with
        ``P(N > n_max) <= tail_prob`` for ``N ~ Poisson(rate_max)``.

    Notes
    -----
    The log-normal efficiency has no finite upper bound, so callers must
    state the rate envelope they use. This helper does not truncate
    simulator draws.
    """
    if not np.isfinite(rate_max) or rate_max < 0 or not 0 < tail_prob < 1:
        raise ValueError("invalid Poisson envelope or tail probability")
    high = int(poisson.ppf(1.0 - tail_prob, rate_max))
    while poisson.sf(high, rate_max) > tail_prob:
        high += 1
    return np.arange(high + 1, dtype=np.int64)


def _three_columns(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert to float64 and check that the last axis has length 3."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    return array


def _parameters(
    mu: ArrayLike,
    b: ArrayLike,
    eps: ArrayLike,
    size: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Broadcast and validate (mu, b, eps); optionally expand scalars to size."""
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (mu, b, eps)))
    if size is not None:
        if size < 0 or arrays[0].ndim != 0:
            raise ValueError("size requires scalar parameters and a nonnegative batch size")
        arrays = [np.broadcast_to(v, (size,)) for v in arrays]
    if any(np.any(~np.isfinite(v)) for v in arrays):
        raise ValueError("parameters must be finite")
    if np.any(arrays[0] < 0) or np.any(arrays[1] < 0) or np.any(arrays[2] <= 0):
        raise ValueError("mu,b must be nonnegative and eps must be positive")
    return arrays[0], arrays[1], arrays[2]


def _poisson_logpmf(n: ArrayLike, rate: ArrayLike) -> NDArray[np.float64]:
    """Return log Pois(n | rate); invalid counts give -inf."""
    counts, rates = np.broadcast_arrays(
        np.asarray(n, dtype=np.float64), np.asarray(rate, dtype=np.float64)
    )
    if np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Poisson rates must be finite and nonnegative")
    valid = np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))
    safe = np.where(valid, counts, 0.0)
    return np.where(valid, xlogy(safe, rates) - rates - gammaln(safe + 1), -np.inf)


def _lognormal_logpdf(
    value: ArrayLike,
    median: ArrayLike,
    sigma: float,
) -> NDArray[np.float64]:
    """Return the log-normal log density; sigma = 0 means a point mass at the median."""
    values, medians = np.broadcast_arrays(
        np.asarray(value, dtype=np.float64), np.asarray(median, dtype=np.float64)
    )
    valid = np.isfinite(values) & (values > 0)
    if sigma == 0:
        return np.where(valid & (values == medians), 0.0, -np.inf)
    safe = np.where(valid, values, 1.0)
    residual = (np.log(safe) - np.log(medians)) / sigma
    density = -0.5 * residual**2 - np.log(safe * sigma * np.sqrt(2 * np.pi))
    return np.where(valid, density, -np.inf)
