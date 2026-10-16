"""Small immutable result records and structural interfaces shared by both paths.

The protocols are structural (``typing.Protocol``): any object with the right
method works, and no inheritance is needed. Observations ``x`` are always
float64 arrays of shape (N, 3) with columns (n_on, n_off, eps_hat). The
records only hold results; they contain no inference logic.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PosteriorMoments:
    """Batch posterior mean and variance of the signal mu.

    Attributes
    ----------
    mean : numpy.ndarray
        Float64 array (N,) of E[mu | x]. Each value lies in [0, mu_max].
    variance : numpy.ndarray
        Float64 array (N,) of Var[mu | x]. Finite and nonnegative.

    Notes
    -----
    WALDO divides by this variance after applying a visible variance floor.
    Predictors do not hide that choice inside the variance.
    """

    mean: NDArray[np.float64]
    variance: NDArray[np.float64]


class PointEstimator(Protocol):
    """Callable taking float64 observations (N,3) and returning mu estimates (N,)."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict a finite signal value for every row of x.

        Parameters
        ----------
        x : numpy.ndarray
            Float64 observations of shape (N, 3).

        Returns
        -------
        numpy.ndarray
            Float64 point estimates delta(x) of shape (N,).
        """
        ...


class PosteriorPredictor(Protocol):
    """Exact Path A -> WALDO interface; input is always raw observations (N,3).

    A predictor owns feature transforms, weights, prior, and config. Path B
    never imports torch, reads weights, or repeats feature preprocessing.
    """

    def moments(self, x: NDArray[np.float64]) -> PosteriorMoments:
        """Return the posterior mean and variance of mu for each observation.

        Parameters
        ----------
        x : numpy.ndarray
            Raw float64 observations of shape (N, 3).

        Returns
        -------
        PosteriorMoments
            Mean and variance arrays of shape (N,).
        """
        ...


class PosteriorSampler(Protocol):
    """Conditional posterior sampler used by simulation-based calibration."""

    def sample(
        self,
        x: NDArray[np.float64],
        n_draws: int,
        *,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw mu from the conditional posterior of each observation.

        Parameters
        ----------
        x : numpy.ndarray
            Float64 observations of shape (N, 3).
        n_draws : int
            Number of posterior draws per observation.
        rng : numpy.random.Generator
            Source of randomness.

        Returns
        -------
        numpy.ndarray
            Float64 draws of shape (N, n_draws). Draws equal exactly 0 when
            the posterior has an atom at zero.
        """
        ...


@dataclass(frozen=True)
class IntervalBatch:
    """Inclusive interval hulls for a batch of observations.

    Attributes
    ----------
    lower, upper : numpy.ndarray
        Float64 endpoints of shape (N,). Empty sets have NaN in both.
    accepted : numpy.ndarray, optional
        Bool array (N, G). For sets built by inverting tests on a grid, it
        records which grid points were accepted. It keeps disconnected sets.
    mu_grid : numpy.ndarray, optional
        Float64 grid (G,) that ``accepted`` refers to.

    Notes
    -----
    The hull is ``[min accepted mu, max accepted mu]``. Coverage of a
    reported hull uses the endpoints; coverage of a raw set uses
    ``accepted``. A hull can cover when the raw set does not, so reports
    must say which one they use.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    accepted: NDArray[np.bool_] | None = None
    mu_grid: NDArray[np.float64] | None = None


class IntervalProcedure(Protocol):
    """Callable from float64 observations (N,3) to N inclusive interval hulls."""

    def __call__(self, x: NDArray[np.float64]) -> IntervalBatch:
        """Return interval hulls and, when available, the raw grid membership.

        Parameters
        ----------
        x : numpy.ndarray
            Float64 observations of shape (N, 3).

        Returns
        -------
        IntervalBatch
            Endpoints of shape (N,) and optional accepted-grid masks.
        """
        ...


@dataclass(frozen=True)
class RiskResult:
    """Mean loss, its Monte Carlo standard error, sample count, and optional cost.

    Attributes
    ----------
    value : float
        Monte Carlo mean loss.
    mc_se : float
        ``sd(loss, ddof=1) / sqrt(n_samples)``. This is Monte Carlo error,
        never posterior uncertainty.
    n_samples : int
        Number of simulated (theta, x) pairs.
    loss : str
        ``"squared"`` for (delta - mu)**2 or ``"absolute"`` for abs(delta - mu).
    inference_us : float, optional
        Wall time per inference in microseconds, when measured.
    """

    value: float
    mc_se: float
    n_samples: int
    loss: str
    inference_us: float | None = None


@dataclass(frozen=True)
class CoverageResult:
    """Coverage, length, and empty-rate diagnostics at K parameter points.

    Attributes
    ----------
    theta : numpy.ndarray
        Float64 parameter rows (K, 3) with columns (mu, b, eps).
    value, mc_se : numpy.ndarray
        Coverage ``C(theta) = P_theta(mu in CI(X))`` and its binomial MC
        error ``sqrt(C * (1 - C) / n_samples)``, each of shape (K,).
    expected_length, length_mc_se : numpy.ndarray
        Expected hull length (empty sets count as length 0) and its MC error.
    empty_rate, empty_mc_se : numpy.ndarray
        Probability of an empty set and its MC error.
    n_samples : int
        Toys per point; 0 for exact sums.
    exact : bool
        True when coverage was computed by summing over the count support.

    Notes
    -----
    Exact sums have zero Monte Carlo error. Tail truncation and grid errors
    are separate and must be reported separately. Coverage uses inclusive
    hull endpoints.
    """

    theta: NDArray[np.float64]
    value: NDArray[np.float64]
    mc_se: NDArray[np.float64]
    expected_length: NDArray[np.float64]
    length_mc_se: NDArray[np.float64]
    empty_rate: NDArray[np.float64]
    empty_mc_se: NDArray[np.float64]
    n_samples: int
    exact: bool
