"""Immutable study constants: the physical problem, grids, and Monte Carlo budgets.

All values are frozen dataclasses. ``FULL_CONFIG`` is the realistic case
(unknown background, uncertain efficiency). ``CORE_CONFIG`` is the exact
check case (known background b = 3, efficiency 1), which matches the
Feldman-Cousins (1998) tables. The rationale for each value is given in
``docs/DESIGN.md`` and ``docs/literature_review.md``.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ProblemConfig:
    """Physical model constants; efficiency is a positive relative scale.

    Parameters
    ----------
    mode : {"core", "full"}
        Core fixes b to ``known_b`` and eps to 1, without calibration noise.
    mu_max : float, default 10.0
        Physical upper bound of the signal; Theta = [0, mu_max].
    tau : float, default 5.0
        Known off/on exposure ratio: n_off ~ Poisson(tau * b).
    eps0 : float, default 1.0
        Median of the relative efficiency.
    sigma_eps : float, default 0.10
        Log-scale s.d. of the efficiency and of its calibration measurement.
    b_min, b_max : float, default 0.5 and 10.0
        Bounds of the full-case uniform background prior.
    known_b : float, default 3.0
        Background used in the core model.
    alpha : float, default 0.10
        Tail probability; nominal interval coverage is ``1 - alpha``.

    Raises
    ------
    ValueError
        If a value is outside its valid range, or if the core mode does not
        have ``eps0 = 1`` and ``sigma_eps = 0``.
    """

    mode: Literal["core", "full"] = "full"
    mu_max: float = 10.0
    tau: float = 5.0
    eps0: float = 1.0
    sigma_eps: float = 0.10
    b_min: float = 0.5
    b_max: float = 10.0
    known_b: float = 3.0
    alpha: float = 0.10

    def __post_init__(self) -> None:
        """Validate the constants; see the class docstring for the rules."""
        if self.mode not in ("core", "full"):
            raise ValueError("mode must be core or full")
        if self.mu_max <= 0 or self.tau <= 0 or self.eps0 <= 0:
            raise ValueError("mu_max, tau, and eps0 must be positive")
        if self.sigma_eps < 0 or not 0 <= self.b_min < self.b_max or self.known_b < 0:
            raise ValueError("invalid noise or background bounds")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if self.mode == "core" and (self.eps0 != 1.0 or self.sigma_eps != 0.0):
            raise ValueError("core requires eps0=1 and sigma_eps=0")


@dataclass(frozen=True)
class GridConfig:
    """Grid resolutions and finite numerical envelopes, not physical eps bounds.

    Parameters
    ----------
    mu_step : float, default 0.01
        Step of the study signal grid on [0, mu_max].
    mu_grid_max_fc, mu_step_fc : float, default 50.0 and 0.005
        Range and step of the Feldman-Cousins validation grid, as in FC (1998).
    mu_step_calib : float, default 0.1
        Step of the signal grid used to calibrate critical values.
    mu_cov_grid : tuple of float
        Coverage-check grid: 0, 0.01, 0.02, 0.05, then 0.1 to 10 in steps of 0.1.
    b_cov_grid : tuple of float
        Background values for coverage checks and nuisance calibration.
    b_points, eps_points : int, default 81 and 31
        Quadrature sizes for background and log-efficiency integration.
    eps_log_span : float, default 5.0
        Half-width, in units of sigma_eps, of the log-efficiency quadrature range.
    n_fc_max : int, default 150
        Largest count used in the Feldman-Cousins table check.
    poisson_tail : float, default 1e-12
        Allowed omitted Poisson tail mass in finite sums.
    fc_table_tol : float, default 0.01
        Tolerance against the published FC values, which are rounded to 0.01.

    Notes
    -----
    Quadrature sizes and the efficiency envelope are numerical choices whose
    convergence is checked in the experiments. They are not physical bounds:
    eps itself remains unbounded in the simulator and in the priors.
    """

    mu_step: float = 0.01
    mu_grid_max_fc: float = 50.0
    mu_step_fc: float = 0.005
    mu_step_calib: float = 0.1
    mu_cov_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05) + tuple(
        round(i / 10, 1) for i in range(1, 101)
    )
    b_cov_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)
    b_points: int = 81
    eps_points: int = 31
    eps_log_span: float = 5.0
    n_fc_max: int = 150
    poisson_tail: float = 1e-12
    fc_table_tol: float = 0.01


@dataclass(frozen=True)
class MonteCarloConfig:
    """Simulation budgets and tolerances; use independent child RNG streams.

    Parameters
    ----------
    n_toys_cov : int, default 4000
        Toys per parameter point in Monte Carlo coverage checks. The binomial
        MC s.e. at coverage 0.90 is sqrt(0.9 * 0.1 / 4000), about 0.0047.
    n_toys_calib : int, default 5000
        Toys per (mu, nuisance) node for critical-value calibration.
    m_bayes_risk : int, default 20000
        Prior draws for each Monte Carlo Bayes risk.
    n_train : int, default 50000
        Simulated pairs used to train the amortized neural posterior.
    n_sbc : int, default 1000
        Prior draws for simulation-based calibration (SBC).
    n_posterior_draws : int, default 200
        Posterior draws per SBC case.
    flagged_multiplier : int, default 10
        Flagged coverage points are rerun with this many times more toys.
    tol_cov_mc : float, default 0.015
        Allowed Monte Carlo undercoverage, about 3 MC s.e. at 4000 toys.
    tol_cov_exact : float, default 0.005
        Allowed undercoverage for exact (summed) coverage; only grid effects.
    """

    n_toys_cov: int = 4000
    n_toys_calib: int = 5000
    m_bayes_risk: int = 20000
    n_train: int = 50000
    n_sbc: int = 1000
    n_posterior_draws: int = 200
    flagged_multiplier: int = 10
    tol_cov_mc: float = 0.015
    tol_cov_exact: float = 0.005


@dataclass(frozen=True)
class StudyConfig:
    """One problem, its numerical grids, simulation budgets, and default seed.

    Parameters
    ----------
    problem : ProblemConfig
        Physical model constants.
    grid : GridConfig
        Grid resolutions and numerical envelopes.
    mc : MonteCarloConfig
        Simulation budgets and coverage tolerances.
    seed : int, default 20261012
        Root seed. Scripts spawn independent child streams from it with
        ``numpy.random.SeedSequence``.

    Notes
    -----
    Use ``dataclasses.replace`` to create a sensitivity configuration. The seed
    is metadata: simulator functions always receive an explicit Generator.
    """

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    mc: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    seed: int = 20261012


FULL_CONFIG = StudyConfig()
CORE_CONFIG = StudyConfig(problem=ProblemConfig(mode="core", sigma_eps=0.0))
