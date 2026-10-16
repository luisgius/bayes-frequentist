# Shared API contract

Version: 1.1, 2026-10-12. This file is the API contract shared by the Path A
(Bayesian) and Path B (frequentist) code. The problem is the on/off Poisson
experiment described below. The sources for the constants and methods are in
`docs/literature_review.md` and `docs/references.bib`.

### Design decisions in brief

- **Constants.** `alpha = 0.10` matches the 90% Feldman–Cousins (FC) tables
  [@feldman1998unified]. The physical signal bound is `mu_max = 10`, so the
  upper bound sometimes binds. `tau = 5` follows the worked examples of Rolke,
  López and Conrad [@rolke2005nuisance]. `eps0 = 1` is a *relative* efficiency
  scale with log-normal spread (log-scale standard deviation `sigma_eps = 0.10`),
  so the known-background core case is nested in the full case. The core case uses `b = 3`, the FC worked
  example. FC validation uses its own grid, `mu` in `[0, 50]` at step 0.005.
- **One data-generating process for both paths.** Observations are
  `(n_on, n_off, eps_hat)`. Path A conditions on all three. Path B profiles the
  background `b` and the efficiency `eps`.
- **Reality prior.** A 0.3 point mass at `mu = 0` plus a log-uniform component on
  `[0.1, 10]`. The prior-sensitivity study varies the weight (0.1, 0.5) and the
  cutoff (0.03, 0.3), and adds flat and Gamma analysis priors.
- **Nuisance box.** Path B checks coverage only at grid points inside
  `b in [0.5, 10]` × `eps in eps0*exp([-2, +2]*sigma_eps)` (about ±20% in
  efficiency). It makes no coverage claim outside this box or between grid
  points. Critical values take the maximum over 7 background values ×
  5 efficiency values. A calibration at `eps0` only (efficiency fixed at 1) is
  kept as an ablation.
- **Coverage criterion.** The target is 90% coverage. It is checked point by
  point on finite grids. The Monte Carlo tolerance is 0.015 at 4,000 toys per
  point (about 3 MC s.e.), so a point passes when its estimated coverage is at
  least 0.885. Flagged points are rerun with 10× more toys before they count as
  failures. Exact (summed) coverage in the core case uses tolerance 0.005. A pass
  is empirical evidence on the checked grid, not a proof of coverage.
- **FC monotone fix.** FC force the upper endpoint to be non-increasing in `b`
  to produce their published tables. The library option `enforce_b_monotone`
  (default `False`) applies this correction when set to `True`, and only for
  `n_on = 0`, which is the row where the plain construction differs from the
  tables.

Path A and Path B implement the signatures below; final inference and experiment
code is available in the package and runners.

## Model and data conventions

The signal is `mu` in `[0, mu_max]`. Background `b` is nonnegative, and relative
efficiency `eps` is positive. The full simulator draws three independent observations:

```text
n_on          ~ Poisson(eps * mu + b)
n_off         ~ Poisson(tau * b)
log(eps_hat)  ~ Normal(log(eps), sigma_eps**2)
```

The reality prior is `0.3 * delta_0 + 0.7 * LogUniform[0.1, 10]` for `mu`,
`Uniform[0.5, 10]` for `b`, and `log(eps) ~ Normal(log(1), 0.1**2)`; the three
components are independent. Prior draws and calibration observations are
separate draws. Path A conditions on all three
observations. Path B profiles the same likelihood, including the auxiliary term;
it resamples `eps_hat` in calibration and coverage toys.

The core case fixes `b=3`, `eps=1`, and `sigma_eps=0`. It retains three observation
columns for a uniform interface; `n_off` is ancillary and `eps_hat=1`. To study a
different known background, replace `CORE_CONFIG.problem.known_b`. The core
likelihood omits the ancillary terms. The full likelihood at zero calibration
noise uses a point measure at `eps_hat=eps`.

| Data | Shape | dtype | Columns / meaning |
| --- | --- | --- | --- |
| `theta` | `(N, 3)` | `float64` | `(mu, b, eps)` |
| `x` | `(N, 3)` | `float64` | `(n_on, n_off, eps_hat)`; first two columns are integer-valued |
| Count-only draws | `(..., 2)` | `int64` | `(n_on, n_off)` |
| Signal grid | `(G,)` | `float64` | sorted, includes both endpoints |
| Estimates / moments | `(N,)` | `float64` | one signal value per observation |
| Posterior masses | `(N, G)` | `float64` | nonnegative, each row sums to 1 |
| Posterior draws | `(N, D)` | `float64` | `D` conditional draws per observation |
| Interval endpoints | `(N,)` | `float64` | inclusive lower/upper; both NaN for empty sets |
| Inverted raw sets | `(N, G)` | `bool` | accepted signal grid points, including disconnected sets |
| Coverage nuisance grid | `(H, 2)` | `float64` | `(b, eps)` |
| FC acceptance table | `(G, J)` | `bool` | signal-grid rows, count-support columns |

Batch inference always takes `(N,3)`, even for one observation. Simulator and
likelihood helpers also support arbitrary leading broadcast axes. Scalar simulation
without `size` returns `(3,)`; use `size=1` for `(1,3)`. For likelihood tables use
`x[:, None, :]` with `mu[None, :]`; for the core use `n_on[:, None]`.

`Prior.mu_density` and `Prior.log_prob` use the signal measure
`delta_0 + Lebesgue`, not a pure Lebesgue density. At zero, the value is the atom
probability (0.3 for reality; 0 for the continuous analysis priors). At positive
`mu`, it is a continuous density. Quadrature assigns the atom weight 1 exactly
once and assigns integration weights only to the continuous component. For a
continuous prior, include the continuous endpoint limit in a quadrature rule if
needed; `mu_density(0)` deliberately represents the atom, not that limit.
Core nuisances use point measures. Gamma analysis priors are truncated and
normalized on `[0, mu_max]`.

All stochastic calls use an explicit `numpy.random.Generator`, except training,
which accepts a seed and initializes its own NumPy and torch seeds. Use
`numpy.random.SeedSequence(config.seed).spawn(...)` for separate training,
calibration, risk, and coverage streams. Never reuse calibration toys as checks.

## Module map

| Module / files | Responsibility |
| --- | --- |
| `src/bayesfreq/config.py` | Immutable defaults |
| `src/bayesfreq/data_simulator.py` | Priors, one DGP, exact likelihoods |
| `src/bayesfreq/interfaces.py` | Shared protocols and result records |
| `src/bayesfreq/plotting.py` | Figure style and saving |
| `src/bayesfreq/__init__.py` | Config exports and version |
| `src/bayesfreq/bayesian_risk.py` | Path A inference and risk |
| `scripts/run_path_a.py`, `tests/test_bayesian_risk.py` | Path A experiment and tests |
| `results/path_a*`, `figures/path_a*` | Path A outputs, including model artifact |
| `src/bayesfreq/frequentist_bounds.py` | Path B belts and coverage |
| `scripts/run_path_b.py`, `tests/test_frequentist_bounds.py` | Path B experiment and tests |
| `scripts/make_summary_figure.py`, `figures/summary.png` | Two-panel summary figure, drawn from saved results only |
| `results/path_b*`, `figures/path_b*` | Path B outputs |
| `tests/test_simulator.py`, `tests/test_inference.py`, `tests/conftest.py` | Cross-path verification |
| `notebooks/01_holy_war_reconciliation.ipynb` | Executed explanation |
| `tests/test_smoke.py`, packaging, tooling, `docs/DESIGN.md` | Scaffold |
| `docs/literature_review.md`, `docs/references.bib` | Checked literature |
| `README.md`, final prose | Presentation and claims backed by results |

The notebook imports only from `bayesfreq` for model, inference, and figure logic,
and reads `results/*.json` for experiment results. It must not copy library
algorithms or silently retrain models. Standard-library file/JSON access and
display utilities may be used.

## Config and foundation API

The type abbreviation `FloatArray` below means `NDArray[np.float64]` and
`IntArray` means `NDArray[np.int64]`; they are documentation shorthand.
Array-like simulator inputs include Python scalars/lists and NumPy arrays.

```python
# config.py (all four dataclasses are frozen)
ProblemConfig(mode="full", mu_max=10.0, tau=5.0, eps0=1.0, sigma_eps=0.10,
              b_min=0.5, b_max=10.0, known_b=3.0, alpha=0.10)
GridConfig(mu_step=0.01, mu_grid_max_fc=50.0, mu_step_fc=0.005,
           mu_step_calib=0.1, mu_cov_grid=(...), b_cov_grid=(...),
           b_points=81, eps_points=31, eps_log_span=5.0, n_fc_max=150,
           poisson_tail=1e-12, fc_table_tol=0.01)
MonteCarloConfig(n_toys_cov=4000, n_toys_calib=5000, m_bayes_risk=20000,
                 n_train=50000, n_sbc=1000, n_posterior_draws=200,
                 flagged_multiplier=10, tol_cov_mc=0.015, tol_cov_exact=0.005)
StudyConfig(problem=ProblemConfig(), grid=GridConfig(), mc=MonteCarloConfig(),
            seed=20261012)
FULL_CONFIG: StudyConfig
CORE_CONFIG: StudyConfig  # ProblemConfig(mode="core", sigma_eps=0.0)

# data_simulator.py
Prior(config=FULL_CONFIG, family="reality", zero_mass=0.3, mu_min=0.1,
      shape=1.0, rate=0.1)
Prior.sample(n: int, *, rng: Generator) -> FloatArray
Prior.mu_density(mu: ArrayLike) -> FloatArray
Prior.log_prob(theta: ArrayLike) -> FloatArray
reality_prior(config=FULL_CONFIG, *, zero_mass=0.3, mu_min=0.1) -> Prior
analysis_priors(config=FULL_CONFIG) -> dict[str, Prior]
simulate_counts(mu, b, eps=1.0, *, rng, config=FULL_CONFIG, size=None) -> IntArray
simulate(mu, b, eps=1.0, *, rng, config=FULL_CONFIG, size=None) -> FloatArray
sample_joint(n: int, *, rng, config=FULL_CONFIG, prior=None) -> tuple[FloatArray, FloatArray]
poisson_pmf(n: ArrayLike, rate: ArrayLike) -> FloatArray
log_likelihood_core(n_on: ArrayLike, mu: ArrayLike, *, b=3.0) -> FloatArray
log_likelihood_full(x, mu, b, eps, *, config=FULL_CONFIG) -> FloatArray
theta_grid(config=FULL_CONFIG, *, validation=False, coverage=False) -> FloatArray
poisson_support(rate_max: float, *, tail_prob=1e-12) -> IntArray

# plotting.py
FIGURES_DIR: Path
METHOD_COLORS: dict   # results-file method key -> colour, used by every figure
METHOD_LABELS: dict   # results-file method key -> display name
set_style() -> None
save_figure(fig: Figure, name: str) -> Path
```

`analysis_priors` has stable keys `reality`, `flat`, `gamma_diffuse`, and
`gamma_shifted`. The Gamma shape/rate pairs are `(1,0.1)` and `(5,0.5)`.
Use `reality_prior(zero_mass=0.1 or 0.5, mu_min=...)` for further sensitivity.
The likelihood helpers allow nonnegative signal outside the study bound for
FC validation; priors constrain signal to the configured support.

`theta_grid(coverage=True)` gives 0, 0.01, 0.02, 0.05, then 0.1 through 10 at
step 0.1. The background check grid is 0.5, 1, 2, 3, 5, 7.5, 10. Check
efficiency across `eps0*exp([-2,-1,0,1,2]*sigma_eps)` in the final Path B study.
Training/SBC sizes, quadrature sizes, and seed are explicit implementation
defaults, not literature values. The inference code checks quadrature convergence.

Efficiency is log-normal with unbounded positive support. The final Path B
coverage study explicitly restricts its coverage checks to the nuisance box
`b in [0.5,10]`, `eps in eps0*exp([-2,+2]*sigma_eps)`. Calibration uses the
seven background values crossed with five log-efficiency offsets `[-2,-1,0,1,2]`.
This study box does not truncate the simulator or the Bayesian efficiency prior.
A calibration at `eps0` only is kept as a documented ablation.
`eps_log_span=5`
is a numerical integration/support envelope, never a physical bound.
For finite count sums call `poisson_support` with an explicit rate envelope,
e.g. `eps0*exp(5*sigma_eps)*mu_max+b_max`, and report the omitted tail.
The default helper tail target is `1e-12`.
FC validation uses `theta_grid(validation=True)` and counts 0 through 150.
Plots use the seaborn colorblind palette, ticks style, and 150 dpi. Each
method keeps one colour and one display name in every figure (`METHOD_COLORS`
and `METHOD_LABELS`).

## Shared records and protocols

```python
# interfaces.py
PosteriorMoments(mean: FloatArray, variance: FloatArray)
PointEstimator.__call__(x: FloatArray) -> FloatArray
PosteriorPredictor.moments(x: FloatArray) -> PosteriorMoments
PosteriorSampler.sample(x: FloatArray, n_draws: int, *, rng: Generator) -> FloatArray
IntervalBatch(lower: FloatArray, upper: FloatArray,
              accepted: NDArray[np.bool_] | None = None, mu_grid: FloatArray | None = None)
IntervalProcedure.__call__(x: FloatArray) -> IntervalBatch
RiskResult(value: float, mc_se: float, n_samples: int, loss: str,
           inference_us: float | None = None)
CoverageResult(theta: FloatArray, value: FloatArray, mc_se: FloatArray,
               expected_length: FloatArray, length_mc_se: FloatArray,
               empty_rate: FloatArray, empty_mc_se: FloatArray,
               n_samples: int, exact: bool)
```

Protocols are structural: no inheritance is required. Frozen result records
are containers, with no inference logic. Arrays are not made read-only by a
frozen dataclass; consumers must not mutate shared inputs/results.
`CoverageResult.theta` is `(K,3)`; every diagnostic/error array is `(K,)`.
`n_samples` is the per-point toy count. Exact sums use zero toys and zero MC
errors; numerical tail/grid errors remain separate. An empty interval has
length zero. `RiskResult.mc_se` and `length_mc_se` are sample s.d. (`ddof=1`)
divided by the square root of the toy count. Probability errors use
`sqrt(p*(1-p)/n)`.

Coverage and expected lengths describe the reported inclusive hull. Keep raw
acceptance grids for nonconvex-set diagnostics and label any raw-set coverage
separately. A hull can cover when the raw set does not. No continuous-domain
guarantee follows from a finite grid alone.

## Path A public API (implemented)

```python
# bayesian_risk.py
GridPosterior(mu_grid: FloatArray, weights: FloatArray,
              mean: FloatArray, variance: FloatArray)
GridPosterior.intervals(*, alpha: float = 0.1) -> IntervalBatch
GridPosterior.sample(n_draws: int, *, rng: Generator) -> FloatArray
exact_posterior(x: FloatArray, *, prior: Prior | None = None,
                config: StudyConfig = FULL_CONFIG) -> GridPosterior
gamma_plugin_moments(x: FloatArray, *, shape: float = 1.0, rate: float = 0.1,
                     config: StudyConfig = FULL_CONFIG) -> PosteriorMoments
laplace_moments(x: FloatArray, *, prior: Prior | None = None,
                config: StudyConfig = FULL_CONFIG) -> PosteriorMoments
clipped_mle(x: FloatArray, *, config: StudyConfig = FULL_CONFIG) -> FloatArray

AmortizedPosterior  # created by train_amortized or load; internal constructor is not a contract
AmortizedPosterior.__call__(x: FloatArray) -> FloatArray
AmortizedPosterior.moments(x: FloatArray) -> PosteriorMoments
AmortizedPosterior.intervals(x: FloatArray, *, alpha: float = 0.1) -> IntervalBatch
AmortizedPosterior.sample(x: FloatArray, n_draws: int, *, rng: Generator) -> FloatArray
AmortizedPosterior.save(path: str | Path) -> Path
AmortizedPosterior.load(path: str | Path) -> AmortizedPosterior  # classmethod
train_amortized(*, prior: Prior | None = None, config: StudyConfig = FULL_CONFIG,
                n_train: int | None = None, epochs: int = 100,
                seed: int = 20261012) -> AmortizedPosterior
estimate_bayes_risk(estimator: PointEstimator, *, rng: Generator,
                    n_samples: int | None = None, prior: Prior | None = None,
                    config: StudyConfig = FULL_CONFIG,
                    loss: Literal["squared", "absolute"] = "squared") -> RiskResult
estimate_frequentist_risk(estimator: PointEstimator, theta: FloatArray, *, rng: Generator,
                          n_samples: int | None = None, config: StudyConfig = FULL_CONFIG,
                          loss: Literal["squared", "absolute"] = "squared") -> list[RiskResult]
simulation_based_calibration(posterior: PosteriorSampler, *, rng: Generator,
                             n_samples: int | None = None, n_draws: int | None = None,
                             prior: Prior | None = None,
                             config: StudyConfig = FULL_CONFIG) -> IntArray
```

`GridPosterior.sample` already owns its observations; wrap the call to
`exact_posterior(x).sample(...)` in a `PosteriorSampler` for SBC. SBC returns
randomized ranks in 0 through `n_draws`, including randomized ties at zero.
If a neural draw distribution is only approximate, label it and test it;
mean and variance heads alone do not define a posterior distribution.

The clipped MLE in full mode profiles the nuisance likelihood. A plug-in
estimate can be an additional baseline but must be named accordingly.
Gamma plug-in inference uses a finite, truncated Gamma mixture (binomial expansion),
not a conjugacy claim for background-subtracted counts. Its analysis prior is a
diffuse Gamma(shape 1, rate 0.1) with no atom at zero, not the reality prior,
and it fixes the nuisances at `b_hat = n_off/tau` and `eps_hat`. Laplace keeps
the reality prior but also plugs in the nuisances. SBC for all four samplers
uses the reality prior as its reference, so the Gamma and Laplace SBC results
combine prior or model mismatch with approximation error; they do not measure
numerical accuracy alone. Laplace must handle the point mass and the physical
boundaries explicitly or state its limitation.

Additional public helpers used by the Path A runner:

```python
GridPosterior.hpd_intervals(*, alpha: float = 0.1) -> IntervalBatch  # smallest-volume grid set + hull
GammaPluginPosterior(config=FULL_CONFIG, shape=1.0, rate=0.1)  # sampler for SBC
GammaPluginPosterior.sample(x, n_draws, *, rng) -> FloatArray
LaplacePosterior(config=FULL_CONFIG, prior=None)                # sampler for SBC
LaplacePosterior.sample(x, n_draws, *, rng) -> FloatArray
time_inference(estimator: PointEstimator, x: FloatArray, *, repeats: int = 7) -> float
```

`time_inference` returns the median batched time per observation in
microseconds; it is a deterministic measurement, not a Monte Carlo estimate.

## Exact Path A to Path B bridge

Path B receives a `PosteriorPredictor`, whose only required method is:

```python
moments(x: NDArray[np.float64]) -> PosteriorMoments
```

Input is raw `(N,3)` data in the fixed column order. Both outputs are `(N,)`
float64 arrays. Mean lies in `[0,mu_max]`; variance is finite and nonnegative.
Variance means `Var(mu | x)`, not the variance of means in the input batch.
The predictor owns all transforms. WALDO uses a visible `variance_floor=1e-8`.

Path A saves `results/path_a_amortized.pt` through `AmortizedPosterior.save`.
Path B loads it through `AmortizedPosterior.load`; it does not inspect torch
state directly. Missing artifacts must produce a clear dependency error or a
clearly labelled temporary stand-in during development. The final experiment
uses Path A's trained model. `make all` runs Path A before Path B to supply it.

## Path B public API (implemented)

```python
# frequentist_bounds.py
TestStatistic = Callable[[FloatArray, FloatArray], FloatArray]  # T(x, mu0)
NeymanBelt(mu_grid: FloatArray, counts: IntArray, acceptance: NDArray[np.bool_])
NeymanBelt.__call__(x: FloatArray) -> IntervalBatch
CalibratedBelt(mu_grid: FloatArray, critical_values: FloatArray,
               statistic: TestStatistic, nuisance_grid: FloatArray, alpha: float)
CalibratedBelt.__call__(x: FloatArray) -> IntervalBatch
feldman_cousins_belt(*, b: float = 3.0, alpha: float = 0.1,
                     mu_grid: FloatArray | None = None, n_max: int | None = None,
                     enforce_b_monotone: bool = False,
                     config: StudyConfig = FULL_CONFIG) -> NeymanBelt
profile_background(n_on: FloatArray, n_off: FloatArray, mu: FloatArray, *,
                   eps: float | FloatArray = 1.0, tau: float = 5.0) -> FloatArray
profile_likelihood_statistic(x: FloatArray, mu: FloatArray, *,
                             config: StudyConfig = FULL_CONFIG) -> FloatArray
waldo_statistic(x: FloatArray, mu: FloatArray, *, posterior: PosteriorPredictor,
                variance_floor: float = 1e-8) -> FloatArray
calibrate_belt(statistic: TestStatistic, *, rng: Generator,
               mu_grid: FloatArray | None = None, nuisance_grid: FloatArray | None = None,
               n_toys: int | None = None, config: StudyConfig = FULL_CONFIG) -> CalibratedBelt
naive_wald_interval(x: FloatArray, *, config: StudyConfig = FULL_CONFIG) -> IntervalBatch
central_neyman_interval(x: FloatArray, *, b: float = 3.0, alpha: float = 0.1,
                        config: StudyConfig = FULL_CONFIG) -> IntervalBatch
upper_limit_interval(x: FloatArray, *, b: float = 3.0, alpha: float = 0.1,
                      config: StudyConfig = FULL_CONFIG) -> IntervalBatch
flat_prior_interval(x: FloatArray, *, config: StudyConfig = FULL_CONFIG) -> IntervalBatch
exact_coverage(procedure: IntervalProcedure, mu_grid: FloatArray, *, b: float = 3.0,
               n_max: int | None = None, config: StudyConfig = FULL_CONFIG) -> CoverageResult
estimate_coverage(procedure: IntervalProcedure, theta: FloatArray, *, rng: Generator,
                  n_samples: int | None = None,
                  config: StudyConfig = FULL_CONFIG) -> CoverageResult
```

Bind any keyword dependencies in a two-argument statistic closure before
passing it to `calibrate_belt`. A statistic broadcasts `x[...,0]` against
`mu`; flatten batch axes when calling `posterior.moments`. The calibration
grid defaults to step 0.1. Full-mode nuisance thresholds take the maximum
over the bounded background/efficiency grid described above. Profile b over
the physical domain `b>=0`, not the Bayesian prior
range. Profile eps over `eps>0` with the auxiliary measurement likelihood.

`CalibratedBelt.__call__` inverts its own grid. An engineer may add an
optional finer inversion helper, preserving this API; critical-value
interpolation forms a three-node maximum upper envelope at every coarse node,
then linearly interpolates this envelope. It dominates both original bracketing
cutoffs and has no isolated downward spikes at exact calibration nodes.
The public helper `refine_belt(belt, mu_grid) -> CalibratedBelt` implements this
convention; it is not a theorem about between-node coverage. Exact core WALDO thresholds
can be obtained by summing the count support, with no MC calibration.
The FC b-monotone option defaults to false and may lengthen upper endpoints.
State its auxiliary b grid and test the published n_on=0 reference rows.

Additional public helpers used by the Path B runner:

```python
exact_waldo_belt(*, posterior: PosteriorPredictor, mu_grid: FloatArray | None = None,
                 config: StudyConfig = FULL_CONFIG) -> CalibratedBelt  # core: exact Poisson sums
LocalPosteriorStandIn(config=FULL_CONFIG)  # exact flat-grid posterior in core mode
LocalPosteriorStandIn.moments(x: FloatArray) -> PosteriorMoments
```

The final full-case WALDO study uses the trained Path A model, not
`LocalPosteriorStandIn`. The core-case WALDO results use `LocalPosteriorStandIn`
with the core configuration.

## Results JSON schema (both scripts)

Write UTF-8 experiment results, not cached invented examples. Path A retains
the complete version-1 payload in `results/path_a.json`. Path B writes an
indented version-2 summary in `results/path_b.json` (under 300 KB), plus the
complete version-1 payload in `results/path_b_details.json.gz`. The gzip file
uses compact JSON separators and preserves every per-point record unchanged.
The complete payloads use this envelope:

```json
{
  "schema_version": 1,
  "path": "a",
  "generated_at": "UTC ISO-8601 timestamp",
  "seed": 20261012,
  "config": {"problem": {}, "grid": {}, "mc": {}, "seed": 20261012},
  "runtime_seconds": 0.0,
  "environment": {"python": "version", "numpy": "version", "torch": "version"},
  "methods": {},
  "diagnostics": {},
  "artifacts": {"figures": [], "model": null},
  "notes": []
}
```

`path` is `"a"` or `"b"`; `config` is `dataclasses.asdict(config)` in full,
including actual overrides. `environment` contains the versions actually used.
`runtime_seconds` includes training/calibration when the script performs them.
`artifacts` paths are relative to the repository, e.g.
`figures/path_a_risk.png` and `results/path_a_amortized.pt`.
`notes` is a list of plain-English limitations/failures.

The Path B summary retains the same envelope keys, with `schema_version=2`,
`format="summary"`, and `artifacts.details="results/path_b_details.json.gz"`.
Config, seed, generation timestamp, run time, environment, model provenance,
Bayes risks, reality-prior interval summaries, FC reference rows, and notes
remain readable. Summary methods omit the arrays `coverage`,
`efficiency_variations`, and `off_grid_coverage`. Each method adds:

| Field | Value schema |
| --- | --- |
| `coverage_check` | compact verdict described below, including minimum coverage and its argmin |
| `maximum_empty_rate` | `{"theta": [mu,b,eps], "empty_rate": Metric}` at the grid maximum |
| `maximum_disconnected_rate` | `{"theta": [mu,b,eps], "disconnected_rate": Metric}`, or null if not recorded |

`interval_summary.expected_length` is mean expected hull length under the
stated reality prior; `interval_summary.empty_rate` is the prior-average empty
rate. Maximum rates are grid extrema, not prior averages. Disconnected rates
come from initial independent full-case toys, before hull-coverage reruns.
An unavailable rate stays null; serialization does not add statistical estimates.
The ablation calibrated at `eps0` only (key `eps0_ablation`), the efficiency-box
checks, and the off-grid checks keep compact verdicts. Calibration grids, cutoffs, initial toy records, rerun rows,
flagged-point lists, and other per-point arrays live in the gzip payload.
The summary keeps calibration node counts and nuisance bounds, and keeps the
FC other-count monotonicity violation count rather than its full list.

Readers needing curves or per-point diagnostics, including the notebook and
figure regeneration, load the complete payload with the standard library:

```python
summary = json.loads((root / "results/path_b.json").read_text())
with gzip.open(root / summary["artifacts"]["details"], "rt", encoding="utf-8") as stream:
    details = json.load(stream)
```

Every stochastic scalar metric has the following record, including MC error:

```json
{"value": 0.0, "mc_se": 0.0, "n_samples": 20000, "method": "mc"}
```

The shown numbers are schema placeholders, not results. `method` is `"mc"`
or `"exact"`. Exact sums have `mc_se=0` and `n_samples=0`; numerical tail/grid
errors belong in diagnostics. Use JSON `null` for unavailable values, with a
reason in notes. Never emit NaN or Infinity (`json.dump(..., allow_nan=False)`).
Wall-clock timings are deterministic measurements with a stated measurement
scheme, not MC expectations, so do not give them a made-up MC error.

`methods` maps stable method identifiers to objects. Use separate identifiers
for core/full or prior variants. Each object has `description`, `mode`
(`"core"` or `"full"`), and the applicable fields below; omit fields that do
not apply rather than inventing values:

| Field | Value schema |
| --- | --- |
| `inference_us` | measured float, or null with explanation |
| `training_seconds` | measured float when trained |
| `analysis_prior` | identifier plus actual family/shape/rate/zero_mass/cutoff settings |
| `bayes_risk` | `{"squared": Metric, "absolute": Metric}` |
| `frequentist_risk` | list of `{"theta": [mu,b,eps], "squared": Metric, "absolute": Metric}` |
| `max_grid_risk` | `{"theta": [mu,b,eps], "squared": Metric}` at the observed grid maximum |
| `interval_summary` | `{"coverage": Metric, "expected_length": Metric, "empty_rate": Metric}` under the reality prior |
| `coverage` | list of `{"theta": [mu,b,eps], "coverage": Metric, "expected_length": Metric, "empty_rate": Metric}` |

For interval-only procedures, estimate the Bayes risk of a stated point
summary (e.g. midpoint) only if it is explicitly defined. Do not call
expected interval length a point-estimation risk. Path A includes risk,
cost, prior sensitivity, and SBC diagnostics. Path B includes coverage,
length, empty rates, FC reference checks, and the provenance of the trained WALDO model.
The scripts may add nested diagnostics while retaining this shared envelope.

Final Path B records the Bayes risk of inclusive-hull midpoints, using the
clipped MLE if a hull is empty, under the stated core/full reality prior.
Path A records fixed-parameter credible coverage alongside its risk grid.
Efficiency-box raw coverage checks evaluate only the statistic at the true
signal (with the same continuous threshold), so they do not report a hull length.
Final Path B requires the saved Path A model and errors on a missing or
physically incompatible model instead of silently using a development stand-in.

In complete payloads, `diagnostics.coverage_check` records `alpha`, `tolerance`, `domain`
(exact grid values, fixed eps and nuisance treatment), `coverage_target`
(`"hull"` or `"raw_set"`), `minimum_coverage`, `flagged_points`,
`rerun_n_samples`, and `passed`. Use a list of such checks for multiple
procedures/domains. At the default full-case grid, require coverage at least
0.90 minus 0.015, rerun flagged points with 10 times the toys, and state
how the reruns change the verdict. Exact-core tolerance is 0.005.

In the Path B summary, checks retain `method`, `alpha`, `tolerance`,
`coverage_target`, `rerun_n_samples`, `passed`, and `initial_minimum` when
recorded. `minimum_coverage` contains the unchanged `theta` and coverage
Metric. `flagged_count_before_reruns` and `flagged_count_after_reruns` count
the original point lists. `domain` reports `n_points`, `bounds` for mu/b/eps,
and the original nuisance description. Bounds describe the finite checked
grid; they do not assert coverage between its nodes.

`diagnostics.sbc` records `n_samples`, `n_draws`, the rank histogram/bins,
tie handling, and any histogram proportions with MC errors. Also record
quadrature convergence, count truncation mass, calibration sample size,
RNG stream seeds, critical-value interpolation, variance floor, and
disconnected-set counts where relevant. Passing SBC is not a fixed-theta
coverage guarantee. A finite-grid coverage pass is not a continuous theorem.

## Tooling and execution

The package requires Python 3.12 or newer. Dependencies have lower bounds
from the installed Python 3.12 environment.
`requirements.txt` pins only direct runtime/notebook dependencies plus pytest
and ruff. The hatchling build uses the src layout and supports editable install.
Setup/CI install torch from the CPU wheel index before the remaining pins.
The editable install itself needs no network download once the pinned packages exist.

```text
make setup       # create/reuse .venv and install pinned deps plus package
make test        # default excludes slow tests
make test-all    # pytest -o addopts= includes slow tests
make lint        # ruff check (E,F,W,I,B,UP,NPY, line length 100) and ruff format --check
make figures     # redraw every figure from the saved results; no simulation, no results/ write
make path-a      # run the Path A experiment (overwrites results/ and figures/)
make path-b      # run Path B using the saved Path A model (overwrites results/ and figures/)
make notebook    # execute the notebook in place with python -m nbconvert
make all         # lint, test, Path A, Path B, summary figure redraw (scripts/make_summary_figure.py), notebook
```

Use `pytest -o addopts= -m slow` for just slow tests; passing `-m` on the
command line also overrides the default marker expression. Run non-slow
tests in under 120 seconds and each experiment in under 10 minutes.
