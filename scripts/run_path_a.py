"""Run the Path A (Bayes risk) experiment on CPU and save its results.

The script trains the amortized neural posterior and compares five estimators of
mu: the grid, Gamma plug-in, Laplace and neural posterior means, and the clipped
MLE. It estimates Bayes risk under the reality prior, frequentist risk on a fixed
signal grid, analysis-prior sensitivity, simulation-based calibration (SBC) and
grid-quadrature convergence. Comparisons on shared draws are also stored as paired
differences against the grid posterior. It writes results/path_a.json, the model file
results/path_a_amortized.pt and four figures. Use --quick for smaller grids and
Monte Carlo budgets; the budgets used are recorded in the JSON file.
Both experiment modes overwrite results/ and figures/. Use --figures-only
to redraw saved figures without training, simulation or results writes.
"""

import argparse
import json
import platform
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import xlogy
from scipy.stats import chisquare

from bayesfreq.bayesian_risk import (
    GammaPluginPosterior,
    GridPosterior,
    LaplacePosterior,
    clipped_mle,
    exact_posterior,
    gamma_plugin_moments,
    laplace_moments,
    simulation_based_calibration,
    time_inference,
    train_amortized,
)
from bayesfreq.config import FULL_CONFIG, StudyConfig
from bayesfreq.data_simulator import Prior, analysis_priors, reality_prior, sample_joint, simulate
from bayesfreq.interfaces import IntervalBatch
from bayesfreq.plotting import (
    METHOD_COLORS,
    METHOD_LABELS,
    PRIOR_LABELS,
    save_figure,
    saved_config,
    set_style,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = METHOD_LABELS


def metric(values: np.ndarray, *, probability: bool = False) -> dict:
    """Summarize a simulated scalar outcome with its Monte Carlo standard error.

    Parameters
    ----------
    values : ndarray, shape (n,)
        Per-sample outcomes, such as losses, lengths or 0/1 indicators.
    probability : bool, default False
        If True, treat values as 0/1 indicators and use the binomial SE
        ``sqrt(p (1 - p) / n)``. Otherwise use ``sd(values, ddof=1) / sqrt(n)``.

    Returns
    -------
    dict
        ``value`` (sample mean), ``mc_se``, ``n_samples`` and ``method`` ("mc").
    """
    n = len(values)
    value = float(np.mean(values))
    se = np.sqrt(value * (1 - value) / n) if probability else np.std(values, ddof=1) / np.sqrt(n)
    return {"value": value, "mc_se": float(se), "n_samples": n, "method": "mc"}


def risks(truth: np.ndarray, estimate: np.ndarray) -> dict:
    """Return Monte Carlo summaries of squared and absolute loss.

    Parameters
    ----------
    truth, estimate : ndarray, shape (n,)
        True signal values and their estimates.

    Returns
    -------
    dict
        ``squared`` and ``absolute``: ``metric`` summaries of
        ``(estimate - truth)**2`` and ``|estimate - truth|``.
    """
    residual = estimate - truth
    return {"squared": metric(residual**2), "absolute": metric(np.abs(residual))}


def prior_metadata(identifier: str, prior: Prior) -> dict:
    """Return the JSON metadata that identifies an analysis prior.

    Parameters
    ----------
    identifier : str
        Name used in the results file, such as ``"reality"`` or ``"cutoff_0.3"``.
    prior : Prior
        Prior to describe.

    Returns
    -------
    dict
        Identifier, family, Gamma shape and rate (stored for every family),
        zero-atom mass, and the log-uniform lower cutoff ``mu_min`` as ``cutoff``.
    """
    return {
        "identifier": identifier,
        "family": prior.family,
        "shape": prior.shape,
        "rate": prior.rate,
        "zero_mass": prior.zero_mass,
        "cutoff": prior.mu_min,
    }


def covered(interval: IntervalBatch, truth: np.ndarray) -> np.ndarray:
    """Return whether each inclusive hull contains its true signal value.

    Parameters
    ----------
    interval : IntervalBatch
        Inclusive hulls, one per sample. A non-finite lower endpoint marks an
        empty set, which never covers.
    truth : ndarray, shape (n,)
        True signal value for each sample.

    Returns
    -------
    ndarray of bool, shape (n,)
        True where the set is non-empty and lower <= truth <= upper.
    """
    empty = ~np.isfinite(interval.lower)
    return (~empty) & (truth >= interval.lower) & (truth <= interval.upper)


def interval_summary(
    interval: IntervalBatch, truth: np.ndarray, *, content: np.ndarray | None = None
) -> dict:
    """Return coverage, expected length and empty rate for a batch of intervals.

    Parameters
    ----------
    interval : IntervalBatch
        Inclusive hulls, one per sample. A non-finite lower endpoint marks an
        empty set.
    truth : ndarray, shape (n,)
        True signal value for each sample.
    content : ndarray, shape (n,), optional
        Posterior mass inside each interval, under the posterior that
        reported it. If given, the summary also describes it.

    Returns
    -------
    dict
        ``metric`` summaries of ``coverage`` (non-empty and
        lower <= truth <= upper), ``expected_length`` (upper - lower, with 0
        for empty sets) and ``empty_rate``. With ``content``, it also has
        ``posterior_content`` (mean mass inside the interval) and
        ``lower_at_zero_rate`` (share of intervals with lower endpoint 0).
    """
    empty = ~np.isfinite(interval.lower)
    coverage = covered(interval, truth)
    widths = np.where(empty, 0, interval.upper - interval.lower)
    summary = {
        "coverage": metric(coverage, probability=True),
        "expected_length": metric(widths),
        "empty_rate": metric(empty, probability=True),
    }
    if content is not None:
        summary["posterior_content"] = metric(content)
        summary["lower_at_zero_rate"] = metric(interval.lower == 0, probability=True)
    return summary


def paired_comparison(
    truth: np.ndarray, estimate: np.ndarray, reference: np.ndarray, *, excess: bool = False
) -> dict:
    """Summarize per-draw loss differences, estimate minus reference, on shared draws.

    Parameters
    ----------
    truth, estimate, reference : ndarray, shape (n,)
        True signal values and two estimates of them from the same draws.
    excess : bool, default False
        If True, also summarize ``(estimate - reference)**2``.

    Returns
    -------
    dict
        ``metric`` summaries of ``squared_loss_difference`` and
        ``absolute_loss_difference``. Their SEs are paired: they use the
        spread of the per-draw differences, so the shared draws cancel. With
        ``excess``, also ``excess_squared_risk``.

    Notes
    -----
    If ``reference`` is the posterior mean under the prior that generated
    the draws, it is the Bayes rule for squared loss, and
    ``E[(estimate - mu)**2] - E[(reference - mu)**2] = E[(estimate - reference)**2]``.
    So ``excess_squared_risk`` estimates the same excess risk as the mean
    squared-loss difference, with a much smaller Monte Carlo SE.
    """
    residual = estimate - truth
    base = reference - truth
    result = {
        "squared_loss_difference": metric(residual**2 - base**2),
        "absolute_loss_difference": metric(np.abs(residual) - np.abs(base)),
    }
    if excess:
        result["excess_squared_risk"] = metric((estimate - reference) ** 2)
    return result


def grid_interval_content(posterior: GridPosterior, interval: IntervalBatch) -> np.ndarray:
    """Return the grid posterior mass on the nodes inside each inclusive interval.

    Parameters
    ----------
    posterior : GridPosterior
        Grid posterior for N observations.
    interval : IntervalBatch
        One inclusive interval per observation, such as ``posterior.intervals()``.

    Returns
    -------
    ndarray, shape (N,)
        Sum of the node masses with lower <= node <= upper.
    """
    grid = posterior.mu_grid
    inside = (grid >= interval.lower[:, None]) & (grid <= interval.upper[:, None])
    return np.sum(posterior.weights * inside, axis=1)


def neural_interval_content(
    probabilities: np.ndarray, edges: np.ndarray, interval: IntervalBatch
) -> np.ndarray:
    """Return the neural posterior's own mass inside each inclusive interval.

    Parameters
    ----------
    probabilities : ndarray, shape (N, K + 1)
        Class probabilities. Class 0 is the zero atom; class k >= 1 is
        uniform on the bin [edges[k - 1], edges[k]].
    edges : ndarray, shape (K + 1,)
        Increasing bin edges, all at or above zero.
    interval : IntervalBatch
        One inclusive interval per observation.

    Returns
    -------
    ndarray, shape (N,)
        Atom mass if the interval starts at 0, plus the bin mass between the
        endpoints.
    """

    def continuous_mass_below(values: np.ndarray) -> np.ndarray:
        fraction = np.clip((values[:, None] - edges[:-1]) / np.diff(edges), 0, 1)
        return np.sum(probabilities[:, 1:] * fraction, axis=1)

    atom = np.where(interval.lower <= 0, probabilities[:, 0], 0.0)
    return atom + continuous_mass_below(interval.upper) - continuous_mass_below(interval.lower)


def class_masses(posterior: GridPosterior, edges: np.ndarray) -> np.ndarray:
    """Bin a grid posterior into the neural posterior's classes.

    Parameters
    ----------
    posterior : GridPosterior
        Grid posterior for N observations.
    edges : ndarray, shape (K + 1,)
        Increasing edges of the neural bins; class 0 is the zero atom.

    Returns
    -------
    ndarray, shape (N, K + 1)
        Posterior mass in each class. Each grid cell is taken as uniform on
        its width, and its mass is split between bins in proportion to their
        overlap. A zero-width node goes to class 0 if it is at 0, and
        otherwise to the bin that contains it.

    Raises
    ------
    ValueError
        If some grid cell is not fully covered by the bins.
    """
    grid = posterior.mu_grid
    widths = np.zeros_like(grid) if posterior.cell_widths is None else posterior.cell_widths
    cells = widths > 0
    split = np.zeros((len(grid), len(edges)))
    overlap = np.minimum(grid[cells, None] + widths[cells, None] / 2, edges[1:]) - np.maximum(
        grid[cells, None] - widths[cells, None] / 2, edges[:-1]
    )
    split[cells, 1:] = np.clip(overlap, 0, None) / widths[cells, None]
    split[~cells & (grid == 0), 0] = 1.0
    points = np.flatnonzero(~cells & (grid > 0))
    split[
        points, np.clip(np.searchsorted(edges, grid[points], side="right"), 1, len(edges) - 1)
    ] = 1
    if not np.allclose(split.sum(axis=1), 1.0, rtol=0, atol=1e-9):
        raise ValueError("every grid cell must lie inside the neural bins")
    return posterior.weights @ split


def neural_vs_exact(
    exact_classes: np.ndarray,
    probabilities: np.ndarray,
    exact_variance: np.ndarray,
    neural_variance: np.ndarray,
) -> dict:
    """Compare the neural posterior with the exact grid posterior on the same draws.

    Parameters
    ----------
    exact_classes : ndarray, shape (N, K + 1)
        Exact posterior binned into the neural classes (``class_masses``).
    probabilities : ndarray, shape (N, K + 1)
        Neural class probabilities for the same observations.
    exact_variance, neural_variance : ndarray, shape (N,)
        Posterior variances of mu under each posterior.

    Returns
    -------
    dict
        ``kl_exact_to_neural``: ``metric`` of KL(exact || neural) in nats,
        sum_k P_k log(P_k / Q_k) per draw; ``kl_quantile_0.95``: its 95th
        percentile; ``variance_correlation``: Pearson correlation of the two
        variances; ``variance_ratio``: ``metric`` of neural / exact variance
        per draw; ``expected_posterior_variance``: ``metric`` of each
        variance. For the exact posterior, E[Var(mu | x)] is the Bayes risk
        of its mean.
    """
    kl = np.sum(xlogy(exact_classes, exact_classes) - xlogy(exact_classes, probabilities), axis=1)
    return {
        "kl_exact_to_neural": metric(kl),
        "kl_quantile_0.95": float(np.quantile(kl, 0.95)),
        "n_classes": exact_classes.shape[1],
        "variance_correlation": float(np.corrcoef(exact_variance, neural_variance)[0, 1]),
        "variance_ratio": metric(neural_variance / exact_variance),
        "expected_posterior_variance": {
            "exact_grid": metric(exact_variance),
            "amortized": metric(neural_variance),
        },
    }


class ExactSampler:
    """Posterior sampler that computes the exact grid posterior on each call.

    ``exact_posterior(x).sample`` already holds its observations, so this
    wrapper gives it the ``PosteriorSampler`` signature
    ``sample(x, n_draws, rng=...)`` that SBC needs. It uses the default
    analysis prior, the reality prior.
    """

    def __init__(self, config: StudyConfig) -> None:
        """Store the study config.

        Parameters
        ----------
        config : StudyConfig
            Model constants and quadrature sizes for ``exact_posterior``.
        """
        self.config = config

    def sample(self, x: np.ndarray, n_draws: int, *, rng: np.random.Generator) -> np.ndarray:
        """Draw mu from the grid posterior of each observation.

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
            Grid-node draws, with exact zeros from the atom.
        """
        return exact_posterior(x, config=self.config).sample(n_draws, rng=rng)


def draw_figures(record: dict) -> list[Path]:
    """Draw four figures using only the values in a results record."""
    config = saved_config(record)
    methods = record["methods"]
    sensitivity = record["diagnostics"]["prior_sensitivity"]
    sbc = record["diagnostics"]["sbc"]
    set_style()
    method_colors = METHOD_COLORS
    figure_paths = []
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for name, entry in methods.items():
        risk = entry["bayes_risk"]["squared"]
        ax.errorbar(
            entry["inference_us"],
            risk["value"],
            yerr=risk["mc_se"],
            fmt="o",
            capsize=3,
            color=method_colors[name],
        )
        ax.annotate(
            LABELS[name],
            (entry["inference_us"], risk["value"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set(
        xscale="log",
        xlabel="Inference time per observation (µs, log scale)",
        ylabel="Average squared error (Bayes risk)",
        title="Accuracy vs speed (lower-left is better)",
    )
    figure_paths.append(save_figure(fig, "path_a_pareto.png"))
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, background in zip(axes, (1.0, config.problem.known_b, 10.0), strict=True):
        for name, entry in methods.items():
            rows = [row for row in entry["frequentist_risk"] if row["theta"][1] == background]
            if not rows:
                continue
            values = np.array([row["squared"]["value"] for row in rows])
            errors = np.array([row["squared"]["mc_se"] for row in rows])
            ax.plot(
                [row["theta"][0] for row in rows],
                values,
                label=LABELS[name],
                color=method_colors[name],
            )
            ax.fill_between(
                [row["theta"][0] for row in rows],
                values - errors,
                values + errors,
                alpha=0.12,
                color=method_colors[name],
            )
        ax.set(
            xlabel="True signal μ",
            title=f"Background b = {background:g}, efficiency ε = {config.problem.eps0:g}",
        )
    axes[0].set_ylabel("Squared frequentist risk (bands: MC SE)")
    axes[1].legend(fontsize=11)
    fig.tight_layout()
    figure_paths.append(save_figure(fig, "path_a_risk_function.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = list(sensitivity)
    metrics = [sensitivity[name]["bayes_risk"]["squared"] for name in names]
    ax.bar(
        range(len(names)),
        [item["value"] for item in metrics],
        yerr=[item["mc_se"] for item in metrics],
        capsize=3,
        color=METHOD_COLORS["exact_grid"],
    )
    ax.set(
        xticks=range(len(names)),
        xticklabels=[PRIOR_LABELS[name] for name in names],
        ylabel="Squared Bayes risk under the reality prior",
        title="Cost of a wrong analysis prior",
    )
    ax.tick_params(axis="x", labelrotation=30)
    figure_paths.append(save_figure(fig, "path_a_prior_sensitivity.png"))
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, (name, summary) in zip(axes.flat, sbc.items(), strict=True):
        bins = np.asarray(summary["bins"])
        ax.bar(
            (bins[:-1] + bins[1:]) / 2,
            summary["histogram"],
            width=np.diff(bins) * 0.85,
            color=METHOD_COLORS[name],
        )
        ax.plot(
            (bins[:-1] + bins[1:]) / 2,
            summary["uniform_expected_counts"],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        p_label = (
            "<1e-300" if summary["p_value_underflow"] else f"={summary['chi_square_p_value']:.3g}"
        )
        ax.set(
            title=f"{LABELS[name]} (chi-square p{p_label})",
            xlabel="Randomized posterior rank",
            ylabel="Count",
        )
    fig.suptitle("Calibration under the reality prior\ndashed: expected count if calibrated")
    fig.tight_layout()
    figure_paths.append(save_figure(fig, "path_a_sbc.png"))
    plt.close(fig)
    return figure_paths


def redraw_saved() -> list[Path]:
    """Redraw saved results without training, simulation or results writes."""
    return draw_figures(json.loads((ROOT / "results/path_a.json").read_text()))


def run(*, quick: bool = False) -> dict:
    """Run the full Path A experiment, then write its JSON results and figures.

    Parameters
    ----------
    quick : bool, default False
        If True, use a coarser grid (mu_step 0.05, 41 background and 15
        efficiency nodes), smaller Monte Carlo budgets and 45 training epochs.

    Returns
    -------
    dict
        The result record, also written to ``results/path_a.json``.

    Notes
    -----
    Six independent seeds spawned from ``config.seed`` drive training, Bayes
    risk, frequentist risk, SBC, the convergence check and the timing batch.
    The stages are:

    1. Train the neural posterior and save ``results/path_a_amortized.pt``.
    2. Time each estimator and estimate its squared and absolute Bayes risk
       under the reality prior. All estimators share one set of draws, so
       each one is also compared with the grid posterior mean by paired
       differences. Record interval coverage and posterior content for the
       grid posterior (equal-tail and HPD hull) and the neural posterior.
       Compare the neural posterior with the binned grid posterior (KL
       divergence and posterior variances).
    3. Estimate frequentist risk and interval coverage on a mu grid at fixed
       b and eps = eps0. All estimators share the toys at each point, so the
       neural-minus-grid coverage difference at b = known_b is paired.
    4. Estimate the Bayes risk of the grid posterior mean under other analysis
       priors, reusing the stage-2 draws from the reality prior. Each prior
       is compared, by paired differences, with the reality-prior analysis
       and with the clipped MLE.
    5. Run SBC for four posterior samplers, with a 10-bin chi-square test.
       All four use one seed, so they share the same prior draws.
    6. Check quadrature convergence against a refined grid on 64 observations.
    7. Save four figures under ``figures/`` and write the JSON file.

    The paired comparisons, interval content and the neural-versus-grid
    comparison reuse existing draws and add no randomness. Progress is
    printed to stdout.
    """
    started = perf_counter()
    config = FULL_CONFIG
    if quick:
        config = replace(
            config,
            grid=replace(config.grid, mu_step=0.05, b_points=41, eps_points=15),
            mc=replace(
                config.mc,
                m_bayes_risk=2000,
                n_train=8000,
                n_toys_cov=300,
                n_sbc=200,
                n_posterior_draws=50,
            ),
        )
    # Six independent child seeds, one per random stage, all derived from config.seed.
    children = np.random.SeedSequence(config.seed).spawn(6)
    seeds = dict(
        zip(
            ("training", "bayes_risk", "frequentist_risk", "sbc", "convergence", "timing"),
            [int(child.generate_state(1)[0]) for child in children],
            strict=True,
        )
    )
    print(f"Path A: {'quick' if quick else 'full'} CPU run", flush=True)
    # Stage 1: train and save the neural posterior.
    model = train_amortized(config=config, seed=seeds["training"], epochs=45 if quick else 100)
    model_path = model.save(ROOT / "results/path_a_amortized.pt")
    print(f"Neural training: {model.training_seconds:.2f}s", flush=True)
    # Stage 2: time estimators and estimate prior-averaged risks and intervals.
    estimators = {
        "exact_grid": lambda x: exact_posterior(x, config=config).mean,
        "gamma_plugin": lambda x: gamma_plugin_moments(x, config=config).mean,
        "laplace": lambda x: laplace_moments(x, config=config).mean,
        "amortized": model,
        "clipped_mle": partial(clipped_mle, config=config),
    }
    theta, x = sample_joint(
        config.mc.m_bayes_risk, rng=np.random.default_rng(seeds["bayes_risk"]), config=config
    )
    methods = {}
    predictions = {}
    priors = analysis_priors(config)
    timing_x = sample_joint(256, rng=np.random.default_rng(seeds["timing"]), config=config)[1]
    # The grid posterior is the reference for the neural comparison, so compute it first.
    reference = exact_posterior(x, config=config)
    exact_classes = class_masses(reference, model.edges)
    neural_comparison = None
    for identifier, estimator in estimators.items():
        posterior = reference if identifier == "exact_grid" else None
        prediction = posterior.mean if posterior is not None else estimator(x)
        predictions[identifier] = prediction
        entry = {
            "description": LABELS[identifier],
            "mode": config.problem.mode,
            "inference_us": time_inference(estimator, timing_x, repeats=7),
            "bayes_risk": risks(theta[:, 0], prediction),
            "frequentist_risk": [],
            "coverage": [],
        }
        if identifier != "clipped_mle":
            selected = (
                priors["gamma_diffuse"] if identifier == "gamma_plugin" else priors["reality"]
            )
            entry["analysis_prior"] = prior_metadata(
                "gamma_diffuse" if identifier == "gamma_plugin" else "reality", selected
            )
        if posterior is not None:
            intervals = posterior.intervals(alpha=config.problem.alpha)
            entry["interval_summary"] = interval_summary(
                intervals, theta[:, 0], content=grid_interval_content(posterior, intervals)
            )
            hpd = posterior.hpd_intervals(alpha=config.problem.alpha)
            entry["hpd_hull_summary"] = interval_summary(hpd, theta[:, 0])
            # Count runs of accepted nodes; more than one run is a disconnected set.
            runs = np.sum(np.diff(hpd.accepted.astype(int), axis=1) == 1, axis=1)
            runs += hpd.accepted[:, 0]
            entry["hpd_disconnected_rate"] = metric(runs > 1, probability=True)
        if identifier == "amortized":
            entry["training_seconds"] = model.training_seconds
            # The class probabilities define the neural intervals; reading them adds no draws.
            probabilities = model._probabilities(x)
            intervals = model.intervals(x, alpha=config.problem.alpha)
            entry["interval_summary"] = interval_summary(
                intervals,
                theta[:, 0],
                content=neural_interval_content(probabilities, model.edges, intervals),
            )
            neural_comparison = neural_vs_exact(
                exact_classes, probabilities, reference.variance, model.moments(x).variance
            )
            neural_comparison["binning"] = (
                "each grid cell split between neural bins by overlap; zero atom is class 0"
            )
        methods[identifier] = entry
        print(
            f"{identifier}: squared risk {entry['bayes_risk']['squared']['value']:.4f}; "
            f"{entry['inference_us']:.2f} us/row",
            flush=True,
        )
    del posterior, reference
    # Paired comparisons with the grid posterior mean, the Bayes rule up to quadrature error.
    for identifier, entry in methods.items():
        if identifier != "exact_grid":
            entry["paired_vs_exact_grid"] = paired_comparison(
                theta[:, 0], predictions[identifier], predictions["exact_grid"], excess=True
            )
    # Stage 3: frequentist risk and coverage at fixed signal values.
    # Risk points: a mu grid at b = known_b (default 3), 1 and 10, with eps = eps0. The slow
    # grid posterior (the reference) runs only at b = known_b; the others run at all three.
    mu_values = np.linspace(0, config.problem.mu_max, 11 if quick else 21)
    backgrounds = (config.problem.known_b, 1.0, 10.0)
    risk_theta = np.concatenate(
        [
            np.column_stack(
                (
                    mu_values,
                    np.full(len(mu_values), background),
                    np.full(len(mu_values), config.problem.eps0),
                )
            )
            for background in backgrounds
        ]
    )
    rng = np.random.default_rng(seeds["frequentist_risk"])
    paired_coverage = []
    for point in risk_theta:
        toys = simulate(*point, rng=rng, config=config, size=config.mc.n_toys_cov)
        truth = np.full(len(toys), point[0])
        hits = {}
        for identifier, estimator in estimators.items():
            if identifier == "exact_grid" and point[1] != config.problem.known_b:
                continue
            posterior = exact_posterior(toys, config=config) if identifier == "exact_grid" else None
            estimate = posterior.mean if posterior is not None else estimator(toys)
            methods[identifier]["frequentist_risk"].append(
                {"theta": point.tolist(), **risks(truth, estimate)}
            )
            if identifier in ("exact_grid", "amortized"):
                if posterior is not None:
                    intervals = posterior.intervals(alpha=config.problem.alpha)
                    content = grid_interval_content(posterior, intervals)
                else:
                    intervals = model.intervals(toys, alpha=config.problem.alpha)
                    content = neural_interval_content(
                        model._probabilities(toys), model.edges, intervals
                    )
                hits[identifier] = covered(intervals, truth)
                methods[identifier]["coverage"].append(
                    {"theta": point.tolist(), **interval_summary(intervals, truth, content=content)}
                )
        if len(hits) == 2:
            # Both posteriors scored the same toys, so the difference has a paired SE.
            difference = hits["amortized"].astype(float) - hits["exact_grid"]
            paired_coverage.append(
                {"theta": point.tolist(), "coverage_difference": metric(difference)}
            )
    methods["amortized"]["paired_coverage_vs_exact_grid"] = paired_coverage
    for entry in methods.values():
        worst = max(entry["frequentist_risk"], key=lambda row: row["squared"]["value"])
        entry["max_grid_risk"] = {"theta": worst["theta"], "squared": worst["squared"]}
        # Every method has the b = known_b slice, so this maximum compares like with like.
        common = [
            row for row in entry["frequentist_risk"] if row["theta"][1] == config.problem.known_b
        ]
        worst = max(common, key=lambda row: row["squared"]["value"])
        entry["max_grid_risk_common_slice"] = {"theta": worst["theta"], "squared": worst["squared"]}
    print("Frequentist risk curves complete", flush=True)
    # Stage 4: analysis-prior sensitivity using the stage-2 draws.
    priors.update(
        {
            "zero_mass_0.1": reality_prior(config, zero_mass=0.1),
            "zero_mass_0.5": reality_prior(config, zero_mass=0.5),
            "cutoff_0.03": reality_prior(config, mu_min=0.03),
            "cutoff_0.3": reality_prior(config, mu_min=0.3),
        }
    )
    sensitivity = {}
    for name, prior in priors.items():
        if name == "reality":
            prediction = predictions["exact_grid"]
            result = methods["exact_grid"]["bayes_risk"]
        else:
            prediction = exact_posterior(x, prior=prior, config=config).mean
            result = risks(theta[:, 0], prediction)
        # Paired on the same draws: against the correct-prior analysis and the clipped MLE.
        paired = {}
        if name != "reality":
            paired["vs_reality"] = paired_comparison(
                theta[:, 0], prediction, predictions["exact_grid"], excess=True
            )
        paired["vs_clipped_mle"] = paired_comparison(
            theta[:, 0], prediction, predictions["clipped_mle"]
        )
        sensitivity[name] = {
            "analysis_prior": prior_metadata(name, prior),
            "bayes_risk": result,
            "paired": paired,
        }
    print("Analysis-prior sensitivity complete", flush=True)
    # Stage 5: simulation-based calibration for four posterior samplers.
    sbc = {}
    samplers = {
        "gamma_plugin": GammaPluginPosterior(config),
        "laplace": LaplacePosterior(config),
        "amortized": model,
        "exact_grid": ExactSampler(config),
    }
    for name, sampler in samplers.items():
        ranks = simulation_based_calibration(
            sampler, rng=np.random.default_rng(seeds["sbc"]), config=config
        )
        # Ten bins over ranks 0..D. Bins hold unequal numbers of integer ranks, so
        # the expected counts under uniformity are computed bin by bin.
        bins = np.linspace(-0.5, config.mc.n_posterior_draws + 0.5, 11)
        histogram, _ = np.histogram(ranks, bins=bins)
        uniform_counts, _ = np.histogram(np.arange(config.mc.n_posterior_draws + 1), bins=bins)
        expected = uniform_counts / (config.mc.n_posterior_draws + 1) * len(ranks)
        chi = chisquare(histogram, expected)
        sbc[name] = {
            "n_samples": len(ranks),
            "n_draws": config.mc.n_posterior_draws,
            "ranks": ranks.tolist(),
            "histogram": histogram.tolist(),
            "bins": bins.tolist(),
            "histogram_proportions": [
                metric((ranks >= bins[i]) & (ranks < bins[i + 1]), probability=True)
                for i in range(10)
            ],
            "tie_handling": "uniform integer from 0 through tied draws, inclusive",
            "uniform_expected_counts": expected.tolist(),
            "chi_square": float(chi.statistic),
            "chi_square_p_value": float(chi.pvalue),
            "p_value_underflow": bool(chi.pvalue == 0),
            "uniformity_rejected_at_0.05": bool(chi.pvalue < 0.05),
        }
    # Stage 6: grid-quadrature convergence on 64 observations.
    # Quadrature check: halve mu_step and use 2n - 1 background and efficiency nodes.
    rng = np.random.default_rng(seeds["convergence"])
    _, check_x = sample_joint(64, rng=rng, config=config)
    coarse = exact_posterior(check_x, config=config)
    refined = replace(
        config,
        grid=replace(
            config.grid,
            mu_step=config.grid.mu_step / 2,
            b_points=2 * config.grid.b_points - 1,
            eps_points=2 * config.grid.eps_points - 1,
        ),
    )
    fine = exact_posterior(check_x, config=refined)
    convergence = {
        "n_observations": len(check_x),
        "refined_grid": asdict(refined.grid),
        "max_abs_mean_difference": float(np.max(np.abs(coarse.mean - fine.mean))),
        "max_abs_variance_difference": float(np.max(np.abs(coarse.variance - fine.variance))),
        "mean_tolerance": 0.005,
        "variance_tolerance": 0.02,
        "passed": bool(
            np.max(np.abs(coarse.mean - fine.mean)) < 0.005
            and np.max(np.abs(coarse.variance - fine.variance)) < 0.02
        ),
    }
    # Stage 7: save figures and the result record.
    figure_paths = draw_figures(
        {
            "config": asdict(config),
            "methods": methods,
            "diagnostics": {"prior_sensitivity": sensitivity, "sbc": sbc},
        }
    )
    result = {
        "schema_version": 1,
        "path": "a",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "config": asdict(config),
        "runtime_seconds": perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "methods": methods,
        "diagnostics": {
            "quick": quick,
            "rng_stream_seeds": seeds,
            "quadrature_convergence": convergence,
            "timing_scheme": {
                "batch_size": len(timing_x),
                "repeats": 7,
                "warmup_runs": 1,
                "summary": "median elapsed time per observation; excludes training",
            },
            "prior_sensitivity": sensitivity,
            "sbc": sbc,
            "neural_vs_exact_posterior": neural_comparison,
            "frequentist_risk_domain": {
                "theta": risk_theta.tolist(),
                "n_samples_per_point": config.mc.n_toys_cov,
                "nuisance_treatment": f"fixed b={','.join(f'{b:g}' for b in sorted(backgrounds))} "
                f"and eps={config.problem.eps0:g}; reference only b={config.problem.known_b:g}",
                "method_background_slices": {
                    name: ([config.problem.known_b] if name == "exact_grid" else list(backgrounds))
                    for name in methods
                },
            },
            "count_truncation_mass": 0.0,
            "efficiency_integration": "Gaussian quadrature of unbounded log efficiency",
            "neural_distribution": "zero atom plus 100 uniform positive cells, learned by an MLP",
        },
        "artifacts": {
            "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
            "model": str(model_path.relative_to(ROOT)),
        },
        "notes": [
            "Absolute loss evaluates means; the posterior median is its Bayes rule.",
            "Max risk describes a finite signal grid at fixed nuisances, not a minimax theorem.",
            "Gamma plug-in ignores nuisance uncertainty and the zero atom; its prior is diffuse.",
            "Laplace plugs in nuisances and retains the atom and boundary slope. "
            "It uses a linear tangent for nonpositive local precision and can fail badly.",
            "The learned neural distribution approximates the posterior; SBC can reject it.",
            "HPD output ranks zero-volume atoms first, then continuous cell densities. Summaries "
            "describe its inclusive hull, while the raw accepted mask can be disconnected.",
            "SBC checks prior-averaged calibration and does not establish fixed-theta coverage.",
            "Zero SBC p-values represent floating-point underflow; figures show p<1e-300.",
            "Risk comparisons use common observations across estimators and analysis priors.",
            "Gaussian efficiency quadrature does not impose a physical efficiency bound.",
            "All four SBC samplers use one seed, so they are scored on the same "
            f"{config.mc.n_sbc} prior draws (common random numbers); their p-values are "
            "not independent tests.",
            "Paired fields use the SE of per-draw differences on shared draws or toys. "
            "excess_squared_risk is E[(d - d_ref)^2]; it equals the excess squared Bayes "
            "risk because the reference, the reality-prior grid posterior mean, is the "
            "Bayes rule up to quadrature error.",
            "posterior_content is the mass that the reporting posterior puts inside its own "
            "equal-tailed interval. When the zero atom holds at least alpha/2, the interval "
            "starts at 0 and holds at least 1 - alpha/2.",
            f"max_grid_risk_common_slice uses only the b={config.problem.known_b:g} rows, "
            "which every method has; max_grid_risk uses each method's own slices.",
            "neural_vs_exact_posterior bins the grid posterior into the network's classes "
            "and reports KL(exact || neural) on the Bayes-risk draws.",
        ],
    }
    if model.training_seconds >= 60:
        result["notes"].append(
            "Neural training exceeded the requested 60-second budget on this machine."
        )
    if not convergence["passed"]:
        result["notes"].append(
            "Quadrature convergence failed the stated tolerances; reference is provisional."
        )
    destination = ROOT / "results/path_a.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination}; experiment wall time {result['runtime_seconds']:.2f}s", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use smaller recorded CPU budgets")
    parser.add_argument("--figures-only", action="store_true", help="redraw saved results only")
    args = parser.parse_args()
    if args.figures_only:
        redraw_saved()
    else:
        run(quick=args.quick)
