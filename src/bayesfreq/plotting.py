"""Shared colorblind-safe figure style and repository-local output paths.

Both experiment scripts and the notebook call ``set_style`` so that all
figures look the same. ``save_figure`` always writes into ``figures/``.
"""

from pathlib import Path

import seaborn as sns
from matplotlib.figure import Figure

from bayesfreq.config import GridConfig, MonteCarloConfig, ProblemConfig, StudyConfig

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


# Related core/full procedures share a colour; estimator colours stay fixed too.
METHOD_COLORS = {
    "fc_core": "#0072B2",
    "profile_lr_full": "#D55E00",
    "waldo_core": "#009E73",
    "waldo_full": "#009E73",
    "wald_core": "#999999",
    "wald_full": "#999999",
    "central_core": "#CC79A7",
    "upper_core": "#E69F00",
    "flat_prior_core": "#56B4E9",
    "exact_grid": "#0072B2",
    "amortized": "#6A3D9A",
    "gamma_plugin": "#E69F00",
    "laplace": "#CC79A7",
    "clipped_mle": "#333333",
}
METHOD_LABELS = {
    "fc_core": "Feldman–Cousins",
    "profile_lr_full": "Profile LR",
    "waldo_core": "WALDO (exact posterior)",
    "waldo_full": "WALDO",
    "wald_core": "Naive Wald",
    "wald_full": "Naive Wald",
    "central_core": "Central Neyman",
    "upper_core": "Upper limit",
    "flat_prior_core": "Flat-prior credible",
    "exact_grid": "Grid posterior",
    "amortized": "Neural posterior",
    "gamma_plugin": "Gamma plug-in",
    "laplace": "Laplace",
    "clipped_mle": "Clipped MLE",
}
PRIOR_LABELS = {
    "reality": "Reality (correct)",
    "flat": "Flat",
    "gamma_diffuse": "Gamma, diffuse",
    "gamma_shifted": "Gamma(5, 0.5)",
    "zero_mass_0.1": "Atom 0.1",
    "zero_mass_0.5": "Atom 0.5",
    "cutoff_0.03": "Cutoff 0.03",
    "cutoff_0.3": "Cutoff 0.3",
}


def _tuple_sequences(fields: dict) -> dict:
    """Return a copy of a config section with JSON lists turned back into tuples."""
    return {
        key: tuple(value) if isinstance(value, list) else value for key, value in fields.items()
    }


def saved_config(record: dict) -> StudyConfig:
    """Reconstruct the study config from a saved results record.

    Parameters
    ----------
    record : dict
        Parsed results JSON whose ``config`` entry was written with
        ``dataclasses.asdict``.

    Returns
    -------
    StudyConfig
        Equal to the config that wrote the record, and hashable. JSON stores
        tuples as lists, so sequence fields are turned back into tuples, as
        ``AmortizedPosterior.load`` does.
    """
    config = record["config"]
    return StudyConfig(
        problem=ProblemConfig(**_tuple_sequences(config["problem"])),
        grid=GridConfig(**_tuple_sequences(config["grid"])),
        mc=MonteCarloConfig(**_tuple_sequences(config["mc"])),
        seed=config["seed"],
    )


def set_style() -> None:
    """Apply seaborn ticks style, colorblind palette, and 150 dpi defaults.

    Returns
    -------
    None

    Notes
    -----
    Call once at the start of an experiment or notebook. Importing this module
    does not modify matplotlib's global settings.
    """
    sns.set_theme(
        style="ticks",
        context="notebook",
        palette="colorblind",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        },
    )


def save_figure(fig: Figure, name: str) -> Path:
    """Save fig at 150 dpi under figures/ and return its absolute path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save. It remains open for notebook display.
    name : str
        Basename, e.g. path_a_risk.png. With no suffix, .png is appended.
        Nested or absolute paths are rejected to keep figures in figures/.

    Returns
    -------
    pathlib.Path
        Saved figure path, independent of the process working directory.

    Raises
    ------
    ValueError
        If ``name`` is empty or contains a directory part.
    """
    candidate = Path(name)
    if candidate.name != name or name in ("", ".", ".."):
        raise ValueError("name must be a nonempty figure basename")
    if not candidate.suffix:
        candidate = candidate.with_suffix(".png")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destination = FIGURES_DIR / candidate
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    return destination
