"""Draw the accuracy/speed and coverage summary from saved results only."""

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesfreq.plotting import METHOD_COLORS, METHOD_LABELS, save_figure, set_style

ROOT = Path(__file__).resolve().parents[1]


def draw_summary(a: dict, b: dict) -> Path:
    """Plot saved risk estimates and the common b = 3, ε = 1 coverage slice."""
    set_style()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    offsets = {
        "exact_grid": (-10, 15),
        "amortized": (5, -27),
        "gamma_plugin": (5, 6),
        "laplace": (-5, -27),
        "clipped_mle": (5, 5),
    }
    for key, method in a["methods"].items():
        risk = method["bayes_risk"]["squared"]
        axes[0].errorbar(
            method["inference_us"],
            risk["value"],
            yerr=risk["mc_se"],
            fmt="o",
            capsize=3,
            color=METHOD_COLORS[key],
        )
        axes[0].annotate(
            METHOD_LABELS[key],
            (method["inference_us"], risk["value"]),
            xytext=offsets[key],
            textcoords="offset points",
            fontsize=14,
            ha="right" if key in ("exact_grid", "laplace") else "left",
        )
    axes[0].set(
        xscale="log",
        xlabel="Inference time (µs, log scale)",
        ylabel="Average squared error",
        title="Accuracy vs speed\n(lower-left is better)",
    )
    axes[0].margins(x=0.15, y=0.35)
    for key in ("exact_grid", "amortized", "profile_lr_full", "waldo_full"):
        source = a if key in a["methods"] else b
        rows = [row for row in source["methods"][key]["coverage"] if row["theta"][1:] == [3.0, 1.0]]
        label = METHOD_LABELS[key]
        if source is a:
            label = "Grid credible" if key == "exact_grid" else "Neural credible"
        axes[1].plot(
            [r["theta"][0] for r in rows],
            [r["coverage"]["value"] for r in rows],
            color=METHOD_COLORS[key],
            label=label,
        )
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="True signal μ",
        ylabel="Coverage",
        ylim=(0, 1.04),
        title="Coverage at fixed truth (target 90%)\nBackground b = 3, efficiency ε = 1",
    )
    axes[1].legend(loc="lower left", frameon=False)
    path = save_figure(fig, "summary.png")
    plt.close(fig)
    return path


def main() -> None:
    """Load the saved files and write figures/summary.png."""
    a = json.loads((ROOT / "results/path_a.json").read_text())
    with gzip.open(ROOT / "results/path_b_details.json.gz", "rt") as handle:
        b = json.load(handle)
    draw_summary(a, b)


if __name__ == "__main__":
    main()
