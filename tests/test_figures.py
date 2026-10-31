"""Saved-result redraws must never invoke experiments or modify result files."""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from bayesfreq import bayesian_risk, data_simulator, frequentist_bounds, plotting
from bayesfreq.config import CORE_CONFIG, FULL_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def result_hashes():
    """Include every saved file so additions and removals also fail the check."""
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "results").rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("script", "args", "expected"),
    [
        ("run_path_a.py", ["--figures-only"], 4),
        ("run_path_b.py", ["--figures-only"], 4),
        ("make_summary_figure.py", [], 1),
    ],
)
def test_saved_redraw_cli_never_runs_experiments(monkeypatch, tmp_path, script, args, expected):
    before = result_hashes()

    def forbidden(*args, **kwargs):
        pytest.fail("saved-result redraw invoked simulation, training or calibration")

    for module, names in (
        (data_simulator, ("simulate", "sample_joint")),
        (bayesian_risk, ("train_amortized", "exact_posterior")),
        (frequentist_bounds, ("calibrate_belt", "estimate_coverage", "exact_waldo_belt")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(plotting, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", [script, *args])
    run_path(str(ROOT / "scripts" / script), run_name="__main__")
    assert result_hashes() == before
    assert len(list(tmp_path.glob("*.png"))) == expected
    assert all(path.stat().st_size > 1000 for path in tmp_path.glob("*.png"))


@pytest.mark.parametrize("config", [FULL_CONFIG, CORE_CONFIG])
def test_saved_config_round_trip_is_equal_and_hashable(config):
    record = json.loads(json.dumps({"config": asdict(config)}))
    rebuilt = plotting.saved_config(record)
    assert rebuilt == config
    assert hash(rebuilt) == hash(config)


def test_saved_config_from_saved_path_a_results_matches_full_config():
    rebuilt = plotting.saved_config(json.loads((ROOT / "results/path_a.json").read_text()))
    assert rebuilt == FULL_CONFIG
    # The lru_cache in the grid posterior hashes the config, so this call must not raise.
    exact = bayesian_risk.exact_posterior(
        np.array([[3.0, 15.0, 1.0]]), config=rebuilt, prior=data_simulator.reality_prior(rebuilt)
    )
    assert exact.mean.shape == (1,)
