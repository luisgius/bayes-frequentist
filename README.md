# Pragmatic Bayes and Bounded Frequentism

**When data are scarce and the answer cannot be negative, which do you trust: a Bayesian answer that is best on average, or a frequentist answer that is built to hold for every possible truth? This project measures both on one simulator and shows where each one wins and where each one fails.**

## Status

Work in progress. The plan has two parts:

- **Path A** (pragmatic computational Bayes): simulate the experiment from a realistic prior and compare fast inference algorithms by Monte Carlo Bayes risk and compute cost.
- **Path B** (domain-restricted frequentism): build 90% confidence sets for $\mu$ and check their coverage at every point of a grid over the physical range and over a box of nuisance values.

## The question

Bayesian and frequentist statistics are often presented as rival camps. A more useful view is that they answer different questions and give different **guarantees**:

- A **Bayesian** procedure is optimal *on average* over a prior distribution of parameter values.
- A **frequentist** procedure aims at a guarantee *for every* parameter value, for example "this interval contains the true value in at least 90% of repeated experiments".

Many statisticians now combine the two. Rubin (1984) and Little (2006) call this *calibrated Bayes*: use Bayesian inference within a model, and use frequentist calculations to check it. Kass (2011) calls a similar position *statistical pragmatism*.

**The test case.** We use an **on/off counting experiment**, as in gamma-ray astronomy or radiation detection.

- The *on* region contains the source. We observe $n_{\text{on}}$ counts from signal plus background.
- The *off* region contains no source. We observe $n_{\text{off}}$ background-only counts. Its exposure is $\tau = 5$ times larger than the on region.
- A separate calibration gives a noisy measurement $\hat\varepsilon$ of the detector efficiency $\varepsilon$.

The unknowns are the **signal** $\mu \in [0, 10]$ (the quantity we want), the **background** $b$ and the **relative efficiency** $\varepsilon$ (two *nuisance parameters*: unknowns that we must account for but do not care about). The upper bound $\mu_{\max} = 10$ stands for a known maximum source activity or detector saturation.

**Why the boundary matters.** A source cannot emit a negative number of photons, so $\mu \ge 0$. Near $\mu = 0$, textbook frequentist intervals can be empty or negative. Choosing between an upper limit and a two-sided interval after seeing the data ("flip-flopping") breaks the coverage guarantee (Feldman & Cousins 1998). The prior also matters most near the boundary, because the data say little there. Note that a constraint does not *force* a Bayesian analysis: frequentist methods can use constraints too, and a prior adds information beyond the constraint (Stark 2015). This repository shows, with numbers and Monte Carlo errors, where the two views agree and where they differ.

We study two cases:

| Case | Background $b$ | Efficiency $\varepsilon$ | Purpose |
|---|---|---|---|
| **Core** | known, $b = 3$ | fixed, $\varepsilon = 1$, no calibration noise | Exact checks against the published Feldman–Cousins tables |
| **Full** | unknown, $b \in [0.5, 10]$ | uncertain, log-normal with log-scale SD 0.1 | Realistic analysis with nuisance parameters |

This problem is a standard test bed in physics intervals (Feldman & Cousins 1998; Rolke et al. 2005) and in machine-learning-based frequentist inference (Dalmasso et al. 2024; Al Kadhim et al. 2023). It is cheap to simulate, so Monte Carlo errors stay small. See [`docs/literature_review.
## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
make setup   # create .venv and install the pinned dependencies and the package
make test    # fast tests
make lint    # ruff check and ruff format --check
```

## License

MIT License, copyright (c) 2026 Luis. See [LICENSE](LICENSE).
