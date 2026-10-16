# Pragmatic Bayes and Bounded Frequentism

**When data are scarce and the answer cannot be negative, which do you trust: a Bayesian answer that is best on average, or a frequentist answer that is built to hold for every possible truth? This project measures both on one simulator and shows where each one wins and where each one fails.**

[![CI](https://github.com/luisgius/bayes-frequentist/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/luisgius/bayes-frequentist/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Status

Work in progress. The plan has two parts:

- **Path A** (pragmatic computational Bayes): simulate the experiment from a realistic prior and compare fast inference algorithms by Monte Carlo Bayes risk and compute cost.
- **Path B** (domain-restricted frequentism): build 90% confidence sets for $\mu$ and check their coverage at every point of a grid over the physical range and over a box of nuisance values.

## 1. The question

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

This problem is a standard test bed in physics intervals (Feldman & Cousins 1998; Rolke et al. 2005) and in machine-learning-based frequentist inference (Dalmasso et al. 2024; Al Kadhim et al. 2023). It is cheap to simulate, so Monte Carlo errors stay small. See [`docs/literature_review.md`](docs/literature_review.md) for the full background.

## 2. Key terms in plain words

- **Prior and posterior:** the prior describes what we believe about the unknowns before seeing the data; the posterior is the updated belief after seeing them.
- **Bayes risk:** the average error of an estimate, averaged over many plausible true values drawn from a prior. Here the error is squared error. Lower is better.
- **Coverage:** how often an interval contains the true value if the experiment is repeated with the truth held fixed. A 90% confidence procedure aims for at least 90% coverage at *every* true value; here we check this on a finite grid of true values.
- **Worst-case coverage:** the lowest coverage over all checked true values: the frequentist success criterion.
- **Credible interval:** a Bayesian interval that holds 90% of the posterior probability. *Equal-tailed* means 5% of the probability lies on each side.
- **Confidence set:** a frequentist set of signal values, built so that it targets 90% coverage at every true value.
- **Nuisance parameter:** an unknown that we do not care about but must account for (here the background $b$ and the detector efficiency $\varepsilon$).
- **Reality prior:** the distribution the simulator uses to draw "true worlds" (30% no source, 70% a source of varying strength).
- **Feldman–Cousins (FC):** a standard 1998 physics recipe for 90% intervals that respect the boundary $\mu \ge 0$ and are never empty.
- **Profile likelihood ratio:** a test that handles nuisance parameters by re-fitting them for each candidate signal value.
- **WALDO:** a 2023 method that turns a posterior (here the neural one) into a confidence set, by calibrating its test statistic with simulations so that it targets frequentist coverage.
- **Amortized neural posterior:** a network trained once on simulated data. It then returns a posterior for any new data set in microseconds.
- **SBC (simulation-based calibration):** if a posterior is computed correctly for the prior that generated the data, the true value looks like a random draw from it. SBC checks this over many simulated data sets.
- **Toys:** simulated repeat experiments, used to estimate coverage.
- **Hull:** the shortest single interval that contains a confidence set (some sets have small gaps).
- **MC SE:** Monte Carlo standard error, the uncertainty that comes from using a finite number of simulations. Every "±" in this README is one MC SE.

## 3. Setup and success criteria

**Model.** In the full case, given the parameters $\theta = (\mu, b, \varepsilon)$, the three observations $x = (n_{\text{on}}, n_{\text{off}}, \hat\varepsilon)$ are independent:

$$
n_{\text{on}} \sim \mathrm{Poisson}(\varepsilon\mu + b), \qquad
n_{\text{off}} \sim \mathrm{Poisson}(\tau b), \qquad
\log\hat\varepsilon \sim \mathcal N(\log\varepsilon,\ \sigma_\varepsilon^2),
$$

with $\tau = 5$ and $\sigma_\varepsilon = 0.1$. In the core case, $b = 3$ is known, $\varepsilon = 1$ and $\sigma_\varepsilon = 0$, so $\hat\varepsilon = 1$ and only $n_{\text{on}} \sim \mathrm{Poisson}(\mu + 3)$ carries information. The physical signal space is $\Theta = [0, \mu_{\max}]$ with $\mu_{\max} = 10$.

**Nuisance box.** The frequentist coverage checks use only points inside the box $\mathcal B = [0.5, 10] \times [e^{-0.2}, e^{0.2}]$ for $(b, \varepsilon)$, that is $\varepsilon \in [0.819, 1.221]$. We make no coverage claim outside this box.

**Reality prior.** The simulator draws the three parameters independently, from a prior $\pi$ that encodes "no source or a weak source is common":

$$
\mu \sim 0.3\,\delta_0 + 0.7\,\mathrm{LogUniform}[0.1, 10], \qquad
b \sim \mathrm{Uniform}[0.5, 10], \qquad
\log\varepsilon \sim \mathcal N(0, 0.1^2),
$$

where $\delta_0$ is a point mass at $\mu = 0$. The efficiency prior has no upper bound; the nuisance box limits only the coverage checks, not the simulator.

**Loss and risk.** An estimator $\delta$ maps data $x$ to an estimate $\delta(x)$ of $\mu$. The primary loss is squared error, $L(\mu, a) = (a - \mu)^2$; we also report absolute error $\lvert a - \mu \rvert$.

- The **frequentist risk** is the expected loss at one fixed parameter value:
  $R(\theta, \delta) = \mathbb E_{X \mid \theta}\, L\big(\mu, \delta(X)\big)$.
- The **Bayes risk** averages the risk over the prior:
  $r(\pi, \delta) = \mathbb E_{\theta \sim \pi}\, R(\theta, \delta)$.
  Under squared loss, the posterior mean $\mathbb E[\mu \mid x]$ minimizes it.

We estimate $r$ by Monte Carlo with $M = 20{,}000$ draws $(\theta_i, x_i)$:

$$
\hat r = \frac{1}{M}\sum_{i=1}^{M} L\big(\mu_i, \delta(x_i)\big), \qquad
\mathrm{SE}(\hat r) = \frac{s_L}{\sqrt M},
$$

where $s_L$ is the sample standard deviation of the losses.

**Coverage and the pass rule.** An interval procedure maps $x$ to a set $\mathrm{CI}(x) \subseteq \Theta$. Its coverage function is

$$
C(\theta) = P_\theta\big(\mu \in \mathrm{CI}(X)\big).
$$

We target $1 - \alpha = 0.90$ and evaluate coverage on finite grids. A procedure passes our empirical check when its estimated coverage is at least $1 - \alpha - \text{tol}$ at every checked point. In the core case we compute $C$ exactly by summing over counts, with $\text{tol} = 0.005$. Otherwise we use $N = 4{,}000$ simulated experiments ("toys") per grid point, with binomial standard error $\sqrt{\hat C (1 - \hat C)/N} \approx 0.0047$ and $\text{tol} = 0.015$ (about 3 standard errors), so the pass threshold is 0.885. Points below the threshold are rerun with 40,000 toys before they count as failures. A pass is empirical evidence on a finite grid, not a proof of 90% coverage everywhere in the continuous parameter space. For a Bayesian credible interval, $C(\theta)$ is a measured property, not a design target.

<details>
<summary><b>Method details: formulas for Feldman–Cousins, profile likelihood, WALDO and calibration</b></summary>

**Feldman–Cousins (FC) ordering.** For known $b$, the FC method (Feldman & Cousins 1998) builds an acceptance set $A(\mu)$ of counts for each $\mu$. Counts $n$ enter in decreasing order of the likelihood ratio

$$
R_{\text{FC}}(n; \mu) = \frac{P(n \mid \mu)}{P(n \mid \mu_{\text{best}})}, \qquad
\mu_{\text{best}} = \min\big(\max(0,\ n - b),\ \mu_{\max}\big),
$$

until $\sum_{n \in A(\mu)} P(n \mid \mu) \ge 1 - \alpha$. Here $\mu_{\text{best}}$ is the best *physically allowed* fit. The interval for an observed count $n_0$ is the set of $\mu$ with $n_0 \in A(\mu)$.

**Profile likelihood ratio.** With unknown nuisances we use

$$
t_\mu(x) = -2 \log \frac{L\big(\mu, \hat{\hat b}(\mu), \hat{\hat\varepsilon}(\mu)\big)}{L\big(\hat\mu, \hat b, \hat\varepsilon\big)},
$$

where the hatted values maximize the likelihood $L$ with $\hat\mu \in \Theta$, and the double-hatted values maximize it at fixed $\mu$.

**WALDO statistic.** WALDO (Masserano et al. 2023) builds a test statistic from any posterior estimator:

$$
T_{\text{W}}(x; \mu_0) = \frac{\big(\mathbb E[\mu \mid x] - \mu_0\big)^2}{\mathbb V[\mu \mid x]},
$$

where $\mathbb E[\mu \mid x]$ and $\mathbb V[\mu \mid x]$ are the posterior mean and variance. In the full case they come from the Path A neural network; in the core case, from an exact flat-prior posterior. A small floor ($10^{-8}$) keeps the variance positive.

**Calibration.** For a statistic $T$ (profile LR or WALDO) in the full case, the *coarse cutoff* at each node $\mu_j$ of a 101-point signal grid (step 0.1 on $[0, 10]$) is the worst case over the nuisance grid:

$$
c_j = \max_{(b, \varepsilon) \in \mathcal B_{\text{grid}}} q_{1-\alpha}\big(T(X; \mu_j) \mid \mu_j, b, \varepsilon\big).
$$

Here $q_{1-\alpha}$ is a conservative empirical 90% quantile from 5,000 toys (order statistic 4,501), and $\mathcal B_{\text{grid}}$ has 7 background × 5 efficiency values. In the core case, WALDO cutoffs come from exact Poisson sums instead of toys.

**Envelope and inversion.** Before inversion, each coarse cutoff is raised to the maximum of itself and its two neighbours (the end values are repeated), and this envelope is interpolated linearly onto a finer signal grid with step 0.01:

$$
e_j = \max\big(c_{j-1},\ c_j,\ c_{j+1}\big), \qquad
\tilde c(\mu_0) = \text{linear interpolation of } (\mu_j, e_j) \text{ at } \mu_0,
\qquad
\mathrm{CI}(x) = \lbrace \mu_0 \in \Theta_{0.01} : T(x; \mu_0) \le \tilde c(\mu_0) \rbrace ,
$$

where $\Theta_{0.01}$ is the signal grid with step 0.01. Between two nodes, $\tilde c$ is at least the larger of the two original cutoffs. The reported interval is the **hull** of $\mathrm{CI}(x)$: from its smallest to its largest accepted grid value, or empty if no value is accepted. The envelope is a numerical convention that removes isolated low cutoffs at the nodes; it does not prove coverage between nodes.

</details>

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
make setup   # create .venv and install the pinned dependencies and the package
make test    # fast tests
make lint    # ruff check and ruff format --check
```

## License

MIT License, copyright (c) 2026 Luis. See [LICENSE](LICENSE).
