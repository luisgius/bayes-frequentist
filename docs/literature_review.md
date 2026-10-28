# Literature review: pragmatic Bayes and bounded frequentism

*Status: checked against sources opened on 2026-10-12; three citations were
rechecked on 2026-10-20. Citation keys such as `[@feldman1998unified]` refer to
`docs/references.bib`.*

## 1. The question behind this repository

People often present Bayesian and frequentist statistics as two rival camps. Many
statisticians now see the choice differently: the real question is *which guarantee
you need*. A Bayesian answer is optimal on average over a prior. A frequentist answer
gives a guarantee for every possible value of the parameter. This review explains
where that view comes from and which recent work makes it practical. It also shows
why a small counting experiment with a physical boundary is a good test case.

## 2. Pragmatic and calibrated Bayes

Rubin argued that an applied Bayesian should also use frequency calculations. They
are useful for checking models and procedures, not only for designing them
[@rubin1984bayesianly]. Little called this compromise *calibrated Bayes*. Inference
under a given model should be Bayesian, while model assessment "can and should
involve frequentist ideas" [@little2006calibrated]. Kass described a similar
position as *statistical pragmatism*. In his view the field has moved beyond the old
controversies, and the important parts are the assumptions that link models to data
[@kass2011big]. Gelman, discussing Kass, added *calibration* as one of the anchors
of probability [@gelman2011pragmatism]. In particle physics, Cousins reviewed why
most analyses have traditionally used the classical framework, and noted that
Bayesian reasoning had begun to make inroads [@cousins1995why]. His later lectures
give a longer introduction [@cousins2018lectures].

**What we take from this:** we evaluate every procedure under both criteria. For a
Bayesian estimator we compute frequentist risk and coverage. For a frequentist
procedure we compute Bayes risk under a prior.

## 3. Small signals near a physical boundary

### 3.1 The Feldman–Cousins unified approach

Take a counting experiment that observes $n \sim \text{Poisson}(\mu + b)$, where the
signal is $\mu \ge 0$ and the background is $b$. If $n$ is smaller than $b$, the
naive estimate $n - b$ is negative. Textbook intervals can then be unphysical.
Feldman and Cousins (FC) also found a subtler problem, which they called
*flip-flopping*. Choosing between an upper limit and a two-sided interval *after*
seeing the data breaks the coverage guarantee [@feldman1998unified].

Their fix is a Neyman construction with a new ordering rule. For each $\mu$, values
of $n$ enter the acceptance region in decreasing order of the likelihood ratio
$R = P(n\mid\mu)/P(n\mid\mu_{\text{best}})$. Here $\mu_{\text{best}} = \max(0, n-b)$
is the best *physically allowed* fit. This gives one coherent set of intervals. It
moves smoothly from upper limits (small $n$) to two-sided intervals (large $n$), and
it never returns an empty interval. Because $n$ is discrete, the intervals still
overcover a little; no ordering rule can avoid that. FC published tables of
intervals for many $(n_0, b)$ pairs, which makes the method easy to test. Their
paper also documents two small "pathologies" and how they handled them, which
matters if you want to reproduce the tables exactly. In particular, for a fixed
$n_0$ the plain upper endpoint does not always decrease as $b$ grows, so FC "force
the function to be non-increasing, by lengthening selected confidence intervals as
necessary" [@feldman1998unified, Sec. IV A]. With this correction applied to every
count, this repository's construction matches all 420 entries of Tables IV–V
(90% C.L., $n_0 = 0$ to 20, $b = 0$ to 15) within 0.01 (see `docs/DESIGN.md`).

### 3.2 Critiques and alternatives

The FC tables show that the upper limit for $n_0 = 0$ gets *smaller* as the
expected background grows. At 90% C.L. it falls from 2.44 at $b = 0$ to 0.92 at
$b = 15$ [@feldman1998unified, Tables IV–V]. Roe and Woodroofe proposed an
improvement to FC. It uses the condition that the number of background events cannot
exceed the total number of events [@roe1999improved]. Later they proposed Bayesian
belts with a uniform prior, and found that these have good frequentist properties
[@roe2001belts]. Mandelkern
reviewed the bounded-parameter problem for statisticians. He showed that the
competing methods give substantially different intervals when the observation is
improbable [@mandelkern2002bounded].

### 3.3 Unknown background: the on/off problem

In practice $b$ is not known. It is estimated from an "off" region with no signal:
$n_{\text{off}} \sim \text{Poisson}(\tau b)$, where $\tau$ is the known exposure
ratio. In gamma-ray astronomy, the standard significance formula for this on/off
set-up comes from Li and Ma [@li1983analysis]. Rolke, López and Conrad (RLC) studied
profile-likelihood intervals for the same model. They also added an uncertain
efficiency, $n_{\text{on}} \sim \text{Poisson}(\varepsilon\mu + b)$. Their coverage
was "surprisingly good" over a wide range of parameters [@rolke2005nuisance;
@rolke2001small].

Another tradition averages over the nuisance parameter with a probability
distribution. Cousins and Highland did this for an uncertain normalisation
[@cousins1992highland]. Conrad and co-authors combined it with FC
[@conrad2003systematics]. Cranmer looked at the hypothesis-testing side and compared
several of these recipes [@cranmer2003frequentist]. Cousins, Linnemann and Tucker
then evaluated three families of methods for the on/off problem. They recommend a
frequentist test based on the ratio of Poisson means, which never undercovers but
can be conservative at very low counts. They found the Li–Ma profile-likelihood
formula "a strikingly good approximation" with at most modest undercoverage
[@cousins2008three]. On the statistics side, Sen, Walker and Woodroofe extended the
FC method to nuisance parameters and implemented hybrid resampling
[@sen2009unified].

**Lesson:** no nuisance-parameter recipe is guaranteed to cover in small samples. You
have to check coverage over a grid of nuisance values.

## 4. Decision theory on a bounded parameter space

The *frequentist risk* $R(\theta,\delta) = \mathbb E_\theta L(\theta,\delta(X))$
describes how an estimator performs at each true value. The *Bayes risk*
$r(\pi,\delta)$ averages that risk over a prior $\pi$. A *minimax* estimator
minimises the worst-case risk. Bounded parameter spaces make these ideas concrete.
For a normal mean restricted to a short interval, the unique minimax estimator under
squared error is the Bayes rule for a *two-point prior* on the endpoints
[@casella1981bounded]. For long intervals, Bickel found the minimax risk
asymptotically; the least favourable prior is then close to a smooth
cosine-squared shape, not uniform [@bickel1981minimax]. So in the bounded world,
"frequentist worst case" and "Bayes under a well-chosen prior" can lead to the
same estimator.

The same bridge exists for intervals. Pratt studied the expected length of
confidence intervals [@pratt1961length]. Schafer and Stark built confidence sets of
minimax expected size for parameters in a compact set, using simulation
[@schafer2009mes]. Hoff and co-authors construct intervals with *exact* frequentist
coverage that use prior information to become shorter when that information is
accurate [@yu2016multigroup; @hoff2021bayesoptimal]. Stark warns against a common
shortcut, though. A constraint such as $\mu \ge 0$ does not require a Bayesian
analysis. A prior adds information beyond the constraint, and Bayesian and
frequentist uncertainties measure different things [@stark2015constraints].

## 5. Simulation-based inference and its trust problem

Simulation-based inference (SBI) uses a simulator, often together with neural
networks, to learn posteriors or likelihood ratios when the likelihood is hard to
compute [@cranmer2020frontier]. Once trained, an *amortised* estimator gives
answers for new data almost instantly.

Speed is not correctness, though. *Simulation-based calibration* (SBC) checks an
inference algorithm in three steps: draw parameters from the prior, simulate data,
and test whether the rank of the true parameter among the posterior draws is
uniform [@talts2018sbc]. The check assumes that the posterior is computed under the
same prior that generated the parameters. A posterior built on a different prior can
therefore fail SBC even when its computation is exact. Which test quantities you check
changes what SBC can detect
[@modrak2023sbc]. Hermans and co-authors showed that all the common SBI algorithms
they benchmarked can produce *overconfident* posteriors. They called this a "trust
crisis" and found that ensembles help [@hermans2022trust]. Later work trains for
calibration directly [@delaunoy2022bnre; @falkiewicz2023calibrating]. TARP tests
the coverage of generative posterior estimators from samples alone [@lemos2023tarp].
Local diagnostics check calibration at specific observations instead of only on
average [@zhao2021diagnostics; @linhart2023lc2st].

There is one key subtlety. SBC and TARP test calibration *averaged over the prior*.
A posterior can pass these tests and still undercover at particular parameter
values, for example near a boundary where the prior has little mass.

## 6. Frequentist guarantees on top of machine learning

A recent line of work keeps the machine-learning machinery but restores
frequentist guarantees through the Neyman construction.

- **ACORE and LF2I.** Dalmasso, Izbicki and Lee estimate a likelihood-ratio
  statistic with a classifier [@dalmasso2020acore]. LF2I generalises this into a
  modular framework [@dalmasso2024lf2i]. It has three steps: choose any test
  statistic, learn its critical values across the parameter space with quantile
  regression, and estimate the coverage *as a function of the parameter*. In its
  on/off Poisson example, profiling the background overcovers (about 96% at a
  nominal 90%). Marginalising it undercovers in some regions. The diagnostics show
  where each one fails.
- **WALDO.** Masserano and co-authors use the posterior mean and variance from any
  neural estimator to build a Wald-like statistic,
  $(\mathbb E[\theta\mid x]-\theta_0)^2/\mathbb V[\theta\mid x]$. They then calibrate
  its critical values for every $\theta_0$ [@masserano2023waldo]. The paper shows
  that the resulting confidence sets keep their coverage whatever prior was used
  for training, provided the critical values are estimated well. A prior that
  agrees with the data gives smaller sets.
- **Newer work.** ALFFI applies a similar amortised method to the on/off problem
  with real, sparse data, where Wilks' theorem is not reliable [@alkadhim2023alffi].
  Other work handles nuisance parameters and distribution shift in classification
  [@masserano2024nuisance]. The 2025/26 "Frequentist-Bayes" (FreB) protocol turns
  posteriors from generative AI models into locally valid confidence regions
  [@carzon2026freb]. Carzon and co-authors also show how to focus a test's power on
  physics-motivated regions, with machine-learned critical values
  [@carzon2025focusing].

For large samples, asymptotic formulas for likelihood-ratio tests remain the
standard tool [@cowan2011asymptotic]. Our problem is small-count and bounded, so
calibration by simulation is the safer choice.

## 7. Consequences for this repository

1. **Problem.** The on/off counting experiment with $\mu \in [0,\mu_{\max}]$, an
   unknown background, and an efficiency nuisance parameter is the standard test bed
   in all three literatures: physics intervals, decision theory, and SBI with
   guarantees. Exact FC tables let us test the core case.
2. **Path A** should report Bayes risk *and* frequentist risk, and check the neural
   estimator with SBC and, because an exact posterior is available here, directly
   against it. Passing SBC does not prove coverage, and a method with a different
   analysis prior can fail SBC under the reference prior for that reason alone.
3. **Path B** should reproduce FC and calibrate profile-likelihood and WALDO
   statistics by simulation over a $(\mu, b)$ grid. It should report the coverage
   function with Monte Carlo error bars and judge it against a stated pass
   threshold. This gives empirical evidence on a finite grid, not the guarantee
   that the papers prove under their own assumptions.
4. **Reconciliation.** WALDO and its relatives show the pragmatic middle ground. In
   that view, the prior affects how *small* the confidence set is, and accurate
   calibration is what controls its *coverage*. In this repository the coverage pass
   comes from the simulation calibration, which makes any statistic pass at the
   calibration nodes; off the nodes both statistics passed. The profile likelihood
   ratio passes the same checks, and the WALDO sets are not shorter. What the
   study shows is that a fast neural posterior can be plugged into a valid
   procedure. It checks coverage by simulation on finite grids; that
   is empirical evidence, not a proof.
