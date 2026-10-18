"""bayesfreq: Bayes risk and bounded frequentist coverage for an on/off counting experiment.

Modules
-------
config
    Frozen study constants (``FULL_CONFIG``, ``CORE_CONFIG``).
data_simulator
    Priors, the shared simulator, and exact likelihoods.
interfaces
    Shared result records and structural protocols.
bayesian_risk
    Path A: posterior approximations, Bayes risk, frequentist risk, and SBC.
"""

from bayesfreq.config import CORE_CONFIG, FULL_CONFIG, StudyConfig

__all__ = ["CORE_CONFIG", "FULL_CONFIG", "StudyConfig"]
__version__ = "0.1.0"
