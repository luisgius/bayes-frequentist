"""bayesfreq: Bayes risk and bounded frequentist coverage for an on/off counting experiment.

Modules
-------
config
    Frozen study constants (``FULL_CONFIG``, ``CORE_CONFIG``).
"""

from bayesfreq.config import CORE_CONFIG, FULL_CONFIG, StudyConfig

__all__ = ["CORE_CONFIG", "FULL_CONFIG", "StudyConfig"]
__version__ = "0.1.0"
