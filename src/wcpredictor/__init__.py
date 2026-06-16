"""World Cup 2026 predictor package.

Provides interpretable statistical models (Elo and Poisson) and a Monte Carlo
tournament simulator for the 48-team 2026 FIFA World Cup.
"""

from wcpredictor.config import Config, default_config

__all__ = ["Config", "default_config", "__version__"]
__version__ = "0.1.0"
