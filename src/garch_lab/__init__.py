"""garch_lab: the small library behind the two notebooks.

Four modules: `data` loads and caches prices, `returns` converts them to log
returns, `volatility` estimates rolling realized volatility, and `evaluation`
holds the chronological split, the two loss functions and the naive variance
benchmark. Every contract is stated in a docstring and enforced by `tests/`.
"""

from garch_lab.data import load_prices
from garch_lab.evaluation import chronological_split, mse, qlike, rw_variance_forecast
from garch_lab.returns import log_returns
from garch_lab.volatility import TRADING_DAYS_PER_YEAR, rolling_vol

__all__ = [
    "load_prices",
    "log_returns",
    "rolling_vol",
    "TRADING_DAYS_PER_YEAR",
    "chronological_split",
    "mse",
    "qlike",
    "rw_variance_forecast",
]
