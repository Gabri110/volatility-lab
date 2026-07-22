"""Rolling realized volatility — YOUR implementation.

Spec: tests/test_volatility.py. Theory: guide §3.2 (docs/concept-guide.md).
"""

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def rolling_vol(r: pd.Series, window: int = 21, annualize: bool = False) -> pd.Series:
    """Rolling volatility: root of the mean of the last `window` squared returns.

    Contract (enforced by tests/test_volatility.py):
    - the value at date t uses r_{t-window+1} ... r_t — t INCLUDED. This is
      an *estimate at* t, not a forecast for t; contrast rw_variance_forecast
      in evaluation.py, whose convention differs by exactly one day.
    - the mean of r is NOT subtracted (guide §3.2 says why it's dropped)
    - output index equals the input index; the first window-1 values are NaN
    - annualize=True multiplies by the correct factor for a σ — careful:
      the factor for a standard deviation is not the factor for a variance.

    Delete the `raise` and write the body.
    """

    r2 = r ** 2
    vol = r2.rolling(window=window).mean() ** 0.5
    if annualize:
        vol *= (TRADING_DAYS_PER_YEAR ** 0.5)

    return vol
