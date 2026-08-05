"""Rolling realized volatility.

Spec: tests/test_volatility.py.
"""

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def rolling_vol(r: pd.Series, window: int = 21, annualize: bool = False) -> pd.Series:
    """Rolling volatility: root of the mean of the last `window` squared returns.

    Contract (enforced by tests/test_volatility.py):
    - the value at date t uses r_{t-window+1} ... r_t, with t included. This
      is an *estimate at* t, not a forecast for t. Contrast
      rw_variance_forecast in evaluation.py, whose convention differs by
      exactly one day.
    - the mean of r is NOT subtracted: the daily mean return is roughly two
      orders of magnitude below daily σ, so estimating it adds more noise
      than the correction removes. With no mean subtracted there is no lost
      degree of freedom, which is why the divisor is n rather than n-1.
    - output index equals the input index; the first window-1 values are NaN
    - annualize=True multiplies by √252, the factor for a standard
      deviation. The factor for a variance is 252.
    """

    r2 = r ** 2
    vol = r2.rolling(window=window).mean() ** 0.5
    if annualize:
        vol *= (TRADING_DAYS_PER_YEAR ** 0.5)

    return vol
