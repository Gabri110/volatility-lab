"""Forecast evaluation: the chronological split, two losses, one benchmark.

Spec: tests/test_evaluation.py. None of these four functions needs a model,
so they can be written and tested before anything is fitted.
"""

import pandas as pd
import numpy as np


def chronological_split(s: pd.Series, train_frac: float = 0.7) -> tuple[pd.Series, pd.Series]:
    """Split a series into (train, test) by time, never randomly.

    Contract (enforced by tests/test_evaluation.py):
    - requires 0 < train_frac < 1, else ValueError
    - train is the first int(train_frac * len(s)) observations, in order
    - no shuffling, no overlap, no dropped observations:
      pd.concat([train, test]) must equal s exactly

    With serially dependent data a random split leaks the test set's
    neighborhood into training, so the split is always by time.
    """
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac should be between 0 and 1 (NOT included).")
    split_point = int(train_frac * len(s))
    s_train = s[:split_point]
    s_test = s[split_point:]
    return s_train, s_test


def mse(proxy: pd.Series, forecast: pd.Series) -> float:
    """Mean squared error between a variance proxy and a variance forecast.

    Contract:
    - proxy and forecast must have identical indexes, else ValueError.
      pandas would otherwise silently align on the union and inject NaNs,
      a bug that produces a number instead of an error.
    - returns the time-average of (proxy - forecast)² as a float
    """
    if not proxy.index.equals(forecast.index):
        raise ValueError("Indexes of proxy and forecast must be identical.")
    return float(((proxy - forecast) ** 2).mean())


def qlike(proxy: pd.Series, forecast: pd.Series) -> float:
    """Mean QLIKE loss:  L(p, h) = p/h - ln(p/h) - 1,  averaged over time.

    Contract:
    - identical indexes required, else ValueError
    - every proxy and forecast value must be strictly positive, else
      ValueError. Zero-return days exist, so ln(0) is a real hazard, and the
      decision of what to do with them (drop with disclosure, or clip)
      belongs at the call site in the notebook, not silently in here.
    - qlike(p, p) == 0, and under-forecasting is punished more than
      over-forecasting, which is the asymmetry a risk manager wants.

    This is the shifted form of QLIKE. Patton (2011) usually writes it as
    ln(h) + p/h; the two differ by (ln p + 1), which depends on p but not on
    h, so model rankings are identical. The shifted form is used here
    because it is 0 at a perfect forecast, but absolute levels will not
    match tables computed with the unshifted version.
    """
    if not proxy.index.equals(forecast.index):
        raise ValueError("Indexes of proxy and forecast must be identical.")
    if (proxy <= 0).any() or (forecast <= 0).any():
        raise ValueError("All proxy and forecast values must be strictly positive.")
    return float((proxy / forecast - (proxy / forecast).apply(np.log) - 1).mean())


def rw_variance_forecast(r: pd.Series, window: int = 21) -> pd.Series:
    """Random-walk variance benchmark: a forecast FOR day t, from the past only.

    Contract:
    - the value at date t is the mean of r² over the `window` days strictly
      before t, with t itself excluded. This is a *forecast for* t. Contrast
      rolling_vol, which is an *estimate at* t. Same arithmetic, different
      timestamp discipline: one shift, and one extra NaN.
    - output index equals the input index; the first `window` values are
      NaN, one more than rolling_vol's warm-up
    - perturbing r_t must not change the value at t. The tests check this by
      shocking one day and asserting the forecast for that day is unmoved
      while the next day's reacts, which is the no-look-ahead property.

    Note on the name: this is a 21-day moving-average variance forecast.
    The random-walk benchmark in the forecasting literature is usually the
    last observed squared return, h_t = p_{t-1}.
    """
    r2 = np.square(r)
    forecast = r2.shift(1).rolling(window=window).mean()
    return forecast
