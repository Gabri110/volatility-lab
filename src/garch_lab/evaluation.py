"""Forecast evaluation: losses, split, naive benchmark — YOUR implementation.

Spec: tests/test_evaluation.py. Theory: guide §5 (docs/concept-guide.md).
These four functions are Sunday morning's warm-up, before any model is fit.
"""

import pandas as pd


def chronological_split(s: pd.Series, train_frac: float = 0.7) -> tuple[pd.Series, pd.Series]:
    """Split a series into (train, test) by TIME — never randomly.

    Contract (enforced by tests/test_evaluation.py):
    - requires 0 < train_frac < 1, else ValueError
    - train is the FIRST int(train_frac * len(s)) observations, in order
    - no shuffling, no overlap, no dropped observations:
      pd.concat([train, test]) must equal s exactly

    Guide §5.1: with serially dependent data, a random split leaks the
    test set's neighborhood into training. Delete the `raise` and write it.
    """
    raise NotImplementedError("Implement me, then run: pytest tests/test_evaluation.py")


def mse(proxy: pd.Series, forecast: pd.Series) -> float:
    """Mean squared error between a variance proxy and a variance forecast.

    Contract:
    - proxy and forecast must have IDENTICAL indexes, else ValueError
      (pandas would otherwise silently align on the union and inject NaNs —
      a bug that produces a number instead of an error)
    - returns the time-average of (proxy - forecast)² as a float
    """
    raise NotImplementedError("Implement me, then run: pytest tests/test_evaluation.py")


def qlike(proxy: pd.Series, forecast: pd.Series) -> float:
    """Mean QLIKE loss:  L(p, h) = p/h - ln(p/h) - 1,  averaged over time.

    Contract:
    - identical indexes required, else ValueError
    - EVERY proxy and forecast value must be strictly positive, else
      ValueError. Zero-return days exist, so ln(0) is a real hazard: the
      decision of what to do with them (drop with disclosure, or clip)
      belongs at the call site in the notebook — not silently in here.
    - qlike(p, p) == 0, and under-forecasting is punished more than
      over-forecasting. Guide §5.3 explains why that asymmetry is the
      right one for a risk manager.
    """
    raise NotImplementedError("Implement me, then run: pytest tests/test_evaluation.py")


def rw_variance_forecast(r: pd.Series, window: int = 21) -> pd.Series:
    """Random-walk variance benchmark: a forecast FOR day t, from the past only.

    Contract:
    - the value at date t is the mean of r² over the `window` days
      STRICTLY BEFORE t — t itself excluded. This is a *forecast for* t;
      contrast rolling_vol, which is an *estimate at* t. Same arithmetic,
      different timestamp discipline: that's one shift, and one extra NaN.
    - output index equals the input index; the first `window` values are
      NaN (one MORE than rolling_vol's warm-up)
    - perturbing r_t must not change the value at t — the tests check this
      by shocking one day and asserting the forecast for that day is
      unmoved while the next day's reacts (guide §5.1, look-ahead bias)
    """
    raise NotImplementedError("Implement me, then run: pytest tests/test_evaluation.py")
