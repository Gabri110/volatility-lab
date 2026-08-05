"""Spec for garch_lab.volatility.rolling_vol."""

import numpy as np
import pandas as pd

from garch_lab.volatility import rolling_vol


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_constant_series_no_mean_subtraction():
    # All returns +1%. This fixture discriminates between the two candidate
    # formulas: root-mean-square of r gives 0.01, while a mean-subtracted
    # rolling std gives 0.0. The mean is deliberately not subtracted.
    r = _series([0.01] * 30)
    v = rolling_vol(r, window=5)
    assert np.allclose(v.dropna(), 0.01)


def test_sign_invariance():
    # volatility measures magnitude; +1%/-1% alternation is constant vol
    r = _series([0.01, -0.01] * 50)
    assert np.allclose(rolling_vol(r, window=5).dropna(), 0.01)


def test_annualization_is_sqrt252_not_252():
    rng = np.random.default_rng(1)
    r = _series(rng.normal(0.0, 0.01, 100))
    ratio = rolling_vol(r, 21, annualize=True) / rolling_vol(r, 21, annualize=False)
    # σ scales with the square root of the horizon, because variances add
    # over uncorrelated periods. 252 is the factor for a variance.
    assert np.allclose(ratio.dropna(), np.sqrt(252.0))


def test_warmup_and_index():
    rng = np.random.default_rng(2)
    r = _series(rng.normal(0.0, 0.01, 50))
    v = rolling_vol(r, window=21)
    assert v.index.equals(r.index)
    assert v.iloc[:20].isna().all()   # window-1 warm-up NaNs, and only that
    assert v.iloc[20:].notna().all()  # many, because the value at t may use
    #                                   r_t: an estimate at t, not a forecast
