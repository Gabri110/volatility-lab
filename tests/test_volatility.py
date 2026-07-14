"""Spec for garch_lab.volatility.rolling_vol — guide §3.2.

Make these green at notebook 01 §3 (roadmap box S3).
"""

import numpy as np
import pandas as pd

from garch_lab.volatility import rolling_vol


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_constant_series_no_mean_subtraction():
    # ALL returns +1%. This fixture discriminates between the two candidate
    # formulas: root-mean-square of r gives 0.01; a mean-subtracted rolling
    # std gives 0.0. Guide §3.2 explains why the mean is dropped.
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
    # σ scales with the SQUARE ROOT of the horizon (variances add over
    # uncorrelated periods — §2.2 earned you that); 252 is the variance factor
    assert np.allclose(ratio.dropna(), np.sqrt(252.0))


def test_warmup_and_index():
    rng = np.random.default_rng(2)
    r = _series(rng.normal(0.0, 0.01, 50))
    v = rolling_vol(r, window=21)
    assert v.index.equals(r.index)
    assert v.iloc[:20].isna().all()   # window-1 warm-up NaNs...
    assert v.iloc[20:].notna().all()  # ...because the value at t MAY use r_t:
    #                                   an estimate at t, not a forecast for t
