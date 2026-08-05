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


def test_estimate_at_t_uses_t_perturbation():
    # The estimate at t must INCLUDE r_t: the mirror image of
    # test_rw_no_look_ahead_perturbation in tests/test_evaluation.py, one
    # day apart on purpose.
    rng = np.random.default_rng(7)
    r1 = _series(rng.normal(0.0, 0.01, 80))
    r2 = r1.copy()
    r2.iloc[40] *= 10.0  # a shock lands on day 40
    v1 = rolling_vol(r1, window=21)
    v2 = rolling_vol(r2, window=21)
    assert not np.isclose(v2.iloc[40], v1.iloc[40])  # estimate AT the shock day: moves
    pd.testing.assert_series_equal(v1.iloc[:40], v2.iloc[:40])  # nothing earlier moves
    pd.testing.assert_series_equal(v1.iloc[61:], v2.iloc[61:])  # the shock leaves the window
