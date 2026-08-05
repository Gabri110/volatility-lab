"""Spec for garch_lab.evaluation.

None of these needs a fitted model, so they can run first. They encode the
evaluation discipline the rest of the project depends on, including the
no-look-ahead perturbation test at the bottom.
"""

import numpy as np
import pandas as pd
import pytest

from garch_lab.evaluation import chronological_split, mse, qlike, rw_variance_forecast


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


@pytest.fixture
def p():
    """A strictly positive 'variance proxy' series."""
    rng = np.random.default_rng(3)
    return _series(rng.uniform(0.5, 2.0, 40))


@pytest.fixture
def s():
    rng = np.random.default_rng(4)
    return _series(rng.normal(0.0, 1.0, 100))


# ------------------------------------------------------------------ mse

def test_mse_known_value():
    assert mse(_series([1.0, 2.0]), _series([2.0, 4.0])) == pytest.approx(2.5)


def test_mse_zero_iff_equal_and_symmetric(p):
    h = 1.3 * p
    assert mse(p, p) == pytest.approx(0.0)
    assert mse(p, h) > 0
    assert mse(p, h) == pytest.approx(mse(h, p))


# ---------------------------------------------------------------- qlike

def test_qlike_nonnegative_zero_iff_perfect(p):
    assert qlike(p, p) == pytest.approx(0.0)
    for c in (0.5, 0.9, 1.1, 3.0):
        assert qlike(p, c * p) > 0


def test_qlike_punishes_underforecast_more(p):
    # Underestimating volatility is worse for a risk manager than paying
    # opportunity cost on an overestimate, and QLIKE encodes that.
    under = qlike(p, p / 2)  # forecast half the true variance
    over = qlike(p, 2 * p)   # forecast double the true variance
    assert under == pytest.approx(1.0 - np.log(2.0))  # 0.30685...
    assert over == pytest.approx(np.log(2.0) - 0.5)   # 0.19315...
    assert under > over


def test_qlike_scale_invariant_mse_is_not(p):
    # why unit sloppiness corrupts MSE rankings but not QLIKE rankings
    h = 1.3 * p
    assert qlike(100.0 * p, 100.0 * h) == pytest.approx(qlike(p, h))
    assert mse(100.0 * p, 100.0 * h) == pytest.approx(100.0**2 * mse(p, h))


def test_qlike_rejects_nonpositive_proxy():
    # zero-return days exist; ln(0) must be a loud error, and the drop/clip
    # decision must happen (and be disclosed) at the call site
    with pytest.raises(ValueError):
        qlike(_series([0.0, 1.0]), _series([1.0, 1.0]))


def test_losses_reject_misaligned_index(p):
    # pandas would silently align on the index union and inject NaNs,
    # a bug that produces a plausible number instead of an error
    h = p.shift(1).dropna()  # same values, shifted dates
    with pytest.raises(ValueError):
        qlike(p, h)
    with pytest.raises(ValueError):
        mse(p, h)


# --------------------------------------------------- chronological_split

def test_split_proportions(s):
    train, test = chronological_split(s, train_frac=0.7)
    assert len(train) == 70
    assert len(test) == 30


def test_split_chronological_no_overlap_no_loss(s):
    train, test = chronological_split(s, train_frac=0.7)
    assert train.index.max() < test.index.min()
    assert len(train.index.intersection(test.index)) == 0
    assert pd.concat([train, test]).equals(s)  # nothing dropped, nothing shuffled


def test_split_no_shuffle(s):
    train, _ = chronological_split(s, train_frac=0.7)
    assert train.equals(s.iloc[:70])


def test_split_point_is_floor_not_round():
    # 0.75 * 10 = 7.5: int() truncates to 7, round() would give 8. The
    # contract pins the floor so the published sample sizes stay stable.
    s10 = _series(np.arange(10.0))
    train, test = chronological_split(s10, train_frac=0.75)
    assert len(train) == 7
    assert len(test) == 3


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.3, -0.2])
def test_split_bad_frac_raises(s, bad):
    with pytest.raises(ValueError):
        chronological_split(s, train_frac=bad)


# ------------------------------------------------- rw_variance_forecast

def test_rw_constant_abs_return_closed_form():
    r = _series([0.02, -0.02] * 100)
    f = rw_variance_forecast(r, window=5)
    assert np.allclose(f.dropna(), 0.0004)  # mean of r² is c² = 0.02²


def test_rw_warmup_is_window_not_window_minus_1():
    rng = np.random.default_rng(5)
    r = _series(rng.normal(0.0, 0.01, 60))
    f = rw_variance_forecast(r, window=21)
    assert f.index.equals(r.index)
    assert f.isna().sum() == 21  # one more NaN than rolling_vol: the shift is real


def test_rw_no_look_ahead_perturbation():
    # The forecast for day t must not see day t.
    rng = np.random.default_rng(6)
    r1 = _series(rng.normal(0.0, 0.01, 60))
    r2 = r1.copy()
    r2.iloc[40] *= 10.0  # a shock lands on day 40
    f1 = rw_variance_forecast(r1, window=21)
    f2 = rw_variance_forecast(r2, window=21)
    assert f2.iloc[40] == pytest.approx(f1.iloc[40])  # forecast for the shock day: unmoved
    assert f2.iloc[41] != pytest.approx(f1.iloc[41])  # the day after: reacts
    pd.testing.assert_series_equal(f1.iloc[:41], f2.iloc[:41])  # nothing earlier moves
