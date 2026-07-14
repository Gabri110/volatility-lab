"""Spec for garch_lab.returns.log_returns — guide §1.

Make these green before notebook 01 §1 (roadmap box S1).
Test names are the contract; read them as sentences.
"""

import numpy as np
import pandas as pd
import pytest

from garch_lab.returns import log_returns


@pytest.fixture
def px():
    idx = pd.bdate_range("2020-01-01", periods=10)
    rng = np.random.default_rng(0)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 10))), index=idx)


def test_known_value_not_simple_return(px):
    two = pd.Series([100.0, 200.0], index=px.index[:2])
    r = log_returns(two)
    # ln(2) = 0.693..., NOT 1.0 — if you got 1.0 you computed simple returns
    assert np.isclose(r.iloc[0], np.log(2.0))


def test_length_and_date_labels(px):
    r = log_returns(px)
    assert len(r) == len(px) - 1
    # a return is labeled by the date it ENDS on
    assert r.index.equals(px.index[1:])


def test_time_additivity(px):
    # guide §1.2 — the property you chose log returns FOR:
    # the sum of daily log returns is exactly the full-period log return
    r = log_returns(px)
    assert np.isclose(r.sum(), np.log(px.iloc[-1] / px.iloc[0]))


def test_no_nans(px):
    assert log_returns(px).notna().all()
