"""Log returns — YOUR implementation.

Spec: tests/test_returns.py. Theory: guide §1 (docs/concept-guide.md).
"""

import pandas as pd
import numpy as np


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns r_t = ln(P_t / P_{t-1}).

    Contract (enforced by tests/test_returns.py):
    - length is len(prices) - 1, no NaNs
    - each return is labeled by the date it ENDS on: index == prices.index[1:]
    - values are natural-log ratios, not simple returns
    - time-additive: the sum over a window equals ln(P_last / P_first)

    Guide §1.2 explains why log (not simple) returns are the right choice
    on the time axis. Delete the `raise` and write the body.
    """
    
    return pd.Series(np.diff(np.log(prices)), index=prices.index[1:])
