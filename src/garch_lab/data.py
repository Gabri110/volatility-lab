"""Price data loading and caching.

This is the ONLY module in the package that ships implemented: it is
plumbing, and the API traps it works around (see comments) are not the
learning content of this project. Read it once — the comments explain
two real-world data gotchas — then forget about it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root = two levels above this file (src/garch_lab/data.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO_ROOT / "data"


def _cache_path(ticker: str, start: str, cache_dir: Path) -> Path:
    safe = ticker.replace("^", "").replace("/", "-").lower()
    return cache_dir / f"{safe}_{start}.csv"


def load_prices(
    ticker: str = "^GSPC",
    start: str = "2005-01-01",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> pd.Series:
    """Daily adjusted close prices as a float Series with a DatetimeIndex.

    Downloads from Yahoo Finance on the first call and caches to a local
    CSV; subsequent calls (and offline work) read the cache. Pass
    ``force_download=True`` to refresh.

    Why "adjusted" matters: raw close prices contain a fake ~-2% "return"
    on every ex-dividend day and a fake -50% "crash" at every 2-for-1
    split. Guide §1.3 (docs/concept-guide.md) explains the trap.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker, start, cache_dir)

    if cache.exists() and not force_download:
        px = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return px.rename(ticker)

    import yfinance as yf  # local import: cached runs never need it

    # GOTCHA 1 — auto_adjust is pinned explicitly, not left to the default.
    # Since yfinance 0.2.51 (Feb 2025) download() adjusts by default: 'Close'
    # already folds in dividends/splits and an 'Adj Close' column no longer
    # exists. Pinning protects this module if the default ever flips again.
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"Yahoo returned no data for {ticker!r}. Rate limit or network "
            "block? Retry in a minute — the result is cached after one success."
        )

    # GOTCHA 2 — download() returns two-level columns (field, ticker) even
    # for a single ticker; select the field and squeeze to a Series.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    px = close.dropna().astype(float).rename(ticker)
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index).date)  # drop tz noise
    px.index.name = "date"
    px.to_csv(cache)
    return px
