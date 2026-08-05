"""Price data loading and caching.

Plumbing. The two yfinance API traps it works around are documented inline
at their call sites.
"""

from pathlib import Path

import pandas as pd

# Repo root: garch_lab -> src -> root, so three levels up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO_ROOT / "data"


def _cache_path(ticker: str, start: str, cache_dir: Path) -> Path:
    safe = ticker.replace("^", "").replace("/", "-").lower()
    return cache_dir / f"{safe}_{start}.csv"


def _normalize(px: pd.Series, ticker: str) -> pd.Series:
    """Coerce either load path to the same contract: float values, tz-naive
    DatetimeIndex of dates, no NaNs, named after the ticker."""
    px = px.dropna().astype(float).rename(ticker)
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index).date)
    px.index.name = "date"
    return px


def load_prices(
    ticker: str = "^GSPC",
    start: str = "2005-01-01",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> pd.Series:
    """Daily split- and dividend-adjusted close prices, as a float Series
    with a DatetimeIndex.

    Downloads from Yahoo Finance on the first call and caches to a local
    CSV; subsequent calls (and offline work) read the cache. Pass
    ``force_download=True`` to refresh.

    Adjustment matters for stocks and ETFs, where raw close prices carry a
    fake ~-2% "return" on every ex-dividend day and a fake -50% "crash" at
    every 2-for-1 split. It is a no-op for a price index such as ^GSPC,
    which has neither.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(ticker, start, cache_dir)

    if cache.exists() and not force_download:
        px = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return _normalize(px, ticker)

    import yfinance as yf  # local import: cached runs never need it

    # Trap 1: auto_adjust is pinned explicitly, not left to the default.
    # Since yfinance 0.2.51 (Feb 2025) download() adjusts by default, so
    # 'Close' already folds in dividends and splits and there is no longer an
    # 'Adj Close' column. Pinning protects this module if the default flips.
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"Yahoo returned no data for {ticker!r}. Rate limit or network "
            "block? Retry in a minute; the result is cached after one success."
        )

    # Trap 2: download() returns two-level columns (field, ticker) even for a
    # single ticker, so select the field and squeeze to a Series.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    px = _normalize(close, ticker)
    px.to_csv(cache)
    return px
