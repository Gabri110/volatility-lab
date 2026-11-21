# Does GARCH(1,1) beat naive volatility forecasts out of sample?

One full turn of the volatility-research loop on ~20 years of S&P 500 daily
returns: verify the stylized facts, fit GARCH(1,1), produce one-day-ahead
variance forecasts on a 30% holdout with frozen parameters, and score them
against two naive benchmarks with a proxy-robust loss.

## TL;DR

Using S&P 500 daily log returns from 2005, with a 70/30 chronological split and
1,629 test days, we find that the QLIKE of GARCH(1,1)-N is 0.50 below that of
the unconditional variance and 0.09 below that of a 21-day rolling variance.
Its MSE is 25.7% lower than the first and 16.8% lower than the second. These
are one-day-ahead variance forecasts scored against squared returns, with
parameters frozen at their training estimates. We are still missing the error
bars that would support the apparent positive verdict.

Our test window opens on 7 February 2020, so training is entirely pre-COVID,
which heavily penalizes the constant baseline. We fit both a normal and a
Student-t innovation distribution and found the Gaussian slightly better
(1.5375 vs 1.5436 QLIKE), a gap of 0.006 that we have no test to tell apart
from zero.

### Summary of findings

- QLIKE over the test period: GARCH-N 1.54, 21-day rolling variance 1.63,
  unconditional 2.04
- QLIKE gap to the unconditional variance 0.50, to the rolling variance 0.09.
  The level of QLIKE depends on the form in which it is written, so we report
  differences and not percentages
- Fitted persistence α+β = 0.9773, shock half-life ≈ 30 trading days
- MSE gives the same ordering as QLIKE, so the ranking is not an artifact of
  the loss we chose

## The question, and why it matters

The sign of tomorrow's return is nearly unpredictable, but its magnitude is
strongly predictable. That is what makes a volatility forecast worth building
at all. Volatility is an input to position sizing, to risk limits and VaR, and
to option pricing, so the question is whether a standard conditional-variance
model earns its complexity against forecasts anyone can produce in one line.

## Data

We use the S&P 500 index (`^GSPC`) daily close from 2005 to 3 August 2026,
obtained via yfinance, which gives 5,428 daily log returns. We request
split- and dividend-adjusted prices, which protects the pipeline if it is ever
pointed at a stock or an ETF, where unadjusted closes carry a fake return on
every ex-dividend day and a fake crash at every split. For a price index such
as `^GSPC` the adjustment is a no-op, since it pays no dividends and does not
split.

We work in trading time, using the series as it comes, and do not reindex to
calendar days, which would inject artificial zero returns across weekends and
holidays. Prices are cached under `data/`, which is not committed.

## What the data demand of a model

- Daily returns show almost no linear autocorrelation. Most lags of the ACF
  sit inside the 95% band of an iid series. The lag-1 autocorrelation is the
  exception: it falls well outside, and we have not explained why. But the
  returns are not independent, and the dependence survives in the higher
  moments.

- Volatility clustering. High-volatility days are followed by high-volatility
  days, and the ACF of |r| stays positive out to 100 lags, decaying far more
  slowly than the exponential a GARCH(1,1) implies.

  ![Volatility clustering](figures/fig1_clustering.png)

- Fat tails. Fitting a Student-t to the returns gives ν̂ = 2.4 degrees of
  freedom. Aggregating to weekly or monthly returns thins the tails, though
  slowly, as the central limit theorem grinds against both the dependence and
  the tail weight.

  ![Fat tails: QQ plot vs normal](figures/fig2_qq_tails.png)

- Asymmetry. The sign of today's return affects tomorrow's volatility:
  negative returns are followed by higher volatility than positive returns of
  the same size. A symmetric model cannot reproduce this.

## Model and evaluation design

We fit GARCH(1,1) with Gaussian and Student-t innovations by maximum likelihood
on the first 70% of the sample. The chronological split puts the last training
day at 6 February 2020 and opens the test window on 7 February 2020.

Over the remaining 30% we produce one-step-ahead variance forecasts with the
parameters frozen at their training estimates, with no refitting. The
benchmarks are a 21-day rolling variance and the unconditional training
variance. The proxy is the squared daily return.

QLIKE is the primary loss and MSE the secondary one. Patton (2011) shows that
under a noisy proxy many intuitive losses systematically misrank models, and
characterizes the class of losses whose rankings survive the noise; MSE and
QLIKE are its two most-used members. There are no zero-return days in the
test window, so no observation had to be dropped or clipped before taking
the logarithm in QLIKE.

## Results

We report one-step-ahead variance forecasts over the 1,629 test days
(2020-02-07 to 2026-08-03). The forecasts are in daily variance units of %²
per day, so the MSE is in the square of those units and QLIKE has none.

| Forecast | QLIKE | MSE |
|---|---|---|
| GARCH(1,1)-N | 1.537518 | 34.937572 |
| GARCH(1,1)-t | 1.543563 | 35.575560 |
| 21-day rolling variance | 1.627217 | 41.970833 |
| Unconditional | 2.038582 | 46.998799 |

GARCH(1,1)-N has the lowest loss on both measures. The gap to GARCH(1,1)-t is
0.006, and we have no test to tell the two apart. The gap to the unconditional
benchmark is large and the gap to the rolling variance is small, which is the
expected shape at a one-day horizon: daily volatility is persistent enough
that "tomorrow looks like the recent past" is genuinely hard to beat.

Plotting the 21-day rolling volatility against the GARCH(1,1)-N conditional
volatility shows the difference in responsiveness between a boxcar filter with
equal weights and an exponentially weighted one whose decay is estimated from
the data.

![Rolling vs GARCH conditional volatility](figures/fig4_rolling_vs_garch.png)

## Limitations, and what I would do next

- The leverage effect measured in notebook 01 is unmodelled, since GARCH(1,1)
  is sign-blind. The model has to be extended to account for this.
- The ACF of |r| decays more slowly than any exponential (fig1), which points
  to HAR-RV on true realized volatility.
- A single split with frozen parameters should become walk-forward
  re-estimation.
- The QLIKE gap carries no significance statement. These should be computed to support the apparent positive verdict.
- We are comparing against the squared returns (a noisy proxy), not the true volatility itself. We could repeat the analysis with other proxies. 

## Reproduce it

Requires Python 3.14 or later.

```
uv sync                 # creates .venv and installs everything, incl. dev tools
uv run pytest           # the suite specifies the library functions
uv run jupyter lab      # run notebooks/01, then notebooks/02
```

## Conventions

- Log returns throughout. Notebook 02 works in %²-per-day variance units.
- `rolling_vol` at t includes t, so it is an estimate at t.
  `rw_variance_forecast` at t excludes t, so it is a forecast for t. The two
  are one day apart on purpose.
- Chronological splits only. Nothing computed at time t may touch data dated
  after t.
- QLIKE primary, MSE secondary, no other losses.

## How this was built

The scaffold of this repository was AI-generated under the constraints written
down in [CLAUDE.md](CLAUDE.md): the test suite as an executable spec, the
notebook skeletons, and a concept guide kept outside the repo. Every line of
analysis, in `src/garch_lab/` and in both notebooks, is mine. The point of the
arrangement was to be able to defend the result rather than to arrive at it
quickly.

## References

Cont, R. (2001). Empirical properties of asset returns: stylized facts and
statistical issues. *Quantitative Finance* 1(2). The stylized-facts catalogue.

Mandelbrot, B. (1963). The variation of certain speculative prices. *Journal of
Business* 36. Volatility clustering, named and shown.

Engle, R. (1982). Autoregressive conditional heteroscedasticity with estimates
of the variance of United Kingdom inflation. *Econometrica* 50(4). ARCH.

Bollerslev, T. (1986). Generalized autoregressive conditional
heteroskedasticity. *Journal of Econometrics* 31. GARCH(1,1).

Patton, A. (2011). Volatility forecast comparison using imperfect volatility
proxies. *Journal of Econometrics* 160(1). Which losses rank models correctly
when the target is a noisy proxy.

Diebold, F. and Mariano, R. (1995). Comparing predictive accuracy. *Journal of
Business & Economic Statistics* 13(3). The forecast-comparison test this study
still owes.

Sheppard, K. The `arch` package. https://arch.readthedocs.io
