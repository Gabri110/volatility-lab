# Does GARCH(1,1) beat naive volatility forecasts out of sample?

One full turn of the volatility-research loop on ~20 years of S&P 500 daily
returns: verify the stylized facts, fit GARCH(1,1), produce one-day-ahead
variance forecasts on a 30% holdout with frozen parameters, and score them
against two naive benchmarks with a proxy-robust loss.

**Status: in progress.** <!-- Delete this line in roadmap box U4. -->

<!-- HOW TO USE THIS SKELETON
Each section carries a comment like this saying what belongs in it and which
ROADMAP.md box produces it. Write in your own voice: conclusion first,
numbers over adjectives, no "This project demonstrates..." — state findings,
not virtues. Delete every skeleton comment before pushing. -->

## TL;DR

<!-- U4. Paste the sober-conclusion paragraph from the last cell of
notebook 02, then compress it into 3–4 bullets, each carrying a NUMBER.
The expected honest shape (guide §5.2): GARCH beats the unconditional
benchmark decisively and beats yesterday's rolling vol modestly or not at
all at a one-day horizon. If that is what you find, say exactly that —
the sober version IS the credible result. -->

- _[QLIKE over the test period: GARCH-t … vs random walk … vs unconditional …]_
- _[Improvement vs unconditional: …% · vs random walk: …%]_
- _[Fitted persistence α+β = …, shock half-life ≈ … trading days]_
- _[One sentence on whether MSE agrees with the QLIKE ranking, and why that matters]_

## The question, and why it matters

The sign of tomorrow's returns is nearly unpredictable, but its magnitude is strongly predictable. Because of this, we build a volatility forecasting model.

This is interesting because volatility is a key input to risk management and option pricing.

## Data

We use the S&P 500 index daily adjusted close from 2005 to present, obtained via yfinance. The adjusted close is preferred as it accounts for corporate actions like dividends and stock splits, providing a more accurate reflection of the stock's value over time. The data is aligned with a trading-day calendar without reindexing to calendar days, and prices are cached locally under the `data/` directory, which is not committed to version control.

## What the data demand of a model

<!-- S4. One short paragraph per verified stylized fact, each anchored to a
number you computed in notebook 01: excess kurtosis (and its subsample
instability), the |r| > 4σ day count vs the Gaussian expectation, fitted
Student-t ν̂, lag-1 ACF of |r| and how far the significance extends, the
leverage correlation. End with the punchline: a model that cannot reproduce
clustering and fat tails is dead on arrival — which is the license for
GARCH. -->

- Volatility clustering: We notice that high-volatility days are followed by high-volatility days.
![Volatility clustering](figures/fig1_clustering.png)

- Fat tails: We fit have fit the returns to Student-t woth estimated ν = 2,4 degrees of freedom.
![Fat tails: QQ plot vs normal](figures/fig2_qq_tails.png)

- Asymmetry of the volatility: The sign of the returns affects tomorrow's volatility differently, and a symmetric model fails to take that into account. In particular, negative returns leads to a higher volatility.

## Model and evaluation design

<!-- U2/U3. Be precise, it is the section readers check: GARCH(1,1)
with Gaussian and Student-t innovations, fit by MLE on the first 70%
(chronological split at [date]); one-step-ahead variance forecasts over the
last 30% with parameters FROZEN at their train estimates (no refitting);
benchmarks = 21-day random-walk variance and unconditional train variance;
proxy = squared daily return; QLIKE primary and MSE secondary, with one
sentence on Patton (2011) — under a noisy proxy, most intuitive losses can
systematically misrank models; these two cannot. Disclose the zero-return-
day count and your drop-vs-clip decision here. -->

## Results

<!-- U4. The loss table plus ONE interpretation paragraph. State the units
(daily variance, %² per day) next to the table. -->

| Forecast | QLIKE | MSE |
|---|---|---|
| GARCH(1,1)-t | _…_ | _…_ |
| GARCH(1,1)-N | _…_ | _…_ |
| Random walk (21d) | _…_ | _…_ |
| Unconditional (train) | _…_ | _…_ |

![Rolling vs GARCH conditional volatility](figures/fig3_rolling_vs_garch.png)

<!-- There is deliberately NO forecast-vs-realized scatter here: the daily
squared-return proxy carries noise ≈ 141% of its signal (guide §3.3), so
that picture is uninformative by construction. Say that in one sentence —
it is worth more than the plot. -->

## Limitations, and what I would do next

<!-- U4. Honest list, one line each, every item tied to something you
MEASURED: leverage effect present in the data but GARCH(1,1) is sign-blind
→ GJR/EGARCH; empirical ACF of |r| decays slower than any exponential →
HAR-RV; a single 70/30 split with frozen parameters → walk-forward
re-estimation; the QLIKE gap has no significance attached → Diebold–Mariano;
zero-return-day handling. -->

## Reproduce it

```
uv sync                 # creates .venv and installs everything, incl. dev tools
uv run pytest           # the suite specifies the library functions
uv run jupyter lab      # run notebooks/01, then notebooks/02
```

<!-- Windows note to keep: `uv run` never needs .venv activation — it runs
inside the project environment directly, so PowerShell execution-policy
trouble is irrelevant. uv.lock is committed: it pins the exact environment
that produced the results, so `uv sync` reproduces it bit-for-bit. -->

<!-- Figures are Plotly, exported to PNG with fig.write_image("figures/figN.png")
via kaleido (already in the dependencies) so they render inline on GitHub. If an
export ever fails asking for a browser, fetch one once with:
.venv\Scripts\python.exe -c "import kaleido; kaleido.get_chrome_sync()" -->

## References

Cont (2001) — stylized facts. Engle (1982) — ARCH. Bollerslev (1986) —
GARCH(1,1). Patton (2011) — loss functions under imperfect volatility
proxies. Annotated list and the full conceptual companion:
[docs/concept-guide.md](docs/concept-guide.md).
