# SOXX Moving-Average Filter Tests

Copies of the 3-day and 5-day HGB models, used to test different SOXX regime filters. Only the regime gate changes: `SOXX close > trailing N-day MA`, with N = 100, 200 (baseline), or 300.

The runner also tests:

- a stricter 200-day gate: SOXX must close at least 20% above its 200-day moving average;
- a stress version of that gate at 40% above;
- a looser gate that stays in the market until SOXX closes more than 10% below its 200-day moving average.

Everything else is the same as the original: selector model, 18 features, hyperparameters, universe, portfolio rules, transaction costs, signal horizons, walk-forward procedure, and out-of-sample end dates.

This folder doesn't place orders or write to the live model, signal, state, or trading-log files.

Run it with:

```bash
python3 semiconductor_100d_filter/research.py
```

Results go to `semiconductor_100d_filter/artifacts/`.
