# SOXX Moving-Average Filter Research Copies

This directory is isolated from the active 3-day and 5-day paper/live system.
It contains physical copies of both fixed HGB model files for isolated 100-day
and 300-day variants. The controlled comparison changes only the regime gate:
`SOXX close > trailing N-day MA`, where N is 100, 200 (baseline), or 300.

The same runner also evaluates a stricter 200-day gate that requires SOXX to
close at least 20% above its trailing 200-session moving average.
An additional stress variant requires SOXX to close at least 40% above that
average.
It also evaluates a looser gate that permits exposure until SOXX closes more
than 10% below its trailing 200-session moving average.

The selector model, 18 selection features, hyperparameters, universe,
portfolio rules, transaction-cost assumption, signal horizons, chronological
walk-forward procedure, and original out-of-sample endpoints are unchanged.

Nothing in this directory submits orders or writes to active model, signal,
state, or trading-log paths.

Run the isolated comparison with:

```bash
python3 semiconductor_100d_filter/research.py
```

Results are written only under `semiconductor_100d_filter/artifacts/`.
