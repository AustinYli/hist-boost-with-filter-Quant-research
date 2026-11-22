# Semiconductor V3 Direction + Selection

Research only. It doesn't use the live/paper executor, place orders, or write to the 3-day/5-day model, state, bar, or log files.

## Hypothesis

At each five-session decision date, after the close:

1. Predict the SOXX return from the next open to the open five sessions later.
2. Predict each eligible stock's return over the same period minus SOXX's return.
3. If the SOXX prediction is positive, go long the ten highest-ranked stocks at +10% each.
4. If it's negative, short the ten lowest-ranked stocks at -10% each.
5. Rebalance at the next open and hold for five open-to-open periods.

There's no neutral state and no combined long-short book in this version.

## Data Modes

Strict mode needs a point-in-time semiconductor membership table, delisting handling, and historical short-borrow data. It refuses to run if any of these are missing.

`seed-existing` builds a test dataset from the current Alpaca/IEX files. It's only good for checking that the pipeline runs: the history is short, it has survivorship bias, and there's no historical borrow data.

```bash
python3 semiconductor_v3_directional/prepare_data.py seed-existing
python3 semiconductor_v3_directional/prepare_data.py validate
python3 semiconductor_v3_directional/research.py run --exploratory
python3 semiconductor_v3_directional/test_research.py
```

To use vendor data, replace the files in `semiconductor_v3_directional/data/` with files matching the schemas below and run without `--exploratory`.

## Required Schemas

`stock_bars.csv` and `market_bars.csv`:

`timestamp,symbol,open,high,low,close,volume,trade_count,vwap`

`universe_membership.csv`:

`symbol,valid_from,valid_to,industry,security_type,us_listed,survivorship_safe,delisted,delisting_handled,source`

`borrow_costs.csv`:

`date,symbol,shortable,annual_borrow_rate,source`

Membership, classifications, delistings, and borrow data have to reflect what was known on each date. Strict mode won't accept today's constituent list applied backward.

## Outputs

Written to `semiconductor_v3_directional/artifacts/`:

- `results.json`: data audit, component metrics, strategy metrics, and the pass/fail result.
- `predictions.csv`: walk-forward direction and stock-selection predictions.
- `period_returns.csv`: primary, stress, and baseline five-session returns.
- `retrain_audit.csv`: label embargo and quarterly refit records.

V3 results are never used for live or paper trading automatically.
