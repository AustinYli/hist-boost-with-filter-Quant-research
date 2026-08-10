# Semiconductor V3 Direction + Selection

This is an isolated research branch. It does not import the live/paper executor,
submit orders, or write to the active 3-day/5-day model, state, bar, or log files.

## Locked Hypothesis

At each five-session decision date after the close:

1. Predict the SOXX return from the next open through the open five sessions later.
2. Predict each eligible stock's return over the same interval minus SOXX's return.
3. If predicted SOXX return is positive, hold the ten highest-ranked stocks at +10% each.
4. If predicted SOXX return is negative, hold the ten lowest-ranked stocks at -10% each.
5. Rebalance at the next open and hold for five complete open-to-open intervals.

There is no neutral state and no simultaneous long-short book in this experiment.

## Data Modes

Strict research requires a point-in-time semiconductor membership table, delisting
handling, and historical short borrow data. Strict mode fails closed when those are
missing.

The bundled `seed-existing` command creates an explicitly non-confirmatory dataset
from the existing current-universe Alpaca/IEX files. It is useful only to verify the
pipeline because it has limited history, survivorship bias, and no historical borrow
availability.

```bash
python3 semiconductor_v3_directional/prepare_data.py seed-existing
python3 semiconductor_v3_directional/prepare_data.py validate
python3 semiconductor_v3_directional/research.py run --exploratory
python3 semiconductor_v3_directional/test_research.py
```

To use institutional/vendor data, replace the files in
`semiconductor_v3_directional/data/` with the documented schemas and run without
`--exploratory`.

## Required Schemas

`stock_bars.csv` and `market_bars.csv`:

`timestamp,symbol,open,high,low,close,volume,trade_count,vwap`

`universe_membership.csv`:

`symbol,valid_from,valid_to,industry,security_type,us_listed,survivorship_safe,delisted,delisting_handled,source`

`borrow_costs.csv`:

`date,symbol,shortable,annual_borrow_rate,source`

Membership, classifications, delistings, and borrow fields must reflect information
known on each historical date. Reconstructing today's constituents backward is not
accepted by strict mode.

## Outputs

Research writes only below `semiconductor_v3_directional/artifacts/`:

- `results.json`: data audit, component metrics, strategy metrics, and pass/fail decision.
- `predictions.csv`: walk-forward direction and stock-selection predictions.
- `period_returns.csv`: primary, stress, and baseline five-session returns.
- `retrain_audit.csv`: label-embargo and quarterly refit evidence.

No V3 artifact is eligible for live or paper trading automatically.
