# Semiconductor V4 Continuous Alpha + Direction Overlay

V4 is an isolated research system. It does not submit orders or modify the
existing 3-day/5-day paper/live system or V3.

## Locked Architecture

V4 decomposes stock returns into sector beta and stock-specific residual alpha:

`stock_return = beta_to_SOXX * SOXX_return + residual`

At every five-session decision date:

1. RidgeCV predicts the next five-session SOXX return.
2. Training-only time-series residuals estimate forecast uncertainty.
3. Conviction is `tanh(prediction / uncertainty)`.
4. HGB predicts each stock's beta-neutral residual return.
5. The top ten and bottom ten form an always-on long-short alpha core.
6. Conviction smoothly changes long and short gross exposure:
   - strong bull: +100% long, -30% short;
   - neutral: +50% long, -50% short;
   - strong bear: +30% long, -100% short.
7. A Ledoit-Wolf covariance model and SLSQP optimizer control concentration,
   SOXX beta, turnover, borrow expense, gross/net exposure, and predicted risk.
8. If predicted annualized volatility exceeds 20%, all weights are scaled down.

The mapping, costs, model families, features, horizon, and constraints are fixed
in `spec.json`; the first result is not used to tune them.

## Data And Commands

Strict mode requires point-in-time semiconductor membership, delisting handling,
and historical borrow data. The existing IEX files can only exercise exploratory
mode.

The historical-data acceptance rules were locked on 2026-08-09 before acquiring
new data. See `DATA_RESEARCH.md` for the source assessment and
`data_contract.json` for the untouched challenge protocol. The preferred complete
route is CRSP/Compustat plus S&P Global Securities Finance; retail OHLCV products
do not provide the required historical borrow tape by themselves.

```bash
python3 semiconductor_v4_continuous/prepare_data.py seed-existing
python3 semiconductor_v4_continuous/prepare_data.py validate
python3 semiconductor_v4_continuous/research.py audit --exploratory
python3 semiconductor_v4_continuous/research.py run --exploratory
python3 semiconductor_v4_continuous/test_research.py
python3 semiconductor_v4_continuous/forward.py status
python3 semiconductor_v4_continuous/historical_data.py verify-lock
python3 semiconductor_v4_continuous/historical_data.py audit
python3 semiconductor_v4_continuous/test_historical_data.py
python3 semiconductor_v4_continuous/test_confirmation.py
python3 semiconductor_v4_continuous/confirmation.py status
```

To prepare a licensed provider export without touching the active exploratory
files:

```bash
python3 semiconductor_v4_continuous/historical_data.py init --stage /path/to/v4-stage
cp /path/to/original/vendor/exports/* /path/to/v4-stage/raw/
python3 semiconductor_v4_continuous/historical_data.py register-raw \
  --stage /path/to/v4-stage \
  --vendor "licensed vendor name" \
  --version "export/database version" \
  --as-of-date YYYY-MM-DD \
  --license-reference "contract or entitlement reference"
python3 semiconductor_v4_continuous/historical_data.py verify-raw --stage /path/to/v4-stage
python3 semiconductor_v4_continuous/historical_data.py audit --stage /path/to/v4-stage
python3 semiconductor_v4_continuous/historical_data.py normalize \
  --stage /path/to/v4-stage --output /path/to/v4-normalized
```

`raw/` contains byte-for-byte original exports. The canonical CSVs live below
`canonical/`. Registration records the exact file set, sizes, hashes, vendor,
version, as-of date, and license reference. Any later raw-file change blocks the
dataset before V4 is allowed to run.

Installation is deliberately separate. It verifies the frozen-hypothesis hashes,
requires every strict data and sample-size check to pass, and backs up the current
exploratory data before replacement. It cannot run without the explicit
`INSTALL_STRICT_V4_DATA` confirmation token.

After a passing bundle is installed, the confirmation workflow has two separate,
irreversible gates:

```bash
python3 semiconductor_v4_continuous/confirmation.py freeze-dataset \
  --confirm FREEZE_CHALLENGE_DATASET
python3 semiconductor_v4_continuous/confirmation.py verify-dataset
python3 semiconductor_v4_continuous/confirmation.py run-once \
  --confirm RUN_FROZEN_V4_ONCE
```

The dataset lock fingerprints every normalized table, source manifest, data
contract, and frozen-hypothesis manifest. `run-once` creates its receipt before
starting the model. A completed, failed, or interrupted claim cannot be silently
retried. The generated report separates pre-2021 from the already inspected
2021-2026 period and reports realized negative SOXX periods, negative direction
predictions, and actual negative-net V4 periods as distinct counts.

Until the licensed bundle passes, `confirmation.py status` remains
`awaiting_confirmation_quality_data`; no dataset lock or one-shot receipt is
created.

Outputs are written only below `semiconductor_v4_continuous/artifacts/`. V4 has
no automatic paper or live trading connection.
