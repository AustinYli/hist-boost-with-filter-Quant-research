# Semiconductor V4 Continuous Alpha + Direction Overlay

Research only. V4 doesn't place orders or change the 3-day/5-day paper/live system or V3.

## Design

V4 splits each stock's return into sector beta and stock-specific residual:

`stock_return = beta_to_SOXX * SOXX_return + residual`

At every five-session decision date:

1. RidgeCV predicts the next five-session SOXX return.
2. Residuals from the training period estimate forecast uncertainty.
3. Conviction is `tanh(prediction / uncertainty)`.
4. HGB predicts each stock's beta-neutral residual return.
5. The top ten and bottom ten stocks form a long-short core that is always on.
6. Conviction shifts long and short gross exposure smoothly:
   - strong bull: +100% long, -30% short;
   - neutral: +50% long, -50% short;
   - strong bear: +30% long, -100% short.
7. A Ledoit-Wolf covariance model and an SLSQP optimizer limit concentration, SOXX beta, turnover, borrow cost, gross/net exposure, and predicted risk.
8. If predicted annualized volatility is above 20%, all weights are scaled down.

The mapping, costs, models, features, horizon, and constraints are fixed in `spec.json` and weren't tuned on the first result.

## Data and Commands

Strict mode needs point-in-time semiconductor membership, delisting handling, and historical borrow data. The current IEX files only work in exploratory mode.

The rules for accepting historical data were fixed on 2026-08-09, before any new data was collected. `DATA_RESEARCH.md` compares data sources and `data_contract.json` has the test protocol. The best full option is CRSP/Compustat plus S&P Global Securities Finance; retail OHLCV products don't include historical borrow data.

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

To prepare a licensed vendor export without touching the exploratory files:

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

`raw/` holds the vendor exports unchanged, and the cleaned CSVs go in `canonical/`. Registering records the file list, sizes, hashes, vendor, version, as-of date, and license reference. If a raw file changes afterward, V4 won't run on that dataset.

Installing the data is a separate step. It checks the frozen-hypothesis hashes, requires every strict data and sample-size check to pass, and backs up the current exploratory data before replacing it. It only runs with the `INSTALL_STRICT_V4_DATA` confirmation token.

Once a dataset is installed, confirmation takes two steps that can't be undone:

```bash
python3 semiconductor_v4_continuous/confirmation.py freeze-dataset \
  --confirm FREEZE_CHALLENGE_DATASET
python3 semiconductor_v4_continuous/confirmation.py verify-dataset
python3 semiconductor_v4_continuous/confirmation.py run-once \
  --confirm RUN_FROZEN_V4_ONCE
```

The dataset lock hashes every normalized table, the source manifest, the data contract, and the frozen-hypothesis manifest. `run-once` writes its receipt before the model starts, so a run that finished, failed, or was interrupted can't be quietly rerun. The report shows pre-2021 separately from 2021–2026 (which was already looked at) and counts negative SOXX periods, negative direction predictions, and negative-net V4 periods separately.

Until a licensed dataset passes, `confirmation.py status` shows `awaiting_confirmation_quality_data`, and no dataset lock or receipt is created.

Outputs go to `semiconductor_v4_continuous/artifacts/`. V4 isn't connected to paper or live trading.
