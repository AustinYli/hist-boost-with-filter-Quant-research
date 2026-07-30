#!/usr/bin/env python3
"""Write the permanent frozen strategy manifest.

This records the chosen specification. It does not tune, backtest, or trade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ml_semiconductor_sector_model import PortfolioRules
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE


SELECTION_FEATURES = [
    "volatility_20d",
    "volatility_60d",
    "atr_20_pct",
    "parkinson_volatility_20d",
    "relative_volume_20d",
    "log_dollar_volume",
    "turnover_proxy",
    "body_pct",
    "range_pct",
    "close_position",
    "gap_from_yesterday",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "beta_20d_to_soxx",
    "beta_60d_to_soxx",
    "cs_rank_relative_volume_20d",
    "cs_rank_volatility_20d",
    "cs_rank_log_dollar_volume",
]


FROZEN_RULES = PortfolioRules(
    top_n=10,
    buy_rank_pct=0.25,
    sell_rank_pct=0.50,
    max_weight=0.10,
    no_trade_band=0.01,
    rebalance_every_days=5,
    transaction_cost_bps=5.0,
    min_predicted_return=0.0,
)


STRATEGY_FILES = [
    "ml_semiconductor_sector_model.py",
    "ml_semiconductor_walkforward_compare.py",
    "ml_semiconductor_regime_overlay_tests.py",
    "ml_semiconductor_robustness_tests.py",
    "ml_semiconductor_frozen_spec_diagnostics.py",
    "ml_semiconductor_frozen_manifest.py",
]


def git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the frozen semiconductor strategy manifest.")
    parser.add_argument("--out", default="models/semiconductor_frozen_spec/frozen_strategy_manifest.json")
    parser.add_argument("--freeze-date", default="2026-07-30")
    args = parser.parse_args()

    root = Path.cwd()
    git_head = git_output(["rev-parse", "--verify", "HEAD"])
    manifest = {
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_date": args.freeze_date,
        "git_commit_hash": git_head,
        "git_status_short": git_output(["status", "--short"]),
        "code_hashes_sha256": {file: sha256_file(root / file) for file in STRATEGY_FILES},
        "strategy_name": "frozen_hgb_ex_momentum_soxx_200d_long_only",
        "purpose": "Untouched forward confirmation test, not further historical optimization.",
        "model": {
            "type": "HistGradientBoostingRegressor",
            "hyperparameters": {
                "max_iter": 220,
                "max_leaf_nodes": 31,
                "learning_rate": 0.035,
                "l2_regularization": 0.2,
                "random_state": 181,
            },
            "training_rule": (
                "Expanding training window. For a signal on date T, train only on rows "
                "whose 5-trading-day future_return label is observable from data available at T."
            ),
            "missing_data": (
                "Rows with missing full 62-feature research matrix values or missing target are excluded before "
                "the frozen 18 selected features are used. This preserves the exact historical warmup/eligibility "
                "rule used when the specification was frozen."
            ),
        },
        "signal": {
            "horizon": "5 trading days",
            "selection_features": SELECTION_FEATURES,
            "execution_convention": "Signal generated after daily close; intended execution next session.",
            "data_sources": {
                "stock_bars": "data/semiconductor_sector_1d_5y.csv",
                "market_bars": "data/semiconductor_market_1d_5y.csv",
                "feed": "Alpaca IEX daily bars used in current research files",
            },
        },
        "universe": SEMICONDUCTOR_UNIVERSE,
        "regime_filter": {
            "name": "SOXX 200-day moving average filter",
            "rule": "Exposure is 1.0 if SOXX close is above its trailing 200-day simple moving average, else 0.0.",
            "unused_allocation": "cash",
        },
        "portfolio": {
            "rules": FROZEN_RULES.__dict__,
            "weighting": "Risk-adjusted predicted score within selected top names; final weights multiplied by regime exposure.",
            "rebalance_schedule": "Every 5 trading days in backtest; forward signal generation should be archived on scheduled rebalance dates.",
            "transaction_cost_formula": "cost = turnover * transaction_cost_bps / 10000; turnover convention is two-way absolute weight change.",
            "one_way_turnover_formula": "0.5 * sum(abs(new_weight_i - old_weight_i))",
        },
        "benchmarks": [
            "Equal-weight semiconductors with same SOXX 200d filter",
            "60-day momentum selection with same SOXX 200d filter",
            "SOXX buy-and-hold",
        ],
        "pass_fail_criteria": {
            "minimum_confirmation_period": "Do not judge from a few weeks; collect an untouched sequence first.",
            "pass_directional": [
                "active return vs filtered equal-weight semis > 0",
                "active information ratio > 0",
                "top-ranked names outperform bottom-ranked names",
                "average Rank IC > 0",
                "turnover and implementation shortfall close to assumptions",
            ],
            "failure_warning": [
                "drawdown exceeds predetermined limit",
                "performance depends on one stock or one month",
                "realized costs materially exceed assumptions",
                "filter-off behavior or filter-on behavior contradicts documented assumptions",
            ],
        },
        "research_selection_bias_note": (
            "The historical period was used to choose this specification. "
            "It cannot independently prove the strategy. Forward data after the freeze date is the confirmation test."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "git_commit_hash": git_head, "strategy_name": manifest["strategy_name"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
