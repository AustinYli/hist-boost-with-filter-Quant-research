#!/usr/bin/env python3
"""SOXX regime-filter research for copied 3d/5d HGB models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE = PROJECT_DIR.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from ml_semiconductor_frozen_manifest import FROZEN_RULES, SELECTION_FEATURES
from ml_semiconductor_frozen_spec_diagnostics import alpha_regression_hac
from ml_semiconductor_regime_overlay_tests import backtest_selection_with_exposure, regime_exposures
from ml_semiconductor_robustness_tests import active_regression, market_returns
from ml_semiconductor_sector_model import PortfolioRules, add_features, add_market_features, load_bars
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE, make_hgb, walk_forward_predictions


SOURCE_MODEL_ROOT = WORKSPACE / "models" / "semiconductor_fixed_weight"
COPY_MODEL_ROOT = PROJECT_DIR / "models"
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts"
DEFAULT_STOCK_BARS = WORKSPACE / "data" / "semiconductor_sector_1d_5y.csv"
DEFAULT_MARKET_BARS = WORKSPACE / "data" / "semiconductor_market_1d_5y.csv"

ORIGINAL_OOS_END = {
    3: pd.Timestamp("2026-07-27").date(),
    5: pd.Timestamp("2026-07-23").date(),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copied_model_manifest() -> dict:
    copies = []
    for moving_average_days in (100, 300):
        for horizon in (3, 5):
            source_dir = SOURCE_MODEL_ROOT / f"hgb_{horizon}d"
            copy_dir = COPY_MODEL_ROOT / f"hgb_{horizon}d_{moving_average_days}d_filter"
            files = []
            for source_name, copy_name in (
                ("model.joblib", "model.joblib"),
                ("metadata.json", "source_metadata.json"),
            ):
                source = source_dir / source_name
                copied = copy_dir / copy_name
                if not source.exists() or not copied.exists():
                    raise FileNotFoundError(f"Missing source/copy pair: {source} -> {copied}")
                source_hash = sha256(source)
                copied_hash = sha256(copied)
                if source_hash != copied_hash:
                    raise ValueError(f"Copied file does not match its source: {copied}")
                if copied.is_symlink():
                    raise ValueError(f"Copied file must not be a symlink: {copied}")
                files.append(
                    {
                        "source": str(source),
                        "copy": str(copied),
                        "source_sha256": source_hash,
                        "copy_sha256": copied_hash,
                        "copy_is_symlink": False,
                    }
                )
            copies.append(
                {
                    "horizon_days": horizon,
                    "moving_average_days": moving_average_days,
                    "files": files,
                }
            )
    return {
        "copy_type": "physical_independent_files",
        "selector_model": "unchanged copied HistGradientBoostingRegressor",
        "variant_change": "SOXX regime moving-average window: 200 sessions -> isolated 100/300-session variants",
        "trading_integration": "none",
        "copies": copies,
    }


def load_horizon_data(stock_bars: Path, market_bars: Path, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    bars = load_bars(stock_bars)
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, market_bars)
    data, all_features = add_features(bars, market_features, horizon)
    data = data.dropna(subset=all_features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    data = data[data["date"] <= ORIGINAL_OOS_END[horizon]].reset_index(drop=True)
    return data, all_features


def add_isolated_exposures(exposure_frame: pd.DataFrame, market_bars: Path, dates: set) -> pd.DataFrame:
    market = load_bars(market_bars)
    soxx = market[market["symbol"] == "SOXX"].sort_values("date").copy()
    soxx["ma_200_local"] = soxx["close"].rolling(200).mean()
    soxx["ma_300"] = soxx["close"].rolling(300).mean()
    soxx["exposure_soxx_200d_minus10pct_filter"] = (
        soxx["close"] >= soxx["ma_200_local"] * 0.90
    ).astype(float)
    soxx["exposure_soxx_200d_plus20pct_filter"] = (
        soxx["close"] >= soxx["ma_200_local"] * 1.20
    ).astype(float)
    soxx["exposure_soxx_200d_plus40pct_filter"] = (
        soxx["close"] >= soxx["ma_200_local"] * 1.40
    ).astype(float)
    soxx["exposure_soxx_300d_filter"] = (soxx["close"] > soxx["ma_300"]).astype(float)
    extra = soxx[soxx["date"].isin(dates)][
        [
            "date",
            "ma_300",
            "exposure_soxx_200d_minus10pct_filter",
            "exposure_soxx_200d_plus20pct_filter",
            "exposure_soxx_200d_plus40pct_filter",
            "exposure_soxx_300d_filter",
        ]
    ]
    merged = exposure_frame.merge(extra, on="date", how="left", validate="one_to_one")
    required = [
        "exposure_soxx_200d_minus10pct_filter",
        "exposure_soxx_200d_plus20pct_filter",
        "exposure_soxx_200d_plus40pct_filter",
        "exposure_soxx_300d_filter",
    ]
    if merged[required].isna().any().any():
        raise ValueError("Missing isolated SOXX exposure for one or more out-of-sample dates.")
    return merged


def evaluate_filter(
    *,
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    market_bars: Path,
    rules: PortfolioRules,
    horizon: int,
    exposure: dict,
) -> dict:
    dates = set(predictions["date"].unique())
    strategy = backtest_selection_with_exposure(predictions, rules, exposure, equal_weight=False)
    equal_weight = backtest_selection_with_exposure(predictions, rules, exposure, equal_weight=True)

    momentum_predictions = data[data["date"].isin(dates)].copy()
    momentum_predictions["predicted_return"] = momentum_predictions["return_60d"]
    momentum = backtest_selection_with_exposure(
        momentum_predictions,
        rules,
        {date: 1.0 for date in dates},
        equal_weight=False,
    )
    soxx = market_returns(market_bars, "SOXX", dates)

    return {
        "strategy": strategy["metrics"],
        "matching_filtered_equal_weight": equal_weight["metrics"],
        "ordinary_factor_alpha": active_regression(
            strategy["daily"], equal_weight["daily"], soxx, momentum["daily"]
        ),
        "newey_west_factor_alpha": alpha_regression_hac(
            strategy["daily"], equal_weight["daily"], soxx, momentum["daily"], lag=horizon
        ),
    }


def run_horizon(stock_bars: Path, market_bars: Path, horizon: int) -> dict:
    data, all_features = load_horizon_data(stock_bars, market_bars, horizon)
    predictions = walk_forward_predictions(
        data,
        SELECTION_FEATURES,
        lambda: make_hgb(181, 220, 0.035),
        0.60,
    )
    dates = set(predictions["date"].unique())
    exposure_frame = regime_exposures(data, market_bars, dates)
    exposure_frame = add_isolated_exposures(exposure_frame, market_bars, dates)
    rules = PortfolioRules(**{**asdict(FROZEN_RULES), "rebalance_every_days": horizon})

    variants = {}
    for label, column in (
        ("soxx_100d_filter", "exposure_soxx_100d_filter"),
        ("soxx_200d_baseline", "exposure_soxx_200d_filter"),
        ("soxx_200d_minus10pct_filter", "exposure_soxx_200d_minus10pct_filter"),
        ("soxx_200d_plus20pct_filter", "exposure_soxx_200d_plus20pct_filter"),
        ("soxx_200d_plus40pct_filter", "exposure_soxx_200d_plus40pct_filter"),
        ("soxx_300d_filter", "exposure_soxx_300d_filter"),
    ):
        exposure = dict(zip(exposure_frame["date"], exposure_frame[column]))
        variants[label] = evaluate_filter(
            predictions=predictions,
            data=data,
            market_bars=market_bars,
            rules=rules,
            horizon=horizon,
            exposure=exposure,
        )

    variant = variants["soxx_100d_filter"]
    baseline = variants["soxx_200d_baseline"]
    variant_minus10 = variants["soxx_200d_minus10pct_filter"]
    variant_plus20 = variants["soxx_200d_plus20pct_filter"]
    variant_plus40 = variants["soxx_200d_plus40pct_filter"]
    variant_300 = variants["soxx_300d_filter"]
    return {
        "horizon_days": horizon,
        "rows": int(len(data)),
        "feature_matrix_count": int(len(all_features)),
        "selection_feature_count": int(len(SELECTION_FEATURES)),
        "oos_rows": int(len(predictions)),
        "oos_dates": int(len(dates)),
        "oos_start": str(min(dates)),
        "oos_end": str(max(dates)),
        "portfolio_rules": asdict(rules),
        "variants": variants,
        "delta_100d_minus_200d": {
            "total_return": variant["strategy"]["total_return"] - baseline["strategy"]["total_return"],
            "annual_return": variant["strategy"]["annual_return"] - baseline["strategy"]["annual_return"],
            "sharpe_like": variant["strategy"]["sharpe_like"] - baseline["strategy"]["sharpe_like"],
            "max_drawdown": variant["strategy"]["max_drawdown"] - baseline["strategy"]["max_drawdown"],
            "annualized_hac_alpha": (
                variant["newey_west_factor_alpha"]["annualized_alpha_simple"]
                - baseline["newey_west_factor_alpha"]["annualized_alpha_simple"]
            ),
        },
        "delta_300d_minus_200d": {
            "total_return": variant_300["strategy"]["total_return"] - baseline["strategy"]["total_return"],
            "annual_return": variant_300["strategy"]["annual_return"] - baseline["strategy"]["annual_return"],
            "sharpe_like": variant_300["strategy"]["sharpe_like"] - baseline["strategy"]["sharpe_like"],
            "max_drawdown": variant_300["strategy"]["max_drawdown"] - baseline["strategy"]["max_drawdown"],
            "annualized_hac_alpha": (
                variant_300["newey_west_factor_alpha"]["annualized_alpha_simple"]
                - baseline["newey_west_factor_alpha"]["annualized_alpha_simple"]
            ),
        },
        "delta_200d_plus20pct_minus_200d": {
            "total_return": variant_plus20["strategy"]["total_return"] - baseline["strategy"]["total_return"],
            "annual_return": variant_plus20["strategy"]["annual_return"] - baseline["strategy"]["annual_return"],
            "sharpe_like": variant_plus20["strategy"]["sharpe_like"] - baseline["strategy"]["sharpe_like"],
            "max_drawdown": variant_plus20["strategy"]["max_drawdown"] - baseline["strategy"]["max_drawdown"],
            "annualized_hac_alpha": (
                variant_plus20["newey_west_factor_alpha"]["annualized_alpha_simple"]
                - baseline["newey_west_factor_alpha"]["annualized_alpha_simple"]
            ),
        },
        "delta_200d_minus10pct_minus_200d": {
            "total_return": variant_minus10["strategy"]["total_return"] - baseline["strategy"]["total_return"],
            "annual_return": variant_minus10["strategy"]["annual_return"] - baseline["strategy"]["annual_return"],
            "sharpe_like": variant_minus10["strategy"]["sharpe_like"] - baseline["strategy"]["sharpe_like"],
            "max_drawdown": variant_minus10["strategy"]["max_drawdown"] - baseline["strategy"]["max_drawdown"],
            "annualized_hac_alpha": (
                variant_minus10["newey_west_factor_alpha"]["annualized_alpha_simple"]
                - baseline["newey_west_factor_alpha"]["annualized_alpha_simple"]
            ),
        },
        "delta_200d_plus40pct_minus_200d": {
            "total_return": variant_plus40["strategy"]["total_return"] - baseline["strategy"]["total_return"],
            "annual_return": variant_plus40["strategy"]["annual_return"] - baseline["strategy"]["annual_return"],
            "sharpe_like": variant_plus40["strategy"]["sharpe_like"] - baseline["strategy"]["sharpe_like"],
            "max_drawdown": variant_plus40["strategy"]["max_drawdown"] - baseline["strategy"]["max_drawdown"],
            "annualized_hac_alpha": (
                variant_plus40["newey_west_factor_alpha"]["annualized_alpha_simple"]
                - baseline["newey_west_factor_alpha"]["annualized_alpha_simple"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate isolated 100d/300d-filter copies of the 3d/5d models.")
    parser.add_argument("--stock-bars", type=Path, default=DEFAULT_STOCK_BARS)
    parser.add_argument("--market-bars", type=Path, default=DEFAULT_MARKET_BARS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "active_models_modified": False,
        "comparison_design": {
            "single_change": "SOXX regime rule variants versus the close-above-MA200 baseline",
            "model": "HGB excluding the frozen momentum-like features",
            "training": "60% initial chronological train, monthly expanding walk-forward refits",
            "execution": "after-close signal, next-session execution assumption",
            "costs_bps": 5,
            "factor_alpha_controls": ["SOXX daily return", "top-10 60-day momentum daily return"],
            "newey_west_lag": "equal to horizon_days",
            "data_endpoints": {str(key): str(value) for key, value in ORIGINAL_OOS_END.items()},
        },
        "model_copy_manifest": copied_model_manifest(),
        "results": [run_horizon(args.stock_bars, args.market_bars, horizon) for horizon in (3, 5)],
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.artifact_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    (args.artifact_dir / "model_copy_manifest.json").write_text(
        json.dumps(output["model_copy_manifest"], indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
