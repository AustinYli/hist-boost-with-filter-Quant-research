#!/usr/bin/env python3
"""Frozen-feature horizon sweep for the semiconductor strategy.

Research-only. This does not submit orders and does not change the frozen
5-day paper bot. It tests whether shorter/longer targets work better under
the same model family, frozen feature list, SOXX 200d regime filter, and
portfolio construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_frozen_manifest import FROZEN_RULES, SELECTION_FEATURES
from ml_semiconductor_regime_overlay_tests import backtest_selection_with_exposure, regime_exposures
from ml_semiconductor_robustness_tests import performance_from_returns
from ml_semiconductor_sector_model import (
    PortfolioRules,
    add_features,
    add_market_features,
    decile_return_table,
    load_bars,
    monthly_rank_ic,
    pick_metrics,
    precision_at_k,
    rank_ic_by_date,
)
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE, make_hgb, walk_forward_predictions


EPSILON = 1e-12


def active_metrics(model_daily: pd.DataFrame, benchmark_daily: pd.DataFrame) -> dict:
    merged = (
        model_daily[["date", "net_return"]]
        .rename(columns={"net_return": "model_return"})
        .merge(
            benchmark_daily[["date", "net_return"]].rename(columns={"net_return": "benchmark_return"}),
            on="date",
            how="inner",
        )
    )
    if merged.empty:
        return {}
    merged["active_return"] = merged["model_return"] - merged["benchmark_return"]
    active = merged["active_return"].astype(float)
    tracking_error = float(active.std() * np.sqrt(252))
    active_ann = float(active.mean() * 252)
    return {
        "active_total_return_sum": float(active.sum()),
        "active_annual_return_simple": active_ann,
        "tracking_error": tracking_error,
        "information_ratio": float(active_ann / tracking_error) if tracking_error > EPSILON else 0.0,
        "positive_active_day_rate": float((active > 0).mean()),
    }


def top_bottom_spread(predictions: pd.DataFrame) -> float | None:
    deciles = decile_return_table(predictions)
    bottom = next((row for row in deciles if row["decile"] == 1), None)
    top = next((row for row in deciles if row["decile"] == 10), None)
    if not bottom or not top:
        return None
    return float(top["avg_future_return"] - bottom["avg_future_return"])


def summarize_predictions(predictions: pd.DataFrame, rules: PortfolioRules) -> dict:
    rank_ic = rank_ic_by_date(predictions)
    monthly_ic = monthly_rank_ic(predictions)
    monthly_values = [row["mean_rank_ic"] for row in monthly_ic]
    precision = precision_at_k(predictions)
    picks = pick_metrics(predictions, rules)
    return {
        "mean_rank_ic": rank_ic.get("mean_rank_ic"),
        "positive_rank_ic_rate": rank_ic.get("positive_rank_ic_rate"),
        "monthly_rank_ic_mean": float(np.mean(monthly_values)) if monthly_values else None,
        "monthly_rank_ic_positive_rate": float((np.array(monthly_values) > 0).mean()) if monthly_values else None,
        "top_minus_bottom_future_return": top_bottom_spread(predictions),
        "precision_at_1": precision.get("precision_at_1"),
        "precision_at_5": precision.get("precision_at_5"),
        "top_pick_win_rate": picks.get("raw_win_rate_5d"),
        "top_pick_avg_future_return": picks.get("avg_future_return_5d"),
        "top_pick_median_future_return": picks.get("median_future_return_5d"),
    }


def run_horizon(args: argparse.Namespace, horizon: int) -> dict:
    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, all_features = add_features(bars, market_features, horizon)
    data = data.dropna(subset=all_features + ["future_return", "next_1d_return"]).reset_index(drop=True)

    missing_frozen = [feature for feature in SELECTION_FEATURES if feature not in data.columns]
    if missing_frozen:
        raise ValueError(f"Missing frozen features for horizon {horizon}: {missing_frozen}")

    predictions = walk_forward_predictions(
        data,
        SELECTION_FEATURES,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    dates = set(predictions["date"].unique())
    exposures = regime_exposures(data, Path(args.market_bars), dates)
    exposure_200d = dict(zip(exposures["date"], exposures["exposure_soxx_200d_filter"]))
    rules = PortfolioRules(**{**asdict(FROZEN_RULES), "rebalance_every_days": horizon})

    strategy = backtest_selection_with_exposure(predictions, rules, exposure_200d, equal_weight=False)
    equal_weight = backtest_selection_with_exposure(predictions, rules, exposure_200d, equal_weight=True)
    momentum_predictions = data[data["date"].isin(dates)].copy()
    momentum_predictions["predicted_return"] = momentum_predictions["return_60d"]
    momentum = backtest_selection_with_exposure(momentum_predictions, rules, exposure_200d, equal_weight=False)

    summary = {
        "horizon_days": horizon,
        "rebalance_every_days": horizon,
        "rows": int(len(data)),
        "oos_rows": int(len(predictions)),
        "oos_start": str(min(dates)) if dates else None,
        "oos_end": str(max(dates)) if dates else None,
        "model": strategy["metrics"],
        "filtered_equal_weight": equal_weight["metrics"],
        "filtered_60d_momentum": momentum["metrics"],
        "active_vs_equal_weight": active_metrics(strategy["daily"], equal_weight["daily"]),
        "active_vs_60d_momentum": active_metrics(strategy["daily"], momentum["daily"]),
        "ranking": summarize_predictions(predictions, rules),
        "portfolio_rules": asdict(rules),
        "feature_count_warmup_matrix": len(all_features),
        "selection_features": SELECTION_FEATURES,
    }
    return summary


def flatten_summary(row: dict) -> dict:
    model = row["model"]
    eq = row["filtered_equal_weight"]
    mom = row["filtered_60d_momentum"]
    rank = row["ranking"]
    active_eq = row["active_vs_equal_weight"]
    active_mom = row["active_vs_60d_momentum"]
    return {
        "horizon_days": row["horizon_days"],
        "rebalance_every_days": row["rebalance_every_days"],
        "model_total_return": model.get("total_return"),
        "model_sharpe": model.get("sharpe_like"),
        "model_max_drawdown": model.get("max_drawdown"),
        "model_avg_turnover": model.get("avg_turnover"),
        "model_avg_exposure": model.get("avg_exposure"),
        "equal_weight_total_return": eq.get("total_return"),
        "momentum_total_return": mom.get("total_return"),
        "active_ir_vs_equal_weight": active_eq.get("information_ratio"),
        "active_sum_vs_equal_weight": active_eq.get("active_total_return_sum"),
        "active_ir_vs_momentum": active_mom.get("information_ratio"),
        "active_sum_vs_momentum": active_mom.get("active_total_return_sum"),
        "mean_rank_ic": rank.get("mean_rank_ic"),
        "positive_rank_ic_rate": rank.get("positive_rank_ic_rate"),
        "monthly_rank_ic_mean": rank.get("monthly_rank_ic_mean"),
        "top_minus_bottom_future_return": rank.get("top_minus_bottom_future_return"),
        "precision_at_1": rank.get("precision_at_1"),
        "precision_at_5": rank.get("precision_at_5"),
        "top_pick_win_rate": rank.get("top_pick_win_rate"),
        "top_pick_avg_future_return": rank.get("top_pick_avg_future_return"),
        "oos_rows": row["oos_rows"],
        "oos_start": row["oos_start"],
        "oos_end": row["oos_end"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen-feature horizon sweep.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_horizon_sweep")
    parser.add_argument("--horizons", default="1,3,5,10")
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    args = parser.parse_args()

    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    results = [run_horizon(args, horizon) for horizon in horizons]
    rows = [flatten_summary(result) for result in results]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "horizon_sweep_results.json").write_text(
        json.dumps({"metadata": vars(args), "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(out_dir / "horizon_sweep_summary.csv", index=False)
    print(json.dumps({"summary": rows, "out_dir": str(out_dir)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
