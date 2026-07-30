#!/usr/bin/env python3
"""Within-date shuffled-target confirmation test for the frozen strategy.

Research-only. This does not change the frozen strategy and does not trade.

The null is:
    features contain no cross-sectional stock-selection information.

For each run, future_return labels are permuted within each date before model
training. Portfolio performance is then evaluated against the real, unshuffled
future returns and real next-day returns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_frozen_manifest import FROZEN_RULES, SELECTION_FEATURES
from ml_semiconductor_regime_overlay_tests import (
    backtest_selection_with_exposure,
    choose_selection_weights,
)
from ml_semiconductor_robustness_tests import (
    active_regression,
    equal_weight_returns,
    market_returns,
)
from ml_semiconductor_sector_model import add_features, add_market_features, load_bars
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE, make_hgb, rank_ic_by_date, walk_forward_predictions


def load_frozen_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, all_features = add_features(bars, market_features, args.horizon_days)
    data = data.dropna(subset=all_features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    return data, all_features


def ma_exposure_map(market_path: Path, dates: set, ma_length: int = 200) -> dict:
    market = load_bars(market_path)
    soxx = market[market["symbol"] == "SOXX"].sort_values("date").copy()
    soxx[f"ma_{ma_length}"] = soxx["close"].rolling(ma_length).mean()
    soxx = soxx[soxx["date"].isin(dates)]
    return {
        row["date"]: float(row["close"] > row[f"ma_{ma_length}"]) if pd.notna(row[f"ma_{ma_length}"]) else 0.0
        for _, row in soxx.iterrows()
    }


def shuffle_targets_within_date(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = data.copy()
    shuffled["future_return"] = shuffled.groupby("date")["future_return"].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    return shuffled


def restore_actual_future_returns(predictions: pd.DataFrame, actual_data: pd.DataFrame) -> pd.DataFrame:
    actual = actual_data[["date", "symbol", "future_return"]].rename(columns={"future_return": "actual_future_return"})
    out = predictions.drop(columns=["future_return"], errors="ignore").merge(actual, on=["date", "symbol"], how="left")
    out = out.rename(columns={"actual_future_return": "future_return"})
    return out


def rank_metrics(predictions: pd.DataFrame) -> dict:
    frame = predictions.dropna(subset=["predicted_return", "future_return"]).copy()
    frame["rank_pct"] = frame.groupby("date")["predicted_return"].rank(pct=True)
    top = frame[frame["rank_pct"] >= 0.75]
    bottom = frame[frame["rank_pct"] <= 0.25]
    return {
        "rank_ic": rank_ic_by_date(frame),
        "top_quartile_return": float(top["future_return"].mean()) if len(top) else None,
        "bottom_quartile_return": float(bottom["future_return"].mean()) if len(bottom) else None,
        "top_minus_bottom_spread": float(top["future_return"].mean() - bottom["future_return"].mean())
        if len(top) and len(bottom)
        else None,
    }


def build_momentum_strategy(data: pd.DataFrame, dates: set, exposure: dict) -> dict:
    frame = data[data["date"].isin(dates)].copy()
    frame["predicted_return"] = frame["return_60d"]
    return backtest_selection_with_exposure(frame, FROZEN_RULES, exposure, equal_weight=False)


def one_way_turnover(current: dict[str, float], target: dict[str, float]) -> float:
    return 0.5 * sum(abs(target.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in set(current) | set(target))


def selection_diagnostics(predictions: pd.DataFrame, exposure: dict) -> dict:
    current: dict[str, float] = {}
    holdings = []
    turnover_rows = []
    for index, date in enumerate(sorted(predictions["date"].unique())[:-1]):
        day = predictions[predictions["date"] == date].dropna(subset=["predicted_return", "next_1d_return"])
        if day.empty:
            continue
        if index % FROZEN_RULES.rebalance_every_days != 0:
            continue
        base = choose_selection_weights(day, current, FROZEN_RULES)
        target = {symbol: weight * float(exposure.get(date, 1.0)) for symbol, weight in base.items()}
        target = {symbol: weight for symbol, weight in target.items() if weight > 1e-12}
        turnover_rows.append(
            {
                "date": str(date),
                "one_way_turnover": one_way_turnover(current, target),
                "two_way_turnover": 2 * one_way_turnover(current, target),
                "exposure": sum(target.values()),
                "holding_count": len(target),
            }
        )
        for symbol, weight in target.items():
            row = day[day["symbol"] == symbol].iloc[0]
            holdings.append(
                {
                    "date": str(date),
                    "symbol": symbol,
                    "weight": float(weight),
                    "predicted_return": float(row["predicted_return"]),
                    "future_return": float(row["future_return"]),
                }
            )
        current = target
    holdings_df = pd.DataFrame(holdings)
    turnover_df = pd.DataFrame(turnover_rows)
    if holdings_df.empty:
        return {}
    date_count = holdings_df["date"].nunique()
    holdings_df["month"] = pd.to_datetime(holdings_df["date"]).dt.to_period("M").astype(str)
    return {
        "most_selected_symbols": holdings_df["symbol"].value_counts().head(12).to_dict(),
        "lite_selection_rate": float((holdings_df["symbol"] == "LITE").sum() / max(date_count, 1)),
        "nvda_selection_rate": float((holdings_df["symbol"] == "NVDA").sum() / max(date_count, 1)),
        "avg_holding_count": float(turnover_df["holding_count"].mean()),
        "avg_one_way_turnover_per_rebalance": float(turnover_df["one_way_turnover"].mean()),
        "avg_two_way_turnover_per_rebalance": float(turnover_df["two_way_turnover"].mean()),
        "avg_exposure": float(turnover_df["exposure"].mean()),
        "exposure_by_month": turnover_df.assign(month=lambda df: pd.to_datetime(df["date"]).dt.to_period("M").astype(str))
        .groupby("month")["exposure"]
        .mean()
        .to_dict(),
        "avg_weight_by_symbol_top12": holdings_df.groupby("symbol")["weight"].mean().sort_values(ascending=False).head(12).to_dict(),
    }


def evaluate_predictions(name: str, predictions: pd.DataFrame, data: pd.DataFrame, exposure: dict, market_path: Path) -> dict:
    dates = set(predictions["date"].unique())
    strategy = backtest_selection_with_exposure(predictions, FROZEN_RULES, exposure, equal_weight=False)
    equal = backtest_selection_with_exposure(predictions, FROZEN_RULES, exposure, equal_weight=True)
    soxx = market_returns(market_path, "SOXX", dates)
    momentum = build_momentum_strategy(data, dates, exposure)
    active = active_regression(strategy["daily"], equal["daily"], soxx, momentum["daily"])
    ranks = rank_metrics(predictions)
    return {
        "name": name,
        "portfolio": strategy["metrics"],
        "matching_equal_weight": equal["metrics"],
        "active": active,
        "rank_metrics": ranks,
        "selection_diagnostics": selection_diagnostics(predictions, exposure),
    }


def p_value(real_value: float | None, null_values: list[float | None]) -> float | None:
    values = [value for value in null_values if value is not None and np.isfinite(value)]
    if real_value is None or not values:
        return None
    return float((sum(value >= real_value for value in values) + 1) / (len(values) + 1))


def summarize_null(values: list[float | None]) -> dict:
    series = pd.Series([value for value in values if value is not None], dtype=float).dropna()
    if series.empty:
        return {}
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "positive_rate": float((series > 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run within-date shuffled target confirmation tests.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_shuffle_confirmation")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    data, _ = load_frozen_data(args)
    real_predictions = walk_forward_predictions(
        data,
        SELECTION_FEATURES,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    dates = set(real_predictions["date"].unique())
    exposure = ma_exposure_map(Path(args.market_bars), dates, 200)
    real = evaluate_predictions("real_frozen", real_predictions, data, exposure, Path(args.market_bars))

    shuffled_runs = []
    for index in range(args.runs):
        run_seed = args.seed + 1000 + index
        shuffled_data = shuffle_targets_within_date(data, run_seed)
        shuffled_predictions = walk_forward_predictions(
            shuffled_data,
            SELECTION_FEATURES,
            lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
            args.initial_train_fraction,
        )
        shuffled_predictions = restore_actual_future_returns(shuffled_predictions, data)
        report = evaluate_predictions(f"within_date_shuffle_{index:03d}", shuffled_predictions, data, exposure, Path(args.market_bars))
        report["seed"] = run_seed
        shuffled_runs.append(report)
        print(
            json.dumps(
                {
                    "run": index + 1,
                    "seed": run_seed,
                    "return": report["portfolio"].get("total_return"),
                    "sharpe": report["portfolio"].get("sharpe_like"),
                    "active_return": report["active"]["active_metrics_vs_equal_weight"].get("total_return"),
                    "rank_ic": report["rank_metrics"]["rank_ic"].get("mean_rank_ic"),
                }
            ),
            flush=True,
        )

    best_by_active = max(
        shuffled_runs,
        key=lambda run: run["active"]["active_metrics_vs_equal_weight"].get("total_return", float("-inf")),
    )
    metrics = {
        "total_return": [run["portfolio"].get("total_return") for run in shuffled_runs],
        "sharpe_like": [run["portfolio"].get("sharpe_like") for run in shuffled_runs],
        "active_return": [run["active"]["active_metrics_vs_equal_weight"].get("total_return") for run in shuffled_runs],
        "active_information_ratio": [run["active"].get("information_ratio") for run in shuffled_runs],
        "alpha_t_stat": [run["active"].get("ols_alpha_t_stat") for run in shuffled_runs],
        "rank_ic": [run["rank_metrics"]["rank_ic"].get("mean_rank_ic") for run in shuffled_runs],
        "top_bottom_spread": [run["rank_metrics"].get("top_minus_bottom_spread") for run in shuffled_runs],
        "max_drawdown": [run["portfolio"].get("max_drawdown") for run in shuffled_runs],
    }
    real_values = {
        "total_return": real["portfolio"].get("total_return"),
        "sharpe_like": real["portfolio"].get("sharpe_like"),
        "active_return": real["active"]["active_metrics_vs_equal_weight"].get("total_return"),
        "active_information_ratio": real["active"].get("information_ratio"),
        "alpha_t_stat": real["active"].get("ols_alpha_t_stat"),
        "rank_ic": real["rank_metrics"]["rank_ic"].get("mean_rank_ic"),
        "top_bottom_spread": real["rank_metrics"].get("top_minus_bottom_spread"),
        "max_drawdown": real["portfolio"].get("max_drawdown"),
    }
    output = {
        "metadata": {
            "runs": args.runs,
            "null": "future_return labels permuted within each date before model training",
            "evaluation": "portfolio and ranking metrics use real unshuffled returns",
            "frozen_rules": asdict(FROZEN_RULES),
        },
        "real_frozen": real,
        "p_values": {metric: p_value(real_values[metric], values) for metric, values in metrics.items()},
        "null_distributions": {metric: summarize_null(values) for metric, values in metrics.items()},
        "best_shuffled_by_active_return": best_by_active,
        "shuffled_runs": shuffled_runs,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shuffle_confirmation_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    summary_rows = []
    for run in shuffled_runs:
        summary_rows.append(
            {
                "name": run["name"],
                "seed": run["seed"],
                "total_return": run["portfolio"].get("total_return"),
                "sharpe_like": run["portfolio"].get("sharpe_like"),
                "active_return": run["active"]["active_metrics_vs_equal_weight"].get("total_return"),
                "active_information_ratio": run["active"].get("information_ratio"),
                "alpha_t_stat": run["active"].get("ols_alpha_t_stat"),
                "rank_ic": run["rank_metrics"]["rank_ic"].get("mean_rank_ic"),
                "top_bottom_spread": run["rank_metrics"].get("top_minus_bottom_spread"),
                "max_drawdown": run["portfolio"].get("max_drawdown"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "shuffle_confirmation_summary.csv", index=False)
    print(json.dumps({"real": real_values, "p_values": output["p_values"], "null_distributions": output["null_distributions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
