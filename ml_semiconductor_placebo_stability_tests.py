#!/usr/bin/env python3
"""Placebo and parameter-stability tests for the frozen semiconductor strategy.

Research-only. These tests diagnose overfitting risk and must not be used to
select a replacement strategy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_frozen_manifest import FROZEN_RULES, SELECTION_FEATURES
from ml_semiconductor_regime_overlay_tests import backtest_selection_with_exposure
from ml_semiconductor_sector_model import PortfolioRules, add_features, add_market_features, load_bars
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE, make_hgb, walk_forward_predictions


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, all_features = add_features(bars, market_features, args.horizon_days)
    return data.dropna(subset=all_features + ["future_return", "next_1d_return"]).reset_index(drop=True)


def ma_exposure_map(market_path: Path, dates: set, ma_length: int) -> dict:
    market = load_bars(market_path)
    soxx = market[market["symbol"] == "SOXX"].sort_values("date").copy()
    soxx[f"ma_{ma_length}"] = soxx["close"].rolling(ma_length).mean()
    soxx = soxx[soxx["date"].isin(dates)]
    return {
        row["date"]: float(row["close"] > row[f"ma_{ma_length}"]) if pd.notna(row[f"ma_{ma_length}"]) else 0.0
        for _, row in soxx.iterrows()
    }


def metric_row(name: str, metrics: dict, **extra) -> dict:
    return {
        "name": name,
        "total_return": metrics.get("total_return"),
        "sharpe_like": metrics.get("sharpe_like"),
        "max_drawdown": metrics.get("max_drawdown"),
        "avg_exposure": metrics.get("avg_exposure"),
        **extra,
    }


def random_ranking_placebo(predictions: pd.DataFrame, exposure: dict, runs: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(runs):
        placebo = predictions.copy()
        placebo["predicted_return"] = rng.normal(size=len(placebo))
        metrics = backtest_selection_with_exposure(placebo, FROZEN_RULES, exposure, equal_weight=False)["metrics"]
        rows.append(metric_row(f"random_ranking_{index:03d}", metrics, placebo_type="random_ranking"))
    return rows


def shuffle_targets_within_date(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = data.copy()
    shuffled["future_return"] = shuffled.groupby("date")["future_return"].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    return shuffled


def shuffled_target_placebo(
    data: pd.DataFrame,
    exposure: dict,
    runs: int,
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for index in range(runs):
        placebo_data = shuffle_targets_within_date(data, args.seed + 1000 + index)
        predictions = walk_forward_predictions(
            placebo_data,
            SELECTION_FEATURES,
            lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
            args.initial_train_fraction,
        )
        metrics = backtest_selection_with_exposure(predictions, FROZEN_RULES, exposure, equal_weight=False)["metrics"]
        rows.append(metric_row(f"shuffled_target_{index:03d}", metrics, placebo_type="shuffled_target"))
    return rows


def p_value(real_metric: float, placebo_values: list[float]) -> float:
    count = sum(value >= real_metric for value in placebo_values if value is not None)
    n = sum(value is not None for value in placebo_values)
    return float((count + 1) / (n + 1)) if n else float("nan")


def stability_map(predictions: pd.DataFrame, market_path: Path, dates: set) -> list[dict]:
    rows = []
    for rebalance_days in (4, 5, 6, 7, 10):
        for top_n in (8, 10, 12, 15):
            for max_weight in (0.08, 0.10, 0.12):
                for ma_length in (180, 200, 220):
                    rules = PortfolioRules(**{**asdict(FROZEN_RULES), "rebalance_every_days": rebalance_days, "top_n": top_n, "max_weight": max_weight})
                    exposure = ma_exposure_map(market_path, dates, ma_length)
                    metrics = backtest_selection_with_exposure(predictions, rules, exposure, equal_weight=False)["metrics"]
                    rows.append(
                        metric_row(
                            f"nearby_r{rebalance_days}_top{top_n}_w{int(max_weight * 100)}_ma{ma_length}",
                            metrics,
                            rebalance_every_days=rebalance_days,
                            top_n=top_n,
                            max_weight=max_weight,
                            ma_length=ma_length,
                        )
                    )
    return rows


def model_neighborhood(data: pd.DataFrame, exposure: dict, args: argparse.Namespace) -> list[dict]:
    configs = [
        {"seed": args.seed, "n_estimators": 180, "learning_rate": 0.035},
        {"seed": args.seed, "n_estimators": 260, "learning_rate": 0.035},
        {"seed": args.seed + 1, "n_estimators": 220, "learning_rate": 0.035},
        {"seed": args.seed + 2, "n_estimators": 220, "learning_rate": 0.030},
        {"seed": args.seed + 3, "n_estimators": 220, "learning_rate": 0.040},
    ]
    rows = []
    for config in configs:
        predictions = walk_forward_predictions(
            data,
            SELECTION_FEATURES,
            lambda cfg=config: make_hgb(cfg["seed"], cfg["n_estimators"], cfg["learning_rate"]),
            args.initial_train_fraction,
        )
        metrics = backtest_selection_with_exposure(predictions, FROZEN_RULES, exposure, equal_weight=False)["metrics"]
        rows.append(metric_row(f"model_neighborhood_{config}", metrics, **config))
    return rows


def summarize_distribution(rows: list[dict], metric: str) -> dict:
    values = pd.Series([row.get(metric) for row in rows], dtype=float).dropna()
    if values.empty:
        return {}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
        "positive_rate": float((values > 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Placebo and stability diagnostics.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_placebo_stability")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--random-runs", type=int, default=100)
    parser.add_argument("--shuffle-runs", type=int, default=25)
    args = parser.parse_args()

    data = load_data(args)
    real_predictions = walk_forward_predictions(
        data,
        SELECTION_FEATURES,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    dates = set(real_predictions["date"].unique())
    exposure_200d = ma_exposure_map(Path(args.market_bars), dates, 200)
    real_metrics = backtest_selection_with_exposure(real_predictions, FROZEN_RULES, exposure_200d, equal_weight=False)["metrics"]
    random_rows = random_ranking_placebo(real_predictions, exposure_200d, args.random_runs, args.seed)
    shuffled_rows = shuffled_target_placebo(data, exposure_200d, args.shuffle_runs, args)
    stability_rows = stability_map(real_predictions, Path(args.market_bars), dates)
    neighborhood_rows = model_neighborhood(data, exposure_200d, args)

    output = {
        "metadata": {
            "purpose": "Diagnostics only. Frozen strategy must not be changed based on this same historical period.",
            "warmup_eligibility": "Rows require complete full 62-feature research matrix, matching the frozen historical implementation.",
            "random_ranking_runs": args.random_runs,
            "shuffled_target_runs": args.shuffle_runs,
            "stability_grid_count": len(stability_rows),
            "model_neighborhood_count": len(neighborhood_rows),
        },
        "real_frozen_metrics": real_metrics,
        "placebo_p_values": {
            "random_ranking_total_return": p_value(real_metrics["total_return"], [row["total_return"] for row in random_rows]),
            "random_ranking_sharpe": p_value(real_metrics["sharpe_like"], [row["sharpe_like"] for row in random_rows]),
            "shuffled_target_total_return": p_value(real_metrics["total_return"], [row["total_return"] for row in shuffled_rows]),
            "shuffled_target_sharpe": p_value(real_metrics["sharpe_like"], [row["sharpe_like"] for row in shuffled_rows]),
        },
        "distributions": {
            "random_ranking_total_return": summarize_distribution(random_rows, "total_return"),
            "random_ranking_sharpe": summarize_distribution(random_rows, "sharpe_like"),
            "shuffled_target_total_return": summarize_distribution(shuffled_rows, "total_return"),
            "shuffled_target_sharpe": summarize_distribution(shuffled_rows, "sharpe_like"),
            "stability_total_return": summarize_distribution(stability_rows, "total_return"),
            "stability_sharpe": summarize_distribution(stability_rows, "sharpe_like"),
            "model_neighborhood_total_return": summarize_distribution(neighborhood_rows, "total_return"),
            "model_neighborhood_sharpe": summarize_distribution(neighborhood_rows, "sharpe_like"),
        },
        "random_ranking_placebos": random_rows,
        "shuffled_target_placebos": shuffled_rows,
        "nearby_specification_stability": stability_rows,
        "model_neighborhood": neighborhood_rows,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "placebo_stability_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(random_rows + shuffled_rows + stability_rows + neighborhood_rows).to_csv(
        out_dir / "placebo_stability_summary.csv", index=False
    )
    print(json.dumps({key: output[key] for key in ("metadata", "real_frozen_metrics", "placebo_p_values", "distributions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
