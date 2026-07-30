#!/usr/bin/env python3
"""Walk-forward semiconductor model comparison.

Research-only. This script does not submit trades.

The goal is to test whether ML adds stable value beyond a simple 60-day
momentum rule under identical portfolio rules and transaction costs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_semiconductor_sector_model import (
    PortfolioRules,
    add_features,
    add_market_features,
    backtest_portfolio,
    benchmark_equal_weight,
    benchmark_market_symbol,
    benchmark_top_momentum,
    decile_return_table,
    load_bars,
    monthly_rank_ic,
    pick_metrics,
    precision_at_k,
    rank_ic_by_date,
)


MOMENTUM_FEATURE_MARKERS = (
    "return_",
    "momentum_",
    "close_to_ma_",
    "ma_",
    "cs_rank_return_",
    "cs_rank_momentum_",
    "excess_return_",
    "spy_return_",
    "qqq_return_",
    "soxx_return_",
    "distance_from_sector_avg_",
)

SEMICONDUCTOR_UNIVERSE = [
    "NVDA",
    "AMD",
    "MU",
    "INTC",
    "TXN",
    "ADI",
    "ASML",
    "AMAT",
    "LRCX",
    "KLAC",
    "TSM",
    "GFS",
    "AVGO",
    "MRVL",
    "ON",
    "MPWR",
    "SWKS",
    "QRVO",
    "MCHP",
    "NXPI",
    "TER",
    "LSCC",
    "ALGM",
    "DIOD",
    "CRUS",
    "RMBS",
    "ACLS",
    "AMKR",
    "COHR",
    "LITE",
    "AAOI",
    "FN",
    "CIEN",
    "STM",
    "UMC",
    "SIMO",
]


def make_xgboost(seed: int, n_estimators: int, max_depth: int, learning_rate: float):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.05,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=4,
    )


def make_hgb(seed: int, n_estimators: int, learning_rate: float):
    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        max_leaf_nodes=31,
        learning_rate=learning_rate,
        l2_regularization=0.2,
        random_state=seed,
    )


def make_ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def non_momentum_features(features: list[str]) -> list[str]:
    output = []
    for feature in features:
        if any(feature.startswith(marker) for marker in MOMENTUM_FEATURE_MARKERS):
            continue
        output.append(feature)
    return output


def split_walk_forward_months(data: pd.DataFrame, initial_train_fraction: float) -> list[tuple[set, set]]:
    dates = np.array(sorted(data["date"].unique()))
    start_index = int(len(dates) * initial_train_fraction)
    oos_dates = pd.Series(pd.to_datetime(dates[start_index:]))
    periods = sorted(oos_dates.dt.to_period("M").unique())
    splits = []
    for period in periods:
        predict_dates = set(pd.Series(oos_dates[oos_dates.dt.to_period("M") == period]).dt.date)
        first_predict_date = min(predict_dates)
        train_dates = set(date for date in dates if date < first_predict_date)
        if train_dates and predict_dates:
            splits.append((train_dates, predict_dates))
    return splits


def walk_forward_predictions(
    data: pd.DataFrame,
    features: list[str],
    model_factory,
    initial_train_fraction: float,
) -> pd.DataFrame:
    predictions = []
    for train_dates, predict_dates in split_walk_forward_months(data, initial_train_fraction):
        train_df = data[data["date"].isin(train_dates)]
        predict_df = data[data["date"].isin(predict_dates)].copy()
        model = model_factory()
        model.fit(train_df[features], train_df["future_return"])
        predict_df["predicted_return"] = model.predict(predict_df[features])
        predictions.append(predict_df)
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)


def rank_blend_by_date(df: pd.DataFrame, model_weight: float, momentum_weight: float) -> pd.Series:
    model_rank = df.groupby("date")["predicted_return"].rank(pct=True)
    momentum_rank = df.groupby("date")["return_60d"].rank(pct=True)
    return model_weight * model_rank + momentum_weight * momentum_rank


def summarize_strategy(name: str, df: pd.DataFrame, rules: PortfolioRules) -> dict:
    backtest = backtest_portfolio(df, rules)
    deciles = decile_return_table(df)
    decile_1 = next((row for row in deciles if row["decile"] == 1), None)
    decile_10 = next((row for row in deciles if row["decile"] == 10), None)
    top_minus_bottom = None
    if decile_1 and decile_10:
        top_minus_bottom = decile_10["avg_future_return"] - decile_1["avg_future_return"]
    return {
        "name": name,
        "rank_ic": rank_ic_by_date(df),
        "monthly_rank_ic": monthly_rank_ic(df),
        "prediction_deciles": deciles,
        "top_decile_minus_bottom_decile_return": top_minus_bottom,
        "precision_at_k": precision_at_k(df),
        "top_pick_metrics": pick_metrics(df, rules),
        "portfolio": backtest["metrics"],
    }


def add_excess_vs_momentum(results: list[dict], baseline_name: str) -> None:
    baseline = next(result for result in results if result["name"] == baseline_name)
    baseline_portfolio = baseline["portfolio"]
    baseline_return = baseline_portfolio.get("total_return", 0.0)
    baseline_sharpe = baseline_portfolio.get("sharpe_like", 0.0)
    for result in results:
        portfolio = result["portfolio"]
        result["excess_vs_60d_momentum"] = {
            "excess_total_return": float(portfolio.get("total_return", 0.0) - baseline_return),
            "excess_sharpe_like": float(portfolio.get("sharpe_like", 0.0) - baseline_sharpe),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled walk-forward semiconductor model comparison.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_walkforward_compare")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--buy-rank-pct", type=float, default=0.25)
    parser.add_argument("--sell-rank-pct", type=float, default=0.50)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--no-trade-band", type=float, default=0.01)
    parser.add_argument("--rebalance-every-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--min-predicted-return", type=float, default=0.0)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=181)
    args = parser.parse_args()

    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, features = add_features(bars, market_features, args.horizon_days)
    data = data.dropna(subset=features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    no_momentum_features = non_momentum_features(features)

    rules = PortfolioRules(
        top_n=args.top_n,
        buy_rank_pct=args.buy_rank_pct,
        sell_rank_pct=args.sell_rank_pct,
        max_weight=args.max_weight,
        no_trade_band=args.no_trade_band,
        rebalance_every_days=args.rebalance_every_days,
        transaction_cost_bps=args.transaction_cost_bps,
        min_predicted_return=args.min_predicted_return,
    )

    sample_oos = walk_forward_predictions(
        data,
        features,
        lambda: make_hgb(args.seed, 1, args.learning_rate),
        args.initial_train_fraction,
    )
    oos_dates = set(sample_oos["date"].unique())
    oos_data = data[data["date"].isin(oos_dates)].copy()

    strategy_frames: dict[str, pd.DataFrame] = {}
    momentum_df = oos_data.copy()
    momentum_df["predicted_return"] = momentum_df["return_60d"]
    strategy_frames["60d_momentum_only"] = momentum_df

    hgb_df = walk_forward_predictions(
        data,
        features,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    strategy_frames["hist_gradient_boosting_all_features"] = hgb_df

    xgb_error = None
    try:
        xgb_df = walk_forward_predictions(
            data,
            features,
            lambda: make_xgboost(args.seed, args.n_estimators, args.max_depth, args.learning_rate),
            args.initial_train_fraction,
        )
        strategy_frames["xgboost_all_features"] = xgb_df
    except Exception as exc:
        xgb_error = str(exc)

    if "xgboost_all_features" in strategy_frames:
        blend_base = strategy_frames["xgboost_all_features"].copy()
    else:
        blend_base = hgb_df.copy()
    blend_base["predicted_return"] = rank_blend_by_date(blend_base, model_weight=0.5, momentum_weight=0.5)
    strategy_frames["momentum_plus_ml_rank_blend"] = blend_base

    no_momentum_df = walk_forward_predictions(
        data,
        no_momentum_features,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    strategy_frames["hist_gradient_boosting_ex_momentum_features"] = no_momentum_df

    ridge_df = walk_forward_predictions(
        data,
        features,
        lambda: make_ridge(args.ridge_alpha),
        args.initial_train_fraction,
    )
    strategy_frames["ridge_same_features"] = ridge_df

    results = [summarize_strategy(name, frame, rules) for name, frame in strategy_frames.items()]
    add_excess_vs_momentum(results, "60d_momentum_only")

    benchmark_results = {
        "equal_weight_semiconductors": benchmark_equal_weight(oos_data)["metrics"],
        "top10_60d_momentum_reference": benchmark_top_momentum(oos_data, rules)["metrics"],
        "soxx_buy_and_hold": benchmark_market_symbol(Path(args.market_bars), "SOXX", oos_dates)["metrics"],
    }

    summary_rows = []
    for result in results:
        portfolio = result["portfolio"]
        pick_metrics = result["top_pick_metrics"]
        rank_ic = result["rank_ic"]
        summary_rows.append(
            {
                "name": result["name"],
                "total_return": portfolio.get("total_return"),
                "sharpe_like": portfolio.get("sharpe_like"),
                "max_drawdown": portfolio.get("max_drawdown"),
                "excess_return_vs_60d_momentum": result["excess_vs_60d_momentum"]["excess_total_return"],
                "excess_sharpe_vs_60d_momentum": result["excess_vs_60d_momentum"]["excess_sharpe_like"],
                "mean_rank_ic": rank_ic.get("mean_rank_ic"),
                "positive_rank_ic_rate": rank_ic.get("positive_rank_ic_rate"),
                "top_decile_minus_bottom_decile_return": result["top_decile_minus_bottom_decile_return"],
                "top_pick_win_rate_5d": pick_metrics.get("raw_win_rate_5d"),
                "top_pick_avg_future_return_5d": pick_metrics.get("avg_future_return_5d"),
            }
        )

    output = {
        "metadata": {
            "bars": args.bars,
            "market_bars": args.market_bars,
            "rows": int(len(data)),
            "oos_rows": int(len(oos_data)),
            "symbols": sorted(data["symbol"].unique().tolist()),
            "date_range": {
                "min": str(data["date"].min()),
                "max": str(data["date"].max()),
                "oos_min": str(min(oos_dates)),
                "oos_max": str(max(oos_dates)),
            },
            "target": f"future_return_{args.horizon_days}d",
            "portfolio_rules": asdict(rules),
            "feature_count_all": len(features),
            "feature_count_ex_momentum": len(no_momentum_features),
            "excluded_momentum_features": [feature for feature in features if feature not in no_momentum_features],
            "xgboost_error": xgb_error,
            "execution_note": "Daily close-derived features imply after-close decision or next-session execution.",
        },
        "summary": summary_rows,
        "benchmarks": benchmark_results,
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walkforward_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary_rows).sort_values("total_return", ascending=False).to_csv(
        out_dir / "walkforward_summary.csv", index=False
    )
    for name, frame in strategy_frames.items():
        frame[["date", "symbol", "future_return", "next_1d_return", "predicted_return"]].to_csv(
            out_dir / f"{name}_predictions.csv", index=False
        )

    print(json.dumps({"metadata": output["metadata"], "summary": summary_rows, "benchmarks": benchmark_results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
