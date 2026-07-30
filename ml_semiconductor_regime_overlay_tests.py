#!/usr/bin/env python3
"""Regime overlay and long-short tests for semiconductor selection.

Research-only. This script does not submit trades.

Layer 1 decides sector exposure. Layer 2 is the existing HGB ex-momentum
stock-selection model that ranks stocks within the semiconductor universe.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_robustness_tests import (
    active_regression,
    apply_no_trade_band,
    equal_weight_returns,
    market_returns,
    performance_from_returns,
    subperiod_tests,
)
from ml_semiconductor_sector_model import (
    PortfolioRules,
    add_features,
    add_market_features,
    load_bars,
)
from ml_semiconductor_walkforward_compare import (
    SEMICONDUCTOR_UNIVERSE,
    make_hgb,
    non_momentum_features,
    walk_forward_predictions,
)


EPSILON = 1e-12


def load_research_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, features = add_features(bars, market_features, args.horizon_days)
    data = data.dropna(subset=features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    return data, features


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(20, window // 4)).mean()
    std = series.rolling(window, min_periods=max(20, window // 4)).std()
    return (series - mean).div(std.replace(0, np.nan)).fillna(0.0)


def regime_exposures(data: pd.DataFrame, market_path: Path, oos_dates: set) -> pd.DataFrame:
    market = load_bars(market_path)
    soxx = market[market["symbol"] == "SOXX"].sort_values("date").copy()
    soxx["ma_100"] = soxx["close"].rolling(100).mean()
    soxx["ma_200"] = soxx["close"].rolling(200).mean()
    soxx["momentum_60d"] = soxx["close"].pct_change(60)

    breadth = (
        data.assign(above_ma_60=lambda frame: frame["close_to_ma_60"] > 0)
        .groupby("date")["above_ma_60"]
        .mean()
        .rename("breadth_above_60d_ma")
        .reset_index()
    )
    regime = soxx[["date", "close", "ma_100", "ma_200", "momentum_60d"]].merge(breadth, on="date", how="left")
    regime["breadth_above_60d_ma"] = regime["breadth_above_60d_ma"].ffill().fillna(0.5)
    condition_100 = regime["close"] > regime["ma_100"]
    condition_200 = regime["close"] > regime["ma_200"]
    condition_momentum = regime["momentum_60d"] > 0
    regime["exposure_original_100pct"] = 1.0
    regime["exposure_soxx_100d_filter"] = condition_100.astype(float)
    regime["exposure_soxx_200d_filter"] = condition_200.astype(float)
    regime["exposure_soxx_100d_and_60d_momentum"] = np.select(
        [condition_100 & condition_momentum, condition_100 | condition_momentum],
        [1.0, 0.5],
        default=0.0,
    )
    momentum_z = rolling_zscore(regime["momentum_60d"], 252)
    breadth_z = rolling_zscore(regime["breadth_above_60d_ma"], 252)
    regime["exposure_smooth_momentum_breadth"] = (0.5 + 0.25 * momentum_z + 0.25 * breadth_z).clip(0.0, 1.0)
    return regime[regime["date"].isin(oos_dates)].reset_index(drop=True)


def normalize_selection_weights(raw_scores: pd.Series, max_weight: float) -> dict[str, float]:
    scores = raw_scores.clip(lower=0).astype(float)
    if scores.sum() <= EPSILON:
        scores = pd.Series(1.0, index=raw_scores.index)
    weights = scores / scores.sum()
    capped: dict[str, float] = {}
    remaining = weights.copy()
    remaining_cap = 1.0
    while len(remaining) > 0:
        proposed = remaining / remaining.sum() * remaining_cap
        over_cap = proposed > max_weight
        if not over_cap.any():
            capped.update(proposed.to_dict())
            break
        capped.update({symbol: max_weight for symbol in proposed[over_cap].index})
        remaining_cap -= float(max_weight * over_cap.sum())
        remaining = remaining[~over_cap]
        if remaining_cap <= EPSILON:
            capped.update({symbol: 0.0 for symbol in remaining.index})
            break
    return {symbol: weight for symbol, weight in capped.items() if weight > EPSILON}


def choose_selection_weights(day: pd.DataFrame, current_weights: dict[str, float], rules: PortfolioRules) -> dict[str, float]:
    day = day.sort_values("predicted_return", ascending=False).copy()
    day["rank_number"] = np.arange(1, len(day) + 1)
    day["rank_pct"] = day["rank_number"] / len(day)
    day["risk_score"] = day["predicted_return"] / day["volatility_20d"].clip(lower=0.005)
    retained = [
        symbol
        for symbol in current_weights
        if symbol in set(day.loc[day["rank_pct"] <= rules.sell_rank_pct, "symbol"])
    ]
    buys = day[
        (day["rank_pct"] <= rules.buy_rank_pct)
        & (day["predicted_return"] >= rules.min_predicted_return)
    ]["symbol"].tolist()
    selected: list[str] = []
    for symbol in retained + buys:
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) >= rules.top_n:
            break
    if len(selected) < rules.top_n:
        for symbol in day["symbol"]:
            if symbol not in selected and day.loc[day["symbol"] == symbol, "predicted_return"].iloc[0] > 0:
                selected.append(symbol)
            if len(selected) >= rules.top_n:
                break
    return normalize_selection_weights(day[day["symbol"].isin(selected)].set_index("symbol")["risk_score"], rules.max_weight)


def backtest_selection_with_exposure(
    predictions: pd.DataFrame,
    rules: PortfolioRules,
    exposure_by_date: dict,
    *,
    equal_weight: bool = False,
) -> dict:
    current_weights: dict[str, float] = {}
    daily_rows = []
    for index, date in enumerate(sorted(predictions["date"].unique())[:-1]):
        day = predictions[predictions["date"] == date].dropna(subset=["predicted_return", "next_1d_return"])
        if day.empty:
            continue
        exposure = float(exposure_by_date.get(date, 1.0))
        cost = 0.0
        turnover = 0.0
        if index % rules.rebalance_every_days == 0:
            if equal_weight:
                symbols = sorted(day["symbol"].unique())
                base = {symbol: 1.0 / len(symbols) for symbol in symbols}
            else:
                base = choose_selection_weights(day, current_weights, rules)
            target = {symbol: weight * exposure for symbol, weight in base.items()}
            target = apply_no_trade_band(current_weights, target, rules.no_trade_band)
            all_symbols = set(current_weights) | set(target)
            turnover = sum(abs(target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
            cost = turnover * rules.transaction_cost_bps / 10000
            current_weights = target

        returns = day.set_index("symbol")["next_1d_return"]
        gross_return = sum(current_weights.get(symbol, 0.0) * float(returns.get(symbol, 0.0)) for symbol in current_weights)
        daily_rows.append(
            {
                "date": date,
                "net_return": float(gross_return - cost),
                "gross_return": float(gross_return),
                "cost": float(cost),
                "turnover": float(turnover),
                "exposure": float(sum(abs(weight) for weight in current_weights.values())),
                "holdings": int(len(current_weights)),
            }
        )
    daily = pd.DataFrame(daily_rows)
    return {
        "daily": daily,
        "metrics": {
            **performance_from_returns(daily["net_return"] if not daily.empty else pd.Series(dtype=float)),
            "avg_exposure": float(daily["exposure"].mean()) if not daily.empty else 0.0,
            "avg_turnover": float(daily["turnover"].mean()) if not daily.empty else 0.0,
        },
    }


def backtest_long_short(predictions: pd.DataFrame, rules: PortfolioRules) -> dict:
    current_weights: dict[str, float] = {}
    daily_rows = []
    for index, date in enumerate(sorted(predictions["date"].unique())[:-1]):
        day = predictions[predictions["date"] == date].dropna(subset=["predicted_return", "next_1d_return"])
        if day.empty:
            continue
        cost = 0.0
        turnover = 0.0
        if index % rules.rebalance_every_days == 0:
            ranked = day.sort_values("predicted_return", ascending=False)
            count = max(1, int(np.ceil(len(ranked) * 0.10)))
            longs = ranked.head(count)["symbol"].tolist()
            shorts = ranked.tail(count)["symbol"].tolist()
            target = {symbol: 0.5 / len(longs) for symbol in longs}
            target.update({symbol: -0.5 / len(shorts) for symbol in shorts})
            all_symbols = set(current_weights) | set(target)
            turnover = sum(abs(target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
            cost = turnover * rules.transaction_cost_bps / 10000
            current_weights = target
        returns = day.set_index("symbol")["next_1d_return"]
        gross_return = sum(current_weights.get(symbol, 0.0) * float(returns.get(symbol, 0.0)) for symbol in current_weights)
        daily_rows.append(
            {
                "date": date,
                "net_return": float(gross_return - cost),
                "gross_return": float(gross_return),
                "cost": float(cost),
                "turnover": float(turnover),
                "gross_exposure": float(sum(abs(weight) for weight in current_weights.values())),
                "net_exposure": float(sum(current_weights.values())),
                "holdings": int(len(current_weights)),
            }
        )
    daily = pd.DataFrame(daily_rows)
    return {
        "daily": daily,
        "metrics": {
            **performance_from_returns(daily["net_return"] if not daily.empty else pd.Series(dtype=float)),
            "avg_gross_exposure": float(daily["gross_exposure"].mean()) if not daily.empty else 0.0,
            "avg_net_exposure": float(daily["net_exposure"].mean()) if not daily.empty else 0.0,
            "avg_turnover": float(daily["turnover"].mean()) if not daily.empty else 0.0,
        },
    }


def strategy_report(name: str, strategy: dict, soxx_daily: pd.DataFrame) -> dict:
    return {
        "name": name,
        "metrics": strategy["metrics"],
        "subperiods": subperiod_tests(strategy["daily"], soxx_daily),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test sector regime overlays and long-short selection.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_regime_overlay")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    args = parser.parse_args()

    data, features = load_research_data(args)
    selection_features = non_momentum_features(features)
    predictions = walk_forward_predictions(
        data,
        selection_features,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    oos_dates = set(predictions["date"].unique())
    exposures = regime_exposures(data, Path(args.market_bars), oos_dates)
    exposure_columns = [column for column in exposures.columns if column.startswith("exposure_")]
    exposure_maps = {
        column.replace("exposure_", ""): dict(zip(exposures["date"], exposures[column]))
        for column in exposure_columns
    }
    rules = PortfolioRules(
        top_n=10,
        buy_rank_pct=0.25,
        sell_rank_pct=0.50,
        max_weight=0.10,
        no_trade_band=0.01,
        rebalance_every_days=5,
        transaction_cost_bps=5.0,
        min_predicted_return=0.0,
    )

    model_strategies = {
        name: backtest_selection_with_exposure(predictions, rules, exposure_map, equal_weight=False)
        for name, exposure_map in exposure_maps.items()
    }
    equal_weight_strategies = {
        name: backtest_selection_with_exposure(predictions, rules, exposure_map, equal_weight=True)
        for name, exposure_map in exposure_maps.items()
    }
    long_short = backtest_long_short(predictions, rules)

    soxx_daily = market_returns(Path(args.market_bars), "SOXX", oos_dates)
    equal_weight_daily = equal_weight_returns(data[data["date"].isin(oos_dates)])
    momentum_predictions = data[data["date"].isin(oos_dates)].copy()
    momentum_predictions["predicted_return"] = momentum_predictions["return_60d"]
    momentum_strategy = backtest_selection_with_exposure(
        momentum_predictions, rules, exposure_maps["original_100pct"], equal_weight=False
    )

    active_tests = {}
    for name, strategy in model_strategies.items():
        matching_equal = equal_weight_strategies[name]
        active_tests[name] = active_regression(
            strategy["daily"], matching_equal["daily"], soxx_daily, momentum_strategy["daily"]
        )

    model_reports = [strategy_report(name, strategy, soxx_daily) for name, strategy in model_strategies.items()]
    equal_reports = [strategy_report(f"equal_weight_{name}", strategy, soxx_daily) for name, strategy in equal_weight_strategies.items()]
    long_short_report = strategy_report("long_short_top_decile_minus_bottom_decile", long_short, soxx_daily)

    summary = []
    for report in model_reports + equal_reports + [long_short_report]:
        metrics = report["metrics"]
        drawdown_months = report["subperiods"].get("drawdown_months", {})
        summary.append(
            {
                "name": report["name"],
                "total_return": metrics.get("total_return"),
                "sharpe_like": metrics.get("sharpe_like"),
                "max_drawdown": metrics.get("max_drawdown"),
                "avg_exposure": metrics.get("avg_exposure", metrics.get("avg_gross_exposure")),
                "drawdown_month_return": drawdown_months.get("total_return"),
                "drawdown_month_sharpe": drawdown_months.get("sharpe_like"),
            }
        )

    output = {
        "metadata": {
            "bars": args.bars,
            "market_bars": args.market_bars,
            "rows": int(len(data)),
            "oos_rows": int(len(predictions)),
            "feature_count": len(selection_features),
            "target": f"future_return_{args.horizon_days}d",
            "portfolio_rules": asdict(rules),
            "exposure_columns": exposure_columns,
            "execution_note": "Regime features use daily close data, so this is an after-close/next-session research assumption.",
        },
        "summary": summary,
        "model_regime_strategies": model_reports,
        "equal_weight_regime_strategies": equal_reports,
        "long_short": long_short_report,
        "active_tests_vs_matching_equal_weight": active_tests,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regime_overlay_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary).sort_values("sharpe_like", ascending=False).to_csv(out_dir / "regime_overlay_summary.csv", index=False)
    print(json.dumps({"metadata": output["metadata"], "summary": summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
