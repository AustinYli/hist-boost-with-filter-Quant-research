#!/usr/bin/env python3
"""Robustness and damage tests for the semiconductor walk-forward model.

Research-only. This script does not submit trades.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_sector_model import (
    PortfolioRules,
    add_features,
    add_market_features,
    load_bars,
    max_drawdown,
)
from ml_semiconductor_walkforward_compare import (
    SEMICONDUCTOR_UNIVERSE,
    make_hgb,
    non_momentum_features,
    walk_forward_predictions,
)


EPSILON = 1e-12


def performance_from_returns(returns: pd.Series) -> dict:
    returns = pd.Series(returns).dropna().astype(float)
    if returns.empty:
        return {}
    equity = (1.0 + returns).cumprod()
    annual_return = float(equity.iloc[-1] ** (252 / len(returns)) - 1)
    annual_volatility = float(returns.std() * np.sqrt(252))
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_like": float(annual_return / annual_volatility) if annual_volatility > EPSILON else 0.0,
        "max_drawdown": max_drawdown(equity),
        "days": int(len(returns)),
    }


def normalize_weights(raw_scores: pd.Series, max_weight: float) -> dict[str, float]:
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
    return capped


def choose_weights(day: pd.DataFrame, current_weights: dict[str, float], rules: PortfolioRules) -> dict[str, float]:
    day = day.sort_values("predicted_return", ascending=False).copy()
    day["rank_number"] = np.arange(1, len(day) + 1)
    day["rank_pct"] = day["rank_number"] / len(day)
    volatility = day["volatility_20d"] if "volatility_20d" in day.columns else pd.Series(0.02, index=day.index)
    day["risk_score"] = day["predicted_return"] / volatility.clip(lower=0.005)

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

    selected_day = day[day["symbol"].isin(selected)].set_index("symbol")
    target = normalize_weights(selected_day["risk_score"], rules.max_weight)
    return {symbol: weight for symbol, weight in target.items() if weight > EPSILON}


def apply_no_trade_band(current: dict[str, float], target: dict[str, float], band: float) -> dict[str, float]:
    output = dict(current)
    for symbol in set(current) | set(target):
        old = current.get(symbol, 0.0)
        new = target.get(symbol, 0.0)
        if abs(new - old) > band or new == 0.0:
            output[symbol] = new
    return {symbol: weight for symbol, weight in output.items() if weight > EPSILON}


def backtest_with_records(df: pd.DataFrame, rules: PortfolioRules, exclude_rows: set[tuple] | None = None) -> dict:
    exclude_rows = exclude_rows or set()
    dates = sorted(df["date"].unique())
    current_weights: dict[str, float] = {}
    daily_records = []
    rebalance_records = []
    contribution_rows = []
    for index, date in enumerate(dates[:-1]):
        day = df[df["date"] == date].dropna(subset=["predicted_return", "next_1d_return"])
        if exclude_rows:
            day = day[~day.apply(lambda row: (row["date"], row["symbol"]) in exclude_rows, axis=1)]
        if day.empty:
            continue
        cost = 0.0
        turnover = 0.0
        if index % rules.rebalance_every_days == 0:
            target = choose_weights(day, current_weights, rules)
            target = apply_no_trade_band(current_weights, target, rules.no_trade_band)
            all_symbols = set(current_weights) | set(target)
            turnover = sum(abs(target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
            cost = turnover * rules.transaction_cost_bps / 10000
            current_weights = target
            lookup = day.set_index("symbol")
            selected = []
            for symbol, weight in current_weights.items():
                if symbol not in lookup.index:
                    continue
                selected.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "weight": float(weight),
                        "future_return": float(lookup.loc[symbol, "future_return"]),
                        "predicted_return": float(lookup.loc[symbol, "predicted_return"]),
                    }
                )
            rebalance_records.extend(selected)

        returns = day.set_index("symbol")["next_1d_return"]
        gross_return = 0.0
        for symbol, weight in current_weights.items():
            symbol_return = float(returns.get(symbol, 0.0))
            contribution = weight * symbol_return
            gross_return += contribution
            contribution_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "weight": float(weight),
                    "return": symbol_return,
                    "contribution": float(contribution),
                }
            )
        net_return = gross_return - cost
        daily_records.append(
            {
                "date": date,
                "net_return": float(net_return),
                "gross_return": float(gross_return),
                "cost": float(cost),
                "turnover": float(turnover),
                "holdings": int(len(current_weights)),
                "gross_exposure": float(sum(current_weights.values())),
            }
        )

    daily = pd.DataFrame(daily_records)
    if not daily.empty:
        daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    return {
        "daily": daily,
        "rebalances": pd.DataFrame(rebalance_records),
        "contributions": pd.DataFrame(contribution_rows),
        "metrics": performance_from_returns(daily["net_return"] if not daily.empty else pd.Series(dtype=float)),
    }


def shift_predictions(df: pd.DataFrame, delay_days: int) -> pd.DataFrame:
    if delay_days <= 0:
        return df.copy()
    shifted = df.sort_values(["symbol", "date"]).copy()
    shifted["predicted_return"] = shifted.groupby("symbol")["predicted_return"].shift(delay_days)
    return shifted.dropna(subset=["predicted_return"]).reset_index(drop=True)


def market_returns(market_path: Path, symbol: str, dates: set) -> pd.DataFrame:
    market = load_bars(market_path)
    market = market[market["symbol"] == symbol].sort_values("date").copy()
    market["net_return"] = market["close"].shift(-1).sub(market["close"]).div(market["close"])
    return market[market["date"].isin(dates)][["date", "net_return"]].dropna()


def equal_weight_returns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, day in df.groupby("date"):
        rows.append({"date": date, "net_return": float(day["next_1d_return"].mean())})
    return pd.DataFrame(rows)


def active_regression(model_daily: pd.DataFrame, equal_daily: pd.DataFrame, soxx_daily: pd.DataFrame, momentum_daily: pd.DataFrame) -> dict:
    merged = (
        model_daily[["date", "net_return"]]
        .rename(columns={"net_return": "model_return"})
        .merge(equal_daily.rename(columns={"net_return": "equal_weight_return"}), on="date", how="inner")
        .merge(soxx_daily.rename(columns={"net_return": "soxx_return"}), on="date", how="inner")
        .merge(momentum_daily[["date", "net_return"]].rename(columns={"net_return": "momentum_return"}), on="date", how="inner")
    )
    merged["active_return"] = merged["model_return"] - merged["equal_weight_return"]
    y = merged["model_return"].to_numpy()
    x = merged[["soxx_return", "momentum_return"]].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    degrees = max(len(y) - x.shape[1], 1)
    sigma2 = float((residuals @ residuals) / degrees)
    covariance = sigma2 * np.linalg.inv(x.T @ x)
    standard_errors = np.sqrt(np.diag(covariance))
    alpha_daily = float(beta[0])
    alpha_t = float(alpha_daily / standard_errors[0]) if standard_errors[0] > EPSILON else None
    active_metrics = performance_from_returns(merged["active_return"])
    tracking_error = float(merged["active_return"].std() * np.sqrt(252))
    active_annual = active_metrics.get("annual_return", 0.0)
    return {
        "active_metrics_vs_equal_weight": active_metrics,
        "tracking_error": tracking_error,
        "information_ratio": float(active_annual / tracking_error) if tracking_error > EPSILON else 0.0,
        "ols_daily_alpha": alpha_daily,
        "ols_annualized_alpha_simple": float(alpha_daily * 252),
        "ols_alpha_t_stat": alpha_t,
        "beta_soxx": float(beta[1]),
        "beta_momentum": float(beta[2]),
        "observations": int(len(merged)),
    }


def subperiod_tests(model_daily: pd.DataFrame, soxx_daily: pd.DataFrame) -> dict:
    daily = model_daily.merge(soxx_daily.rename(columns={"net_return": "soxx_return"}), on="date", how="inner")
    daily = daily.sort_values("date").reset_index(drop=True)
    midpoint = len(daily) // 2
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
    monthly_soxx = daily.groupby("month")["soxx_return"].sum()
    daily["soxx_vol_20d"] = daily["soxx_return"].rolling(20).std()
    median_vol = daily["soxx_vol_20d"].median()
    tests = {
        "first_half": daily.iloc[:midpoint],
        "second_half": daily.iloc[midpoint:],
        "bull_months": daily[daily["month"].isin(monthly_soxx[monthly_soxx > 0].index)],
        "drawdown_months": daily[daily["month"].isin(monthly_soxx[monthly_soxx <= 0].index)],
        "high_volatility_days": daily[daily["soxx_vol_20d"] > median_vol],
        "low_volatility_days": daily[daily["soxx_vol_20d"] <= median_vol],
    }
    return {name: performance_from_returns(frame["net_return"]) for name, frame in tests.items()}


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    return {
        "remove_volatility": [
            feature
            for feature in features
            if "volatility" in feature or feature.startswith("atr_") or feature.startswith("range_")
        ],
        "remove_volume": [
            feature
            for feature in features
            if "volume" in feature or "dollar" in feature or "turnover" in feature
        ],
        "remove_market_relative": [feature for feature in features if "beta_" in feature or "soxx" in feature],
        "remove_ohlc_shape": [
            feature
            for feature in features
            if feature
            in {
                "body_pct",
                "close_position",
                "gap_from_yesterday",
                "upper_shadow_pct",
                "lower_shadow_pct",
            }
        ],
        "remove_cross_sectional": [feature for feature in features if feature.startswith("cs_rank_")],
    }


def run_feature_ablation(data: pd.DataFrame, base_features: list[str], rules: PortfolioRules, args: argparse.Namespace) -> list[dict]:
    rows = []
    for name, removed in feature_groups(base_features).items():
        kept = [feature for feature in base_features if feature not in set(removed)]
        if not kept:
            continue
        preds = walk_forward_predictions(
            data,
            kept,
            lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
            args.initial_train_fraction,
        )
        backtest = backtest_with_records(preds, rules)
        rows.append(
            {
                "name": name,
                "removed_features": removed,
                "feature_count": len(kept),
                "metrics": backtest["metrics"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Damage-test the best semiconductor walk-forward model.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_robustness")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    args = parser.parse_args()

    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, features = add_features(bars, market_features, args.horizon_days)
    data = data.dropna(subset=features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    best_features = non_momentum_features(features)

    base_rules = PortfolioRules(
        top_n=10,
        buy_rank_pct=0.25,
        sell_rank_pct=0.50,
        max_weight=0.10,
        no_trade_band=0.01,
        rebalance_every_days=5,
        transaction_cost_bps=5.0,
        min_predicted_return=0.0,
    )

    best_predictions = walk_forward_predictions(
        data,
        best_features,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    baseline = backtest_with_records(best_predictions, base_rules)
    oos_dates = set(best_predictions["date"].unique())
    oos_data = data[data["date"].isin(oos_dates)].copy()

    selected = baseline["rebalances"].sort_values("future_return", ascending=False)
    remove_tests = {}
    for count in (1, 5, 10):
        excluded = set(map(tuple, selected.head(count)[["date", "symbol"]].to_numpy()))
        remove_tests[f"remove_best_{count}_selected_5d_picks"] = backtest_with_records(
            best_predictions, base_rules, excluded
        )["metrics"]
    best_month = (
        baseline["daily"]
        .assign(month=lambda frame: pd.to_datetime(frame["date"]).dt.to_period("M").astype(str))
        .groupby("month")["net_return"]
        .apply(lambda returns: float((1.0 + returns).prod() - 1.0))
        .sort_values(ascending=False)
        .index[0]
    )
    without_best_month = baseline["daily"][
        pd.to_datetime(baseline["daily"]["date"]).dt.to_period("M").astype(str) != best_month
    ]
    remove_tests["remove_best_month"] = {
        "removed_month": best_month,
        **performance_from_returns(without_best_month["net_return"]),
    }
    best_stock = (
        baseline["contributions"].groupby("symbol")["contribution"].sum().sort_values(ascending=False).index[0]
    )
    remove_tests["remove_best_stock"] = {
        "removed_stock": best_stock,
        **backtest_with_records(best_predictions[best_predictions["symbol"] != best_stock], base_rules)["metrics"],
    }

    concentration_tests = {}
    for max_weight in (0.05, 0.10, 0.15):
        rules = PortfolioRules(**{**asdict(base_rules), "max_weight": max_weight})
        concentration_tests[f"max_weight_{int(max_weight * 100)}pct"] = backtest_with_records(best_predictions, rules)[
            "metrics"
        ]

    friction_tests = {}
    for bps in (5.0, 10.0, 25.0):
        rules = PortfolioRules(**{**asdict(base_rules), "transaction_cost_bps": bps})
        friction_tests[f"transaction_cost_{int(bps)}bps"] = backtest_with_records(best_predictions, rules)["metrics"]

    delay_tests = {
        "same_close_baseline": baseline["metrics"],
        "trade_next_close_signal_delayed_1_day": backtest_with_records(shift_predictions(best_predictions, 1), base_rules)[
            "metrics"
        ],
        "trade_one_full_day_later_signal_delayed_2_days": backtest_with_records(
            shift_predictions(best_predictions, 2), base_rules
        )["metrics"],
    }

    soxx = market_returns(Path(args.market_bars), "SOXX", oos_dates)
    equal_weight = equal_weight_returns(oos_data)
    momentum_predictions = oos_data.copy()
    momentum_predictions["predicted_return"] = momentum_predictions["return_60d"]
    momentum_backtest = backtest_with_records(momentum_predictions, base_rules)
    active_tests = active_regression(baseline["daily"], equal_weight, soxx, momentum_backtest["daily"])
    subperiods = subperiod_tests(baseline["daily"], soxx)
    ablations = run_feature_ablation(data, best_features, base_rules, args)

    output = {
        "metadata": {
            "bars": args.bars,
            "market_bars": args.market_bars,
            "strategy": "hist_gradient_boosting_ex_momentum_features",
            "target": f"future_return_{args.horizon_days}d",
            "rows": int(len(data)),
            "oos_rows": int(len(best_predictions)),
            "feature_count": len(best_features),
            "portfolio_rules": asdict(base_rules),
            "execution_note": "Baseline uses same-close-to-next-close returns; delay tests shift signals to reduce this optimism.",
        },
        "baseline": baseline["metrics"],
        "damage_tests": remove_tests,
        "concentration_tests": concentration_tests,
        "friction_tests": friction_tests,
        "delay_tests": delay_tests,
        "subperiod_tests": subperiods,
        "active_return_tests": active_tests,
        "feature_group_ablation": ablations,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    rows = []
    for section in ("damage_tests", "concentration_tests", "friction_tests", "delay_tests", "subperiod_tests"):
        for name, metrics in output[section].items():
            row = {"section": section, "name": name}
            row.update(metrics)
            rows.append(row)
    for item in ablations:
        row = {"section": "feature_group_ablation", "name": item["name"], "feature_count": item["feature_count"]}
        row.update(item["metrics"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "robustness_summary.csv", index=False)
    print(json.dumps({key: output[key] for key in output if key != "feature_group_ablation"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
