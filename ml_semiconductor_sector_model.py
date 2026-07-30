#!/usr/bin/env python3
"""Daily semiconductor sector model and weekly portfolio backtest.

Research-only. This script does not submit trades.

It follows a standard quant-equity workflow:
1. Build daily cross-sectional features for a semiconductor universe.
2. Train a return regressor for next 5 trading days.
3. Rank stocks by predicted risk-adjusted return.
4. Convert ranks into weekly target weights with hysteresis and a no-trade band.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


EPSILON = 1e-12
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


@dataclass(frozen=True)
class PortfolioRules:
    top_n: int
    buy_rank_pct: float
    sell_rank_pct: float
    max_weight: float
    no_trade_band: float
    rebalance_every_days: int
    transaction_cost_bps: float
    min_predicted_return: float


def safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    numeric_columns = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def add_market_features(df: pd.DataFrame, market_path: Path) -> tuple[pd.DataFrame, list[str]]:
    market = load_bars(market_path)
    market = market.sort_values(["symbol", "timestamp"]).copy()
    grouped = market.groupby("symbol", group_keys=False)
    for window in (1, 2, 5, 10, 20, 60):
        market[f"return_{window}d"] = grouped["close"].pct_change(window)
    market_features = []
    pieces = []
    for symbol, prefix in (("SPY", "spy"), ("QQQ", "qqq"), ("SOXX", "soxx")):
        one = market[market["symbol"] == symbol].copy()
        if one.empty:
            continue
        keep = ["date"] + [f"return_{window}d" for window in (1, 2, 5, 10, 20, 60)]
        renamed = {
            f"return_{window}d": f"{prefix}_return_{window}d"
            for window in (1, 2, 5, 10, 20, 60)
        }
        pieces.append(one[keep].rename(columns=renamed))
        market_features.extend(renamed.values())
    if not pieces:
        return df, []
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="date", how="outer")
    merged = merged.sort_values("date").ffill()
    return df.merge(merged, on="date", how="left"), market_features


def rolling_beta(stock_return: pd.Series, sector_return: pd.Series, window: int) -> pd.Series:
    covariance = stock_return.rolling(window).cov(sector_return)
    variance = sector_return.rolling(window).var()
    return safe_div(covariance, variance)


def add_features(df: pd.DataFrame, market_features: list[str], horizon_days: int) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    grouped = df.groupby("symbol", group_keys=False)
    features: list[str] = []

    for window in (1, 2, 5, 10, 20, 60, 120):
        name = f"return_{window}d"
        df[name] = grouped["close"].pct_change(window)
        features.append(name)

    for window in (5, 20, 60):
        ma_name = f"ma_{window}"
        ratio_name = f"close_to_ma_{window}"
        df[ma_name] = grouped["close"].transform(lambda series: series.rolling(window).mean())
        df[ratio_name] = safe_div(df["close"], df[ma_name]) - 1.0
        features.append(ratio_name)
    df["ma_5_to_20"] = safe_div(df["ma_5"], df["ma_20"]) - 1.0
    df["ma_20_to_60"] = safe_div(df["ma_20"], df["ma_60"]) - 1.0
    features.extend(["ma_5_to_20", "ma_20_to_60"])

    for window in (63, 126, 252):
        name = f"momentum_{window}d"
        df[name] = grouped["close"].pct_change(window)
        features.append(name)

    daily_return = grouped["close"].pct_change()
    for window in (20, 60):
        name = f"volatility_{window}d"
        df[name] = daily_return.groupby(df["symbol"]).transform(lambda series: series.rolling(window).std())
        features.append(name)

    prev_close = grouped["close"].shift(1)
    tr_1 = df["high"] - df["low"]
    tr_2 = (df["high"] - prev_close).abs()
    tr_3 = (df["low"] - prev_close).abs()
    df["atr_20_pct"] = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1).groupby(df["symbol"]).transform(
        lambda series: series.rolling(20).mean()
    ) / df["close"]
    df["parkinson_volatility_20d"] = np.sqrt(
        ((np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))).groupby(df["symbol"]).transform(
            lambda series: series.rolling(20).mean()
        )
    )
    features.extend(["atr_20_pct", "parkinson_volatility_20d"])

    avg_volume_20 = grouped["volume"].transform(lambda series: series.shift(1).rolling(20).mean())
    df["relative_volume_20d"] = safe_div(df["volume"], avg_volume_20)
    df["dollar_volume"] = df["close"] * df["volume"]
    df["log_dollar_volume"] = np.log1p(df["dollar_volume"])
    df["turnover_proxy"] = safe_div(df["volume"], avg_volume_20)
    features.extend(["relative_volume_20d", "log_dollar_volume", "turnover_proxy"])

    df["body_pct"] = safe_div(df["close"] - df["open"], df["close"])
    df["range_pct"] = safe_div(df["high"] - df["low"], df["close"])
    df["close_position"] = safe_div(df["close"] - df["low"], df["high"] - df["low"]).fillna(0.5)
    df["gap_from_yesterday"] = safe_div(df["open"] - prev_close, prev_close)
    df["upper_shadow_pct"] = safe_div(df["high"] - df[["open", "close"]].max(axis=1), df["close"])
    df["lower_shadow_pct"] = safe_div(df[["open", "close"]].min(axis=1) - df["low"], df["close"])
    features.extend(
        [
            "body_pct",
            "range_pct",
            "close_position",
            "gap_from_yesterday",
            "upper_shadow_pct",
            "lower_shadow_pct",
        ]
    )

    if "soxx_return_1d" in df.columns:
        for window in (1, 5, 20, 60):
            stock = df[f"return_{window}d"]
            sector = df[f"soxx_return_{window}d"]
            name = f"excess_return_{window}d_vs_soxx"
            df[name] = stock - sector
            features.append(name)
        for window in (20, 60):
            beta_name = f"beta_{window}d_to_soxx"
            df[beta_name] = grouped.apply(
                lambda group: rolling_beta(group["return_1d"], group["soxx_return_1d"], window),
                include_groups=False,
            ).reset_index(level=0, drop=True)
            features.append(beta_name)

    rank_sources = [
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "momentum_63d",
        "momentum_126d",
        "relative_volume_20d",
        "volatility_20d",
        "log_dollar_volume",
    ]
    for source in rank_sources:
        if source not in df.columns:
            continue
        rank_name = f"cs_rank_{source}"
        df[rank_name] = df.groupby("date")[source].rank(pct=True)
        features.append(rank_name)

    if "return_1d" in df.columns:
        df["sector_avg_return_1d"] = df.groupby("date")["return_1d"].transform("mean")
        df["distance_from_sector_avg_1d"] = df["return_1d"] - df["sector_avg_return_1d"]
        features.append("distance_from_sector_avg_1d")

    features.extend(feature for feature in market_features if feature in df.columns)

    df["future_return"] = grouped["close"].shift(-horizon_days).sub(df["close"]).div(df["close"])
    df["next_1d_return"] = grouped["close"].shift(-1).sub(df["close"]).div(df["close"])
    df = df.replace([np.inf, -np.inf], np.nan)
    features = list(dict.fromkeys(features))
    return df, features


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(df["date"].unique()))
    train_end = int(len(dates) * 0.6)
    validation_end = int(len(dates) * 0.8)
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(validation_dates)].copy(),
        df[df["date"].isin(test_dates)].copy(),
    )


def make_regressor(seed: int, args: argparse.Namespace):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.05,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )
    except Exception as exc:
        print(f"XGBoost unavailable, using HistGradientBoostingRegressor fallback: {exc}")
        return HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            max_leaf_nodes=31,
            learning_rate=args.learning_rate,
            l2_regularization=0.2,
            random_state=seed,
        )


def normalize_weights(raw_scores: pd.Series, *, max_weight: float) -> dict[str, float]:
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


def choose_target_weights(
    day: pd.DataFrame,
    current_weights: dict[str, float],
    rules: PortfolioRules,
) -> dict[str, float]:
    day = day.sort_values("predicted_return", ascending=False).copy()
    day["rank_number"] = np.arange(1, len(day) + 1)
    day["rank_pct"] = day["rank_number"] / len(day)
    day["risk_score"] = day["predicted_return"] / day["volatility_20d"].clip(lower=0.005)

    retained = [
        symbol
        for symbol in current_weights
        if symbol in set(day.loc[day["rank_pct"] <= rules.sell_rank_pct, "symbol"])
    ]
    buy_candidates = day[
        (day["rank_pct"] <= rules.buy_rank_pct)
        & (day["predicted_return"] >= rules.min_predicted_return)
    ]["symbol"].tolist()

    selected: list[str] = []
    for symbol in retained + buy_candidates:
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) >= rules.top_n:
            break

    if len(selected) < rules.top_n:
        for symbol in day["symbol"].tolist():
            if symbol not in selected and day.loc[day["symbol"] == symbol, "predicted_return"].iloc[0] > 0:
                selected.append(symbol)
            if len(selected) >= rules.top_n:
                break

    selected_day = day[day["symbol"].isin(selected)].set_index("symbol")
    target = normalize_weights(selected_day["risk_score"], max_weight=rules.max_weight)
    for symbol in day["symbol"]:
        target.setdefault(symbol, 0.0)
    return target


def apply_no_trade_band(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    no_trade_band: float,
) -> dict[str, float]:
    output = dict(current_weights)
    all_symbols = set(current_weights) | set(target_weights)
    for symbol in all_symbols:
        current = current_weights.get(symbol, 0.0)
        target = target_weights.get(symbol, 0.0)
        if abs(target - current) > no_trade_band or target == 0.0:
            output[symbol] = target
    return {symbol: weight for symbol, weight in output.items() if weight > EPSILON}


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def backtest_portfolio(test_df: pd.DataFrame, rules: PortfolioRules) -> dict:
    dates = sorted(test_df["date"].unique())
    current_weights: dict[str, float] = {}
    daily_records = []
    rebalance_records = []
    equity = 1.0
    for index, date in enumerate(dates[:-1]):
        day = test_df[test_df["date"] == date].dropna(subset=["predicted_return", "next_1d_return", "volatility_20d"])
        if day.empty:
            continue
        cost = 0.0
        rebalanced = index % rules.rebalance_every_days == 0
        if rebalanced:
            target_weights = choose_target_weights(day, current_weights, rules)
            banded_target = apply_no_trade_band(current_weights, target_weights, rules.no_trade_band)
            all_symbols = set(current_weights) | set(banded_target)
            turnover = sum(abs(banded_target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
            cost = turnover * rules.transaction_cost_bps / 10000
            current_weights = banded_target
            ranked = day.sort_values("predicted_return", ascending=False)
            selected = [symbol for symbol, weight in current_weights.items() if weight > EPSILON]
            rebalance_records.append(
                {
                    "date": str(date),
                    "turnover": float(turnover),
                    "cost": float(cost),
                    "holdings": {
                        symbol: float(current_weights[symbol])
                        for symbol in sorted(selected, key=lambda sym: current_weights[sym], reverse=True)
                    },
                    "top_predictions": ranked[["symbol", "predicted_return", "future_return"]].head(rules.top_n).to_dict(
                        orient="records"
                    ),
                }
            )

        returns = day.set_index("symbol")["next_1d_return"]
        gross_return = sum(current_weights.get(symbol, 0.0) * float(returns.get(symbol, 0.0)) for symbol in current_weights)
        net_return = gross_return - cost
        equity *= 1.0 + net_return
        daily_records.append(
            {
                "date": str(date),
                "gross_return": float(gross_return),
                "cost": float(cost),
                "net_return": float(net_return),
                "equity": float(equity),
                "holdings": int(len(current_weights)),
                "gross_exposure": float(sum(current_weights.values())),
            }
        )

    daily = pd.DataFrame(daily_records)
    if daily.empty:
        return {"daily": [], "rebalances": [], "metrics": {}}
    daily_return = daily["net_return"]
    annual_return = float((daily["equity"].iloc[-1]) ** (252 / max(len(daily), 1)) - 1)
    annual_volatility = float(daily_return.std() * np.sqrt(252))
    sharpe = annual_return / annual_volatility if annual_volatility > EPSILON else 0.0
    metrics = {
        "total_return": float(daily["equity"].iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_like": float(sharpe),
        "max_drawdown": max_drawdown(daily["equity"]),
        "avg_daily_turnover_cost": float(daily["cost"].mean()),
        "rebalance_count": len(rebalance_records),
        "avg_holdings": float(daily["holdings"].mean()),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
    }
    return {
        "daily": daily_records,
        "rebalances": rebalance_records,
        "metrics": metrics,
    }


def equity_metrics(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {}
    daily_return = daily["net_return"]
    annual_return = float((daily["equity"].iloc[-1]) ** (252 / max(len(daily), 1)) - 1)
    annual_volatility = float(daily_return.std() * np.sqrt(252))
    return {
        "total_return": float(daily["equity"].iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_like": float(annual_return / annual_volatility) if annual_volatility > EPSILON else 0.0,
        "max_drawdown": max_drawdown(daily["equity"]),
        "days": int(len(daily)),
    }


def benchmark_equal_weight(test_df: pd.DataFrame) -> dict:
    records = []
    equity = 1.0
    for date in sorted(test_df["date"].unique())[:-1]:
        day = test_df[test_df["date"] == date].dropna(subset=["next_1d_return"])
        if day.empty:
            continue
        net_return = float(day["next_1d_return"].mean())
        equity *= 1.0 + net_return
        records.append({"date": str(date), "net_return": net_return, "equity": float(equity)})
    return {"daily": records, "metrics": equity_metrics(pd.DataFrame(records))}


def benchmark_top_momentum(test_df: pd.DataFrame, rules: PortfolioRules, momentum_column: str = "return_60d") -> dict:
    records = []
    rebalances = []
    current_weights: dict[str, float] = {}
    equity = 1.0
    for index, date in enumerate(sorted(test_df["date"].unique())[:-1]):
        day = test_df[test_df["date"] == date].dropna(subset=[momentum_column, "next_1d_return"])
        if day.empty:
            continue
        cost = 0.0
        if index % rules.rebalance_every_days == 0:
            selected = day.sort_values(momentum_column, ascending=False).head(rules.top_n)["symbol"].tolist()
            target = {symbol: min(1.0 / max(len(selected), 1), rules.max_weight) for symbol in selected}
            all_symbols = set(current_weights) | set(target)
            turnover = sum(abs(target.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
            cost = turnover * rules.transaction_cost_bps / 10000
            current_weights = target
            rebalances.append({"date": str(date), "holdings": current_weights, "turnover": float(turnover)})
        returns = day.set_index("symbol")["next_1d_return"]
        gross_return = sum(current_weights.get(symbol, 0.0) * float(returns.get(symbol, 0.0)) for symbol in current_weights)
        net_return = gross_return - cost
        equity *= 1.0 + net_return
        records.append({"date": str(date), "net_return": float(net_return), "equity": float(equity)})
    return {"daily": records, "rebalances": rebalances, "metrics": equity_metrics(pd.DataFrame(records))}


def benchmark_market_symbol(market_path: Path, symbol: str, test_dates: set) -> dict:
    market = load_bars(market_path)
    market = market[market["symbol"] == symbol].sort_values("date").copy()
    market["next_1d_return"] = market["close"].shift(-1).sub(market["close"]).div(market["close"])
    market = market[market["date"].isin(test_dates)].dropna(subset=["next_1d_return"])
    records = []
    equity = 1.0
    for _, row in market.iterrows():
        net_return = float(row["next_1d_return"])
        equity *= 1.0 + net_return
        records.append({"date": str(row["date"]), "net_return": net_return, "equity": float(equity)})
    return {"daily": records, "metrics": equity_metrics(pd.DataFrame(records))}


def decile_return_table(df: pd.DataFrame) -> list[dict]:
    frame = df.dropna(subset=["predicted_return", "future_return"]).copy()
    frame["prediction_decile"] = np.ceil(frame.groupby("date")["predicted_return"].rank(pct=True) * 10).clip(1, 10)
    rows = []
    for decile, group in frame.groupby("prediction_decile"):
        rows.append(
            {
                "decile": int(decile),
                "samples": int(len(group)),
                "avg_future_return": float(group["future_return"].mean()),
                "median_future_return": float(group["future_return"].median()),
                "win_rate": float((group["future_return"] > 0).mean()),
                "profit_15bp_rate": float((group["future_return"] > 0.0015).mean()),
            }
        )
    return sorted(rows, key=lambda row: row["decile"])


def precision_at_k(df: pd.DataFrame, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    rows: dict[str, dict] = {}
    grouped = df.dropna(subset=["predicted_return", "future_return"]).groupby("date")
    for k in ks:
        picks = []
        for _, day in grouped:
            picks.append(day.sort_values("predicted_return", ascending=False).head(k))
        if not picks:
            continue
        picked = pd.concat(picks, ignore_index=True)
        key = f"precision_at_{k}"
        rows[key] = {
            "picked_samples": int(len(picked)),
            "win_rate": float((picked["future_return"] > 0).mean()),
            "profit_15bp_rate": float((picked["future_return"] > 0.0015).mean()),
            "avg_future_return": float(picked["future_return"].mean()),
            "median_future_return": float(picked["future_return"].median()),
        }
    return rows


def monthly_rank_ic(df: pd.DataFrame) -> list[dict]:
    daily_rows = []
    for date, day in df.groupby("date"):
        day = day.dropna(subset=["predicted_return", "future_return"])
        if len(day) < 5:
            continue
        correlation = spearmanr(day["predicted_return"], day["future_return"]).correlation
        if np.isfinite(correlation):
            daily_rows.append({"date": pd.Timestamp(date), "rank_ic": float(correlation)})
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return []
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    output = []
    for month, group in daily.groupby("month"):
        output.append(
            {
                "month": month,
                "mean_rank_ic": float(group["rank_ic"].mean()),
                "median_rank_ic": float(group["rank_ic"].median()),
                "positive_rank_ic_rate": float((group["rank_ic"] > 0).mean()),
                "dates": int(len(group)),
            }
        )
    return output


def pick_metrics(test_df: pd.DataFrame, rules: PortfolioRules) -> dict:
    picks = []
    for index, date in enumerate(sorted(test_df["date"].unique())):
        if index % rules.rebalance_every_days != 0:
            continue
        day = test_df[test_df["date"] == date].dropna(subset=["predicted_return", "future_return", "volatility_20d"])
        if day.empty:
            continue
        day = day.sort_values("predicted_return", ascending=False).head(rules.top_n)
        picks.append(day)
    if not picks:
        return {}
    picked = pd.concat(picks, ignore_index=True)
    wins = picked["future_return"] > 0
    profit_wins = picked["future_return"] > 0.0015
    losses = picked.loc[picked["future_return"] <= 0, "future_return"]
    gains = picked.loc[picked["future_return"] > 0, "future_return"]
    return {
        "picked_samples": int(len(picked)),
        "raw_win_rate_5d": float(wins.mean()),
        "profit_15bp_rate_5d": float(profit_wins.mean()),
        "avg_future_return_5d": float(picked["future_return"].mean()),
        "median_future_return_5d": float(picked["future_return"].median()),
        "profit_factor_5d": float(gains.sum() / abs(losses.sum())) if abs(losses.sum()) > EPSILON else None,
    }


def rank_ic_by_date(df: pd.DataFrame) -> dict:
    values = []
    for _, day in df.groupby("date"):
        day = day.dropna(subset=["predicted_return", "future_return"])
        if len(day) < 5:
            continue
        correlation = spearmanr(day["predicted_return"], day["future_return"]).correlation
        if np.isfinite(correlation):
            values.append(float(correlation))
    if not values:
        return {}
    return {
        "mean_rank_ic": float(np.mean(values)),
        "median_rank_ic": float(np.median(values)),
        "positive_rank_ic_rate": float((np.array(values) > 0).mean()),
        "dates": int(len(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a daily semiconductor sector ranking model.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_sector_xgb_5d")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--buy-rank-pct", type=float, default=0.25)
    parser.add_argument("--sell-rank-pct", type=float, default=0.50)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--no-trade-band", type=float, default=0.01)
    parser.add_argument("--rebalance-every-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--min-predicted-return", type=float, default=0.0)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    args = parser.parse_args()

    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, features = add_features(bars, market_features, args.horizon_days)
    data = data.dropna(subset=features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    train_df, validation_df, test_df = split_by_date(data)

    model = make_regressor(args.seed, args)
    model.fit(train_df[features], train_df["future_return"])
    for frame in (train_df, validation_df, test_df):
        frame["predicted_return"] = model.predict(frame[features])

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
    backtest = backtest_portfolio(test_df, rules)
    test_dates = set(test_df["date"].unique())
    latest_data = data.assign(predicted_return=model.predict(data[features]))
    latest_date = latest_data["date"].max()
    metrics = {
        "train": {
            "rmse": float(mean_squared_error(train_df["future_return"], train_df["predicted_return"]) ** 0.5),
            "mae": float(mean_absolute_error(train_df["future_return"], train_df["predicted_return"])),
            "r2": float(r2_score(train_df["future_return"], train_df["predicted_return"])),
        },
        "validation": {
            "rmse": float(mean_squared_error(validation_df["future_return"], validation_df["predicted_return"]) ** 0.5),
            "mae": float(mean_absolute_error(validation_df["future_return"], validation_df["predicted_return"])),
            "r2": float(r2_score(validation_df["future_return"], validation_df["predicted_return"])),
            "rank_ic": rank_ic_by_date(validation_df),
            "monthly_rank_ic": monthly_rank_ic(validation_df),
            "prediction_deciles": decile_return_table(validation_df),
            "precision_at_k": precision_at_k(validation_df),
        },
        "test": {
            "rmse": float(mean_squared_error(test_df["future_return"], test_df["predicted_return"]) ** 0.5),
            "mae": float(mean_absolute_error(test_df["future_return"], test_df["predicted_return"])),
            "r2": float(r2_score(test_df["future_return"], test_df["predicted_return"])),
            "rank_ic": rank_ic_by_date(test_df),
            "monthly_rank_ic": monthly_rank_ic(test_df),
            "prediction_deciles": decile_return_table(test_df),
            "precision_at_k": precision_at_k(test_df),
            "top_pick_metrics": pick_metrics(test_df, rules),
            "portfolio": backtest["metrics"],
            "benchmarks": {
                "equal_weight_semiconductors": benchmark_equal_weight(test_df)["metrics"],
                "top10_60d_momentum": benchmark_top_momentum(test_df, rules)["metrics"],
                "soxx_buy_and_hold": benchmark_market_symbol(Path(args.market_bars), "SOXX", test_dates)["metrics"],
            },
        },
    }
    output = {
        "metadata": {
            "bars": args.bars,
            "market_bars": args.market_bars,
            "symbols": sorted(data["symbol"].unique().tolist()),
            "rows": int(len(data)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "date_range": {
                "min": str(data["date"].min()),
                "max": str(data["date"].max()),
                "train_max": str(train_df["date"].max()),
                "validation_max": str(validation_df["date"].max()),
                "test_min": str(test_df["date"].min()),
            },
            "target": f"future_return_{args.horizon_days}d",
            "model": type(model).__name__,
            "features": features,
            "feature_count": len(features),
            "portfolio_rules": asdict(rules),
            "leakage_notes": [
                "Features use same-day close-derived values, so signals assume execution after the close or next session.",
                "future_return and next_1d_return are targets only and are excluded from the feature list.",
                "The universe is a fixed current research universe, so survivorship/selection bias is still possible.",
            ],
        },
        "metrics": metrics,
        "latest_ranking": latest_data[latest_data["date"] == latest_date]
        .sort_values("predicted_return", ascending=False)[
            ["symbol", "date", "predicted_return", "volatility_20d", "future_return"]
        ]
        .to_dict(orient="records"),
        "backtest": backtest,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / "model.joblib")
    (out_dir / "results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    pd.concat([train_df, validation_df, test_df], ignore_index=True)[
        ["date", "symbol", "future_return", "next_1d_return", "predicted_return"] + features
    ].to_csv(out_dir / "dataset_with_predictions.csv", index=False)
    print(json.dumps({"metadata": output["metadata"], "metrics": output["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
