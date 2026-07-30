#!/usr/bin/env python3
"""Frozen-spec diagnostics for the semiconductor strategy.

Research-only. This script does not submit trades or tune the strategy.

The specification is frozen:
- Model: HGB excluding predefined momentum features.
- Horizon: 5 trading days.
- Execution assumption: after-close signal / next-session implementation.
- Regime filter: SOXX close above trailing 200-day moving average.
- Universe and portfolio construction: fixed as documented below.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_regime_overlay_tests import (
    backtest_long_short,
    backtest_selection_with_exposure,
    regime_exposures,
)
from ml_semiconductor_robustness_tests import (
    active_regression,
    equal_weight_returns,
    market_returns,
    performance_from_returns,
)
from ml_semiconductor_sector_model import PortfolioRules, add_features, add_market_features, load_bars
from ml_semiconductor_walkforward_compare import (
    SEMICONDUCTOR_UNIVERSE,
    make_hgb,
    non_momentum_features,
    walk_forward_predictions,
)


EPSILON = 1e-12


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    n = len(residuals)
    xtx_inv = np.linalg.inv(x.T @ x)
    middle = np.zeros((x.shape[1], x.shape[1]))
    for t in range(n):
        xt = x[t : t + 1].T
        middle += residuals[t] ** 2 * (xt @ xt.T)
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1)
        gamma = np.zeros((x.shape[1], x.shape[1]))
        for t in range(l, n):
            xt = x[t : t + 1].T
            xt_lag = x[t - l : t - l + 1].T
            gamma += residuals[t] * residuals[t - l] * (xt @ xt_lag.T)
        middle += weight * (gamma + gamma.T)
    return xtx_inv @ middle @ xtx_inv


def alpha_regression_hac(
    model_daily: pd.DataFrame,
    equal_daily: pd.DataFrame,
    soxx_daily: pd.DataFrame,
    momentum_daily: pd.DataFrame,
    *,
    lag: int,
) -> dict:
    merged = (
        model_daily[["date", "net_return"]]
        .rename(columns={"net_return": "model_return"})
        .merge(equal_daily.rename(columns={"net_return": "equal_weight_return"}), on="date", how="inner")
        .merge(soxx_daily.rename(columns={"net_return": "soxx_return"}), on="date", how="inner")
        .merge(momentum_daily[["date", "net_return"]].rename(columns={"net_return": "momentum_return"}), on="date", how="inner")
    )
    y = merged["model_return"].to_numpy()
    x = merged[["soxx_return", "momentum_return"]].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    covariance = newey_west_covariance(x, residuals, lag)
    standard_errors = np.sqrt(np.diag(covariance))
    return {
        "lag": lag,
        "daily_alpha": float(beta[0]),
        "annualized_alpha_simple": float(beta[0] * 252),
        "alpha_hac_standard_error": float(standard_errors[0]),
        "alpha_hac_t_stat": float(beta[0] / standard_errors[0]) if standard_errors[0] > EPSILON else None,
        "beta_soxx": float(beta[1]),
        "beta_momentum": float(beta[2]),
        "observations": int(len(merged)),
    }


def deflated_sharpe_approx(strategy_returns: pd.Series, candidate_count: int) -> dict:
    returns = pd.Series(strategy_returns).dropna().astype(float)
    if returns.empty:
        return {}
    sharpe_daily = returns.mean() / returns.std() if returns.std() > EPSILON else 0.0
    sharpe_annual = sharpe_daily * np.sqrt(252)
    skew = float(returns.skew())
    kurtosis = float(returns.kurtosis() + 3.0)
    n = len(returns)
    sharpe_std = np.sqrt(max((1 - skew * sharpe_daily + ((kurtosis - 1) / 4) * sharpe_daily**2) / max(n - 1, 1), 0))
    expected_max_noise_sharpe = np.sqrt(2 * np.log(max(candidate_count, 2))) / np.sqrt(252)
    deflated_sharpe_daily = sharpe_daily - expected_max_noise_sharpe
    return {
        "candidate_count_assumption": int(candidate_count),
        "ordinary_annual_sharpe_from_mean": float(sharpe_annual),
        "daily_sharpe": float(sharpe_daily),
        "skew": skew,
        "kurtosis": kurtosis,
        "daily_sharpe_standard_error": float(sharpe_std),
        "expected_max_noise_daily_sharpe": float(expected_max_noise_sharpe),
        "deflated_daily_sharpe_approx": float(deflated_sharpe_daily),
        "deflated_sharpe_z_approx": float(deflated_sharpe_daily / sharpe_std) if sharpe_std > EPSILON else None,
        "note": "Approximation only; use White/Hansen/bootstrap for publication-grade multiple-testing correction.",
    }


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    covariance = y.rolling(window).cov(x)
    variance = x.rolling(window).var()
    return covariance.div(variance.replace(0, np.nan))


def long_short_exposure_report(
    long_short_daily: pd.DataFrame,
    long_short_prediction_frame: pd.DataFrame,
    market_path: Path,
    dates: set,
) -> dict:
    soxx = market_returns(market_path, "SOXX", dates).rename(columns={"net_return": "soxx_return"})
    spy = market_returns(market_path, "SPY", dates).rename(columns={"net_return": "spy_return"})
    qqq = market_returns(market_path, "QQQ", dates).rename(columns={"net_return": "qqq_return"})
    merged = (
        long_short_daily[["date", "net_return", "gross_exposure", "net_exposure", "turnover"]]
        .merge(soxx, on="date", how="inner")
        .merge(spy, on="date", how="inner")
        .merge(qqq, on="date", how="inner")
    )
    x = np.column_stack([np.ones(len(merged)), merged[["soxx_return", "spy_return", "qqq_return"]].to_numpy()])
    y = merged["net_return"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    merged["rolling_soxx_beta_60d"] = rolling_beta(merged["net_return"], merged["soxx_return"], 60)

    ranked_rows = []
    for date, day in long_short_prediction_frame.groupby("date"):
        ranked = day.sort_values("predicted_return", ascending=False)
        count = max(1, int(np.ceil(len(ranked) * 0.10)))
        for side, frame in (("long", ranked.head(count)), ("short", ranked.tail(count))):
            for _, row in frame.iterrows():
                ranked_rows.append(
                    {
                        "date": date,
                        "symbol": row["symbol"],
                        "side": side,
                        "volatility_20d": row["volatility_20d"],
                        "log_dollar_volume": row["log_dollar_volume"],
                    }
                )
    holdings = pd.DataFrame(ranked_rows)
    return {
        "avg_gross_exposure": float(merged["gross_exposure"].mean()),
        "avg_net_exposure": float(merged["net_exposure"].mean()),
        "avg_turnover": float(merged["turnover"].mean()),
        "soxx_beta": float(beta[1]),
        "spy_beta": float(beta[2]),
        "qqq_beta": float(beta[3]),
        "rolling_soxx_beta_60d_mean": float(merged["rolling_soxx_beta_60d"].mean()),
        "rolling_soxx_beta_60d_min": float(merged["rolling_soxx_beta_60d"].min()),
        "rolling_soxx_beta_60d_max": float(merged["rolling_soxx_beta_60d"].max()),
        "long_avg_volatility_20d": float(holdings.loc[holdings["side"] == "long", "volatility_20d"].mean()),
        "short_avg_volatility_20d": float(holdings.loc[holdings["side"] == "short", "volatility_20d"].mean()),
        "long_avg_log_dollar_volume": float(holdings.loc[holdings["side"] == "long", "log_dollar_volume"].mean()),
        "short_avg_log_dollar_volume": float(holdings.loc[holdings["side"] == "short", "log_dollar_volume"].mean()),
        "most_common_long_symbols": holdings[holdings["side"] == "long"]["symbol"].value_counts().head(10).to_dict(),
        "most_common_short_symbols": holdings[holdings["side"] == "short"]["symbol"].value_counts().head(10).to_dict(),
    }


def long_short_cost_stress(predictions: pd.DataFrame, base_rules: PortfolioRules) -> dict:
    results = {}
    for bps in (5.0, 25.0, 50.0, 100.0):
        rules = PortfolioRules(**{**asdict(base_rules), "transaction_cost_bps": bps})
        results[f"transaction_and_short_cost_{int(bps)}bps"] = backtest_long_short(predictions, rules)["metrics"]
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-spec diagnostics.")
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--out-dir", default="models/semiconductor_frozen_spec")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--candidate-count", type=int, default=30)
    args = parser.parse_args()

    bars = load_bars(Path(args.bars))
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, Path(args.market_bars))
    data, features = add_features(bars, market_features, args.horizon_days)
    data = data.dropna(subset=features + ["future_return", "next_1d_return"]).reset_index(drop=True)
    selection_features = non_momentum_features(features)
    predictions = walk_forward_predictions(
        data,
        selection_features,
        lambda: make_hgb(args.seed, args.n_estimators, args.learning_rate),
        args.initial_train_fraction,
    )
    dates = set(predictions["date"].unique())
    exposures = regime_exposures(data, Path(args.market_bars), dates)
    exposure_200d = dict(zip(exposures["date"], exposures["exposure_soxx_200d_filter"]))
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

    frozen_long_only = backtest_selection_with_exposure(predictions, rules, exposure_200d, equal_weight=False)
    matching_equal_weight = backtest_selection_with_exposure(predictions, rules, exposure_200d, equal_weight=True)
    long_short = backtest_long_short(predictions, rules)
    soxx = market_returns(Path(args.market_bars), "SOXX", dates)
    equal_daily = equal_weight_returns(data[data["date"].isin(dates)])
    momentum_predictions = data[data["date"].isin(dates)].copy()
    momentum_predictions["predicted_return"] = momentum_predictions["return_60d"]
    momentum = backtest_selection_with_exposure(momentum_predictions, rules, {date: 1.0 for date in dates})

    ordinary_alpha = active_regression(frozen_long_only["daily"], matching_equal_weight["daily"], soxx, momentum["daily"])
    hac_alpha = alpha_regression_hac(
        frozen_long_only["daily"], matching_equal_weight["daily"], soxx, momentum["daily"], lag=args.horizon_days
    )
    output = {
        "frozen_specification": {
            "model": "HistGradientBoostingRegressor excluding predefined momentum-like features",
            "signal_horizon": f"{args.horizon_days} trading days",
            "execution": "after-close signal, next-session execution assumption",
            "regime_filter": "SOXX close above trailing 200-day moving average",
            "universe": SEMICONDUCTOR_UNIVERSE,
            "portfolio_construction": asdict(rules),
            "costs": "5 bps transaction cost assumption in baseline",
            "selection_features": selection_features,
            "warning": "This specification should now be frozen and evaluated on new data without further tuning.",
        },
        "long_only_200d_filter": {
            "metrics": frozen_long_only["metrics"],
            "ordinary_active_alpha": ordinary_alpha,
            "newey_west_active_alpha": hac_alpha,
            "deflated_sharpe_approx": deflated_sharpe_approx(frozen_long_only["daily"]["net_return"], args.candidate_count),
        },
        "matching_equal_weight_200d_filter": matching_equal_weight["metrics"],
        "long_short_research_sleeve": {
            "metrics": long_short["metrics"],
            "cost_stress": long_short_cost_stress(predictions, rules),
            "exposure_report": long_short_exposure_report(long_short["daily"], predictions, Path(args.market_bars), dates),
        },
        "selection_bias_note": {
            "issue": "Many model families, feature sets, overlays, and portfolio constructions were explored before freezing this spec.",
            "practical_response": "Do not tune further on this same test period; track paper/live-forward performance and/or test on a later untouched period.",
            "stronger_methods_to_add": [
                "White Reality Check",
                "Hansen SPA test",
                "stationary bootstrap over daily returns",
                "probability of backtest overfitting",
            ],
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_spec_diagnostics.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
