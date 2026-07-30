#!/usr/bin/env python3
"""Untouched forward signal journal for the frozen semiconductor strategy.

Research/paper-only. This script does not submit orders.

Use:
  python3 ml_semiconductor_forward_test.py signal
  python3 ml_semiconductor_forward_test.py evaluate

The `signal` command appends an immutable archived signal to JSONL. The
`evaluate` command fills a separate realized-outcome file only after enough
future bars exist.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml_semiconductor_frozen_manifest import FROZEN_RULES, SELECTION_FEATURES
from ml_semiconductor_regime_overlay_tests import choose_selection_weights
from ml_semiconductor_sector_model import add_features, add_market_features, load_bars
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE, make_hgb


FORWARD_DIR = Path("models/semiconductor_forward_test")
SIGNAL_LOG = FORWARD_DIR / "forward_signals.jsonl"
EVALUATION_LOG = FORWARD_DIR / "forward_evaluations.jsonl"


def normalize_date(value) -> str:
    return str(pd.Timestamp(value).date())


def load_feature_frame(stock_bars: Path, market_bars: Path, horizon_days: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    bars = load_bars(stock_bars)
    bars = bars[bars["symbol"].isin(SEMICONDUCTOR_UNIVERSE)].copy()
    bars, market_features = add_market_features(bars, market_bars)
    feature_frame, all_features = add_features(bars, market_features, horizon_days)
    market = load_bars(market_bars)
    return feature_frame.replace([np.inf, -np.inf], np.nan), market, all_features


def available_signal_dates(data: pd.DataFrame) -> list:
    clean = data.dropna(subset=SELECTION_FEATURES)
    counts = clean.groupby("date")["symbol"].nunique()
    return sorted(counts[counts >= 10].index)


def train_cutoff_for_signal(signal_date, data: pd.DataFrame, horizon_days: int):
    dates = sorted(data["date"].unique())
    signal_index = dates.index(signal_date)
    cutoff_index = signal_index - horizon_days
    if cutoff_index < 0:
        raise ValueError("Not enough prior dates to form observable 5-day training labels.")
    return dates[cutoff_index]


def regime_exposure_for_date(market: pd.DataFrame, signal_date) -> dict:
    soxx = market[market["symbol"] == "SOXX"].sort_values("date").copy()
    soxx["ma_200"] = soxx["close"].rolling(200).mean()
    row = soxx[soxx["date"] == signal_date]
    if row.empty:
        raise ValueError(f"Missing SOXX market row for signal date {signal_date}")
    latest = row.iloc[-1]
    exposure = float(latest["close"] > latest["ma_200"]) if pd.notna(latest["ma_200"]) else 0.0
    return {
        "soxx_close": float(latest["close"]),
        "soxx_ma_200": float(latest["ma_200"]) if pd.notna(latest["ma_200"]) else None,
        "exposure": exposure,
        "filter_on": bool(exposure > 0),
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def latest_archived_weights() -> dict[str, float]:
    records = read_jsonl(SIGNAL_LOG)
    if not records:
        return {}
    return records[-1].get("intended_weights", {})


def rank_benchmarks(day: pd.DataFrame, exposure: float) -> dict:
    symbols = sorted(day["symbol"].unique().tolist())
    equal_weight = {symbol: exposure / len(symbols) for symbol in symbols} if symbols and exposure > 0 else {}
    momentum = (
        day.dropna(subset=["return_60d"])
        .sort_values("return_60d", ascending=False)
        .head(FROZEN_RULES.top_n)["symbol"]
        .tolist()
    )
    momentum_weight = {symbol: exposure / max(len(momentum), 1) for symbol in momentum} if exposure > 0 else {}
    return {
        "equal_weight_semis_200d_filter": equal_weight,
        "top10_60d_momentum_200d_filter": momentum_weight,
        "soxx": {"SOXX": exposure},
    }


def make_forward_signal(args: argparse.Namespace) -> dict:
    data, market, all_features = load_feature_frame(Path(args.bars), Path(args.market_bars), args.horizon_days)
    dates = available_signal_dates(data)
    signal_date = pd.Timestamp(args.signal_date).date() if args.signal_date else dates[-1]
    if signal_date not in dates:
        raise ValueError(f"Signal date {signal_date} is unavailable or lacks enough feature rows.")

    train_cutoff = train_cutoff_for_signal(signal_date, data, args.horizon_days)
    train = data[
        (data["date"] <= train_cutoff)
        & data["future_return"].notna()
    ].dropna(subset=all_features + ["future_return"])
    predict = data[data["date"] == signal_date].dropna(subset=all_features).copy()
    if train.empty or predict.empty:
        raise ValueError("Insufficient train or prediction rows.")

    model = make_hgb(args.seed, args.n_estimators, args.learning_rate)
    model.fit(train[SELECTION_FEATURES], train["future_return"])
    predict["predicted_return"] = model.predict(predict[SELECTION_FEATURES])
    predict = predict.sort_values("predicted_return", ascending=False)

    regime = regime_exposure_for_date(market, signal_date)
    current_weights = latest_archived_weights()
    base_weights = choose_selection_weights(predict, current_weights, FROZEN_RULES)
    intended_weights = {symbol: float(weight * regime["exposure"]) for symbol, weight in base_weights.items()}
    intended_weights = {symbol: weight for symbol, weight in intended_weights.items() if weight > 1e-12}
    all_symbols = set(current_weights) | set(intended_weights)
    two_way_turnover = sum(abs(intended_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in all_symbols)
    one_way_turnover = 0.5 * two_way_turnover

    signal = {
        "strategy_name": "frozen_hgb_ex_momentum_soxx_200d_long_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_asof_date": normalize_date(signal_date),
        "train_label_cutoff_date": normalize_date(train_cutoff),
        "execution_assumption": "after-close signal, intended next-session execution",
        "model": {
            "type": "HistGradientBoostingRegressor",
            "hyperparameters": {
                "max_iter": args.n_estimators,
                "max_leaf_nodes": 31,
                "learning_rate": args.learning_rate,
                "l2_regularization": 0.2,
                "random_state": args.seed,
            },
            "train_rows": int(len(train)),
            "predict_rows": int(len(predict)),
            "warmup_eligibility": "Rows require complete full 62-feature research matrix, then use frozen 18 features.",
        },
        "regime": regime,
        "scores": predict[["symbol", "predicted_return", "volatility_20d", "relative_volume_20d", "log_dollar_volume"]]
        .to_dict(orient="records"),
        "selected_top10": list(intended_weights.keys()),
        "intended_weights": intended_weights,
        "current_weights_before_signal": current_weights,
        "turnover": {
            "two_way_abs_weight_change": float(two_way_turnover),
            "one_way_abs_weight_change": float(one_way_turnover),
            "estimated_cost": float(two_way_turnover * FROZEN_RULES.transaction_cost_bps / 10000),
            "transaction_cost_bps": FROZEN_RULES.transaction_cost_bps,
        },
        "benchmarks": rank_benchmarks(predict, regime["exposure"]),
        "status": "archived_signal_not_yet_evaluated",
    }
    append_jsonl(SIGNAL_LOG, signal)
    return signal


def evaluate_signal(signal: dict, data: pd.DataFrame, market: pd.DataFrame, horizon_days: int) -> dict | None:
    signal_date = pd.Timestamp(signal["data_asof_date"]).date()
    dates = sorted(data["date"].unique())
    if signal_date not in dates:
        return None
    signal_index = dates.index(signal_date)
    realized_index = signal_index + horizon_days
    if realized_index >= len(dates):
        return None
    realized_date = dates[realized_index]
    by_symbol = data[data["date"].isin([signal_date, realized_date])].pivot(index="symbol", columns="date", values="close")
    if signal_date not in by_symbol.columns or realized_date not in by_symbol.columns:
        return None
    realized_returns = by_symbol[realized_date].sub(by_symbol[signal_date]).div(by_symbol[signal_date]).dropna()

    def portfolio_return(weights: dict[str, float]) -> float:
        return float(sum(float(weight) * float(realized_returns.get(symbol, 0.0)) for symbol, weight in weights.items()))

    soxx = market[market["symbol"] == "SOXX"].sort_values("date")
    soxx_pivot = soxx[soxx["date"].isin([signal_date, realized_date])].set_index("date")["close"]
    soxx_return = None
    if signal_date in soxx_pivot.index and realized_date in soxx_pivot.index:
        soxx_return = float((soxx_pivot.loc[realized_date] - soxx_pivot.loc[signal_date]) / soxx_pivot.loc[signal_date])

    selected = signal.get("intended_weights", {})
    top = [row["symbol"] for row in signal.get("scores", [])[: FROZEN_RULES.top_n]]
    bottom = [row["symbol"] for row in signal.get("scores", [])[-FROZEN_RULES.top_n :]]
    top_return = float(realized_returns.reindex(top).dropna().mean()) if top else None
    bottom_return = float(realized_returns.reindex(bottom).dropna().mean()) if bottom else None
    rank_ic = None
    score_frame = pd.DataFrame(signal.get("scores", []))
    if not score_frame.empty:
        merged = score_frame.set_index("symbol")["predicted_return"].to_frame().join(realized_returns.rename("realized_return"))
        merged = merged.dropna()
        if len(merged) >= 5:
            rank_ic = float(merged["predicted_return"].rank().corr(merged["realized_return"].rank()))

    return {
        "strategy_name": signal["strategy_name"],
        "signal_created_at_utc": signal["created_at_utc"],
        "data_asof_date": normalize_date(signal_date),
        "realized_date": normalize_date(realized_date),
        "horizon_days": horizon_days,
        "realized_strategy_return_before_cost": portfolio_return(selected),
        "estimated_signal_cost": signal.get("turnover", {}).get("estimated_cost", 0.0),
        "realized_strategy_return_after_estimated_cost": portfolio_return(selected)
        - float(signal.get("turnover", {}).get("estimated_cost", 0.0)),
        "benchmark_returns": {
            name: portfolio_return(weights)
            for name, weights in signal.get("benchmarks", {}).items()
            if isinstance(weights, dict)
        },
        "soxx_return": soxx_return,
        "rank_ic": rank_ic,
        "top10_avg_return": top_return,
        "bottom10_avg_return": bottom_return,
        "top_minus_bottom_return": top_return - bottom_return if top_return is not None and bottom_return is not None else None,
        "selected_symbol_returns": {
            symbol: float(realized_returns.get(symbol, 0.0)) for symbol in selected
        },
    }


def evaluate_archived_signals(args: argparse.Namespace) -> list[dict]:
    data, market, _ = load_feature_frame(Path(args.bars), Path(args.market_bars), args.horizon_days)
    signals = read_jsonl(SIGNAL_LOG)
    existing = {
        (record["signal_created_at_utc"], record["data_asof_date"])
        for record in read_jsonl(EVALUATION_LOG)
    }
    new_records = []
    for signal in signals:
        key = (signal["created_at_utc"], signal["data_asof_date"])
        if key in existing:
            continue
        evaluation = evaluate_signal(signal, data, market, args.horizon_days)
        if evaluation is not None:
            append_jsonl(EVALUATION_LOG, evaluation)
            new_records.append(evaluation)
    return new_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward journal for the frozen semiconductor strategy.")
    parser.add_argument("command", choices=["signal", "evaluate", "status"])
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--signal-date", default=None)
    args = parser.parse_args()

    if args.command == "signal":
        print(json.dumps(make_forward_signal(args), indent=2, default=str))
    elif args.command == "evaluate":
        print(json.dumps({"new_evaluations": evaluate_archived_signals(args)}, indent=2, default=str))
    else:
        print(
            json.dumps(
                {
                    "signal_log": str(SIGNAL_LOG),
                    "signal_count": len(read_jsonl(SIGNAL_LOG)),
                    "evaluation_log": str(EVALUATION_LOG),
                    "evaluation_count": len(read_jsonl(EVALUATION_LOG)),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
