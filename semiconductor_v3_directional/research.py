#!/usr/bin/env python3
"""Walk-forward V3 semiconductor direction-plus-selection research."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
EPSILON = 1e-12


@dataclass
class ResearchData:
    panel: pd.DataFrame
    direction: pd.DataFrame
    market: pd.DataFrame
    decision_dates: list
    market_session_index: dict
    audit: dict


def load_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["date"] = frame["timestamp"].dt.date
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for column in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def load_membership(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "symbol",
        "valid_from",
        "valid_to",
        "industry",
        "security_type",
        "us_listed",
        "survivorship_safe",
        "delisted",
        "delisting_handled",
        "source",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Membership file is missing: {sorted(missing)}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"], errors="raise").dt.date
    frame["valid_to"] = pd.to_datetime(frame["valid_to"], errors="coerce").dt.date
    for column in ["us_listed", "survivorship_safe", "delisted", "delisting_handled"]:
        frame[column] = boolean_series(frame[column])
    return frame


def load_borrow(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"date", "symbol", "shortable", "annual_borrow_rate", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Borrow file is missing: {sorted(missing)}")
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["shortable"] = boolean_series(frame["shortable"])
    frame["annual_borrow_rate"] = pd.to_numeric(frame["annual_borrow_rate"], errors="coerce")
    return frame


def membership_mask(frame: pd.DataFrame, membership: pd.DataFrame, spec: dict) -> pd.Series:
    universe = spec["universe"]
    active = pd.Series(False, index=frame.index)
    eligible_membership = membership[
        membership["industry"].eq(universe["industry"])
        & membership["security_type"].eq(universe["security_type"])
        & membership["us_listed"].eq(bool(universe["us_listed"]))
    ]
    for row in eligible_membership.itertuples(index=False):
        mask = frame["symbol"].eq(row.symbol) & frame["date"].ge(row.valid_from)
        if not pd.isna(row.valid_to):
            mask &= frame["date"].le(row.valid_to)
        active |= mask
    return active


def rolling_beta(stock: pd.Series, market: pd.Series, window: int) -> pd.Series:
    covariance = stock.rolling(window, min_periods=window).cov(market)
    variance = market.rolling(window, min_periods=window).var()
    return covariance.div(variance.replace(0, np.nan))


def make_decision_grid(
    session_dates: list,
    desired_start,
    requested_anchor,
    every: int,
) -> tuple[list, int, bool]:
    if not session_dates:
        return [], 0, False
    session_index = {date: index for index, date in enumerate(session_dates)}
    anchor_in_data = requested_anchor in session_index
    if anchor_in_data:
        anchor_index = session_index[requested_anchor]
    elif requested_anchor > session_dates[-1]:
        estimated_future_sessions = len(
            pd.bdate_range(pd.Timestamp(session_dates[-1]) + pd.Timedelta(days=1), requested_anchor)
        )
        anchor_index = len(session_dates) - 1 + estimated_future_sessions
    else:
        raise ValueError(
            f"Decision-grid anchor {requested_anchor} is inside the data range but is not a market session."
        )
    dates = [
        date
        for index, date in enumerate(session_dates)
        if date >= desired_start and (index - anchor_index) % every == 0
    ]
    return dates, anchor_index, anchor_in_data


def market_feature_frame(market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pieces = []
    for symbol in ["SOXX", "SPY", "QQQ"]:
        one = market[market["symbol"].eq(symbol)].sort_values("date").copy()
        if one.empty:
            raise ValueError(f"Missing required market symbol {symbol}.")
        prefix = symbol.lower()
        for window in [1, 5, 20, 60]:
            one[f"{prefix}_return_{window}d"] = one["close"].pct_change(window)
        keep = ["date", "open", "close", "volume"] + [
            f"{prefix}_return_{window}d" for window in [1, 5, 20, 60]
        ]
        one = one[keep].rename(
            columns={
                "open": f"{prefix}_open",
                "close": f"{prefix}_close",
                "volume": f"{prefix}_volume",
            }
        )
        pieces.append(one)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on="date", how="outer")
    result = result.sort_values("date").reset_index(drop=True)

    for window in [20, 50, 200]:
        average = result["soxx_close"].rolling(window, min_periods=window).mean()
        result[f"soxx_distance_ma_{window}"] = result["soxx_close"].div(average) - 1.0
    for window in [10, 20, 60]:
        result[f"soxx_volatility_{window}d"] = result["soxx_return_1d"].rolling(
            window, min_periods=window
        ).std()
    rolling_high = result["soxx_close"].rolling(60, min_periods=60).max()
    result["soxx_drawdown_60d"] = result["soxx_close"].div(rolling_high) - 1.0
    prior_volume = result["soxx_volume"].shift(1).rolling(20, min_periods=20).mean()
    result["soxx_relative_volume_20d"] = result["soxx_volume"].div(prior_volume)

    result["direction_target"] = result["soxx_open"].shift(-(horizon + 1)).div(
        result["soxx_open"].shift(-1)
    ) - 1.0
    result["soxx_entry_open"] = result["soxx_open"].shift(-1)
    result["soxx_exit_open"] = result["soxx_open"].shift(-(horizon + 1))
    result["label_entry_date"] = result["date"].shift(-1)
    result["label_exit_date"] = result["date"].shift(-(horizon + 1))
    result["future_spy_return_5d"] = result["spy_open"].shift(-(horizon + 1)).div(
        result["spy_open"].shift(-1)
    ) - 1.0
    result["future_qqq_return_5d"] = result["qqq_open"].shift(-(horizon + 1)).div(
        result["qqq_open"].shift(-1)
    ) - 1.0
    result["soxx_above_ma_200"] = result["soxx_distance_ma_200"].gt(0)
    return result.replace([np.inf, -np.inf], np.nan)


def stock_feature_frame(stocks: pd.DataFrame, market_features: pd.DataFrame, spec: dict) -> pd.DataFrame:
    horizon = int(spec["execution"]["horizon_sessions"])
    frame = stocks.merge(market_features, on="date", how="left")
    frame = frame.sort_values(["symbol", "date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["history_sessions"] = grouped.cumcount() + 1
    for window in [1, 5, 20, 60]:
        frame[f"return_{window}d"] = grouped["close"].pct_change(window)
    for window in [20, 60]:
        frame[f"volatility_{window}d"] = frame["return_1d"].groupby(frame["symbol"]).transform(
            lambda values, w=window: values.rolling(w, min_periods=w).std()
        )

    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_20_pct"] = true_range.groupby(frame["symbol"]).transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    ).div(frame["close"])
    frame["parkinson_volatility_20d"] = np.sqrt(
        (np.log(frame["high"].div(frame["low"])) ** 2 / (4 * np.log(2))).groupby(
            frame["symbol"]
        ).transform(lambda values: values.rolling(20, min_periods=20).mean())
    )
    prior_average_volume = grouped["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )
    frame["relative_volume_20d"] = frame["volume"].div(prior_average_volume)
    frame["turnover_proxy"] = frame["volume"].div(prior_average_volume)
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["log_dollar_volume"] = np.log1p(frame["dollar_volume"])
    liquidity_window = int(spec["universe"]["liquidity_window_sessions"])
    frame["median_dollar_volume_60d"] = frame["dollar_volume"].groupby(frame["symbol"]).transform(
        lambda values: values.rolling(liquidity_window, min_periods=liquidity_window).median()
    )

    frame["body_pct"] = (frame["close"] - frame["open"]).div(frame["close"])
    frame["range_pct"] = (frame["high"] - frame["low"]).div(frame["close"])
    frame["close_position"] = safe_div(frame["close"] - frame["low"], frame["high"] - frame["low"]).fillna(0.5)
    frame["gap_from_yesterday"] = (frame["open"] - previous_close).div(previous_close)
    frame["upper_shadow_pct"] = (frame["high"] - frame[["open", "close"]].max(axis=1)).div(frame["close"])
    frame["lower_shadow_pct"] = (frame[["open", "close"]].min(axis=1) - frame["low"]).div(frame["close"])

    frame["beta_20d_to_soxx"] = np.nan
    frame["beta_60d_to_soxx"] = np.nan
    for _, indexes in frame.groupby("symbol").groups.items():
        one = frame.loc[indexes].sort_values("date")
        for window in [20, 60]:
            frame.loc[one.index, f"beta_{window}d_to_soxx"] = rolling_beta(
                one["return_1d"], one["soxx_return_1d"], window
            ).to_numpy()

    for source in ["relative_volume_20d", "volatility_20d", "log_dollar_volume"]:
        frame[f"cs_rank_{source}"] = frame.groupby("date")[source].rank(pct=True)

    frame["entry_open"] = grouped["open"].shift(-1)
    frame["exit_open"] = grouped["open"].shift(-(horizon + 1))
    frame["future_stock_return_5d"] = frame["exit_open"].div(frame["entry_open"]) - 1.0
    frame["selection_target"] = frame["future_stock_return_5d"] - frame["direction_target"]
    frame["stock_label_entry_date"] = grouped["date"].shift(-1)
    frame["stock_label_exit_date"] = grouped["date"].shift(-(horizon + 1))
    return frame.replace([np.inf, -np.inf], np.nan)


def build_research_data(spec: dict, exploratory: bool) -> ResearchData:
    stocks = load_bars(project_path(spec["data"]["stock_bars"]))
    market = load_bars(project_path(spec["data"]["market_bars"]))
    membership = load_membership(project_path(spec["data"]["universe_membership"]))
    borrow = load_borrow(project_path(spec["data"]["borrow_costs"]))
    required_market = set(spec["data"]["required_market_symbols"])
    missing_market = required_market - set(market["symbol"].unique())
    if missing_market:
        raise ValueError(f"Missing market symbols: {sorted(missing_market)}")

    point_in_time = bool(membership["survivorship_safe"].all())
    delistings_handled = bool(membership["delisting_handled"].all())
    borrow_available = not borrow.empty
    strict_blockers = []
    if not point_in_time:
        strict_blockers.append("universe membership is not point-in-time/survivorship-safe")
    if not delistings_handled:
        strict_blockers.append("delisting outcomes are not documented as handled")
    if not borrow_available:
        strict_blockers.append("historical borrow availability and fees are absent")
    if strict_blockers and not exploratory:
        raise ValueError("Strict V3 research blocked: " + "; ".join(strict_blockers))

    horizon = int(spec["execution"]["horizon_sessions"])
    market_features = market_feature_frame(market, horizon)
    panel = stock_feature_frame(stocks, market_features, spec)
    panel["membership_active"] = membership_mask(panel, membership, spec)
    universe = spec["universe"]
    panel["eligible_before_cap"] = (
        panel["membership_active"]
        & panel["history_sessions"].ge(int(universe["minimum_history_sessions"]))
        & panel["close"].ge(float(universe["minimum_price"]))
        & panel["median_dollar_volume_60d"].ge(float(universe["minimum_median_dollar_volume"]))
    )
    panel["liquidity_rank"] = panel.loc[panel["eligible_before_cap"]].groupby("date")[
        "median_dollar_volume_60d"
    ].rank(ascending=False, method="first")
    panel["eligible"] = panel["eligible_before_cap"] & panel["liquidity_rank"].le(
        int(universe["maximum_names"])
    )

    eligible = panel[panel["eligible"]]
    breadth = eligible.groupby("date").agg(
        sector_breadth_positive_20d=("return_20d", lambda values: float((values > 0).mean())),
        sector_median_return_1d=("return_1d", "median"),
        sector_median_return_20d=("return_20d", "median"),
        sector_dispersion_1d=("return_1d", "std"),
        sector_dispersion_20d=("return_20d", "std"),
        eligible_names=("symbol", "nunique"),
    ).reset_index()
    ma60 = panel.groupby("symbol")["close"].transform(lambda values: values.rolling(60, min_periods=60).mean())
    panel["above_ma_60"] = panel["close"].gt(ma60)
    above = panel[panel["eligible"]].groupby("date")["above_ma_60"].mean().rename(
        "sector_breadth_above_ma_60"
    ).reset_index()
    breadth = breadth.merge(above, on="date", how="left")
    direction = market_features.merge(breadth, on="date", how="left")

    if borrow_available:
        panel = panel.merge(
            borrow[["date", "symbol", "shortable", "annual_borrow_rate"]],
            on=["date", "symbol"],
            how="left",
        )
    else:
        panel["shortable"] = True if exploratory else False
        panel["annual_borrow_rate"] = float(spec["costs"]["primary_annual_borrow_rate_fallback"])

    session_dates = sorted(direction["date"].dropna().unique())
    session_index = {date: index for index, date in enumerate(session_dates)}
    desired_start = pd.Timestamp(spec["data"]["desired_start_date"]).date()
    requested_anchor = pd.Timestamp(spec["execution"]["decision_grid_anchor_date"]).date()
    every = int(spec["execution"]["decision_grid_every_sessions"])
    decision_dates, anchor_index, anchor_in_data = make_decision_grid(
        session_dates, desired_start, requested_anchor, every
    )
    decision_set = set(decision_dates)
    panel["on_decision_grid"] = panel["date"].isin(decision_set)
    direction["on_decision_grid"] = direction["date"].isin(decision_set)

    counts = eligible.groupby("date")["symbol"].nunique()
    audit = {
        "exploratory": bool(exploratory),
        "strict_blockers": strict_blockers,
        "confirmatory_eligible": bool(not strict_blockers and not exploratory),
        "stock_rows": int(len(stocks)),
        "stock_symbols": int(stocks["symbol"].nunique()),
        "stock_start": str(stocks["date"].min()),
        "stock_end": str(stocks["date"].max()),
        "market_start": str(market["date"].min()),
        "market_end": str(market["date"].max()),
        "membership_rows": int(len(membership)),
        "membership_sources": sorted(membership["source"].unique().tolist()),
        "point_in_time_universe": point_in_time,
        "delistings_handled": delistings_handled,
        "historical_borrow_available": borrow_available,
        "eligible_rows": int(panel["eligible"].sum()),
        "eligible_symbols": int(panel.loc[panel["eligible"], "symbol"].nunique()),
        "median_eligible_names": float(counts.median()) if not counts.empty else 0.0,
        "maximum_eligible_names": int(counts.max()) if not counts.empty else 0,
        "decision_grid_dates": int(len(decision_dates)),
        "decision_grid_anchor_requested": str(requested_anchor),
        "decision_grid_anchor_present_in_data": anchor_in_data,
        "decision_grid_phase_reference_index": int(anchor_index),
    }
    return ResearchData(panel, direction, market_features, decision_dates, session_index, audit)


def make_direction_hgb(spec: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**spec["models"]["direction_hgb"])


def make_direction_ridge(spec: dict):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=float(spec["models"]["direction_ridge"]["alpha"])),
    )


def make_selector(spec: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**spec["models"]["selection_hgb"])


def cross_section_sample_weights(frame: pd.DataFrame) -> pd.Series:
    counts = frame.groupby("date")["symbol"].transform("count")
    return 1.0 / counts


def walk_forward(data: ResearchData, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction_features = spec["direction_features"]
    selection_features = spec["selection_features"]
    minimum_dates = int(spec["models"]["minimum_training_decision_dates"])
    retrain_every = int(spec["models"]["retrain_every_sessions"])
    minimum_names = int(spec["universe"]["minimum_names_to_trade"])
    oos_start = pd.Timestamp(spec["data"]["historical_oos_start"]).date()
    research_cutoff = pd.Timestamp(spec["data"]["research_label_cutoff"]).date()

    direction_grid = data.direction[data.direction["on_decision_grid"]].copy()
    panel_grid = data.panel[data.panel["on_decision_grid"] & data.panel["eligible"]].copy()
    models = None
    last_retrain_session = None
    prediction_rows = []
    audits = []

    for signal_date in data.decision_dates:
        if signal_date < oos_start:
            continue
        direction_train = direction_grid[
            direction_grid["label_exit_date"].notna()
            & direction_grid["label_exit_date"].le(signal_date)
            & direction_grid["date"].lt(signal_date)
            & direction_grid["date"].le(research_cutoff)
        ].dropna(subset=direction_features + ["direction_target"])
        selector_train = panel_grid[
            panel_grid["stock_label_exit_date"].notna()
            & panel_grid["stock_label_exit_date"].le(signal_date)
            & panel_grid["date"].lt(signal_date)
            & panel_grid["date"].le(research_cutoff)
        ].dropna(subset=selection_features + ["selection_target"])
        if direction_train["date"].nunique() < minimum_dates or selector_train["date"].nunique() < minimum_dates:
            continue

        current_session = data.market_session_index[signal_date]
        retrain_due = models is None or current_session - int(last_retrain_session) >= retrain_every
        if retrain_due:
            direction_hgb = make_direction_hgb(spec)
            direction_ridge = make_direction_ridge(spec)
            selector = make_selector(spec)
            direction_hgb.fit(direction_train[direction_features], direction_train["direction_target"])
            direction_ridge.fit(direction_train[direction_features], direction_train["direction_target"])
            sample_weight = cross_section_sample_weights(selector_train)
            selector.fit(
                selector_train[selection_features],
                selector_train["selection_target"],
                sample_weight=sample_weight,
            )
            models = (direction_hgb, direction_ridge, selector)
            last_retrain_session = current_session
            audits.append(
                {
                    "retrain_signal_date": signal_date,
                    "direction_train_start": direction_train["date"].min(),
                    "direction_train_end": direction_train["date"].max(),
                    "direction_last_label_exit": direction_train["label_exit_date"].max(),
                    "direction_train_dates": int(direction_train["date"].nunique()),
                    "selector_train_start": selector_train["date"].min(),
                    "selector_train_end": selector_train["date"].max(),
                    "selector_last_label_exit": selector_train["stock_label_exit_date"].max(),
                    "selector_train_dates": int(selector_train["date"].nunique()),
                    "selector_train_rows": int(len(selector_train)),
                    "selector_total_weight": float(sample_weight.sum()),
                }
            )

        direction_day = direction_grid[direction_grid["date"].eq(signal_date)].dropna(subset=direction_features)
        stock_day = panel_grid[panel_grid["date"].eq(signal_date)].dropna(subset=selection_features)
        if direction_day.empty or len(stock_day) < minimum_names:
            continue
        direction_hgb, direction_ridge, selector = models
        hgb_prediction = float(direction_hgb.predict(direction_day[direction_features])[0])
        ridge_prediction = float(direction_ridge.predict(direction_day[direction_features])[0])
        stock_day = stock_day.copy()
        stock_day["selection_prediction"] = selector.predict(stock_day[selection_features])
        market_row = direction_day.iloc[0]
        for row in stock_day.itertuples(index=False):
            prediction_rows.append(
                {
                    "signal_date": signal_date,
                    "symbol": row.symbol,
                    "direction_hgb_prediction": hgb_prediction,
                    "direction_ridge_prediction": ridge_prediction,
                    "direction_target": market_row.direction_target,
                    "selection_prediction": row.selection_prediction,
                    "selection_target": row.selection_target,
                    "future_stock_return_5d": row.future_stock_return_5d,
                    "future_soxx_return_5d": market_row.direction_target,
                    "future_spy_return_5d": market_row.future_spy_return_5d,
                    "future_qqq_return_5d": market_row.future_qqq_return_5d,
                    "entry_date": row.stock_label_entry_date,
                    "exit_date": row.stock_label_exit_date,
                    "shortable": row.shortable,
                    "annual_borrow_rate": row.annual_borrow_rate,
                    "soxx_above_ma_200": bool(market_row.soxx_above_ma_200),
                    "eligible_names": int(len(stock_day)),
                    "model_retrain_signal_date": data.decision_dates[
                        max(
                            index
                            for index, candidate_date in enumerate(data.decision_dates)
                            if data.market_session_index[candidate_date] <= int(last_retrain_session)
                        )
                    ],
                }
            )
    if not prediction_rows:
        raise RuntimeError("No walk-forward predictions were produced. Check history and universe eligibility.")
    return pd.DataFrame(prediction_rows), pd.DataFrame(audits)


def equal_weights(symbols: list[str], side: float, top_n: int) -> dict[str, float]:
    selected = symbols[:top_n]
    if len(selected) < top_n:
        return {}
    return {symbol: side / top_n for symbol in selected}


def weights_for_strategy(day: pd.DataFrame, strategy: str, spec: dict) -> tuple[dict[str, float], str]:
    top_n = int(spec["universe"]["minimum_names_to_trade"]) // 2
    ordered = day.sort_values("selection_prediction", ascending=False)
    top = ordered["symbol"].head(top_n).tolist()
    bottom = ordered["symbol"].tail(top_n).tolist()[::-1]
    if strategy == "hgb_directional":
        positive = float(day["direction_hgb_prediction"].iloc[0]) > 0
    elif strategy == "ridge_directional":
        positive = float(day["direction_ridge_prediction"].iloc[0]) > 0
    elif strategy == "ma200_directional":
        positive = bool(day["soxx_above_ma_200"].iloc[0])
    elif strategy == "ma200_long_cash":
        if not bool(day["soxx_above_ma_200"].iloc[0]):
            return {}, "cash"
        return equal_weights(top, 1.0, top_n), "long"
    elif strategy == "selector_always_long":
        return equal_weights(top, 1.0, top_n), "long"
    elif strategy == "soxx_buy_hold":
        return {"SOXX": 1.0}, "long"
    else:
        raise ValueError(strategy)
    if positive:
        return equal_weights(top, 1.0, top_n), "long"
    short_day = ordered.set_index("symbol").reindex(bottom)
    if not boolean_series(short_day["shortable"]).all():
        return {}, "unavailable_short"
    return equal_weights(bottom, -1.0, top_n), "short"


def backtest(
    predictions: pd.DataFrame,
    strategy: str,
    spec: dict,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    cost_rate = float(spec["costs"]["primary_one_way_bps"]) * cost_multiplier / 10000
    fallback_borrow = float(spec["costs"]["primary_annual_borrow_rate_fallback"]) * cost_multiplier
    horizon = int(spec["execution"]["horizon_sessions"])
    current: dict[str, float] = {}
    records = []
    realized_dates = predictions.dropna(
        subset=["future_stock_return_5d", "future_soxx_return_5d", "exit_date"]
    )["signal_date"].unique()
    for signal_date in sorted(realized_dates):
        day = predictions[predictions["signal_date"].eq(signal_date)].copy()
        target, side = weights_for_strategy(day, strategy, spec)
        if side == "unavailable_short":
            raise ValueError(f"Historical borrow data marks a selected short unavailable on {signal_date}.")
        turnover = sum(
            abs(target.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in set(target) | set(current)
        )
        transaction_cost = turnover * cost_rate
        stock_returns = day.set_index("symbol")["future_stock_return_5d"]
        gross_return = 0.0
        for symbol, weight in target.items():
            realized = (
                float(day["future_soxx_return_5d"].iloc[0])
                if symbol == "SOXX"
                else float(stock_returns.get(symbol, np.nan))
            )
            if not np.isfinite(realized):
                raise ValueError(f"Missing realized return for selected {symbol} on {signal_date}.")
            gross_return += weight * realized
        borrow_cost = 0.0
        if side == "short":
            borrow_lookup = day.set_index("symbol")["annual_borrow_rate"]
            for symbol, weight in target.items():
                rate = float(borrow_lookup.get(symbol, fallback_borrow))
                if not np.isfinite(rate):
                    rate = fallback_borrow
                borrow_cost += abs(weight) * rate * horizon / 252
        net_return = gross_return - transaction_cost - borrow_cost
        records.append(
            {
                "signal_date": signal_date,
                "entry_date": day["entry_date"].iloc[0],
                "exit_date": day["exit_date"].iloc[0],
                "strategy": strategy,
                "side": side,
                "gross_return": gross_return,
                "net_return": net_return,
                "transaction_cost": transaction_cost,
                "borrow_cost": borrow_cost,
                "turnover": turnover,
                "gross_exposure": sum(abs(weight) for weight in target.values()),
                "net_exposure": sum(target.values()),
                "direction_hgb_prediction": float(day["direction_hgb_prediction"].iloc[0]),
                "direction_ridge_prediction": float(day["direction_ridge_prediction"].iloc[0]),
                "future_soxx_return_5d": float(day["future_soxx_return_5d"].iloc[0]),
                "future_spy_return_5d": float(day["future_spy_return_5d"].iloc[0]),
                "future_qqq_return_5d": float(day["future_qqq_return_5d"].iloc[0]),
                "weights": json.dumps(target, sort_keys=True),
                "cost_multiplier": cost_multiplier,
            }
        )
        current = target
    result = pd.DataFrame(records)
    if not result.empty and current:
        final_cost = sum(abs(weight) for weight in current.values()) * cost_rate
        result.loc[result.index[-1], "transaction_cost"] += final_cost
        result.loc[result.index[-1], "net_return"] -= final_cost
        result.loc[result.index[-1], "turnover"] += sum(abs(weight) for weight in current.values())
    return result


def performance(returns: pd.Series, periods_per_year: float) -> dict:
    values = pd.Series(returns).dropna().astype(float)
    if values.empty:
        return {}
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    annual_return = float(equity.iloc[-1] ** (periods_per_year / len(values)) - 1.0)
    volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    return {
        "periods": int(len(values)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe": float(values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year))
        if values.std(ddof=1) > EPSILON
        else 0.0,
        "max_drawdown": float((equity / peak - 1.0).min()),
        "win_rate": float((values > 0).mean()),
        "mean_period_return": float(values.mean()),
        "median_period_return": float(values.median()),
    }


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    inverse = np.linalg.pinv(x.T @ x)
    middle = np.zeros((x.shape[1], x.shape[1]))
    for index in range(len(residuals)):
        vector = x[index : index + 1].T
        middle += residuals[index] ** 2 * (vector @ vector.T)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1)
        gamma = np.zeros_like(middle)
        for index in range(offset, len(residuals)):
            current = x[index : index + 1].T
            previous = x[index - offset : index - offset + 1].T
            gamma += residuals[index] * residuals[index - offset] * (current @ previous.T)
        middle += weight * (gamma + gamma.T)
    return inverse @ middle @ inverse


def factor_alpha(periods: pd.DataFrame, spec: dict) -> dict:
    clean = periods.dropna(subset=["net_return", "future_soxx_return_5d", "future_qqq_return_5d"])
    if len(clean) < 10:
        return {}
    y = clean["net_return"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(clean)),
            clean["future_soxx_return_5d"].to_numpy(dtype=float),
            clean["future_qqq_return_5d"].to_numpy(dtype=float),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    covariance = newey_west_covariance(x, residuals, int(spec["evaluation"]["hac_lag_periods"]))
    standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    periods_per_year = float(spec["evaluation"]["periods_per_year"])
    return {
        "period_alpha": float(coefficients[0]),
        "annualized_alpha": float(coefficients[0] * periods_per_year),
        "alpha_hac_t": float(coefficients[0] / standard_error) if standard_error > EPSILON else None,
        "beta_soxx": float(coefficients[1]),
        "beta_qqq": float(coefficients[2]),
        "observations": int(len(clean)),
        "factors": ["SOXX open-to-open 5d return", "QQQ open-to-open 5d return"],
    }


def direction_metrics(predictions: pd.DataFrame, prediction_column: str) -> dict:
    dates = predictions.drop_duplicates("signal_date").dropna(subset=[prediction_column, "direction_target"])
    predicted_positive = dates[prediction_column].gt(0)
    actual_positive = dates["direction_target"].gt(0)
    return {
        "periods": int(len(dates)),
        "sign_accuracy": float(predicted_positive.eq(actual_positive).mean()),
        "prediction_target_correlation": float(dates[prediction_column].corr(dates["direction_target"])),
        "mean_actual_soxx_when_predicted_positive": float(dates.loc[predicted_positive, "direction_target"].mean()),
        "mean_actual_soxx_when_predicted_negative": float(dates.loc[~predicted_positive, "direction_target"].mean()),
        "predicted_positive_periods": int(predicted_positive.sum()),
        "predicted_negative_periods": int((~predicted_positive).sum()),
    }


def selection_metrics(predictions: pd.DataFrame, direction_column: str) -> dict:
    rows = []
    for signal_date, day in predictions.dropna(subset=["selection_prediction", "selection_target"]).groupby(
        "signal_date"
    ):
        if len(day) < 20 or day["selection_prediction"].nunique() < 2:
            continue
        ordered = day.sort_values("selection_prediction", ascending=False)
        rows.append(
            {
                "signal_date": signal_date,
                "rank_ic": float(day["selection_prediction"].rank().corr(day["selection_target"].rank())),
                "top_minus_bottom": float(
                    ordered.head(10)["selection_target"].mean()
                    - ordered.tail(10)["selection_target"].mean()
                ),
                "predicted_regime": "long" if float(day[direction_column].iloc[0]) > 0 else "short",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    result = {
        "periods": int(len(frame)),
        "mean_rank_ic": float(frame["rank_ic"].mean()),
        "mean_top_minus_bottom": float(frame["top_minus_bottom"].mean()),
        "positive_rank_ic_fraction": float(frame["rank_ic"].gt(0).mean()),
    }
    for regime in ["long", "short"]:
        subset = frame[frame["predicted_regime"].eq(regime)]
        result[f"{regime}_periods"] = int(len(subset))
        result[f"{regime}_mean_rank_ic"] = float(subset["rank_ic"].mean()) if not subset.empty else None
        result[f"{regime}_mean_top_minus_bottom"] = (
            float(subset["top_minus_bottom"].mean()) if not subset.empty else None
        )
    return result


def strategy_summary(periods: pd.DataFrame, spec: dict) -> dict:
    periods_per_year = float(spec["evaluation"]["periods_per_year"])
    summary = performance(periods["net_return"], periods_per_year)
    summary.update(factor_alpha(periods, spec))
    summary.update(
        {
            "total_turnover": float(periods["turnover"].sum()),
            "average_turnover": float(periods["turnover"].mean()),
            "total_transaction_cost": float(periods["transaction_cost"].sum()),
            "total_borrow_cost": float(periods["borrow_cost"].sum()),
            "long_periods": int(periods["side"].eq("long").sum()),
            "short_periods": int(periods["side"].eq("short").sum()),
            "cash_periods": int(periods["side"].eq("cash").sum()),
            "long_regime": performance(periods.loc[periods["side"].eq("long"), "net_return"], periods_per_year),
            "short_regime": performance(periods.loc[periods["side"].eq("short"), "net_return"], periods_per_year),
        }
    )
    yearly = periods.assign(year=pd.to_datetime(periods["signal_date"]).dt.year).groupby("year")["net_return"].apply(
        lambda values: float(np.log1p(values).sum())
    )
    positive = yearly.clip(lower=0)
    share = float(positive.max() / positive.sum()) if positive.sum() > EPSILON else 1.0
    summary["yearly_log_returns"] = {str(year): value for year, value in yearly.items()}
    summary["largest_positive_year_share"] = share
    return summary


def pass_fail(
    primary: dict,
    stress: dict,
    selector: dict,
    audit: dict,
    spec: dict,
) -> dict:
    threshold = float(spec["evaluation"]["minimum_hac_alpha_t_for_strong_claim"])
    checks = {
        "long_regime_after_cost_positive": primary.get("long_regime", {}).get("total_return", -1) > 0,
        "short_regime_after_cost_positive": primary.get("short_regime", {}).get("total_return", -1) > 0,
        "rank_ic_predicted_long_positive": (selector.get("long_mean_rank_ic") or -1) > 0,
        "rank_ic_predicted_short_positive": (selector.get("short_mean_rank_ic") or -1) > 0,
        "factor_alpha_positive": primary.get("annualized_alpha", -1) > 0,
        "hac_alpha_t_at_least_2": (primary.get("alpha_hac_t") or -999) >= threshold,
        "stress_total_return_positive": stress.get("total_return", -1) > 0,
        "not_dominated_by_one_positive_year": primary.get("largest_positive_year_share", 1.0)
        <= float(spec["evaluation"]["maximum_single_year_share_of_positive_log_return"]),
        "confirmatory_data": bool(audit.get("confirmatory_eligible")),
    }
    return {
        "checks": checks,
        "statistical_strategy_pass": bool(all(value for key, value in checks.items() if key != "confirmatory_data")),
        "production_evidence_pass": bool(all(checks.values())),
    }


def run(args: argparse.Namespace) -> dict:
    spec = load_spec(Path(args.spec))
    data = build_research_data(spec, args.exploratory)
    predictions, retrain_audit = walk_forward(data, spec)
    strategies = [
        "hgb_directional",
        "ridge_directional",
        "ma200_long_cash",
        "ma200_directional",
        "selector_always_long",
        "soxx_buy_hold",
    ]
    period_frames = []
    summaries = {}
    for strategy in strategies:
        periods = backtest(predictions, strategy, spec)
        period_frames.append(periods)
        summaries[strategy] = strategy_summary(periods, spec)
    stress_multiplier = float(spec["costs"]["stress_multiplier"])
    stress = backtest(predictions, "hgb_directional", spec, stress_multiplier)
    stress["strategy"] = "hgb_directional_cost_stress"
    period_frames.append(stress)
    stress_summary = strategy_summary(stress, spec)

    realized_predictions = predictions.dropna(subset=["direction_target", "selection_target"]).copy()
    direction_hgb = direction_metrics(realized_predictions, "direction_hgb_prediction")
    direction_ridge = direction_metrics(realized_predictions, "direction_ridge_prediction")
    selector = selection_metrics(realized_predictions, "direction_hgb_prediction")
    decision = pass_fail(summaries["hgb_directional"], stress_summary, selector, data.audit, spec)
    result = {
        "project": spec["project"],
        "research_only": True,
        "trading_integration": False,
        "specification": spec,
        "data_audit": data.audit,
        "walk_forward": {
            "prediction_rows": int(len(predictions)),
            "prediction_dates": int(predictions["signal_date"].nunique()),
            "realized_prediction_dates": int(realized_predictions["signal_date"].nunique()),
            "retrain_events": int(len(retrain_audit)),
            "first_prediction_date": str(predictions["signal_date"].min()),
            "last_prediction_date": str(predictions["signal_date"].max()),
            "last_realized_signal_date": str(realized_predictions["signal_date"].max()),
        },
        "components": {
            "direction_hgb": direction_hgb,
            "direction_ridge": direction_ridge,
            "selection_hgb_relative_return": selector,
        },
        "strategies": summaries,
        "cost_stress": stress_summary,
        "decision": decision,
        "interpretation_guardrails": [
            "Exploratory current-universe results are survivorship-biased and not confirmatory.",
            "Short-side results using fallback borrow assumptions are provisional.",
            "The architecture is not connected to paper or live trading.",
            "Do not tune the sign threshold, horizon, feature sets, or portfolio rules from this first result.",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.concat(period_frames, ignore_index=True).to_csv(out / "period_returns.csv", index=False)
    retrain_audit.to_csv(out / "retrain_audit.csv", index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "audit"])
    parser.add_argument("--spec", default=str(ROOT / "spec.json"))
    parser.add_argument("--out", default=str(ROOT / "artifacts"))
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    spec = load_spec(Path(args.spec))
    if args.command == "audit":
        data = build_research_data(spec, args.exploratory)
        print(json.dumps(data.audit, indent=2, default=str))
        return 0
    result = run(args)
    print(json.dumps({"data_audit": result["data_audit"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
