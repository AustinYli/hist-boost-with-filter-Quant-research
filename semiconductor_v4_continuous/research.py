#!/usr/bin/env python3
"""V4 continuous semiconductor alpha-core and direction-overlay research."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
EPSILON = 1e-12


def load_v3_module():
    path = PROJECT_ROOT / "semiconductor_v3_directional" / "research.py"
    module_spec = importlib.util.spec_from_file_location("v3_frozen_data_pipeline", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load frozen V3 data pipeline from {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


V3 = load_v3_module()


@dataclass
class V4Data:
    base: object
    audit: dict

    @property
    def panel(self) -> pd.DataFrame:
        return self.base.panel

    @property
    def direction(self) -> pd.DataFrame:
        return self.base.direction

    @property
    def decision_dates(self) -> list:
        return self.base.decision_dates

    @property
    def market_session_index(self) -> dict:
        return self.base.market_session_index


def load_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_optional_table(path: Path, required: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def interval_dividend_return(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    symbol_column: str,
    entry_date_column: str,
    exit_date_column: str,
    entry_price_column: str,
) -> pd.Series:
    result = pd.Series(0.0, index=frame.index)
    if dividends.empty:
        return result
    dividends = dividends.copy()
    dividends["date"] = pd.to_datetime(dividends["date"], errors="raise").dt.date
    dividends["symbol"] = dividends["symbol"].astype(str).str.upper()
    dividends["cash_amount"] = pd.to_numeric(dividends["cash_amount"], errors="coerce").fillna(0.0)
    for symbol, indexes in frame.groupby(symbol_column).groups.items():
        payouts = dividends[dividends["symbol"].eq(symbol)].sort_values("date")
        if payouts.empty:
            continue
        payout_dates = np.array(payouts["date"], dtype="datetime64[D]")
        cumulative = np.concatenate([[0.0], payouts["cash_amount"].cumsum().to_numpy(dtype=float)])
        one = frame.loc[indexes]
        entries = pd.to_datetime(one[entry_date_column], errors="coerce").to_numpy(dtype="datetime64[D]")
        exits = pd.to_datetime(one[exit_date_column], errors="coerce").to_numpy(dtype="datetime64[D]")
        left = np.searchsorted(payout_dates, entries, side="right")
        right = np.searchsorted(payout_dates, exits, side="right")
        cash = cumulative[right] - cumulative[left]
        entry_prices = pd.to_numeric(one[entry_price_column], errors="coerce").to_numpy(dtype=float)
        result.loc[indexes] = np.divide(
            cash,
            entry_prices,
            out=np.zeros_like(cash, dtype=float),
            where=np.isfinite(entry_prices) & (entry_prices > 0),
        )
    return result


def build_data(spec: dict, exploratory: bool) -> V4Data:
    base = V3.build_research_data(spec, exploratory=True)
    dividends = load_optional_table(
        project_path(spec["data"]["dividends"]), ["date", "symbol", "cash_amount", "source"]
    )
    actions = load_optional_table(
        project_path(spec["data"]["corporate_actions"]),
        ["date", "symbol", "action_type", "value", "handled", "source"],
    )
    blockers = list(base.audit["strict_blockers"])
    if dividends.empty:
        blockers.append("explicit dividend cash flows are absent")
    if actions.empty:
        blockers.append("corporate-action ledger is absent")
    elif not actions["handled"].astype(str).str.lower().eq("true").all():
        blockers.append("at least one corporate action is not handled")
    if blockers and not exploratory:
        raise ValueError("Strict V4 research blocked: " + "; ".join(blockers))

    panel = base.panel
    direction = base.direction
    panel["stock_dividend_return_5d"] = interval_dividend_return(
        panel,
        dividends,
        "symbol",
        "stock_label_entry_date",
        "stock_label_exit_date",
        "entry_open",
    )
    soxx_frame = direction.assign(symbol="SOXX")
    direction["soxx_dividend_return_5d"] = interval_dividend_return(
        soxx_frame,
        dividends,
        "symbol",
        "label_entry_date",
        "label_exit_date",
        "soxx_entry_open",
    )
    direction["direction_total_return_target"] = (
        direction["direction_target"] + direction["soxx_dividend_return_5d"]
    )
    soxx_total = direction.set_index("date")["direction_total_return_target"]
    panel["future_soxx_total_return_5d"] = panel["date"].map(soxx_total)
    panel["future_stock_total_return_5d"] = (
        panel["future_stock_return_5d"] + panel["stock_dividend_return_5d"]
    )
    panel["beta_neutral_target"] = (
        panel["future_stock_total_return_5d"]
        - panel["beta_60d_to_soxx"] * panel["future_soxx_total_return_5d"]
    )
    audit = {
        **base.audit,
        "exploratory": bool(exploratory),
        "strict_blockers": blockers,
        "dividend_rows": int(len(dividends)),
        "corporate_action_rows": int(len(actions)),
        "confirmatory_eligible": bool(not blockers and not exploratory),
        "selection_target": "stock total return - trailing beta_60d * SOXX total return",
    }
    return V4Data(base, audit)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def fit_direction_models(train: pd.DataFrame, spec: dict):
    features = spec["direction_features"]
    x = train[features]
    y = train["direction_total_return_target"]
    splits = min(int(spec["models"]["direction_cv_splits"]), max(2, len(train) // 12))
    time_cv = TimeSeriesSplit(n_splits=splits)
    search = GridSearchCV(
        ridge_pipeline(1.0),
        {"ridge__alpha": [float(value) for value in spec["models"]["ridge_alphas"]]},
        cv=time_cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    search.fit(x, y)
    best_alpha = float(search.best_params_["ridge__alpha"])
    residuals = []
    for train_indexes, validation_indexes in time_cv.split(x):
        model = ridge_pipeline(best_alpha)
        model.fit(x.iloc[train_indexes], y.iloc[train_indexes])
        residuals.extend(
            (y.iloc[validation_indexes].to_numpy() - model.predict(x.iloc[validation_indexes])).tolist()
        )
    minimum_sigma = float(spec["models"]["uncertainty_minimum_sigma"])
    sigma = max(float(np.std(residuals, ddof=1)), minimum_sigma) if len(residuals) > 1 else minimum_sigma
    challenger = HistGradientBoostingRegressor(**spec["models"]["direction_hgb_challenger"])
    challenger.fit(x, y)
    return search.best_estimator_, challenger, sigma, best_alpha


def fit_selector(train: pd.DataFrame, spec: dict):
    model = HistGradientBoostingRegressor(**spec["models"]["selection_hgb"])
    weights = V3.cross_section_sample_weights(train)
    model.fit(train[spec["selection_features"]], train["beta_neutral_target"], sample_weight=weights)
    return model, weights


def walk_forward(data: V4Data, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction_grid = data.direction[data.direction["on_decision_grid"]].copy()
    panel_grid = data.panel[data.panel["on_decision_grid"] & data.panel["eligible"]].copy()
    cutoff = pd.Timestamp(spec["data"]["research_label_cutoff"]).date()
    oos_start = pd.Timestamp(spec["data"]["historical_oos_start"]).date()
    minimum_dates = int(spec["models"]["minimum_training_decision_dates"])
    retrain_every = int(spec["models"]["retrain_every_sessions"])
    minimum_names = int(spec["universe"]["minimum_names_to_trade"])
    models = None
    last_retrain_session = None
    rows = []
    audits = []

    for signal_date in data.decision_dates:
        if signal_date < oos_start:
            continue
        direction_train = direction_grid[
            direction_grid["label_exit_date"].notna()
            & direction_grid["label_exit_date"].le(signal_date)
            & direction_grid["date"].lt(signal_date)
            & direction_grid["date"].le(cutoff)
        ].dropna(subset=spec["direction_features"] + ["direction_total_return_target"])
        selector_train = panel_grid[
            panel_grid["stock_label_exit_date"].notna()
            & panel_grid["stock_label_exit_date"].le(signal_date)
            & panel_grid["date"].lt(signal_date)
            & panel_grid["date"].le(cutoff)
        ].dropna(subset=spec["selection_features"] + ["beta_neutral_target"])
        if direction_train["date"].nunique() < minimum_dates or selector_train["date"].nunique() < minimum_dates:
            continue
        session_index = data.market_session_index[signal_date]
        if models is None or session_index - int(last_retrain_session) >= retrain_every:
            ridge, hgb, sigma, ridge_alpha = fit_direction_models(direction_train, spec)
            selector, selector_weights = fit_selector(selector_train, spec)
            models = ridge, hgb, selector, sigma, ridge_alpha
            last_retrain_session = session_index
            audits.append(
                {
                    "retrain_signal_date": signal_date,
                    "direction_train_start": direction_train["date"].min(),
                    "direction_train_end": direction_train["date"].max(),
                    "direction_last_label_exit": direction_train["label_exit_date"].max(),
                    "direction_train_dates": int(direction_train["date"].nunique()),
                    "selected_ridge_alpha": ridge_alpha,
                    "forecast_error_sigma": sigma,
                    "selector_train_start": selector_train["date"].min(),
                    "selector_train_end": selector_train["date"].max(),
                    "selector_last_label_exit": selector_train["stock_label_exit_date"].max(),
                    "selector_train_dates": int(selector_train["date"].nunique()),
                    "selector_train_rows": int(len(selector_train)),
                    "selector_total_sample_weight": float(selector_weights.sum()),
                }
            )
        direction_day = direction_grid[direction_grid["date"].eq(signal_date)].dropna(
            subset=spec["direction_features"]
        )
        stock_day = panel_grid[panel_grid["date"].eq(signal_date)].dropna(subset=spec["selection_features"])
        if direction_day.empty or len(stock_day) < minimum_names:
            continue
        ridge, hgb, selector, sigma, ridge_alpha = models
        ridge_prediction = float(ridge.predict(direction_day[spec["direction_features"]])[0])
        hgb_prediction = float(hgb.predict(direction_day[spec["direction_features"]])[0])
        conviction = float(np.tanh(ridge_prediction / sigma))
        stock_day = stock_day.copy()
        stock_day["alpha_prediction"] = selector.predict(stock_day[spec["selection_features"]])
        market_row = direction_day.iloc[0]
        for row in stock_day.itertuples(index=False):
            rows.append(
                {
                    "signal_date": signal_date,
                    "symbol": row.symbol,
                    "direction_prediction": ridge_prediction,
                    "direction_hgb_challenger": hgb_prediction,
                    "forecast_error_sigma": sigma,
                    "direction_conviction": conviction,
                    "ridge_alpha": ridge_alpha,
                    "direction_target": market_row.direction_total_return_target,
                    "alpha_prediction": row.alpha_prediction,
                    "alpha_target": row.beta_neutral_target,
                    "future_stock_return_5d": row.future_stock_total_return_5d,
                    "future_soxx_return_5d": market_row.direction_total_return_target,
                    "future_spy_return_5d": market_row.future_spy_return_5d,
                    "future_qqq_return_5d": market_row.future_qqq_return_5d,
                    "entry_date": row.stock_label_entry_date,
                    "exit_date": row.stock_label_exit_date,
                    "beta_60d_to_soxx": row.beta_60d_to_soxx,
                    "volatility_20d": row.volatility_20d,
                    "shortable": row.shortable,
                    "annual_borrow_rate": row.annual_borrow_rate,
                    "eligible_names": int(len(stock_day)),
                }
            )
    if not rows:
        raise RuntimeError("No V4 walk-forward predictions were produced.")
    return pd.DataFrame(rows), pd.DataFrame(audits)


def overlay_exposures(conviction: float, spec: dict) -> tuple[float, float]:
    conviction = float(np.clip(conviction, -1.0, 1.0))
    neutral = float(spec["overlay"]["neutral_long_gross"])
    strong = float(spec["overlay"]["strong_side_gross"])
    weak = float(spec["overlay"]["weak_side_gross"])
    if conviction >= 0:
        long_gross = neutral + (strong - neutral) * conviction
        short_gross = neutral - (neutral - weak) * conviction
    else:
        magnitude = abs(conviction)
        long_gross = neutral - (neutral - weak) * magnitude
        short_gross = neutral + (strong - neutral) * magnitude
    return long_gross, short_gross


def risk_covariance(data: V4Data, signal_date, symbols: list[str], spec: dict) -> np.ndarray:
    lookback = int(spec["risk"]["lookback_sessions"])
    minimum = int(spec["risk"]["minimum_return_observations"])
    history = data.panel[
        data.panel["symbol"].isin(symbols) & data.panel["date"].le(signal_date)
    ].pivot(index="date", columns="symbol", values="return_1d").sort_index().tail(lookback)
    history = history.reindex(columns=symbols)
    if len(history) < minimum:
        raise ValueError(f"Only {len(history)} risk observations on {signal_date}; need {minimum}.")
    history = history.apply(lambda column: column.fillna(column.mean())).fillna(0.0)
    return LedoitWolf().fit(history.to_numpy(dtype=float)).covariance_


def optimize_portfolio(
    day: pd.DataFrame,
    data: V4Data,
    conviction: float,
    old_weights: dict[str, float],
    spec: dict,
) -> tuple[dict[str, float], dict]:
    top_n = int(spec["universe"]["top_n"])
    bottom_n = int(spec["universe"]["bottom_n"])
    ordered = day.sort_values("alpha_prediction", ascending=False)
    top = ordered.head(top_n)
    shortable = ordered[ordered["shortable"].astype(bool)]
    bottom = shortable.tail(bottom_n).sort_values("alpha_prediction")
    if len(top) < top_n or len(bottom) < bottom_n:
        raise ValueError("Insufficient long or short candidates for V4.")
    candidates = pd.concat([top, bottom], ignore_index=True)
    symbols = candidates["symbol"].tolist()
    alpha = candidates["alpha_prediction"].to_numpy(dtype=float)
    beta = candidates["beta_60d_to_soxx"].to_numpy(dtype=float)
    borrow = candidates["annual_borrow_rate"].to_numpy(dtype=float)
    covariance = risk_covariance(data, day["signal_date"].iloc[0], symbols, spec)
    long_gross, short_gross = overlay_exposures(conviction, spec)
    cap = float(spec["risk"]["maximum_single_stock_abs_weight"])
    initial = np.concatenate(
        [np.full(top_n, long_gross / top_n), np.full(bottom_n, -short_gross / bottom_n)]
    )
    old = np.array([old_weights.get(symbol, 0.0) for symbol in symbols], dtype=float)
    desired_beta = long_gross - short_gross
    horizon = int(spec["execution"]["horizon_sessions"])
    cost_rate = float(spec["costs"]["primary_one_way_bps"]) / 10000
    risk_aversion = float(spec["risk"]["risk_aversion"])
    beta_penalty = float(spec["risk"]["beta_target_penalty"])
    turnover_penalty = float(spec["risk"]["turnover_penalty_multiplier"])

    def objective(weights: np.ndarray) -> float:
        alpha_term = -float(weights @ alpha)
        variance_term = risk_aversion * float(weights @ covariance @ weights) * horizon
        beta_term = beta_penalty * float(weights @ beta - desired_beta) ** 2
        trading_term = turnover_penalty * cost_rate * float(np.sqrt((weights - old) ** 2 + 1e-10).sum())
        borrow_term = float(np.maximum(-weights, 0.0) @ np.nan_to_num(borrow, nan=0.0)) * horizon / 252
        return alpha_term + variance_term + beta_term + trading_term + borrow_term

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights[:top_n].sum() - long_gross)},
        {"type": "eq", "fun": lambda weights: float(weights[top_n:].sum() + short_gross)},
    ]
    bounds = [(0.0, cap)] * top_n + [(-cap, 0.0)] * bottom_n
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(spec["risk"]["optimizer_max_iterations"]), "ftol": 1e-10},
    )
    raw = result.x if result.success else initial
    predicted_volatility = float(np.sqrt(max(raw @ covariance @ raw, 0.0) * 252))
    target_volatility = float(spec["risk"]["annualized_volatility_target"])
    scale = min(1.0, target_volatility / predicted_volatility) if predicted_volatility > EPSILON else 1.0
    weights = raw * scale
    final = {symbol: float(weight) for symbol, weight in zip(symbols, weights) if abs(weight) > EPSILON}
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "conviction": conviction,
        "requested_long_gross": long_gross,
        "requested_short_gross": short_gross,
        "risk_scale": scale,
        "predicted_volatility_before_scale": predicted_volatility,
        "predicted_volatility_after_scale": predicted_volatility * scale,
        "gross_exposure": float(np.abs(weights).sum()),
        "net_exposure": float(weights.sum()),
        "estimated_soxx_beta": float(weights @ beta),
        "maximum_abs_weight": float(np.abs(weights).max()),
    }
    return final, diagnostics


def equal_side_weights(day: pd.DataFrame, conviction: float, spec: dict) -> tuple[dict[str, float], dict]:
    top_n = int(spec["universe"]["top_n"])
    bottom_n = int(spec["universe"]["bottom_n"])
    ordered = day.sort_values("alpha_prediction", ascending=False)
    top = ordered.head(top_n)["symbol"].tolist()
    bottom = ordered[ordered["shortable"].astype(bool)].tail(bottom_n)["symbol"].tolist()
    long_gross, short_gross = overlay_exposures(conviction, spec)
    weights = {symbol: long_gross / top_n for symbol in top}
    weights.update({symbol: -short_gross / bottom_n for symbol in bottom})
    beta = day.set_index("symbol")["beta_60d_to_soxx"]
    return weights, {
        "optimizer_success": True,
        "conviction": conviction,
        "risk_scale": 1.0,
        "predicted_volatility_after_scale": None,
        "gross_exposure": sum(abs(value) for value in weights.values()),
        "net_exposure": sum(weights.values()),
        "estimated_soxx_beta": sum(value * float(beta.get(symbol, 0.0)) for symbol, value in weights.items()),
        "maximum_abs_weight": max(abs(value) for value in weights.values()),
    }


def strategy_weights(
    strategy: str,
    day: pd.DataFrame,
    data: V4Data,
    old_weights: dict[str, float],
    spec: dict,
) -> tuple[dict[str, float], dict]:
    conviction = float(day["direction_conviction"].iloc[0])
    if strategy == "v4_continuous":
        return optimize_portfolio(day, data, conviction, old_weights, spec)
    if strategy == "alpha_core":
        return optimize_portfolio(day, data, 0.0, old_weights, spec)
    if strategy == "continuous_no_risk":
        return equal_side_weights(day, conviction, spec)
    if strategy == "v3_binary":
        binary = 1.0 if float(day["direction_prediction"].iloc[0]) > 0 else -1.0
        ordered = day.sort_values("alpha_prediction", ascending=False)
        count = int(spec["universe"]["top_n"])
        selected = ordered.head(count) if binary > 0 else ordered[ordered["shortable"].astype(bool)].tail(count)
        return (
            {symbol: binary / count for symbol in selected["symbol"]},
            {
                "optimizer_success": True,
                "conviction": binary,
                "risk_scale": 1.0,
                "predicted_volatility_after_scale": None,
                "gross_exposure": 1.0,
                "net_exposure": binary,
                "estimated_soxx_beta": float(
                    sum(
                        binary / count * value
                        for value in selected["beta_60d_to_soxx"].to_numpy(dtype=float)
                    )
                ),
                "maximum_abs_weight": 0.1,
            },
        )
    if strategy == "direction_only":
        weight = float(spec["overlay"]["maximum_abs_net_exposure"]) * conviction
        return {"SOXX": weight}, {
            "optimizer_success": True,
            "conviction": conviction,
            "risk_scale": 1.0,
            "predicted_volatility_after_scale": None,
            "gross_exposure": abs(weight),
            "net_exposure": weight,
            "estimated_soxx_beta": weight,
            "maximum_abs_weight": abs(weight),
        }
    if strategy == "soxx_buy_hold":
        return {"SOXX": 1.0}, {
            "optimizer_success": True,
            "conviction": 1.0,
            "risk_scale": 1.0,
            "predicted_volatility_after_scale": None,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
            "estimated_soxx_beta": 1.0,
            "maximum_abs_weight": 1.0,
        }
    raise ValueError(strategy)


def backtest(
    predictions: pd.DataFrame,
    data: V4Data,
    strategy: str,
    spec: dict,
    stress: bool = False,
) -> pd.DataFrame:
    cost_rate = float(
        spec["costs"]["stress_one_way_bps"] if stress else spec["costs"]["primary_one_way_bps"]
    ) / 10000
    fallback_borrow = float(
        spec["costs"]["stress_annual_borrow_rate_fallback"]
        if stress
        else spec["costs"]["primary_annual_borrow_rate_fallback"]
    )
    actual_borrow = bool(data.audit["historical_borrow_available"])
    horizon = int(spec["execution"]["horizon_sessions"])
    current: dict[str, float] = {}
    records = []
    realized_dates = predictions.dropna(subset=["future_stock_return_5d", "exit_date"])["signal_date"].unique()
    for signal_date in sorted(realized_dates):
        day = predictions[predictions["signal_date"].eq(signal_date)].copy()
        target, diagnostics = strategy_weights(strategy, day, data, current, spec)
        turnover = sum(
            abs(target.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in set(target) | set(current)
        )
        transaction_cost = turnover * cost_rate
        stock_returns = day.set_index("symbol")["future_stock_return_5d"]
        borrow_lookup = day.set_index("symbol")["annual_borrow_rate"]
        gross_return = 0.0
        borrow_cost = 0.0
        for symbol, weight in target.items():
            realized = (
                float(day["future_soxx_return_5d"].iloc[0])
                if symbol == "SOXX"
                else float(stock_returns.get(symbol, np.nan))
            )
            if not np.isfinite(realized):
                raise ValueError(f"Missing realized return for selected {symbol} on {signal_date}.")
            gross_return += weight * realized
            if weight < 0:
                rate = float(borrow_lookup.get(symbol, fallback_borrow)) if actual_borrow else fallback_borrow
                borrow_cost += abs(weight) * rate * horizon / 252
        records.append(
            {
                "signal_date": signal_date,
                "entry_date": day["entry_date"].iloc[0],
                "exit_date": day["exit_date"].iloc[0],
                "strategy": strategy + ("_stress" if stress else ""),
                "gross_return": gross_return,
                "net_return": gross_return - transaction_cost - borrow_cost,
                "transaction_cost": transaction_cost,
                "borrow_cost": borrow_cost,
                "turnover": turnover,
                "direction_prediction": float(day["direction_prediction"].iloc[0]),
                "direction_target": float(day["direction_target"].iloc[0]),
                "direction_conviction": float(day["direction_conviction"].iloc[0]),
                "future_soxx_return_5d": float(day["future_soxx_return_5d"].iloc[0]),
                "future_spy_return_5d": float(day["future_spy_return_5d"].iloc[0]),
                "future_qqq_return_5d": float(day["future_qqq_return_5d"].iloc[0]),
                "weights": json.dumps(target, sort_keys=True),
                **diagnostics,
            }
        )
        current = target
    result = pd.DataFrame(records)
    if not result.empty and current:
        exit_cost = sum(abs(value) for value in current.values()) * cost_rate
        result.loc[result.index[-1], "transaction_cost"] += exit_cost
        result.loc[result.index[-1], "net_return"] -= exit_cost
        result.loc[result.index[-1], "turnover"] += sum(abs(value) for value in current.values())
    return result


def factor_alpha(periods: pd.DataFrame, spec: dict) -> dict:
    clean = periods.dropna(subset=["net_return", "future_soxx_return_5d", "future_qqq_return_5d"])
    if len(clean) < 10:
        return {}
    y = clean["net_return"].to_numpy(dtype=float)
    x = np.column_stack(
        [np.ones(len(clean)), clean["future_soxx_return_5d"], clean["future_qqq_return_5d"]]
    )
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    covariance = V3.newey_west_covariance(x, residuals, int(spec["evaluation"]["hac_lag_periods"]))
    standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    annualization = float(spec["evaluation"]["periods_per_year"])
    return {
        "annualized_alpha": float(coefficients[0] * annualization),
        "alpha_hac_t": float(coefficients[0] / standard_error) if standard_error > EPSILON else None,
        "beta_soxx": float(coefficients[1]),
        "beta_qqq": float(coefficients[2]),
        "alpha_observations": int(len(clean)),
    }


def strategy_summary(periods: pd.DataFrame, spec: dict) -> dict:
    summary = V3.performance(periods["net_return"], float(spec["evaluation"]["periods_per_year"]))
    summary.update(factor_alpha(periods, spec))
    summary.update(
        {
            "total_turnover": float(periods["turnover"].sum()),
            "average_turnover": float(periods["turnover"].mean()),
            "total_transaction_cost": float(periods["transaction_cost"].sum()),
            "total_borrow_cost": float(periods["borrow_cost"].sum()),
            "average_gross_exposure": float(periods["gross_exposure"].mean()),
            "maximum_gross_exposure": float(periods["gross_exposure"].max()),
            "average_net_exposure": float(periods["net_exposure"].mean()),
            "average_abs_net_exposure": float(periods["net_exposure"].abs().mean()),
            "maximum_abs_net_exposure": float(periods["net_exposure"].abs().max()),
            "average_estimated_soxx_beta": float(periods["estimated_soxx_beta"].mean()),
            "maximum_abs_weight": float(periods["maximum_abs_weight"].max()),
            "optimizer_failure_count": int((~periods["optimizer_success"].astype(bool)).sum()),
            "positive_net_periods": V3.performance(
                periods.loc[periods["direction_conviction"].ge(0), "net_return"],
                float(spec["evaluation"]["periods_per_year"]),
            ),
            "negative_net_periods": V3.performance(
                periods.loc[periods["direction_conviction"].lt(0), "net_return"],
                float(spec["evaluation"]["periods_per_year"]),
            ),
        }
    )
    vol = pd.to_numeric(periods["predicted_volatility_after_scale"], errors="coerce")
    summary["average_predicted_volatility"] = float(vol.mean()) if vol.notna().any() else None
    yearly = periods.assign(year=pd.to_datetime(periods["signal_date"]).dt.year).groupby("year")["net_return"].apply(
        lambda values: float(np.log1p(values).sum())
    )
    positive = yearly.clip(lower=0)
    summary["yearly_log_returns"] = {str(year): value for year, value in yearly.items()}
    summary["largest_positive_year_share"] = (
        float(positive.max() / positive.sum()) if positive.sum() > EPSILON else 1.0
    )
    return summary


def direction_metrics(predictions: pd.DataFrame) -> dict:
    dates = predictions.drop_duplicates("signal_date").dropna(subset=["direction_prediction", "direction_target"])
    positive = dates["direction_prediction"].gt(0)
    error = dates["direction_target"] - dates["direction_prediction"]
    return {
        "periods": int(len(dates)),
        "sign_accuracy": float(positive.eq(dates["direction_target"].gt(0)).mean()),
        "prediction_target_correlation": float(dates["direction_prediction"].corr(dates["direction_target"])),
        "mean_actual_when_predicted_positive": float(dates.loc[positive, "direction_target"].mean()),
        "mean_actual_when_predicted_negative": float(dates.loc[~positive, "direction_target"].mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_forecast_sigma": float(dates["forecast_error_sigma"].mean()),
        "within_one_sigma_fraction": float((error.abs() <= dates["forecast_error_sigma"]).mean()),
        "mean_abs_conviction": float(dates["direction_conviction"].abs().mean()),
    }


def alpha_metrics(predictions: pd.DataFrame) -> dict:
    rows = []
    for signal_date, day in predictions.dropna(subset=["alpha_prediction", "alpha_target"]).groupby("signal_date"):
        if len(day) < 20:
            continue
        ordered = day.sort_values("alpha_prediction", ascending=False)
        rows.append(
            {
                "signal_date": signal_date,
                "rank_ic": float(day["alpha_prediction"].rank().corr(day["alpha_target"].rank())),
                "top_minus_bottom": float(
                    ordered.head(10)["alpha_target"].mean() - ordered.tail(10)["alpha_target"].mean()
                ),
                "conviction": float(day["direction_conviction"].iloc[0]),
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
    for regime, mask in [("positive_direction", frame["conviction"].ge(0)), ("negative_direction", frame["conviction"].lt(0))]:
        subset = frame[mask]
        result[f"{regime}_periods"] = int(len(subset))
        result[f"{regime}_rank_ic"] = float(subset["rank_ic"].mean()) if not subset.empty else None
        result[f"{regime}_top_minus_bottom"] = (
            float(subset["top_minus_bottom"].mean()) if not subset.empty else None
        )
    return result


def pass_fail(
    summaries: dict,
    stress: dict,
    alpha: dict,
    direction: dict,
    audit: dict,
    spec: dict,
) -> dict:
    v4 = summaries["v4_continuous"]
    core = summaries["alpha_core"]
    overlay_increment = v4.get("total_return", -1) - core.get("total_return", -1)
    checks = {
        "alpha_core_profitable": core.get("total_return", -1) > 0,
        "direction_overlay_adds_return": overlay_increment > 0,
        "direction_conditional_sign_positive": direction.get("mean_actual_when_predicted_positive", -1) > 0,
        "direction_conditional_sign_negative": direction.get("mean_actual_when_predicted_negative", 1) < 0,
        "overall_rank_ic_positive": alpha.get("mean_rank_ic", -1) > 0,
        "overall_top_minus_bottom_positive": alpha.get("mean_top_minus_bottom", -1) > 0,
        "positive_direction_rank_ic_positive": (alpha.get("positive_direction_rank_ic") or -1) > 0,
        "negative_direction_rank_ic_positive": (alpha.get("negative_direction_rank_ic") or -1) > 0,
        "positive_direction_spread_positive": (alpha.get("positive_direction_top_minus_bottom") or -1) > 0,
        "negative_direction_spread_positive": (alpha.get("negative_direction_top_minus_bottom") or -1) > 0,
        "factor_alpha_positive": v4.get("annualized_alpha", -1) > 0,
        "hac_alpha_t_at_least_2": (v4.get("alpha_hac_t") or -999)
        >= float(spec["evaluation"]["minimum_hac_alpha_t_for_strong_claim"]),
        "cost_stress_profitable": stress.get("total_return", -1) > 0,
        "single_year_not_dominant": v4.get("largest_positive_year_share", 1.0)
        <= float(spec["evaluation"]["maximum_single_year_share_of_positive_log_return"]),
        "risk_constraints_respected": v4.get("maximum_abs_weight", 999)
        <= float(spec["risk"]["maximum_single_stock_abs_weight"]) + 1e-8
        and v4.get("maximum_gross_exposure", 999) <= float(spec["overlay"]["maximum_gross_exposure"]) + 1e-8
        and v4.get("maximum_abs_net_exposure", 999)
        <= float(spec["overlay"]["maximum_abs_net_exposure"]) + 1e-8,
        "confirmatory_data": bool(audit.get("confirmatory_eligible")),
    }
    return {
        "overlay_total_return_increment": overlay_increment,
        "checks": checks,
        "statistical_strategy_pass": bool(all(value for key, value in checks.items() if key != "confirmatory_data")),
        "production_evidence_pass": bool(all(checks.values())),
    }


def run(args: argparse.Namespace) -> dict:
    spec = load_spec(Path(args.spec))
    data = build_data(spec, args.exploratory)
    predictions, retrain_audit = walk_forward(data, spec)
    strategies = [
        "v4_continuous",
        "alpha_core",
        "continuous_no_risk",
        "v3_binary",
        "direction_only",
        "soxx_buy_hold",
    ]
    period_frames = []
    summaries = {}
    for strategy in strategies:
        periods = backtest(predictions, data, strategy, spec)
        period_frames.append(periods)
        summaries[strategy] = strategy_summary(periods, spec)
    stress_periods = backtest(predictions, data, "v4_continuous", spec, stress=True)
    period_frames.append(stress_periods)
    stress_summary = strategy_summary(stress_periods, spec)
    realized = predictions.dropna(subset=["direction_target", "alpha_target"]).copy()
    direction = direction_metrics(realized)
    alpha = alpha_metrics(realized)
    decision = pass_fail(summaries, stress_summary, alpha, direction, data.audit, spec)
    result = {
        "project": spec["project"],
        "research_only": True,
        "trading_integration": False,
        "specification": spec,
        "data_audit": data.audit,
        "walk_forward": {
            "prediction_rows": int(len(predictions)),
            "prediction_dates": int(predictions["signal_date"].nunique()),
            "realized_prediction_dates": int(realized["signal_date"].nunique()),
            "retrain_events": int(len(retrain_audit)),
            "first_prediction_date": str(predictions["signal_date"].min()),
            "last_prediction_date": str(predictions["signal_date"].max()),
            "last_realized_signal_date": str(realized["signal_date"].max()),
        },
        "components": {
            "direction_ridgecv": direction,
            "residual_alpha_hgb": alpha,
        },
        "strategies": summaries,
        "cost_stress": stress_summary,
        "decision": decision,
        "guardrails": [
            "V4 is not connected to paper or live trading.",
            "Current-universe IEX results are exploratory and survivorship-biased.",
            "Fallback borrow rates and absent dividend history make short results provisional.",
            "Do not tune the continuous exposure map, volatility target, or optimizer from this first result.",
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
        data = build_data(spec, args.exploratory)
        print(json.dumps(data.audit, indent=2, default=str))
        return 0
    result = run(args)
    print(json.dumps({"data_audit": result["data_audit"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
