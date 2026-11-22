#!/usr/bin/env python3
"""Freeze and journal V4 research signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

import research


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def training_frames(data: research.V4Data, configuration: dict, as_of_date):
    cutoff = pd.Timestamp(configuration["data"]["research_label_cutoff"]).date()
    direction = data.direction[
        data.direction["on_decision_grid"]
        & data.direction["date"].le(cutoff)
        & data.direction["label_exit_date"].notna()
        & data.direction["label_exit_date"].le(as_of_date)
    ].dropna(subset=configuration["direction_features"] + ["direction_total_return_target"])
    selector = data.panel[
        data.panel["on_decision_grid"]
        & data.panel["eligible"]
        & data.panel["date"].le(cutoff)
        & data.panel["stock_label_exit_date"].notna()
        & data.panel["stock_label_exit_date"].le(as_of_date)
    ].dropna(subset=configuration["selection_features"] + ["beta_neutral_target"])
    return direction, selector


def freeze(args: argparse.Namespace) -> dict:
    spec_path = Path(args.spec)
    configuration = research.load_spec(spec_path)
    data = research.build_data(configuration, args.exploratory)
    as_of_date = pd.Timestamp(args.as_of_date).date()
    direction_train, selector_train = training_frames(data, configuration, as_of_date)
    minimum = int(configuration["models"]["minimum_training_decision_dates"])
    if direction_train["date"].nunique() < minimum or selector_train["date"].nunique() < minimum:
        raise ValueError("Insufficient fully observable decision dates to freeze V4.")
    ridge, challenger, sigma, alpha = research.fit_direction_models(direction_train, configuration)
    selector, selector_weights = research.fit_selector(selector_train, configuration)
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge, out / "direction_ridgecv.joblib")
    joblib.dump(challenger, out / "direction_hgb_challenger.joblib")
    joblib.dump(selector, out / "residual_alpha_hgb.joblib")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(as_of_date),
        "research_only": True,
        "trading_integration": False,
        "exploratory": bool(args.exploratory),
        "confirmatory_eligible": bool(data.audit["confirmatory_eligible"]),
        "spec_sha256": sha256(spec_path),
        "selected_ridge_alpha": alpha,
        "forecast_error_sigma": sigma,
        "direction_train_dates": int(direction_train["date"].nunique()),
        "direction_last_label_exit": str(direction_train["label_exit_date"].max()),
        "selector_train_dates": int(selector_train["date"].nunique()),
        "selector_train_rows": int(len(selector_train)),
        "selector_last_label_exit": str(selector_train["stock_label_exit_date"].max()),
        "selector_total_sample_weight": float(selector_weights.sum()),
        "data_audit": data.audit,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def signal(args: argparse.Namespace) -> dict:
    spec_path = Path(args.spec)
    configuration = research.load_spec(spec_path)
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["spec_sha256"] != sha256(spec_path):
        raise ValueError("V4 specification changed after freeze.")
    if manifest["exploratory"] and not args.exploratory:
        raise ValueError("The frozen V4 model is exploratory; explicit --exploratory is required.")
    data = research.build_data(configuration, args.exploratory)
    signal_date = pd.Timestamp(args.signal_date).date()
    if signal_date not in data.decision_dates:
        raise ValueError(f"{signal_date} is not on the locked V4 decision grid.")
    direction_day = data.direction[data.direction["date"].eq(signal_date)].dropna(
        subset=configuration["direction_features"]
    )
    stock_day = data.panel[data.panel["date"].eq(signal_date) & data.panel["eligible"]].dropna(
        subset=configuration["selection_features"]
    )
    if direction_day.empty or len(stock_day) < int(configuration["universe"]["minimum_names_to_trade"]):
        raise ValueError("Signal date lacks complete direction features or enough eligible stocks.")
    ridge = joblib.load(model_dir / "direction_ridgecv.joblib")
    challenger = joblib.load(model_dir / "direction_hgb_challenger.joblib")
    selector = joblib.load(model_dir / "residual_alpha_hgb.joblib")
    prediction = float(ridge.predict(direction_day[configuration["direction_features"]])[0])
    challenger_prediction = float(challenger.predict(direction_day[configuration["direction_features"]])[0])
    conviction = float(__import__("numpy").tanh(prediction / float(manifest["forecast_error_sigma"])))
    stock_day = stock_day.copy()
    stock_day["alpha_prediction"] = selector.predict(stock_day[configuration["selection_features"]])
    stock_day["signal_date"] = signal_date
    journal = Path(args.journal)
    existing = read_jsonl(journal)
    if any(item.get("signal_date") == str(signal_date) for item in existing):
        raise ValueError(f"Signal {signal_date} is already journaled; overwrite/backfill is disabled.")
    old_weights = existing[-1].get("intended_weights", {}) if existing else {}
    weights, diagnostics = research.optimize_portfolio(
        stock_day,
        data,
        conviction,
        {symbol: float(weight) for symbol, weight in old_weights.items()},
        configuration,
    )
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(signal_date),
        "intended_execution": "next_session_open",
        "intended_exit": "open_t_plus_6",
        "direction_prediction": prediction,
        "direction_hgb_challenger": challenger_prediction,
        "forecast_error_sigma": manifest["forecast_error_sigma"],
        "direction_conviction": conviction,
        "intended_weights": weights,
        "risk_diagnostics": diagnostics,
        "research_only": True,
        "orders_submitted": False,
    }
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "signal", "status"])
    parser.add_argument("--spec", default=str(ROOT / "spec.json"))
    parser.add_argument("--model-dir", default=str(ROOT / "frozen_model"))
    parser.add_argument("--journal", default=str(ROOT / "forward" / "signals.jsonl"))
    parser.add_argument("--as-of-date", default="2026-08-10")
    parser.add_argument("--signal-date", default="2026-08-10")
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args)
    elif args.command == "signal":
        result = signal(args)
    else:
        manifest_path = Path(args.model_dir) / "manifest.json"
        result = {
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
            "signals": read_jsonl(Path(args.journal)),
        }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
