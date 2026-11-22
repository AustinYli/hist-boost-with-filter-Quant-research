#!/usr/bin/env python3
"""Freeze and journal V3 signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from research import (
    ROOT,
    build_research_data,
    cross_section_sample_weights,
    load_spec,
    make_direction_hgb,
    make_direction_ridge,
    make_selector,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_frames(data, spec: dict, as_of_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction_features = spec["direction_features"]
    selection_features = spec["selection_features"]
    cutoff = pd.Timestamp(spec["data"]["research_label_cutoff"]).date()
    direction = data.direction[
        data.direction["on_decision_grid"]
        & data.direction["date"].le(cutoff)
        & data.direction["label_exit_date"].notna()
        & data.direction["label_exit_date"].le(as_of_date)
    ].dropna(subset=direction_features + ["direction_target"])
    selector = data.panel[
        data.panel["on_decision_grid"]
        & data.panel["eligible"]
        & data.panel["date"].le(cutoff)
        & data.panel["stock_label_exit_date"].notna()
        & data.panel["stock_label_exit_date"].le(as_of_date)
    ].dropna(subset=selection_features + ["selection_target"])
    return direction, selector


def freeze(args: argparse.Namespace) -> dict:
    spec_path = Path(args.spec)
    spec = load_spec(spec_path)
    data = build_research_data(spec, args.exploratory)
    as_of_date = pd.Timestamp(args.as_of_date).date()
    direction_train, selector_train = training_frames(data, spec, as_of_date)
    minimum = int(spec["models"]["minimum_training_decision_dates"])
    if direction_train["date"].nunique() < minimum or selector_train["date"].nunique() < minimum:
        raise ValueError("Insufficient safe decision dates to freeze V3 models.")

    direction_hgb = make_direction_hgb(spec)
    direction_ridge = make_direction_ridge(spec)
    selector = make_selector(spec)
    direction_hgb.fit(direction_train[spec["direction_features"]], direction_train["direction_target"])
    direction_ridge.fit(direction_train[spec["direction_features"]], direction_train["direction_target"])
    selector_weights = cross_section_sample_weights(selector_train)
    selector.fit(
        selector_train[spec["selection_features"]],
        selector_train["selection_target"],
        sample_weight=selector_weights,
    )

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(direction_hgb, out / "direction_hgb.joblib")
    joblib.dump(direction_ridge, out / "direction_ridge.joblib")
    joblib.dump(selector, out / "selection_hgb.joblib")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(as_of_date),
        "research_only": True,
        "trading_integration": False,
        "exploratory": bool(args.exploratory),
        "confirmatory_eligible": bool(data.audit["confirmatory_eligible"]),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256(spec_path),
        "data_audit": data.audit,
        "direction_train_dates": int(direction_train["date"].nunique()),
        "direction_train_start": str(direction_train["date"].min()),
        "direction_train_end": str(direction_train["date"].max()),
        "direction_last_label_exit": str(direction_train["label_exit_date"].max()),
        "selector_train_dates": int(selector_train["date"].nunique()),
        "selector_train_rows": int(len(selector_train)),
        "selector_train_start": str(selector_train["date"].min()),
        "selector_train_end": str(selector_train["date"].max()),
        "selector_last_label_exit": str(selector_train["stock_label_exit_date"].max()),
        "selector_total_sample_weight": float(selector_weights.sum()),
        "model_files": ["direction_hgb.joblib", "direction_ridge.joblib", "selection_hgb.joblib"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def signal(args: argparse.Namespace) -> dict:
    spec_path = Path(args.spec)
    spec = load_spec(spec_path)
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(spec_path) != manifest["spec_sha256"]:
        raise ValueError("V3 specification changed after the model freeze.")
    if manifest["exploratory"] and not args.exploratory:
        raise ValueError("This model was frozen with exploratory data; pass --exploratory to journal a research signal.")
    data = build_research_data(spec, args.exploratory)
    signal_date = pd.Timestamp(args.signal_date).date()
    if signal_date not in data.decision_dates:
        raise ValueError(f"{signal_date} is not on the locked five-session decision grid.")
    direction_day = data.direction[data.direction["date"].eq(signal_date)].dropna(
        subset=spec["direction_features"]
    )
    stock_day = data.panel[
        data.panel["date"].eq(signal_date) & data.panel["eligible"]
    ].dropna(subset=spec["selection_features"])
    minimum = int(spec["universe"]["minimum_names_to_trade"])
    if direction_day.empty or len(stock_day) < minimum:
        raise ValueError(f"Signal date has only {len(stock_day)} eligible stocks; need {minimum}.")

    direction_hgb = joblib.load(model_dir / "direction_hgb.joblib")
    direction_ridge = joblib.load(model_dir / "direction_ridge.joblib")
    selector = joblib.load(model_dir / "selection_hgb.joblib")
    hgb_prediction = float(direction_hgb.predict(direction_day[spec["direction_features"]])[0])
    ridge_prediction = float(direction_ridge.predict(direction_day[spec["direction_features"]])[0])
    stock_day = stock_day.copy()
    stock_day["selection_prediction"] = selector.predict(stock_day[spec["selection_features"]])
    ranked = stock_day.sort_values("selection_prediction", ascending=False)
    top_n = int(round(1.0 / float(spec["hypothesis"]["single_name_weight"])))
    side = "long" if hgb_prediction > 0 else "short"
    selected = ranked.head(top_n) if side == "long" else ranked.tail(top_n).sort_values(
        "selection_prediction"
    )
    if side == "short" and not selected["shortable"].astype(bool).all():
        raise ValueError("At least one selected short is unavailable to borrow.")
    signed_weight = float(spec["hypothesis"]["single_name_weight"]) * (1 if side == "long" else -1)
    weights = {symbol: signed_weight for symbol in selected["symbol"]}
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(signal_date),
        "intended_execution": "next_session_open",
        "intended_exit": "open_t_plus_6",
        "model_manifest": str((model_dir / "manifest.json").resolve()),
        "direction_hgb_prediction": hgb_prediction,
        "direction_ridge_benchmark_prediction": ridge_prediction,
        "side": side,
        "selected": selected[["symbol", "selection_prediction", "shortable", "annual_borrow_rate"]].to_dict(
            orient="records"
        ),
        "intended_weights": weights,
        "research_only": True,
        "orders_submitted": False,
    }
    journal = Path(args.journal)
    existing = read_jsonl(journal)
    if any(item.get("signal_date") == record["signal_date"] for item in existing):
        raise ValueError(f"Signal {signal_date} is already journaled; no backfill or overwrite is allowed.")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "signal", "status"])
    parser.add_argument("--spec", default=str(ROOT / "spec.json"))
    parser.add_argument("--model-dir", default=str(ROOT / "frozen_model"))
    parser.add_argument("--journal", default=str(ROOT / "forward/signals.jsonl"))
    parser.add_argument("--as-of-date", default="2026-08-10")
    parser.add_argument("--signal-date", default="2026-08-10")
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args)
    elif args.command == "signal":
        result = signal(args)
    else:
        model_dir = Path(args.model_dir)
        result = {
            "manifest": json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
            if (model_dir / "manifest.json").exists()
            else None,
            "signals": read_jsonl(Path(args.journal)),
        }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
