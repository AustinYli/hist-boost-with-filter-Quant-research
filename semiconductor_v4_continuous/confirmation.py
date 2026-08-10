#!/usr/bin/env python3
"""Freeze a challenge dataset and run the frozen V4 confirmation exactly once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import historical_data as HD
import research as V4


ROOT = Path(__file__).resolve().parent
DATASET_LOCK_PATH = ROOT / "frozen_challenge_dataset_manifest.json"
CONFIRMATION_DIR = ROOT / "confirmation"
RUN_DIR = CONFIRMATION_DIR / "one_shot"
RUN_RECEIPT_PATH = CONFIRMATION_DIR / "one_shot_receipt.json"
FREEZE_CONFIRMATION = "FREEZE_CHALLENGE_DATASET"
RUN_CONFIRMATION = "RUN_FROZEN_V4_ONCE"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def dataset_files(data_dir: Path) -> list[Path]:
    return [data_dir / name for name in HD.INSTALL_TABLES + ["source_manifest.json"]]


def freeze_dataset(
    data_dir: Path = HD.DATA_DIR,
    lock_path: Path = DATASET_LOCK_PATH,
    confirmation: str = "",
) -> dict:
    if confirmation != FREEZE_CONFIRMATION:
        raise ValueError(f"Dataset freeze requires --confirm {FREEZE_CONFIRMATION}")
    if lock_path.exists():
        raise FileExistsError(f"Frozen challenge dataset already exists at {lock_path}")
    hypothesis = HD.verify_lock()
    if not hypothesis["valid"]:
        raise ValueError("Frozen V4 hypothesis hash verification failed")
    readiness = HD.audit_bundle(data_dir)
    if not readiness["v4_unchanged_rerun_ready"]:
        raise ValueError("Challenge dataset does not pass the locked readiness contract")
    missing = [str(path) for path in dataset_files(data_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Challenge dataset files are missing: {missing}")
    files = {
        path.name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": HD.sha256(path),
        }
        for path in dataset_files(data_dir)
    }
    manifest = {
        "schema_version": 1,
        "frozen_at_utc": utc_now(),
        "purpose": "Immutable challenge dataset for the sole unchanged V4 confirmation rerun.",
        "data_contract_path": str(HD.CONTRACT_PATH),
        "data_contract_sha256": HD.sha256(HD.CONTRACT_PATH),
        "frozen_hypothesis_manifest_sha256": HD.sha256(HD.LOCK_PATH),
        "source_manifest_sha256": files["source_manifest.json"]["sha256"],
        "files": files,
        "readiness": readiness,
        "challenge_splits": HD.load_json(HD.CONTRACT_PATH)["challenge_protocol"],
    }
    write_json(lock_path, manifest)
    return {**manifest, "dataset_manifest_path": str(lock_path), "dataset_manifest_sha256": HD.sha256(lock_path)}


def verify_dataset_lock(lock_path: Path = DATASET_LOCK_PATH) -> dict:
    if not lock_path.exists():
        return {"valid": False, "reason": "frozen challenge dataset manifest is missing", "files": {}}
    try:
        manifest = HD.load_json(lock_path)
    except Exception as error:
        return {"valid": False, "reason": f"invalid dataset manifest: {error}", "files": {}}
    files = {}
    for name, expected in manifest.get("files", {}).items():
        path = Path(expected["path"])
        observed_hash = HD.sha256(path) if path.exists() else None
        observed_size = path.stat().st_size if path.exists() else None
        files[name] = {
            "path": str(path),
            "expected_sha256": expected.get("sha256"),
            "observed_sha256": observed_hash,
            "expected_size_bytes": expected.get("size_bytes"),
            "observed_size_bytes": observed_size,
            "matches": bool(
                observed_hash == expected.get("sha256")
                and observed_size == expected.get("size_bytes")
            ),
        }
    contract_matches = manifest.get("data_contract_sha256") == HD.sha256(HD.CONTRACT_PATH)
    hypothesis_manifest_matches = (
        HD.LOCK_PATH.exists()
        and manifest.get("frozen_hypothesis_manifest_sha256") == HD.sha256(HD.LOCK_PATH)
    )
    hypothesis = HD.verify_lock()
    valid = bool(
        files
        and all(item["matches"] for item in files.values())
        and contract_matches
        and hypothesis_manifest_matches
        and hypothesis["valid"]
    )
    return {
        "valid": valid,
        "dataset_manifest_path": str(lock_path),
        "dataset_manifest_sha256": HD.sha256(lock_path),
        "contract_matches": contract_matches,
        "hypothesis_manifest_matches": hypothesis_manifest_matches,
        "hypothesis_lock": hypothesis,
        "files": files,
    }


def performance(frame: pd.DataFrame, spec: dict) -> dict:
    if frame.empty:
        return {"periods": 0}
    result = V4.strategy_summary(frame, spec)
    result["start_signal_date"] = str(frame["signal_date"].min())
    result["end_signal_date"] = str(frame["signal_date"].max())
    return result


def selector_metrics(predictions: pd.DataFrame) -> dict:
    result = V4.alpha_metrics(predictions)
    rows = []
    for signal_date, day in predictions.dropna(subset=["alpha_prediction", "alpha_target"]).groupby("signal_date"):
        if len(day) < 20 or day["alpha_prediction"].nunique() < 2:
            continue
        ordered = day.sort_values("alpha_prediction", ascending=False)
        rows.append(
            {
                "signal_date": str(signal_date),
                "rank_ic": float(day["alpha_prediction"].rank().corr(day["alpha_target"].rank())),
                "top_minus_bottom": float(
                    ordered.head(10)["alpha_target"].mean() - ordered.tail(10)["alpha_target"].mean()
                ),
            }
        )
    return {**result, "monthly": monthly_selector_metrics(pd.DataFrame(rows))}


def monthly_selector_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["signal_date"]).dt.to_period("M").astype(str)
    return {
        month: {
            "periods": int(len(one)),
            "mean_rank_ic": float(one["rank_ic"].mean()),
            "mean_top_minus_bottom": float(one["top_minus_bottom"].mean()),
        }
        for month, one in frame.groupby("month")
    }


def direction_report(predictions: pd.DataFrame) -> dict:
    dates = predictions.drop_duplicates("signal_date").dropna(
        subset=["direction_prediction", "direction_target"]
    ).copy()
    if dates.empty:
        return {"periods": 0, "calibration_bins": []}
    dates["prediction_bin"] = pd.qcut(
        dates["direction_prediction"].rank(method="first"), min(5, len(dates)), labels=False
    )
    calibration = [
        {
            "bin": int(index),
            "periods": int(len(one)),
            "mean_prediction": float(one["direction_prediction"].mean()),
            "mean_realized_soxx_return": float(one["direction_target"].mean()),
            "realized_positive_fraction": float(one["direction_target"].gt(0).mean()),
        }
        for index, one in dates.groupby("prediction_bin", observed=True)
    ]
    return {**V4.direction_metrics(dates), "calibration_bins": calibration}


def regime_report(frame: pd.DataFrame, spec: dict) -> dict:
    if frame.empty:
        return {}
    realized_negative = frame["future_soxx_return_5d"].lt(0)
    predicted_negative = frame["direction_prediction"].lt(0)
    negative_net = frame["net_exposure"].lt(0)
    return {
        "counts": {
            "all_realized_periods": int(len(frame)),
            "realized_negative_soxx_periods": int(realized_negative.sum()),
            "realized_positive_or_flat_soxx_periods": int((~realized_negative).sum()),
            "predicted_negative_soxx_periods": int(predicted_negative.sum()),
            "predicted_positive_or_flat_soxx_periods": int((~predicted_negative).sum()),
            "negative_net_v4_periods": int(negative_net.sum()),
            "nonnegative_net_v4_periods": int((~negative_net).sum()),
        },
        "realized_negative_soxx": performance(frame[realized_negative], spec),
        "realized_positive_or_flat_soxx": performance(frame[~realized_negative], spec),
        "predicted_negative_soxx": performance(frame[predicted_negative], spec),
        "predicted_positive_or_flat_soxx": performance(frame[~predicted_negative], spec),
        "negative_net_v4": performance(frame[negative_net], spec),
        "nonnegative_net_v4": performance(frame[~negative_net], spec),
    }


def book_contributions(v4_periods: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        str(signal_date): day.set_index("symbol")["future_stock_return_5d"].to_dict()
        for signal_date, day in predictions.groupby("signal_date")
    }
    rows = []
    for period in v4_periods.itertuples(index=False):
        weights = json.loads(period.weights)
        realized = lookup.get(str(period.signal_date), {})
        long_contribution = 0.0
        short_contribution = 0.0
        for symbol, weight in weights.items():
            value = float(period.future_soxx_return_5d) if symbol == "SOXX" else float(realized[symbol])
            contribution = float(weight) * value
            if weight >= 0:
                long_contribution += contribution
            else:
                short_contribution += contribution
        rows.append(
            {
                "signal_date": period.signal_date,
                "long_book_contribution": long_contribution,
                "short_book_contribution": short_contribution,
                "gross_return_reconstructed": long_contribution + short_contribution,
                "gross_return_reported": float(period.gross_return),
                "transaction_cost": float(period.transaction_cost),
                "borrow_cost": float(period.borrow_cost),
                "net_return": float(period.net_return),
            }
        )
    return pd.DataFrame(rows)


def contribution_summary(
    books: pd.DataFrame,
    primary: pd.DataFrame,
    alpha_core: pd.DataFrame,
    direction_only: pd.DataFrame,
) -> dict:
    aligned = primary[["signal_date", "net_return"]].merge(
        alpha_core[["signal_date", "net_return"]], on="signal_date", suffixes=("_v4", "_core")
    )
    aligned["direction_overlay_increment"] = aligned["net_return_v4"] - aligned["net_return_core"]
    return {
        "long_book_arithmetic_contribution": float(books["long_book_contribution"].sum()),
        "short_book_arithmetic_contribution": float(books["short_book_contribution"].sum()),
        "long_book_period_contribution_statistics": V4.V3.performance(
            books["long_book_contribution"], float(V4.load_spec(HD.SPEC_PATH)["evaluation"]["periods_per_year"])
        ),
        "short_book_period_contribution_statistics": V4.V3.performance(
            books["short_book_contribution"], float(V4.load_spec(HD.SPEC_PATH)["evaluation"]["periods_per_year"])
        ),
        "transaction_cost_drag": float(books["transaction_cost"].sum()),
        "borrow_cost_drag": float(books["borrow_cost"].sum()),
        "reconstruction_max_abs_error": float(
            (books["gross_return_reconstructed"] - books["gross_return_reported"]).abs().max()
        ),
        "market_neutral_selection_only": performance(alpha_core, V4.load_spec(HD.SPEC_PATH)),
        "direction_only": performance(direction_only, V4.load_spec(HD.SPEC_PATH)),
        "direction_overlay_arithmetic_increment": float(aligned["direction_overlay_increment"].sum()),
    }


def weight_records(v4_periods: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    returns = {
        str(signal_date): day.set_index("symbol")["future_stock_return_5d"].to_dict()
        for signal_date, day in predictions.groupby("signal_date")
    }
    rows = []
    for period in v4_periods.itertuples(index=False):
        for symbol, weight in json.loads(period.weights).items():
            realized = (
                float(period.future_soxx_return_5d)
                if symbol == "SOXX"
                else float(returns[str(period.signal_date)][symbol])
            )
            rows.append(
                {
                    "signal_date": period.signal_date,
                    "year": pd.Timestamp(period.signal_date).year,
                    "symbol": symbol,
                    "weight": float(weight),
                    "abs_weight": abs(float(weight)),
                    "contribution": float(weight) * realized,
                }
            )
    return pd.DataFrame(rows)


def security_concentration(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    grouped = raw.groupby("symbol").agg(
        periods=("signal_date", "nunique"),
        mean_abs_weight=("abs_weight", "mean"),
        max_abs_weight=("abs_weight", "max"),
        arithmetic_contribution=("contribution", "sum"),
        long_periods=("weight", lambda values: int((values > 0).sum())),
        short_periods=("weight", lambda values: int((values < 0).sum())),
    )
    return grouped.reset_index().sort_values("mean_abs_weight", ascending=False)


def grouped_concentration(raw: pd.DataFrame, group_column: str) -> dict:
    output = {}
    for group, one in raw.groupby(group_column):
        exposure = one.groupby("symbol")["abs_weight"].sum().sort_values(ascending=False)
        total = float(exposure.sum())
        shares = exposure / total if total > 0 else exposure
        output[str(group)] = {
            "periods": int(one["signal_date"].nunique()),
            "securities": int(one["symbol"].nunique()),
            "largest_single_position_weight": float(one["abs_weight"].max()),
            "largest_cumulative_exposure_symbol": str(exposure.index[0]) if len(exposure) else None,
            "largest_cumulative_exposure_share": float(shares.iloc[0]) if len(shares) else None,
            "exposure_herfindahl": float((shares**2).sum()) if len(shares) else None,
        }
    return output


def drawdown_episodes(market_path: Path, threshold: float) -> list[dict]:
    market = pd.read_csv(market_path)
    market["date"] = pd.to_datetime(market["timestamp"], utc=True).dt.date
    soxx = market[market["symbol"].eq("SOXX")].sort_values("date").drop_duplicates("date")
    episodes = []
    peak_value = -np.inf
    peak_date = None
    active = None
    for row in soxx.itertuples(index=False):
        close = float(row.close)
        if active is None:
            if close > peak_value:
                peak_value = close
                peak_date = row.date
            decline = close / peak_value - 1.0 if peak_value > 0 else 0.0
            if decline <= threshold:
                active = {
                    "peak_date": peak_date,
                    "peak_close": peak_value,
                    "start_date": row.date,
                    "trough_date": row.date,
                    "trough_drawdown": decline,
                }
        else:
            decline = close / active["peak_close"] - 1.0
            if decline < active["trough_drawdown"]:
                active["trough_drawdown"] = decline
                active["trough_date"] = row.date
            if close >= active["peak_close"]:
                active["end_date"] = row.date
                episodes.append(active)
                peak_value = close
                peak_date = row.date
                active = None
    if active is not None:
        active["end_date"] = soxx.iloc[-1]["date"]
        episodes.append(active)
    return episodes


def episode_report(
    v4_periods: pd.DataFrame,
    predictions: pd.DataFrame,
    books: pd.DataFrame,
    weights: pd.DataFrame,
    market_path: Path,
    spec: dict,
    threshold: float,
) -> list[dict]:
    result = []
    signal_dates = pd.to_datetime(v4_periods["signal_date"]).dt.date
    for number, episode in enumerate(drawdown_episodes(market_path, threshold), start=1):
        mask = signal_dates.ge(episode["peak_date"]) & signal_dates.le(episode["end_date"])
        one = v4_periods[mask]
        prediction_dates = pd.to_datetime(predictions["signal_date"]).dt.date
        pred = predictions[
            prediction_dates.ge(episode["peak_date"]) & prediction_dates.le(episode["end_date"])
        ]
        book_dates = pd.to_datetime(books["signal_date"]).dt.date
        book = books[book_dates.ge(episode["peak_date"]) & book_dates.le(episode["end_date"])]
        weight_dates = pd.to_datetime(weights["signal_date"]).dt.date
        episode_weights = weights[
            weight_dates.ge(episode["peak_date"]) & weight_dates.le(episode["end_date"])
        ]
        top_security = security_concentration(episode_weights).head(5)
        result.append(
            {
                "episode": number,
                **episode,
                "strategy": performance(one, spec),
                "soxx_benchmark": V4.V3.performance(
                    one["future_soxx_return_5d"], float(spec["evaluation"]["periods_per_year"])
                ),
                "regimes": regime_report(one, spec),
                "residual_selector": selector_metrics(pred),
                "direction_forecast": direction_report(pred),
                "long_book_arithmetic_contribution": float(book["long_book_contribution"].sum()),
                "short_book_arithmetic_contribution": float(book["short_book_contribution"].sum()),
                "transaction_cost_drag": float(book["transaction_cost"].sum()),
                "borrow_cost_drag": float(book["borrow_cost"].sum()),
                "top_security_concentration": top_security.to_dict(orient="records"),
            }
        )
    return result


def split_report(
    periods: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: dict,
    contract: dict,
) -> dict:
    signal = pd.to_datetime(periods["signal_date"])
    prediction_signal = pd.to_datetime(predictions["signal_date"])
    protocol = contract["challenge_protocol"]
    splits = {
        "pre_2021_genuinely_new": (
            pd.Timestamp(protocol["historical_unseen_start"]),
            pd.Timestamp(protocol["historical_unseen_end"]),
        ),
        "2021_2026_previously_inspected": (
            pd.Timestamp(protocol["previously_inspected_start"]),
            pd.Timestamp(protocol["previously_inspected_end"]),
        ),
        "full_history": (signal.min(), signal.max()),
    }
    output = {}
    for name, (start, end) in splits.items():
        one = periods[signal.ge(start) & signal.le(end)]
        pred = predictions[prediction_signal.ge(start) & prediction_signal.le(end)]
        output[name] = {
            "strategy": performance(one, spec),
            "regimes": regime_report(one, spec),
            "residual_selector": selector_metrics(pred),
            "direction_forecast": direction_report(pred),
        }
    return output


def generate_report(run_dir: Path = RUN_DIR, data_dir: Path = HD.DATA_DIR) -> dict:
    results = HD.load_json(run_dir / "results.json")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    all_periods = pd.read_csv(run_dir / "period_returns.csv")
    spec = V4.load_spec(HD.SPEC_PATH)
    contract = HD.load_json(HD.CONTRACT_PATH)
    primary = all_periods[all_periods["strategy"].eq("v4_continuous")].copy()
    alpha_core = all_periods[all_periods["strategy"].eq("alpha_core")].copy()
    direction_only = all_periods[all_periods["strategy"].eq("direction_only")].copy()
    stress = all_periods[all_periods["strategy"].eq("v4_continuous_stress")].copy()
    if primary.empty:
        raise ValueError("V4 confirmation output has no primary period returns")
    books = book_contributions(primary, predictions)
    weights = weight_records(primary, predictions)
    concentration = security_concentration(weights)
    episodes = episode_report(
        primary,
        predictions,
        books,
        weights,
        data_dir / "market_bars.csv",
        spec,
        float(contract["acceptance"]["bear_drawdown_threshold"]),
    )
    yearly = {
        str(year): performance(one, spec)
        for year, one in primary.assign(year=pd.to_datetime(primary["signal_date"]).dt.year).groupby("year")
    }
    report = {
        "created_at_utc": utc_now(),
        "confirmation_outcome": (
            "data_passed_v4_passed"
            if results["decision"]["production_evidence_pass"]
            else "data_passed_v4_failed"
        ),
        "dataset_lock": verify_dataset_lock(),
        "hypothesis_lock": HD.verify_lock(),
        "full_period": performance(primary, spec),
        "regime_distinction": regime_report(primary, spec),
        "residual_selector": selector_metrics(predictions),
        "direction_forecast": direction_report(predictions),
        "components": contribution_summary(books, primary, alpha_core, direction_only),
        "cost_and_borrow_stress": {
            "primary": performance(primary, spec),
            "stress_25bps_and_15pct_fallback_borrow": performance(stress, spec),
            "primary_transaction_cost_drag": float(primary["transaction_cost"].sum()),
            "primary_borrow_cost_drag": float(primary["borrow_cost"].sum()),
            "stress_transaction_cost_drag": float(stress["transaction_cost"].sum()),
            "stress_borrow_cost_drag": float(stress["borrow_cost"].sum()),
        },
        "time_splits": split_report(primary, predictions, spec, contract),
        "major_drawdown_episodes": episodes,
        "yearly": yearly,
        "concentration_by_year": grouped_concentration(weights, "year"),
        "concentration_by_episode": {
            str(item["episode"]): item["top_security_concentration"] for item in episodes
        },
        "security_concentration_top_20": concentration.head(20).to_dict(orient="records"),
        "frozen_pass_fail": results["decision"],
        "interpretation_guardrails": [
            "Realized negative-SOXX periods and predicted negative-direction periods answer different questions.",
            "Negative direction predictions and actual negative-net V4 positioning are reported separately.",
            "The pre-2021 split is the primary genuinely new historical challenge period.",
            "A failed V4 is not modified and rerun on this now-inspected challenge dataset.",
            "A passed V4 proceeds to prospective paper testing, not immediate live deployment.",
        ],
    }
    write_json(run_dir / "confirmation_report.json", report)
    books.to_csv(run_dir / "book_contributions.csv", index=False)
    concentration.to_csv(run_dir / "security_concentration.csv", index=False)
    weights.to_csv(run_dir / "position_contributions.csv", index=False)
    pd.DataFrame(episodes).to_csv(run_dir / "drawdown_episodes.csv", index=False)
    return report


def claim_receipt(receipt_path: Path, value: dict) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def run_once(
    confirmation: str,
    run_dir: Path = RUN_DIR,
    receipt_path: Path = RUN_RECEIPT_PATH,
    dataset_lock_path: Path = DATASET_LOCK_PATH,
) -> dict:
    if confirmation != RUN_CONFIRMATION:
        raise ValueError(f"One-shot rerun requires --confirm {RUN_CONFIRMATION}")
    if receipt_path.exists():
        raise FileExistsError(f"The sole V4 confirmation run was already claimed: {receipt_path}")
    dataset = verify_dataset_lock(dataset_lock_path)
    if not dataset["valid"]:
        raise ValueError("Frozen challenge dataset hash verification failed")
    hypothesis = HD.verify_lock()
    if not hypothesis["valid"]:
        raise ValueError("Frozen V4 hypothesis hash verification failed")
    readiness = HD.audit_bundle(HD.DATA_DIR)
    if not readiness["v4_unchanged_rerun_ready"]:
        raise ValueError("Installed challenge data no longer passes readiness checks")
    if run_dir.exists():
        raise FileExistsError(f"Confirmation output directory already exists: {run_dir}")
    receipt = {
        "claimed_at_utc": utc_now(),
        "status": "claimed",
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "hypothesis_manifest_sha256": HD.sha256(HD.LOCK_PATH),
        "command": [
            sys.executable,
            str(ROOT / "research.py"),
            "run",
            "--spec",
            str(HD.SPEC_PATH),
            "--out",
            str(run_dir),
        ],
    }
    claim_receipt(receipt_path, receipt)
    process = subprocess.run(receipt["command"], cwd=ROOT.parent, capture_output=True, text=True)
    receipt.update(
        {
            "finished_at_utc": utc_now(),
            "return_code": process.returncode,
            "stdout": process.stdout[-20000:],
            "stderr": process.stderr[-20000:],
        }
    )
    if process.returncode != 0:
        receipt["status"] = "technical_failure_no_automatic_retry"
        write_json(receipt_path, receipt)
        raise RuntimeError("Frozen V4 process failed; receipt permanently blocks an automatic retry")
    report = generate_report(run_dir)
    receipt["status"] = "completed"
    receipt["outcome"] = report["confirmation_outcome"]
    receipt["output_hashes"] = {
        path.name: HD.sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    write_json(receipt_path, receipt)
    return receipt


def status() -> dict:
    readiness = HD.audit_bundle(HD.DATA_DIR)
    return {
        "status": (
            "awaiting_confirmation_quality_data"
            if not readiness["v4_unchanged_rerun_ready"]
            else "data_ready_not_frozen"
            if not DATASET_LOCK_PATH.exists()
            else "dataset_frozen_awaiting_one_shot"
            if not RUN_RECEIPT_PATH.exists()
            else "confirmation_run_claimed"
        ),
        "hypothesis_lock": HD.verify_lock(),
        "data_readiness": readiness,
        "dataset_lock": verify_dataset_lock(),
        "one_shot_receipt": HD.load_json(RUN_RECEIPT_PATH) if RUN_RECEIPT_PATH.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    freeze = commands.add_parser("freeze-dataset")
    freeze.add_argument("--confirm", default="")
    commands.add_parser("verify-dataset")
    run = commands.add_parser("run-once")
    run.add_argument("--confirm", default="")
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "status":
        result = status()
    elif args.command == "freeze-dataset":
        result = freeze_dataset(confirmation=args.confirm)
    elif args.command == "verify-dataset":
        result = verify_dataset_lock()
    elif args.command == "run-once":
        result = run_once(args.confirm)
    else:
        if not RUN_RECEIPT_PATH.exists():
            raise ValueError("No one-shot confirmation receipt exists")
        result = generate_report()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
