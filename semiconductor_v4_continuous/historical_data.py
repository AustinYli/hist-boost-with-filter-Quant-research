#!/usr/bin/env python3
"""Audit and install provider-neutral historical data for frozen V4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
CONTRACT_PATH = ROOT / "data_contract.json"
SPEC_PATH = ROOT / "spec.json"
LOCK_PATH = ROOT / "frozen_hypothesis_manifest.json"
INSTALL_CONFIRMATION = "INSTALL_STRICT_V4_DATA"
RAW_MANIFEST_NAME = "raw_manifest.json"

TABLE_COLUMNS = {
    "security_master.csv": [
        "symbol",
        "identifier_type",
        "identifier_value",
        "ticker",
        "valid_from",
        "valid_to",
        "cusip",
        "isin",
        "source",
    ],
    "stock_bars.csv": [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ],
    "market_bars.csv": [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ],
    "universe_membership.csv": [
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
    ],
    "borrow_costs.csv": ["date", "symbol", "shortable", "annual_borrow_rate", "source"],
    "dividends.csv": ["date", "symbol", "cash_amount", "source"],
    "corporate_actions.csv": ["date", "symbol", "action_type", "value", "handled", "source"],
}

INSTALL_TABLES = list(TABLE_COLUMNS)

CORE_LOCK_FILES = {
    "v4_spec": SPEC_PATH,
    "v4_research": ROOT / "research.py",
    "v3_data_dependency": PROJECT_ROOT / "semiconductor_v3_directional" / "research.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_directory(stage: Path) -> Path:
    canonical = stage / "canonical"
    return canonical if canonical.is_dir() else stage


def provenance_path(stage: Path) -> Path | None:
    candidates = [
        stage / "provenance.json",
        canonical_directory(stage) / "provenance.json",
        stage / "source_manifest.json",
        canonical_directory(stage) / "source_manifest.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def raw_export_files(stage: Path) -> list[Path]:
    raw = stage / "raw"
    if not raw.is_dir():
        return []
    return sorted(path for path in raw.rglob("*") if path.is_file())


def register_raw_exports(
    stage: Path,
    vendor: str,
    version: str,
    as_of_date: str,
    license_reference: str,
) -> dict:
    """Fingerprint vendor exports."""
    stage = stage.resolve()
    manifest_path = stage / RAW_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace registered raw manifest {manifest_path}")
    metadata = {
        "vendor": vendor.strip(),
        "version": version.strip(),
        "as_of_date": as_of_date.strip(),
        "license_reference": license_reference.strip(),
    }
    if any(not value for value in metadata.values()):
        raise ValueError("vendor, version, as-of date, and license reference are all required")
    pd.Timestamp(metadata["as_of_date"])
    files = raw_export_files(stage)
    if not files:
        raise ValueError(f"No original vendor files found below {stage / 'raw'}")
    entries = [
        {
            "path": path.relative_to(stage).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "registered_at_utc": utc_now(),
        "source": metadata,
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path)}


def verify_raw_exports(stage: Path) -> dict:
    stage = stage.resolve()
    manifest_path = stage / RAW_MANIFEST_NAME
    if not manifest_path.exists():
        return {"valid": False, "reason": f"{RAW_MANIFEST_NAME} is missing", "files": []}
    try:
        manifest = load_json(manifest_path)
    except Exception as error:
        return {"valid": False, "reason": f"invalid manifest JSON: {error}", "files": []}
    registered = {item.get("path"): item for item in manifest.get("files", [])}
    observed_paths = {path.relative_to(stage).as_posix(): path for path in raw_export_files(stage)}
    checks = []
    for relative_path in sorted(set(registered) | set(observed_paths)):
        expected = registered.get(relative_path)
        path = observed_paths.get(relative_path)
        observed_hash = sha256(path) if path is not None else None
        observed_size = path.stat().st_size if path is not None else None
        checks.append(
            {
                "path": relative_path,
                "expected_sha256": expected.get("sha256") if expected else None,
                "observed_sha256": observed_hash,
                "expected_size_bytes": expected.get("size_bytes") if expected else None,
                "observed_size_bytes": observed_size,
                "matches": bool(
                    expected
                    and path
                    and expected.get("sha256") == observed_hash
                    and int(expected.get("size_bytes", -1)) == observed_size
                ),
            }
        )
    source = manifest.get("source", {})
    metadata_complete = all(source.get(field) for field in ["vendor", "version", "as_of_date", "license_reference"])
    valid = bool(checks and metadata_complete and all(item["matches"] for item in checks))
    return {
        "valid": valid,
        "reason": "registered raw exports match" if valid else "raw export set, hashes, sizes, or metadata do not match",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source": source,
        "files": checks,
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_check(report: dict, name: str, passed: bool, detail: str, blocking: bool = True) -> None:
    item = {"name": name, "passed": bool(passed), "blocking": bool(blocking), "detail": detail}
    report["checks"].append(item)
    if blocking and not passed:
        report["blockers"].append(f"{name}: {detail}")
    elif not passed:
        report["warnings"].append(f"{name}: {detail}")


def load_table(stage: Path, name: str, report: dict) -> pd.DataFrame:
    path = stage / name
    if not path.exists():
        add_check(report, f"file:{name}", False, "file is missing")
        return pd.DataFrame(columns=TABLE_COLUMNS[name])
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as error:
        add_check(report, f"file:{name}", False, f"cannot read CSV: {error}")
        return pd.DataFrame(columns=TABLE_COLUMNS[name])
    missing = sorted(set(TABLE_COLUMNS[name]) - set(frame.columns))
    add_check(
        report,
        f"schema:{name}",
        not missing,
        "required columns present" if not missing else f"missing columns {missing}",
    )
    if missing:
        return pd.DataFrame(columns=TABLE_COLUMNS[name])
    return frame


def parse_bars(frame: pd.DataFrame, name: str, report: dict) -> pd.DataFrame:
    if frame.empty:
        add_check(report, f"rows:{name}", False, "table is empty")
        return frame.assign(date=pd.Series(dtype="object"))
    result = frame.copy()
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    except Exception as error:
        add_check(report, f"dates:{name}", False, f"invalid timestamp: {error}")
        return result.assign(date=pd.NaT)
    result["date"] = result["timestamp"].dt.date
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    numeric = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    price_columns = ["open", "high", "low", "close"]
    finite = np.isfinite(result[price_columns].to_numpy(dtype=float)).all()
    nonnegative = bool((result[price_columns] >= 0).all().all() and (result["volume"] >= 0).all())
    consistent = bool(
        (result["low"] <= result[["open", "close", "high"]].min(axis=1)).all()
        and (result["high"] >= result[["open", "close", "low"]].max(axis=1)).all()
    )
    duplicates = int(result.duplicated(["symbol", "date"]).sum())
    add_check(report, f"numeric:{name}", finite and nonnegative, "finite nonnegative OHLCV values")
    add_check(report, f"ohlc:{name}", consistent, "low <= open/close <= high")
    add_check(report, f"unique:{name}", duplicates == 0, f"duplicate symbol-date rows: {duplicates}")
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def parse_membership(frame: pd.DataFrame, report: dict) -> pd.DataFrame:
    if frame.empty:
        add_check(report, "rows:universe_membership.csv", False, "table is empty")
        return frame
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    try:
        result["valid_from"] = pd.to_datetime(result["valid_from"], errors="raise").dt.date
        result["valid_to"] = pd.to_datetime(result["valid_to"], errors="coerce").dt.date
    except Exception as error:
        add_check(report, "dates:universe_membership.csv", False, f"invalid interval date: {error}")
        return result
    for column in ["us_listed", "survivorship_safe", "delisted", "delisting_handled"]:
        result[column] = normalized_bool(result[column])
    bad_ranges = result["valid_to"].notna() & (result["valid_to"] < result["valid_from"])
    overlaps = 0
    for _, one in result.sort_values(["symbol", "valid_from"]).groupby("symbol"):
        prior_end = None
        for row in one.itertuples(index=False):
            if prior_end is not None and row.valid_from <= prior_end:
                overlaps += 1
            if pd.isna(row.valid_to):
                prior_end = pd.Timestamp.max.date()
            elif prior_end is None or row.valid_to > prior_end:
                prior_end = row.valid_to
    add_check(report, "membership_intervals", not bad_ranges.any() and overlaps == 0, f"bad ranges={int(bad_ranges.sum())}, overlaps={overlaps}")
    add_check(report, "point_in_time_membership_rows", bool(result["survivorship_safe"].all()), "every interval is explicitly survivorship-safe")
    add_check(report, "delisting_handled_rows", bool(result["delisting_handled"].all()), "every interval has documented delisting treatment")
    return result


def parse_master(frame: pd.DataFrame, contract: dict, report: dict) -> pd.DataFrame:
    if frame.empty:
        add_check(report, "rows:security_master.csv", False, "table is empty")
        return frame
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    result["identifier_type"] = result["identifier_type"].astype(str).str.strip().str.lower()
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    try:
        result["valid_from"] = pd.to_datetime(result["valid_from"], errors="raise").dt.date
        result["valid_to"] = pd.to_datetime(result["valid_to"], errors="coerce").dt.date
    except Exception as error:
        add_check(report, "dates:security_master.csv", False, f"invalid identifier interval: {error}")
        return result
    allowed = set(contract["required_stable_identifier_types"])
    stable = result["identifier_type"].isin(allowed) & result["identifier_value"].astype(str).str.strip().ne("")
    add_check(report, "stable_security_identity", bool(stable.all()), f"allowed identifier types: {sorted(allowed)}")
    duplicates = int(result.duplicated(["symbol", "valid_from", "ticker"]).sum())
    add_check(report, "unique_security_master", duplicates == 0, f"duplicate identity intervals: {duplicates}")
    return result


def parse_borrow(frame: pd.DataFrame, report: dict) -> pd.DataFrame:
    if frame.empty:
        add_check(report, "rows:borrow_costs.csv", False, "table is empty")
        return frame
    result = frame.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.date
    except Exception as error:
        add_check(report, "dates:borrow_costs.csv", False, f"invalid date: {error}")
        return result
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    result["shortable"] = normalized_bool(result["shortable"])
    result["annual_borrow_rate"] = pd.to_numeric(result["annual_borrow_rate"], errors="coerce")
    valid_rates = result.loc[result["shortable"], "annual_borrow_rate"].notna().all() and (
        result.loc[result["shortable"], "annual_borrow_rate"] >= 0
    ).all()
    duplicates = int(result.duplicated(["date", "symbol"]).sum())
    add_check(report, "borrow_rates", bool(valid_rates), "every shortable row has a nonnegative annual fee")
    add_check(report, "unique_borrow_panel", duplicates == 0, f"duplicate date-symbol rows: {duplicates}")
    return result


def parse_dividends(frame: pd.DataFrame, report: dict) -> pd.DataFrame:
    if frame.empty:
        add_check(report, "rows:dividends.csv", False, "table is empty")
        return frame
    result = frame.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.date
    except Exception as error:
        add_check(report, "dates:dividends.csv", False, f"invalid date: {error}")
        return result
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    result["cash_amount"] = pd.to_numeric(result["cash_amount"], errors="coerce")
    valid = result["cash_amount"].notna().all() and (result["cash_amount"] >= 0).all()
    add_check(report, "dividend_amounts", bool(valid), "cash amounts are finite and nonnegative")
    return result


def parse_actions(frame: pd.DataFrame, report: dict) -> pd.DataFrame:
    if frame.empty:
        add_check(report, "rows:corporate_actions.csv", False, "table is empty")
        return frame
    result = frame.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.date
    except Exception as error:
        add_check(report, "dates:corporate_actions.csv", False, f"invalid date: {error}")
        return result
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    result["action_type"] = result["action_type"].astype(str).str.strip().str.lower()
    result["handled"] = normalized_bool(result["handled"])
    add_check(report, "corporate_actions_handled", bool(result["handled"].all()), "every action is marked handled")
    return result


def membership_active_keys(stocks: pd.DataFrame, membership: pd.DataFrame, start, end) -> pd.MultiIndex:
    pieces = []
    limited = stocks[stocks["date"].ge(start) & stocks["date"].le(end)]
    for row in membership.itertuples(index=False):
        mask = limited["symbol"].eq(row.symbol) & limited["date"].ge(row.valid_from)
        if not pd.isna(row.valid_to):
            mask &= limited["date"].le(row.valid_to)
        one = limited.loc[mask, ["date", "symbol"]]
        if not one.empty:
            pieces.append(one)
    if not pieces:
        return pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
    return pd.MultiIndex.from_frame(pd.concat(pieces, ignore_index=True).drop_duplicates())


def decision_period_diagnostics(
    market: pd.DataFrame,
    stocks: pd.DataFrame,
    membership: pd.DataFrame,
    spec: dict,
    contract: dict,
) -> dict:
    soxx = (
        market[market["symbol"].eq("SOXX")]
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if soxx.empty:
        return {"realized_periods": 0, "negative_periods": 0, "drawdown_episodes": 0}
    dates = soxx["date"].tolist()
    index = {date: position for position, date in enumerate(dates)}
    anchor = pd.Timestamp(spec["execution"]["decision_grid_anchor_date"]).date()
    every = int(spec["execution"]["decision_grid_every_sessions"])
    if anchor in index:
        anchor_index = index[anchor]
    elif anchor > dates[-1]:
        estimated = len(pd.bdate_range(pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), anchor))
        anchor_index = len(dates) - 1 + estimated
    else:
        return {"realized_periods": 0, "negative_periods": 0, "drawdown_episodes": 0}
    horizon = int(spec["execution"]["horizon_sessions"])
    target = soxx["open"].shift(-(horizon + 1)).div(soxx["open"].shift(-1)) - 1.0
    oos_start = pd.Timestamp(contract["challenge_protocol"]["historical_unseen_start"]).date()
    cutoff = pd.Timestamp(spec["data"]["research_label_cutoff"]).date()
    grid = pd.Series([(position - anchor_index) % every == 0 for position in range(len(soxx))])
    eligible = grid & soxx["date"].ge(oos_start) & soxx["date"].le(cutoff) & target.notna()
    if not stocks.empty and not membership.empty:
        stock = stocks.sort_values(["symbol", "date"]).copy()
        stock["history_sessions"] = stock.groupby("symbol").cumcount() + 1
        stock["dollar_volume"] = stock["close"] * stock["volume"]
        liquidity_window = int(spec["universe"]["liquidity_window_sessions"])
        stock["median_dollar_volume"] = stock.groupby("symbol")["dollar_volume"].transform(
            lambda values: values.rolling(liquidity_window, min_periods=liquidity_window).median()
        )
        active = pd.Series(False, index=stock.index)
        for row in membership.itertuples(index=False):
            mask = stock["symbol"].eq(row.symbol) & stock["date"].ge(row.valid_from)
            if not pd.isna(row.valid_to):
                mask &= stock["date"].le(row.valid_to)
            active |= mask
        universe = spec["universe"]
        stock["eligible"] = (
            active
            & stock["history_sessions"].ge(int(universe["minimum_history_sessions"]))
            & stock["close"].ge(float(universe["minimum_price"]))
            & stock["median_dollar_volume"].ge(float(universe["minimum_median_dollar_volume"]))
        )
        counts = stock[stock["eligible"]].groupby("date")["symbol"].nunique()
        tradable_dates = set(counts[counts.ge(int(universe["minimum_names_to_trade"]))].index)
        eligible &= soxx["date"].isin(tradable_dates)
    drawdown = soxx["close"].div(soxx["close"].cummax()) - 1.0
    threshold = float(contract["acceptance"]["bear_drawdown_threshold"])
    in_bear = drawdown.le(threshold)
    recovery_threshold = threshold / 2.0
    episodes = 0
    armed = True
    for is_bear, recovered in zip(in_bear, drawdown.ge(recovery_threshold)):
        if armed and is_bear:
            episodes += 1
            armed = False
        elif not armed and recovered:
            armed = True
    return {
        "realized_periods": int(eligible.sum()),
        "negative_periods": int((target[eligible] < 0).sum()),
        "drawdown_episodes": episodes,
    }


def audit_bundle(stage: Path, contract: dict | None = None, spec: dict | None = None) -> dict:
    stage = stage.resolve()
    canonical = canonical_directory(stage)
    contract = contract or load_json(CONTRACT_PATH)
    spec = spec or load_json(SPEC_PATH)
    report = {
        "created_at_utc": utc_now(),
        "stage": str(stage),
        "canonical_directory": str(canonical),
        "contract_locked_at": contract["contract_locked_at"],
        "checks": [],
        "blockers": [],
        "warnings": [],
        "metrics": {},
    }

    source_path = provenance_path(stage)
    loaded_provenance = {}
    if source_path is not None:
        try:
            loaded_provenance = load_json(source_path)
            provenance = loaded_provenance.get("provenance", loaded_provenance)
            add_check(report, "provenance_file", True, str(source_path.name))
        except Exception as error:
            provenance = {}
            add_check(report, "provenance_file", False, f"invalid JSON: {error}")
    else:
        provenance = {}
        add_check(report, "provenance_file", False, "provenance.json is missing")

    physical_raw = verify_raw_exports(stage)
    preserved_raw = loaded_provenance.get("raw_export", {})
    preserved_valid = bool(
        preserved_raw.get("verified_at_normalization")
        and preserved_raw.get("manifest_sha256")
        and preserved_raw.get("manifest", {}).get("files")
        and preserved_raw.get("manifest_content_sha256")
        == json_sha256(preserved_raw.get("manifest", {}))
    )
    raw_valid = physical_raw["valid"] or preserved_valid
    raw_detail = (
        physical_raw["reason"]
        if physical_raw["valid"] or not preserved_valid
        else f"verified normalized provenance; manifest={preserved_raw.get('manifest_sha256')}"
    )
    add_check(report, "raw_export_fingerprint", raw_valid, raw_detail)

    coverage = provenance.get("coverage", {})
    for field in contract["required_source_attestations"]:
        add_check(report, f"attestation:{field}", coverage.get(field) is True, "provider/export attestation must be true")
    add_check(
        report,
        "price_adjustment",
        provenance.get("price_adjustment") == contract["canonical_price_adjustment"],
        f"required={contract['canonical_price_adjustment']}, observed={provenance.get('price_adjustment')}",
    )

    frames = {name: load_table(canonical, name, report) for name in TABLE_COLUMNS}
    master = parse_master(frames["security_master.csv"], contract, report)
    stocks = parse_bars(frames["stock_bars.csv"], "stock_bars.csv", report)
    market = parse_bars(frames["market_bars.csv"], "market_bars.csv", report)
    membership = parse_membership(frames["universe_membership.csv"], report)
    borrow = parse_borrow(frames["borrow_costs.csv"], report)
    dividends = parse_dividends(frames["dividends.csv"], report)
    actions = parse_actions(frames["corporate_actions.csv"], report)

    master_symbols = set(master.get("symbol", pd.Series(dtype=str)))
    stock_symbols = set(stocks.get("symbol", pd.Series(dtype=str)))
    membership_symbols = set(membership.get("symbol", pd.Series(dtype=str)))
    linked = bool(stock_symbols) and stock_symbols.issubset(master_symbols) and stock_symbols.issubset(membership_symbols)
    add_check(report, "identity_linkage", linked, f"stock={len(stock_symbols)}, master={len(master_symbols)}, membership={len(membership_symbols)}")

    required_market = set(spec["data"]["required_market_symbols"])
    observed_market = set(market.get("symbol", pd.Series(dtype=str)))
    add_check(report, "required_market_symbols", required_market.issubset(observed_market), f"required={sorted(required_market)}, observed={sorted(observed_market)}")

    acceptance = contract["acceptance"]
    min_stocks = int(acceptance["minimum_stock_symbols"])
    add_check(report, "stock_symbol_count", len(stock_symbols) >= min_stocks, f"observed={len(stock_symbols)}, required={min_stocks}")

    if not market.empty and "date" in market:
        soxx_dates = set(market.loc[market["symbol"].eq("SOXX"), "date"])
        overlap = {}
        for symbol in required_market:
            dates = set(market.loc[market["symbol"].eq(symbol), "date"])
            overlap[symbol] = len(dates & soxx_dates) / len(soxx_dates) if soxx_dates else 0.0
        minimum_overlap = min(overlap.values()) if overlap else 0.0
        add_check(report, "market_calendar_overlap", minimum_overlap >= float(acceptance["minimum_market_calendar_overlap"]), f"overlap={overlap}")
        market_sessions = len(soxx_dates)
        history_years = (
            (max(soxx_dates) - min(soxx_dates)).days / 365.25 if len(soxx_dates) > 1 else 0.0
        )
    else:
        overlap = {}
        market_sessions = 0
        history_years = 0.0
    add_check(report, "history_years", history_years >= float(acceptance["minimum_history_years"]), f"observed={history_years:.2f}")
    add_check(report, "market_sessions", market_sessions >= int(acceptance["minimum_market_sessions"]), f"observed={market_sessions}")

    oos_start = pd.Timestamp(contract["challenge_protocol"]["historical_unseen_start"]).date()
    cutoff = pd.Timestamp(spec["data"]["research_label_cutoff"]).date()
    sessions_before = 0
    if not market.empty and "date" in market:
        sessions_before = market.loc[market["symbol"].eq("SOXX") & market["date"].lt(oos_start), "date"].nunique()
    add_check(
        report,
        "pre_oos_warmup",
        sessions_before >= int(acceptance["minimum_sessions_before_historical_oos"]),
        f"observed={sessions_before}",
    )

    if not stocks.empty and not membership.empty:
        expected = membership_active_keys(stocks, membership, oos_start, cutoff)
        observed = (
            pd.MultiIndex.from_frame(borrow[["date", "symbol"]].drop_duplicates())
            if not borrow.empty
            else pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        )
        covered = len(expected.intersection(observed))
        missing_fraction = 1.0 - covered / len(expected) if len(expected) else 1.0
    else:
        expected = pd.MultiIndex.from_arrays([[], []])
        missing_fraction = 1.0
    add_check(
        report,
        "borrow_panel_coverage",
        missing_fraction <= float(acceptance["maximum_missing_borrow_fraction"]),
        f"expected_stock_dates={len(expected)}, missing_fraction={missing_fraction:.6f}",
    )

    delisted = set(membership.loc[membership.get("delisted", False), "symbol"]) if not membership.empty else set()
    delisting_actions = set(
        actions.loc[
            actions.get("action_type", pd.Series(dtype=str)).isin(
                contract["delisting_materialization"]["supported_action_types"]
            )
            & actions.get("handled", pd.Series(dtype=bool)),
            "symbol",
        ]
    ) if not actions.empty else set()
    missing_delistings = sorted(delisted - delisting_actions)
    add_check(
        report,
        "inactive_security_presence",
        bool(delisted),
        f"historical delisted securities represented={len(delisted)}",
    )
    add_check(report, "delisting_outcome_coverage", not missing_delistings, f"delisted={len(delisted)}, missing={missing_delistings[:20]}")

    diagnostics = decision_period_diagnostics(market, stocks, membership, spec, contract) if not market.empty else {
        "realized_periods": 0,
        "negative_periods": 0,
        "drawdown_episodes": 0,
    }
    add_check(
        report,
        "realized_five_session_periods",
        diagnostics["realized_periods"] >= int(acceptance["minimum_realized_five_session_periods"]),
        f"observed={diagnostics['realized_periods']}",
    )
    add_check(
        report,
        "negative_soxx_periods",
        diagnostics["negative_periods"] >= int(acceptance["minimum_negative_soxx_periods"]),
        f"observed={diagnostics['negative_periods']}",
    )
    add_check(
        report,
        "bear_drawdown_episodes",
        diagnostics["drawdown_episodes"] >= int(acceptance["minimum_distinct_twenty_percent_drawdown_episodes"]),
        f"observed={diagnostics['drawdown_episodes']}",
    )

    source_checks = [item for item in report["checks"] if item["blocking"] and item["name"] not in {
        "history_years",
        "market_sessions",
        "pre_oos_warmup",
        "realized_five_session_periods",
        "negative_soxx_periods",
        "bear_drawdown_episodes",
    }]
    adequacy_names = {
        "history_years",
        "market_sessions",
        "pre_oos_warmup",
        "realized_five_session_periods",
        "negative_soxx_periods",
        "bear_drawdown_episodes",
    }
    adequacy_checks = [item for item in report["checks"] if item["name"] in adequacy_names]
    report["source_strict_ready"] = bool(source_checks and all(item["passed"] for item in source_checks))
    report["research_sample_adequate"] = bool(adequacy_checks and all(item["passed"] for item in adequacy_checks))
    report["v4_unchanged_rerun_ready"] = bool(report["source_strict_ready"] and report["research_sample_adequate"])
    report["metrics"] = {
        "stock_rows": int(len(stocks)),
        "stock_symbols": int(len(stock_symbols)),
        "market_rows": int(len(market)),
        "market_sessions": int(market_sessions),
        "history_years": float(history_years),
        "membership_rows": int(len(membership)),
        "delisted_symbols": int(len(delisted)),
        "borrow_rows": int(len(borrow)),
        "borrow_missing_fraction": float(missing_fraction),
        "dividend_rows": int(len(dividends)),
        "corporate_action_rows": int(len(actions)),
        **diagnostics,
    }
    return report


def init_bundle(stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    raw = stage / "raw"
    canonical = stage / "canonical"
    raw.mkdir()
    canonical.mkdir()
    for name, columns in TABLE_COLUMNS.items():
        path = canonical / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    provenance = {
        "bundle_id": "replace-with-licensed-export-id",
        "created_at_utc": utc_now(),
        "price_adjustment": "split_adjusted_dividend_unadjusted",
        "coverage": {field: False for field in load_json(CONTRACT_PATH)["required_source_attestations"]},
        "sources": [],
        "notes": [
            "Place unmodified original vendor exports below raw/, then run register-raw exactly once.",
            "Write provider-normalized copies only below canonical/; never edit raw files.",
            "Set a coverage field true only when the licensed source or export procedure supports it.",
            "Use a permanent research symbol, not a ticker, in every canonical table.",
        ],
    }
    (stage / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def create_lock() -> dict:
    import pandas
    import scipy
    import sklearn

    manifest = {
        "locked_at_utc": utc_now(),
        "hypothesis_locked_at": load_json(SPEC_PATH)["hypothesis_locked_at"],
        "purpose": "Verify that V4 and its shared feature pipeline are unchanged before the new-data rerun.",
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in CORE_LOCK_FILES.items()},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    LOCK_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_lock() -> dict:
    if not LOCK_PATH.exists():
        return {"valid": False, "reason": "frozen_hypothesis_manifest.json is missing", "files": {}}
    manifest = load_json(LOCK_PATH)
    files = {}
    for name, item in manifest.get("files", {}).items():
        path = Path(item["path"])
        observed = sha256(path) if path.exists() else None
        files[name] = {
            "path": str(path),
            "expected": item["sha256"],
            "observed": observed,
            "matches": observed == item["sha256"],
        }
    return {"valid": bool(files and all(item["matches"] for item in files.values())), "files": files}


def materialize_delisting_tails(
    stocks: pd.DataFrame,
    market: pd.DataFrame,
    membership: pd.DataFrame,
    actions: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Carry documented delisting recovery through the fixed label horizon."""
    stock = stocks.copy()
    member = membership.copy()
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True, errors="raise")
    stock["date"] = stock["timestamp"].dt.date
    market_dates = sorted(pd.to_datetime(market["timestamp"], utc=True, errors="raise").dt.date.unique())
    market_index = {date: index for index, date in enumerate(market_dates)}
    action = actions.copy()
    action["date"] = pd.to_datetime(action["date"], errors="raise").dt.date
    action["action_type"] = action["action_type"].astype(str).str.lower()
    action = action[action["action_type"].isin({"delisting_return", "delisting_cash"})]
    synthetic = []
    audit = []

    for row in action.sort_values(["symbol", "date"]).itertuples(index=False):
        one = stock[stock["symbol"].eq(row.symbol) & stock["date"].le(row.date)].sort_values("date")
        if one.empty:
            raise ValueError(f"No last traded bar for delisted security {row.symbol} on {row.date}")
        last = one.iloc[-1]
        last_date = last["date"]
        if last_date not in market_index:
            raise ValueError(f"Last traded date {last_date} for {row.symbol} is not a market session")
        value = float(row.value)
        terminal = value if row.action_type == "delisting_cash" else float(last["close"]) * (1.0 + value)
        if not np.isfinite(terminal) or terminal < 0:
            raise ValueError(f"Invalid delisting terminal value for {row.symbol}: {terminal}")
        start = market_index[last_date] + 1
        tail_dates = market_dates[start : start + horizon + 1]
        existing = set(stock.loc[stock["symbol"].eq(row.symbol), "date"])
        added = 0
        for date in tail_dates:
            if date in existing:
                continue
            synthetic.append(
                {
                    "timestamp": pd.Timestamp(date, tz="UTC"),
                    "symbol": row.symbol,
                    "open": terminal,
                    "high": terminal,
                    "low": terminal,
                    "close": terminal,
                    "volume": 0.0,
                    "trade_count": 0.0,
                    "vwap": terminal,
                    "date": date,
                }
            )
            added += 1
        previous_index = max(market_index[last_date] - 1, 0)
        last_signal_eligible = market_dates[previous_index]
        delisted_mask = member["symbol"].eq(row.symbol) & normalized_bool(member["delisted"])
        for index in member.index[delisted_mask]:
            current = pd.to_datetime(member.at[index, "valid_to"], errors="coerce")
            if pd.isna(current) or current.date() > last_signal_eligible:
                member.at[index, "valid_to"] = str(last_signal_eligible)
        audit.append(
            {
                "symbol": row.symbol,
                "action_date": str(row.date),
                "last_traded_date": str(last_date),
                "last_signal_eligible_date": str(last_signal_eligible),
                "terminal_value": terminal,
                "synthetic_tail_rows": added,
            }
        )
    if synthetic:
        stock = pd.concat([stock, pd.DataFrame(synthetic)], ignore_index=True)
    stock = stock.drop(columns=["date"], errors="ignore").sort_values(["symbol", "timestamp"])
    return stock, member, audit


def normalized_tables(stage: Path, spec: dict) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    canonical = canonical_directory(stage)
    tables = {name: pd.read_csv(canonical / name, keep_default_na=False) for name in TABLE_COLUMNS}
    stock, membership, delisting_audit = materialize_delisting_tails(
        tables["stock_bars.csv"],
        tables["market_bars.csv"],
        tables["universe_membership.csv"],
        tables["corporate_actions.csv"],
        int(spec["execution"]["horizon_sessions"]),
    )
    tables["stock_bars.csv"] = stock[TABLE_COLUMNS["stock_bars.csv"]]
    tables["universe_membership.csv"] = membership[TABLE_COLUMNS["universe_membership.csv"]]
    return tables, delisting_audit


def write_normalized(stage: Path, output: Path, report: dict, replace: bool = False) -> dict:
    if output.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --replace explicitly")
    spec = load_json(SPEC_PATH)
    tables, delisting_audit = normalized_tables(stage, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v4-data-", dir=output.parent) as temporary:
        temp = Path(temporary)
        for name in INSTALL_TABLES:
            tables[name].to_csv(temp / name, index=False)
        source_path = provenance_path(stage)
        if source_path is None:
            raise ValueError("Cannot normalize without provenance.json")
        loaded = load_json(source_path)
        provenance = loaded.get("provenance", loaded)
        raw_check = verify_raw_exports(stage)
        if raw_check["valid"]:
            raw_manifest_path = stage / RAW_MANIFEST_NAME
            raw_manifest = load_json(raw_manifest_path)
            raw_manifest_sha = sha256(raw_manifest_path)
        else:
            preserved = loaded.get("raw_export", {})
            raw_manifest = preserved.get("manifest")
            raw_manifest_sha = preserved.get("manifest_sha256")
        manifest = {
            "created_at_utc": utc_now(),
            "mode": "strict_historical_bundle",
            "confirmatory_eligible": bool(report["v4_unchanged_rerun_ready"]),
            "provenance": provenance,
            "raw_export": {
                "verified_at_normalization": bool(raw_check["valid"] or loaded.get("raw_export", {}).get("verified_at_normalization")),
                "manifest_sha256": raw_manifest_sha,
                "manifest_content_sha256": json_sha256(raw_manifest) if raw_manifest else None,
                "manifest": raw_manifest,
            },
            "data_contract_sha256": sha256(CONTRACT_PATH),
            "frozen_hypothesis_lock": verify_lock(),
            "audit": report,
            "delisting_materialization": delisting_audit,
            "table_hashes": {name: sha256(temp / name) for name in INSTALL_TABLES},
        }
        (temp / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(temp, output)
    return {"output": str(output), "rows": {name: int(len(tables[name])) for name in INSTALL_TABLES}}


def install_bundle(stage: Path, confirmation: str) -> dict:
    if confirmation != INSTALL_CONFIRMATION:
        raise ValueError(f"Installation requires --confirm {INSTALL_CONFIRMATION}")
    lock = verify_lock()
    if not lock["valid"]:
        raise ValueError("Frozen V4 lock does not match; refusing installation")
    report = audit_bundle(stage)
    if not report["v4_unchanged_rerun_ready"]:
        raise ValueError("Historical bundle does not pass the locked data contract")
    backup = ROOT / "data_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DATA_DIR, backup)
    with tempfile.TemporaryDirectory(prefix="v4-install-", dir=ROOT) as temporary:
        normalized = Path(temporary) / "data"
        write_normalized(stage, normalized, report)
        for name in INSTALL_TABLES + ["source_manifest.json"]:
            os.replace(normalized / name, DATA_DIR / name)
    return {"installed": True, "backup": str(backup), "data_dir": str(DATA_DIR)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create an empty canonical staging bundle")
    init.add_argument("--stage", required=True)

    register = commands.add_parser("register-raw", help="Fingerprint immutable original vendor exports")
    register.add_argument("--stage", required=True)
    register.add_argument("--vendor", required=True)
    register.add_argument("--version", required=True)
    register.add_argument("--as-of-date", required=True)
    register.add_argument("--license-reference", required=True)

    verify_raw = commands.add_parser("verify-raw", help="Verify exact raw-export files against their fingerprint")
    verify_raw.add_argument("--stage", required=True)

    audit = commands.add_parser("audit", help="Audit a staging bundle without running V4")
    audit.add_argument("--stage", default=str(DATA_DIR))
    audit.add_argument("--output", default=str(ARTIFACTS_DIR / "data_readiness.json"))

    commands.add_parser("lock", help="Record hashes of the frozen V4 hypothesis")
    commands.add_parser("verify-lock", help="Verify frozen V4 hashes")

    normalize = commands.add_parser("normalize", help="Materialize canonical V4 files outside the active data directory")
    normalize.add_argument("--stage", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--replace", action="store_true")
    normalize.add_argument("--allow-incomplete", action="store_true")

    install = commands.add_parser("install", help="Install a passing bundle after backing up exploratory data")
    install.add_argument("--stage", required=True)
    install.add_argument("--confirm", default="")

    args = parser.parse_args()
    if args.command == "init":
        init_bundle(Path(args.stage))
        result = {"created": str(Path(args.stage).resolve())}
    elif args.command == "register-raw":
        result = register_raw_exports(
            Path(args.stage), args.vendor, args.version, args.as_of_date, args.license_reference
        )
    elif args.command == "verify-raw":
        result = verify_raw_exports(Path(args.stage))
    elif args.command == "audit":
        result = audit_bundle(Path(args.stage))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "lock":
        result = create_lock()
    elif args.command == "verify-lock":
        result = verify_lock()
    elif args.command == "normalize":
        report = audit_bundle(Path(args.stage))
        if not report["v4_unchanged_rerun_ready"] and not args.allow_incomplete:
            raise ValueError("Bundle is not ready; use audit for blockers")
        result = write_normalized(Path(args.stage), Path(args.output), report, replace=args.replace)
    else:
        result = install_bundle(Path(args.stage), args.confirm)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
