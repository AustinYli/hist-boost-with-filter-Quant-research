#!/usr/bin/env python3
"""Prepare or validate isolated V3 research inputs without touching trading data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REQUIRED_BAR_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]
MEMBERSHIP_COLUMNS = [
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
]
BORROW_COLUMNS = ["date", "symbol", "shortable", "annual_borrow_rate", "source"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing bar columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")


def seed_existing(args: argparse.Namespace) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_stock = Path(args.source_stock).resolve()
    source_market = Path(args.source_market).resolve()
    stock = load_bars(source_stock)
    market = load_bars(source_market)
    stock_out = DATA_DIR / "stock_bars.csv"
    market_out = DATA_DIR / "market_bars.csv"
    shutil.copy2(source_stock, stock_out)
    shutil.copy2(source_market, market_out)

    membership_rows = []
    for symbol, one in stock.groupby("symbol"):
        membership_rows.append(
            {
                "symbol": symbol,
                "valid_from": one["timestamp"].min().date(),
                "valid_to": "",
                "industry": "Semiconductors & Semiconductor Equipment",
                "security_type": "common_equity",
                "us_listed": True,
                "survivorship_safe": False,
                "delisted": False,
                "delisting_handled": False,
                "source": "current-universe exploratory seed",
            }
        )
    pd.DataFrame(membership_rows, columns=MEMBERSHIP_COLUMNS).to_csv(
        DATA_DIR / "universe_membership.csv", index=False
    )
    pd.DataFrame(columns=BORROW_COLUMNS).to_csv(DATA_DIR / "borrow_costs.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "exploratory_current_universe_seed",
        "confirmatory_eligible": False,
        "source_stock": str(source_stock),
        "source_market": str(source_market),
        "source_hashes": {"stock": sha256(source_stock), "market": sha256(source_market)},
        "stock_rows": int(len(stock)),
        "stock_symbols": int(stock["symbol"].nunique()),
        "stock_start": str(stock["timestamp"].min().date()),
        "stock_end": str(stock["timestamp"].max().date()),
        "market_rows": int(len(market)),
        "market_symbols": sorted(market["symbol"].unique().tolist()),
        "market_start": str(market["timestamp"].min().date()),
        "market_end": str(market["timestamp"].max().date()),
        "known_limitations": [
            "current constituents reconstructed backward",
            "no delisted equities",
            "no point-in-time classifications",
            "no historical borrow availability or fees",
            "Alpaca IEX history is shorter than the requested 2001 start",
        ],
    }
    (DATA_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate(_args: argparse.Namespace) -> dict:
    stock_path = DATA_DIR / "stock_bars.csv"
    market_path = DATA_DIR / "market_bars.csv"
    membership_path = DATA_DIR / "universe_membership.csv"
    borrow_path = DATA_DIR / "borrow_costs.csv"
    stock = load_bars(stock_path)
    market = load_bars(market_path)
    membership = pd.read_csv(membership_path, keep_default_na=False)
    borrow = pd.read_csv(borrow_path, keep_default_na=False)
    missing_membership = set(MEMBERSHIP_COLUMNS) - set(membership.columns)
    missing_borrow = set(BORROW_COLUMNS) - set(borrow.columns)
    if missing_membership:
        raise ValueError(f"Membership file is missing: {sorted(missing_membership)}")
    if missing_borrow:
        raise ValueError(f"Borrow file is missing: {sorted(missing_borrow)}")
    required_market = {"SOXX", "SPY", "QQQ"}
    missing_market = required_market - set(market["symbol"])
    if missing_market:
        raise ValueError(f"Market bars are missing: {sorted(missing_market)}")
    safe = membership["survivorship_safe"].astype(str).str.lower().eq("true").all()
    delist = membership["delisting_handled"].astype(str).str.lower().eq("true").all()
    report = {
        "stock_rows": int(len(stock)),
        "stock_symbols": int(stock["symbol"].nunique()),
        "stock_start": str(stock["timestamp"].min().date()),
        "stock_end": str(stock["timestamp"].max().date()),
        "market_symbols": sorted(market["symbol"].unique().tolist()),
        "point_in_time_universe": bool(safe),
        "delistings_handled": bool(delist),
        "borrow_rows": int(len(borrow)),
        "strict_ready": bool(safe and delist and not borrow.empty),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("seed-existing")
    seed.add_argument("--source-stock", default="data/semiconductor_sector_1d_5y.csv")
    seed.add_argument("--source-market", default="data/semiconductor_market_1d_5y.csv")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    result = seed_existing(args) if args.command == "seed-existing" else validate(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
