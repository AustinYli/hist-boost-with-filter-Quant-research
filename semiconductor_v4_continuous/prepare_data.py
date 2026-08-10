#!/usr/bin/env python3
"""Prepare isolated V4 research inputs without touching trading or V3 files."""

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
BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
MEMBERSHIP_COLUMNS = [
    "symbol", "valid_from", "valid_to", "industry", "security_type", "us_listed",
    "survivorship_safe", "delisted", "delisting_handled", "source",
]
BORROW_COLUMNS = ["date", "symbol", "shortable", "annual_borrow_rate", "source"]
DIVIDEND_COLUMNS = ["date", "symbol", "cash_amount", "source"]
ACTION_COLUMNS = ["date", "symbol", "action_type", "value", "handled", "source"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")


def seed_existing(args: argparse.Namespace) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stock_source = Path(args.source_stock).resolve()
    market_source = Path(args.source_market).resolve()
    stock = load_bars(stock_source)
    market = load_bars(market_source)
    shutil.copy2(stock_source, DATA_DIR / "stock_bars.csv")
    shutil.copy2(market_source, DATA_DIR / "market_bars.csv")
    membership = []
    for symbol, one in stock.groupby("symbol"):
        membership.append(
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
    pd.DataFrame(membership, columns=MEMBERSHIP_COLUMNS).to_csv(DATA_DIR / "universe_membership.csv", index=False)
    pd.DataFrame(columns=BORROW_COLUMNS).to_csv(DATA_DIR / "borrow_costs.csv", index=False)
    pd.DataFrame(columns=DIVIDEND_COLUMNS).to_csv(DATA_DIR / "dividends.csv", index=False)
    pd.DataFrame(columns=ACTION_COLUMNS).to_csv(DATA_DIR / "corporate_actions.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "exploratory_current_universe_seed",
        "confirmatory_eligible": False,
        "source_hashes": {"stock": sha256(stock_source), "market": sha256(market_source)},
        "stock_rows": int(len(stock)),
        "stock_symbols": int(stock["symbol"].nunique()),
        "stock_start": str(stock["timestamp"].min().date()),
        "stock_end": str(stock["timestamp"].max().date()),
        "market_start": str(market["timestamp"].min().date()),
        "market_end": str(market["timestamp"].max().date()),
        "limitations": [
            "current constituents reconstructed backward",
            "no delisted equities or point-in-time classifications",
            "no historical borrow tape",
            "no explicit dividend or corporate-action ledger",
            "history starts in 2021 rather than the requested early regimes",
        ],
    }
    (DATA_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate(_args: argparse.Namespace) -> dict:
    stock = load_bars(DATA_DIR / "stock_bars.csv")
    market = load_bars(DATA_DIR / "market_bars.csv")
    membership = pd.read_csv(DATA_DIR / "universe_membership.csv", keep_default_na=False)
    borrow = pd.read_csv(DATA_DIR / "borrow_costs.csv", keep_default_na=False)
    dividends = pd.read_csv(DATA_DIR / "dividends.csv", keep_default_na=False)
    actions = pd.read_csv(DATA_DIR / "corporate_actions.csv", keep_default_na=False)
    point_in_time = membership["survivorship_safe"].astype(str).str.lower().eq("true").all()
    delistings = membership["delisting_handled"].astype(str).str.lower().eq("true").all()
    return {
        "stock_rows": int(len(stock)),
        "stock_symbols": int(stock["symbol"].nunique()),
        "market_symbols": sorted(market["symbol"].unique().tolist()),
        "point_in_time_universe": bool(point_in_time),
        "delistings_handled": bool(delistings),
        "borrow_rows": int(len(borrow)),
        "dividend_rows": int(len(dividends)),
        "corporate_action_rows": int(len(actions)),
        "strict_ready": bool(point_in_time and delistings and not borrow.empty and not dividends.empty and not actions.empty),
    }


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
