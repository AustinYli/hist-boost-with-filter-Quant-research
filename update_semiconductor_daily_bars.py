#!/usr/bin/env python3
"""Update local semiconductor daily OHLCV files from Alpaca data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE
from trailing_stop_bot import AlpacaClient, AlpacaCredentials, load_dotenv


MARKET_SYMBOLS = ["SPY", "QQQ", "SOXX"]
COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"]


def normalize_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["symbol"] = frame["symbol"].str.upper()
    return frame


def merge_rows(path: Path, rows: list[dict[str, str]]) -> pd.DataFrame:
    incoming = normalize_frame(rows)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def fetch_rows(client: AlpacaClient, symbols: list[str], args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for start_index in range(0, len(symbols), args.batch_size):
        batch = symbols[start_index : start_index + args.batch_size]
        rows.extend(
            client.stock_bar_rows_multi(
                batch,
                timeframe="1Day",
                feed=args.feed,
                start=args.start,
                end=args.end,
                limit=args.limit,
            )
        )
    return rows


def latest_dates(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {}
    return frame.groupby("symbol")["timestamp"].max().to_dict()


def earliest_dates(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {}
    return frame.groupby("symbol")["timestamp"].min().to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Update semiconductor daily bars from Alpaca.")
    parser.add_argument("--stock-out", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-out", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--start", default="2026-07-31T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stock-symbols", default=None)
    parser.add_argument("--market-symbols", default=",".join(MARKET_SYMBOLS))
    args = parser.parse_args()

    load_dotenv()
    client = AlpacaClient(AlpacaCredentials.from_env())
    stock_symbols = (
        [symbol.strip().upper() for symbol in args.stock_symbols.split(",") if symbol.strip()]
        if args.stock_symbols
        else SEMICONDUCTOR_UNIVERSE
    )
    market_symbols = [symbol.strip().upper() for symbol in args.market_symbols.split(",") if symbol.strip()]
    stock_rows = fetch_rows(client, stock_symbols, args)
    market_rows = fetch_rows(client, market_symbols, args)
    stock_incoming = normalize_frame(stock_rows)
    market_incoming = normalize_frame(market_rows)
    if args.dry_run:
        stock_file_rows = None
        market_file_rows = None
    else:
        stock_frame = merge_rows(Path(args.stock_out), stock_rows)
        market_frame = merge_rows(Path(args.market_out), market_rows)
        stock_file_rows = len(stock_frame)
        market_file_rows = len(market_frame)

    print(
        {
            "stock_rows_fetched": len(stock_rows),
            "market_rows_fetched": len(market_rows),
            "stock_earliest_dates": earliest_dates(stock_incoming),
            "stock_latest_dates": latest_dates(stock_incoming),
            "market_earliest_dates": earliest_dates(market_incoming),
            "market_latest_dates": latest_dates(market_incoming),
            "stock_file_rows": stock_file_rows,
            "market_file_rows": market_file_rows,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
