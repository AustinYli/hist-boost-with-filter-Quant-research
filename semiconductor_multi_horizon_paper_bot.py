#!/usr/bin/env python3
"""Executor for parallel semiconductor strategy horizons.

Alpaca has one net position book, so this runner combines multiple strategy
sleeves into one target portfolio while tracking each sleeve's virtual equity
and PnL separately. Paper trading is the default; live trading requires
explicit live credentials and a live-order confirmation flag.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import pandas as pd

from ml_semiconductor_forward_test import (
    available_signal_dates,
    fixed_model_dir_for_horizon,
    load_feature_frame,
    make_forward_signal,
    read_jsonl,
)
from ml_semiconductor_walkforward_compare import SEMICONDUCTOR_UNIVERSE
from trailing_stop_bot import (
    ALPACA_LIVE_TRADING_URL,
    ALPACA_PAPER_TRADING_URL,
    AlpacaClient,
    AlpacaCredentials,
    BotError,
    decimal_to_api_string,
    load_dotenv,
)
from update_semiconductor_daily_bars import MARKET_SYMBOLS, fetch_rows, merge_rows


PAPER_RUN_LOG = Path("data/semiconductor_multi_horizon_paper_runs.jsonl")
PAPER_STATE_PATH = Path("data/semiconductor_multi_horizon_paper_state.json")
PAPER_LOG_PATH = Path("data/semiconductor_multi_horizon_paper_bot.log")
LIVE_RUN_LOG = Path("data/semiconductor_multi_horizon_live_runs.jsonl")
LIVE_STATE_PATH = Path("data/semiconductor_multi_horizon_live_state.json")
LIVE_LOG_PATH = Path("data/semiconductor_multi_horizon_live_bot.log")
RUN_LOG = PAPER_RUN_LOG
STATE_PATH = PAPER_STATE_PATH
LOG_PATH = PAPER_LOG_PATH
DEFAULT_SLEEVES = "5:0.5,3:0.5"
SIGNAL_MODE_EXPANDING = "expanding"
SIGNAL_MODE_FIXED = "fixed"
NEW_YORK_TZ = "America/New_York"
LIVE_ORDER_CONFIRMATION = "LIVE_ORDERS_ENABLED"


def configure_runtime_paths(live_trading: bool) -> None:
    global RUN_LOG, STATE_PATH, LOG_PATH
    if live_trading:
        RUN_LOG = LIVE_RUN_LOG
        STATE_PATH = LIVE_STATE_PATH
        LOG_PATH = LIVE_LOG_PATH
    else:
        RUN_LOG = PAPER_RUN_LOG
        STATE_PATH = PAPER_STATE_PATH
        LOG_PATH = PAPER_LOG_PATH


def account_mode(args: argparse.Namespace) -> str:
    return "live" if getattr(args, "live_trading", False) else "paper"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def latest_csv_date(path: Path):
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["timestamp"])
    if frame.empty:
        return None
    return pd.to_datetime(frame["timestamp"], utc=True).dt.date.max()


def closed_daily_bar_end(clock) -> str:
    timestamp = pd.Timestamp(clock.timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    local = timestamp.tz_convert(NEW_YORK_TZ)
    end_date = local.date()
    if not clock.is_open and local.weekday() < 5 and local.hour >= 17:
        end_date = (local + pd.Timedelta(days=1)).date()
    return f"{end_date.isoformat()}T00:00:00Z"


def maybe_update_daily_bars(args: argparse.Namespace, client: AlpacaClient, clock, state: dict) -> None:
    if not args.auto_update_bars:
        return
    end = closed_daily_bar_end(clock)
    if state.get("last_daily_bar_update_end") == end:
        return

    latest_stock = latest_csv_date(Path(args.bars))
    latest_market = latest_csv_date(Path(args.market_bars))
    existing_dates = [date for date in [latest_stock, latest_market] if date is not None]
    if not existing_dates:
        log("Skipping daily bar auto-update because no existing local daily-bar start date was found.")
        return
    latest_existing = min(existing_dates)
    start = f"{latest_existing.isoformat()}T00:00:00Z"
    update_args = argparse.Namespace(
        feed=args.feed,
        start=start,
        end=end,
        limit=args.bar_update_limit,
        batch_size=args.bar_update_batch_size,
    )
    stock_rows = fetch_rows(client, SEMICONDUCTOR_UNIVERSE, update_args)
    market_rows = fetch_rows(client, MARKET_SYMBOLS, update_args)
    stock_frame = merge_rows(Path(args.bars), stock_rows)
    market_frame = merge_rows(Path(args.market_bars), market_rows)
    state["last_daily_bar_update_end"] = end
    state["last_daily_bar_update_at_utc"] = utc_now()
    log(
        "Updated daily bars "
        f"start={start} end={end} stock_rows={len(stock_rows)} market_rows={len(market_rows)} "
        f"stock_file_rows={len(stock_frame)} market_file_rows={len(market_frame)}."
    )


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def decimal_field(data: dict, name: str, default: str = "0") -> Decimal:
    return Decimal(str(data.get(name, default)))


def account_snapshot(account: dict) -> dict:
    return {
        "equity": str(account.get("equity")),
        "portfolio_value": str(account.get("portfolio_value")),
        "cash": str(account.get("cash")),
        "buying_power": str(account.get("buying_power")),
        "last_equity": str(account.get("last_equity")),
    }


def quantize_qty(qty: Decimal) -> Decimal:
    return qty.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)


def position_qty(position: dict) -> Decimal:
    return decimal_field(position, "qty")


def position_value(position: dict) -> Decimal:
    return decimal_field(position, "market_value")


def parse_sleeves(raw: str) -> list[dict]:
    sleeves = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        horizon, fraction = item.split(":", 1)
        sleeves.append({"horizon_days": int(horizon), "capital_fraction": float(fraction)})
    total = sum(sleeve["capital_fraction"] for sleeve in sleeves)
    if not sleeves or total <= 0 or total > 1.0 + 1e-9:
        raise ValueError("Sleeve fractions must be positive and sum to 1.0 or less.")
    return sleeves


def signal_log_for_horizon(horizon_days: int, signal_mode: str) -> Path:
    if signal_mode == SIGNAL_MODE_EXPANDING:
        if horizon_days == 5:
            return Path("models/semiconductor_active_expanding/forward_signals.jsonl")
        return Path(f"models/semiconductor_active_expanding_{horizon_days}d/forward_signals.jsonl")
    if horizon_days == 5:
        return Path("models/semiconductor_forward_test/forward_signals.jsonl")
    return Path(f"models/semiconductor_forward_test_{horizon_days}d/forward_signals.jsonl")


def evaluation_log_for_horizon(horizon_days: int, signal_mode: str) -> Path:
    return signal_log_for_horizon(horizon_days, signal_mode).parent / "forward_evaluations.jsonl"


def is_fixed_weight_signal(record: dict) -> bool:
    return record.get("model", {}).get("training_mode") == "fixed_weight"


def is_expanding_signal(record: dict) -> bool:
    return record.get("model", {}).get("training_mode") == "expanding_window"


def latest_available_date(args: argparse.Namespace, horizon_days: int):
    data, _, _ = load_feature_frame(Path(args.bars), Path(args.market_bars), horizon_days)
    dates = available_signal_dates(data)
    if not dates:
        raise BotError(f"No available signal dates for {horizon_days}d sleeve.")
    return dates[-1], dates


def should_retrain_expanding_signal(
    args: argparse.Namespace,
    horizon_days: int,
    records: list[dict],
    sleeve_state: dict | None = None,
) -> tuple[bool, object]:
    latest_date, dates = latest_available_date(args, horizon_days)
    if getattr(args, "force_latest_signals", False):
        return True, latest_date
    state_date = (sleeve_state or {}).get("last_data_asof_date")
    if state_date:
        try:
            state_index = dates.index(pd.Timestamp(state_date).date())
            latest_index = dates.index(latest_date)
        except ValueError:
            return True, latest_date
        return latest_index - state_index >= horizon_days, latest_date
    if getattr(args, "latest_signal_on_new_sleeve", True):
        return True, latest_date
    if not records:
        return True, latest_date
    latest_record = records[-1]
    if not is_expanding_signal(latest_record):
        return True, latest_date
    latest_record_date = latest_record.get("data_asof_date")
    if not latest_record_date:
        return True, latest_date
    try:
        last_index = dates.index(pd.Timestamp(latest_record_date).date())
        latest_index = dates.index(latest_date)
    except ValueError:
        return True, latest_date
    return latest_index - last_index >= horizon_days, latest_date


def record_for_signal_date(records: list[dict], signal_date) -> dict | None:
    normalized = str(pd.Timestamp(signal_date).date())
    for record in reversed(records):
        if record.get("data_asof_date") == normalized:
            return record
    return None


def record_for_created_at(records: list[dict], created_at_utc: str | None) -> dict | None:
    if not created_at_utc:
        return None
    for record in reversed(records):
        if record.get("created_at_utc") == created_at_utc:
            return record
    return None


def create_signal_for_date(
    args: argparse.Namespace,
    horizon_days: int,
    signal_date,
    signal_log: Path,
    model_dir: Path | None,
) -> dict:
    signal_args = argparse.Namespace(
        bars=args.bars,
        market_bars=args.market_bars,
        horizon_days=horizon_days,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        signal_date=signal_date,
        model_dir=str(model_dir) if model_dir else None,
        signal_log=str(signal_log),
        evaluation_log=str(evaluation_log_for_horizon(horizon_days, args.signal_mode)),
    )
    return make_forward_signal(signal_args)


def latest_or_create_signal(args: argparse.Namespace, horizon_days: int, state: dict | None = None) -> dict:
    signal_log = signal_log_for_horizon(horizon_days, args.signal_mode)
    records = read_jsonl(signal_log)

    if args.signal_mode == SIGNAL_MODE_EXPANDING:
        sleeve_state = (state or {}).get("sleeves", {}).get(str(horizon_days), {})
        needs_retrain, signal_date = should_retrain_expanding_signal(args, horizon_days, records, sleeve_state)
        if records and not needs_retrain:
            executed_id = (state or {}).get("last_executed_signal_ids", {}).get(str(horizon_days))
            account_signal = record_for_created_at(records, executed_id)
            if account_signal is None:
                account_signal = record_for_created_at(records, sleeve_state.get("last_signal_created_at_utc"))
            return account_signal or records[-1]
        existing = record_for_signal_date(records, signal_date)
        if existing is not None:
            return existing
        if records:
            log(f"Retraining {horizon_days}d expanding signal using data through {signal_date}.")
        else:
            log(f"Creating first {horizon_days}d expanding signal using data through {signal_date}.")
        model_dir = None
    else:
        if records and is_fixed_weight_signal(records[-1]):
            return records[-1]
        if records:
            log(f"Latest {horizon_days}d signal is not fixed-weight; generating fixed-weight replacement.")
        else:
            log(f"No {horizon_days}d signal found; generating fixed-weight signal from local daily data.")
        model_dir = fixed_model_dir_for_horizon(horizon_days)
        if not model_dir.exists():
            raise BotError(f"Missing fixed-weight model for {horizon_days}d sleeve: {model_dir}")
        signal_date = args.signal_date

    return create_signal_for_date(args, horizon_days, signal_date, signal_log, model_dir)


def ensure_pending_signal_is_fresh(
    args: argparse.Namespace,
    horizon_days: int,
    state: dict,
    signal_record: dict,
) -> tuple[dict, str]:
    """Never allow an unexecuted expanding signal to trade after a newer close exists."""
    latest_date, _ = latest_available_date(args, horizon_days)
    latest_date_text = str(pd.Timestamp(latest_date).date())
    executed_id = state.get("last_executed_signal_ids", {}).get(str(horizon_days))
    signal_id = signal_record.get("created_at_utc")
    signal_date = signal_record.get("data_asof_date")

    if signal_id == executed_id or signal_date == latest_date_text:
        return signal_record, latest_date_text
    if args.signal_mode != SIGNAL_MODE_EXPANDING:
        raise BotError(
            f"Refusing stale {horizon_days}d fixed signal dated {signal_date}; "
            f"latest completed-session features are dated {latest_date_text}."
        )

    signal_log = signal_log_for_horizon(horizon_days, args.signal_mode)
    records = read_jsonl(signal_log)
    fresh_record = record_for_signal_date(records, latest_date)
    if fresh_record is None or not is_expanding_signal(fresh_record):
        log(
            f"Replacing stale pending {horizon_days}d signal dated {signal_date} "
            f"with latest completed-session features dated {latest_date_text}."
        )
        fresh_record = create_signal_for_date(args, horizon_days, latest_date, signal_log, None)
    return fresh_record, latest_date_text


def price_lookup(client: AlpacaClient, cache: dict[str, Decimal], symbol: str, feed: str) -> Decimal:
    if symbol not in cache:
        cache[symbol] = client.latest_trade_price(symbol, feed=feed)
    return cache[symbol]


def combined_target_weights(sleeve_records: list[dict]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for sleeve in sleeve_records:
        fraction = float(sleeve["capital_fraction"])
        for symbol, weight in sleeve["signal"].get("intended_weights", {}).items():
            combined[symbol] = combined.get(symbol, 0.0) + fraction * float(weight)
    return {symbol: weight for symbol, weight in combined.items() if weight > 1e-12}


def build_orders(
    *,
    client: AlpacaClient,
    target_weights: dict[str, float],
    account: dict,
    min_trade_dollars: Decimal,
    cash_reserve_dollars: Decimal,
    prevent_margin: bool,
    feed: str,
    price_cache: dict[str, Decimal],
) -> tuple[list[dict], list[dict]]:
    portfolio_value = decimal_field(account, "portfolio_value")
    cash = decimal_field(account, "cash")
    positions = client.positions()
    universe = set(SEMICONDUCTOR_UNIVERSE)
    symbols = sorted((set(target_weights) | (set(positions) & universe)))
    sells: list[dict] = []
    buy_candidates: list[dict] = []
    estimated_sell_proceeds = Decimal("0")

    for symbol in symbols:
        target_value = portfolio_value * Decimal(str(target_weights.get(symbol, 0.0)))
        current_position = positions.get(symbol, {})
        current_value = position_value(current_position) if current_position else Decimal("0")
        diff = target_value - current_value
        if abs(diff) < min_trade_dollars:
            continue

        price = price_lookup(client, price_cache, symbol, feed)
        if price <= 0:
            continue

        if diff < 0:
            held_qty = position_qty(current_position)
            qty = min(held_qty, quantize_qty(abs(diff) / price))
            if qty > 0:
                sells.append({"symbol": symbol, "qty": qty, "dollars": str(abs(diff)), "price": str(price)})
                estimated_sell_proceeds += qty * price
        else:
            qty = quantize_qty(diff / price)
            if qty > 0:
                buy_candidates.append({"symbol": symbol, "qty": qty, "dollars": str(diff), "price": str(price)})

    if not prevent_margin:
        return sells, buy_candidates

    available_cash = max(Decimal("0"), cash + estimated_sell_proceeds - cash_reserve_dollars)
    buys: list[dict] = []
    for order in sorted(buy_candidates, key=lambda item: item["symbol"]):
        requested_dollars = Decimal(str(order["dollars"]))
        price = Decimal(str(order["price"]))
        if available_cash < min_trade_dollars or price <= 0:
            break
        allowed_dollars = min(requested_dollars, available_cash)
        qty = quantize_qty(allowed_dollars / price)
        if qty <= 0:
            continue
        actual_dollars = qty * price
        if actual_dollars < min_trade_dollars:
            continue
        trimmed = {**order, "qty": qty, "dollars": str(actual_dollars)}
        if actual_dollars < requested_dollars:
            trimmed["cash_capped_from_dollars"] = str(requested_dollars)
        buys.append(trimmed)
        available_cash -= actual_dollars
    return sells, buys


def execute_orders(
    client: AlpacaClient,
    orders: list[dict],
    *,
    side: str,
    time_in_force: str,
    mode: str,
) -> list[dict]:
    responses = []
    for order in orders:
        symbol = order["symbol"]
        qty = Decimal(str(order["qty"]))
        if side == "sell":
            response = client.place_market_sell(symbol=symbol, qty=qty, time_in_force=time_in_force)
        else:
            response = client.place_market_buy(symbol=symbol, qty=qty, time_in_force=time_in_force)
        responses.append({"symbol": symbol, "qty": decimal_to_api_string(qty), "side": side, "response": response})
        log(f"Submitted combined {mode} {side} {decimal_to_api_string(qty)} {symbol}.")
    return responses


def ensure_sleeve_state(state: dict, sleeve_records: list[dict], account: dict) -> dict:
    sleeves = state.setdefault("sleeves", {})
    portfolio_value = decimal_field(account, "portfolio_value")
    for sleeve in sleeve_records:
        key = str(sleeve["horizon_days"])
        allocated = portfolio_value * Decimal(str(sleeve["capital_fraction"]))
        sleeves.setdefault(
            key,
            {
                "initial_equity": str(allocated),
                "cash": str(allocated),
                "positions": {},
                "last_equity": str(allocated),
                "last_pnl": "0",
                "last_pnl_pct": "0",
            },
        )
    return state


def mark_sleeves(state: dict, client: AlpacaClient, feed: str, price_cache: dict[str, Decimal]) -> dict:
    snapshots = {}
    for key, sleeve_state in state.get("sleeves", {}).items():
        cash = Decimal(str(sleeve_state.get("cash", "0")))
        equity = cash
        positions = sleeve_state.get("positions", {})
        for symbol, qty_raw in positions.items():
            qty = Decimal(str(qty_raw))
            price = price_lookup(client, price_cache, symbol, feed)
            equity += qty * price
        initial = Decimal(str(sleeve_state.get("initial_equity", equity)))
        pnl = equity - initial
        pnl_pct = pnl / initial if initial else Decimal("0")
        sleeve_state["last_equity"] = str(equity)
        sleeve_state["last_pnl"] = str(pnl)
        sleeve_state["last_pnl_pct"] = str(pnl_pct)
        snapshots[key] = {
            "equity": str(equity),
            "initial_equity": str(initial),
            "pnl": str(pnl),
            "pnl_pct": str(pnl_pct),
            "cash": str(cash),
            "positions": positions,
        }
    return snapshots


def rebalance_virtual_sleeves(
    state: dict,
    sleeve_records: list[dict],
    client: AlpacaClient,
    feed: str,
    price_cache: dict[str, Decimal],
) -> dict:
    snapshots = mark_sleeves(state, client, feed, price_cache)
    for sleeve in sleeve_records:
        key = str(sleeve["horizon_days"])
        sleeve_state = state["sleeves"][key]
        equity = Decimal(str(snapshots[key]["equity"]))
        target_weights = sleeve["signal"].get("intended_weights", {})
        target_positions = {}
        used = Decimal("0")
        for symbol, weight in target_weights.items():
            target_value = equity * Decimal(str(weight))
            price = price_lookup(client, price_cache, symbol, feed)
            qty = quantize_qty(target_value / price) if price > 0 else Decimal("0")
            if qty > 0:
                target_positions[symbol] = str(qty)
                used += qty * price
        sleeve_state["positions"] = target_positions
        sleeve_state["cash"] = str(equity - used)
        sleeve_state["last_signal_created_at_utc"] = sleeve["signal"]["created_at_utc"]
        sleeve_state["last_data_asof_date"] = sleeve["signal"].get("data_asof_date")
    return mark_sleeves(state, client, feed, price_cache)


def run_once(args: argparse.Namespace, client: AlpacaClient) -> dict:
    mode = account_mode(args)
    account = client.account()
    clock = client.market_clock()
    state = load_state()
    if args.signal_mode == SIGNAL_MODE_EXPANDING:
        maybe_update_daily_bars(args, client, clock, state)

    sleeve_records = []
    for sleeve in parse_sleeves(args.sleeves):
        signal_record = latest_or_create_signal(args, sleeve["horizon_days"], state)
        signal_record, latest_feature_date = ensure_pending_signal_is_fresh(
            args,
            sleeve["horizon_days"],
            state,
            signal_record,
        )
        sleeve_records.append(
            {
                **sleeve,
                "signal": signal_record,
                "latest_completed_feature_date": latest_feature_date,
            }
        )

    price_cache: dict[str, Decimal] = {}
    state = ensure_sleeve_state(state, sleeve_records, account)
    sleeve_marks = mark_sleeves(state, client, args.feed, price_cache)
    target_weights = combined_target_weights(sleeve_records)
    signal_ids = {str(sleeve["horizon_days"]): sleeve["signal"]["created_at_utc"] for sleeve in sleeve_records}

    record = {
        "created_at_utc": utc_now(),
        "account_mode": mode,
        "account": account_snapshot(account),
        "market_clock": clock.__dict__,
        "risk_controls": {
            "prevent_margin": bool(args.prevent_margin),
            "cash_reserve_dollars": str(args.cash_reserve_dollars),
            "live_trading": bool(args.live_trading),
        },
        "sleeves": [
            {
                "horizon_days": sleeve["horizon_days"],
                "capital_fraction": sleeve["capital_fraction"],
                "signal_created_at_utc": sleeve["signal"]["created_at_utc"],
                "data_asof_date": sleeve["signal"].get("data_asof_date"),
                "latest_completed_feature_date": sleeve["latest_completed_feature_date"],
                "target_weights": sleeve["signal"].get("intended_weights", {}),
                "pnl": sleeve_marks.get(str(sleeve["horizon_days"]), {}),
            }
            for sleeve in sleeve_records
        ],
        "combined_target_weights": target_weights,
        "orders": [],
        "status": "pending",
    }

    if not clock.is_open:
        record["status"] = "market_closed_waiting"
        log(f"Market closed. Multi-horizon bot recorded PnL only. Next open: {clock.next_open}.")
        append_jsonl(RUN_LOG, record)
        save_state(state)
        return record

    if state.get("last_executed_signal_ids") == signal_ids and not args.force:
        record["status"] = "already_executed"
        append_jsonl(RUN_LOG, record)
        save_state(state)
        return record

    sells, buys = build_orders(
        client=client,
        target_weights=target_weights,
        account=account,
        min_trade_dollars=Decimal(str(args.min_trade_dollars)),
        cash_reserve_dollars=Decimal(str(args.cash_reserve_dollars)),
        prevent_margin=bool(args.prevent_margin),
        feed=args.feed,
        price_cache=price_cache,
    )
    responses = []
    responses.extend(execute_orders(client, sells, side="sell", time_in_force=args.time_in_force, mode=mode))
    responses.extend(execute_orders(client, buys, side="buy", time_in_force=args.time_in_force, mode=mode))
    record["orders"] = responses
    record["end_account"] = account_snapshot(client.account())
    record["sleeve_pnl_after_virtual_rebalance"] = rebalance_virtual_sleeves(state, sleeve_records, client, args.feed, price_cache)
    record["status"] = "executed"
    state["last_executed_signal_ids"] = signal_ids
    state["last_executed_at_utc"] = utc_now()
    append_jsonl(RUN_LOG, record)
    save_state(state)
    return record


def run_loop(args: argparse.Namespace) -> int:
    load_dotenv()
    configure_runtime_paths(args.live_trading)
    if args.live_trading and args.confirm_live_orders != LIVE_ORDER_CONFIRMATION:
        raise BotError(
            "Live trading requires "
            f"--confirm-live-orders {LIVE_ORDER_CONFIRMATION} "
            "so live orders cannot be submitted accidentally."
        )
    trading_url = ALPACA_LIVE_TRADING_URL if args.live_trading else ALPACA_PAPER_TRADING_URL
    client = AlpacaClient(
        AlpacaCredentials.from_env(live_trading=args.live_trading),
        trading_base_url=trading_url,
    )
    stop = {"requested": False}

    def request_stop(_signum, _frame) -> None:
        stop["requested"] = True
        log("Stop requested.")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(f"Started multi-horizon semiconductor {account_mode(args)} bot with sleeves {args.sleeves}.")
    while not stop["requested"]:
        try:
            record = run_once(args, client)
            if args.once:
                print(json.dumps(record, indent=2, default=str))
                return 0
        except BotError as error:
            log(f"Bot error: {error}")
            if args.once:
                raise
        time.sleep(args.poll_seconds)
    log(f"Multi-horizon semiconductor {account_mode(args)} bot stopped.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run combined 3d/5d semiconductor sleeves.")
    parser.add_argument("command", choices=["run", "once", "status"])
    parser.add_argument("--sleeves", default=DEFAULT_SLEEVES)
    parser.add_argument(
        "--signal-mode",
        choices=[SIGNAL_MODE_EXPANDING, SIGNAL_MODE_FIXED],
        default=SIGNAL_MODE_EXPANDING,
        help="expanding retrains the active paper signal on schedule; fixed uses saved frozen model weights.",
    )
    parser.add_argument("--bars", default="data/semiconductor_sector_1d_5y.csv")
    parser.add_argument("--market-bars", default="data/semiconductor_market_1d_5y.csv")
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--time-in-force", default="day")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--min-trade-dollars", type=float, default=100.0)
    parser.add_argument("--prevent-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cash-reserve-dollars", type=float, default=250.0)
    parser.add_argument("--poll-seconds", type=float, default=900.0)
    parser.add_argument("--auto-update-bars", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bar-update-limit", type=int, default=1000)
    parser.add_argument("--bar-update-batch-size", type=int, default=25)
    parser.add_argument("--latest-signal-on-new-sleeve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-latest-signals",
        action="store_true",
        help="Generate/use latest available close signals even if the sleeve is not due by horizon spacing.",
    )
    parser.add_argument(
        "--live-trading",
        action="store_true",
        help="Use Alpaca live trading credentials and endpoint instead of paper trading.",
    )
    parser.add_argument(
        "--confirm-live-orders",
        default="",
        help=f"Required with --live-trading: pass {LIVE_ORDER_CONFIRMATION} to allow live orders.",
    )
    parser.add_argument(
        "--allow-expanding-signals",
        action="store_true",
        help="Deprecated compatibility flag; active paper trading now uses --signal-mode expanding by default.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.once = args.command == "once"
    configure_runtime_paths(args.live_trading)

    if args.command == "status":
        print(
            json.dumps(
                {
                    "state": load_state(),
                    "run_log": str(RUN_LOG),
                    "run_count": len(read_jsonl(RUN_LOG)),
                    "log_path": str(LOG_PATH),
                    "account_mode": account_mode(args),
                    "sleeves": parse_sleeves(args.sleeves),
                    "signal_mode": args.signal_mode,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
