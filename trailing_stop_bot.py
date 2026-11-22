#!/usr/bin/env python3
"""
Trailing stop helper for a long stock position.

Runs in dry-run mode unless --live is passed.
Alpaca paper trading is supported through environment variables:

    ALPACA_API_KEY_ID
    ALPACA_API_SECRET_KEY

Live trading uses separate environment variables:

    ALPACA_LIVE_API_KEY_ID
    ALPACA_LIVE_API_SECRET_KEY

Examples:

    # Place a broker-native $1 trailing stop sell order in paper mode.
    python scripts/trailing_stop_bot.py native --symbol AAPL --qty 10 --trail 1

    # Locally monitor Alpaca market data and submit a market sell in dry-run mode.
    python scripts/trailing_stop_bot.py monitor --symbol AAPL --qty 10 --trail 1

    # Test the logic with known prices, without any broker credentials.
    python scripts/trailing_stop_bot.py simulate --symbol TEST --qty 1 --trail 1 \
      --prices 210,215,220,219
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

import certifi
import pandas as pd
from joblib import load as load_joblib

try:
    from ml_offline_train import BAR_FIELDS, BotError as MlModelError
    from ml_offline_train import build_latest_features, load_model, predict_model
    from ml_experiments import build_feature_frame_from_dataframes
    from ml_feature_research import add_enhanced_features
    from ml_worldquant_alpha_runs import add_worldquant_alpha_features
except ModuleNotFoundError:
    BAR_FIELDS = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"]

    class MlModelError(RuntimeError):
        pass

    def _missing_legacy_ml(*_args, **_kwargs):
        raise MlModelError("Legacy intraday ML helpers were removed from this focused semiconductor workspace.")

    build_latest_features = _missing_legacy_ml
    load_model = _missing_legacy_ml
    predict_model = _missing_legacy_ml
    build_feature_frame_from_dataframes = _missing_legacy_ml
    add_enhanced_features = _missing_legacy_ml
    add_worldquant_alpha_features = _missing_legacy_ml


ALPACA_PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_TRADING_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"
DEFAULT_DOTENV_PATH = ".env"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8765
DEFAULT_LINEAR_MODEL_PATH = "models/year_1m/linear_regression_15m.json"
DEFAULT_LOGISTIC_MODEL_PATH = "models/year_1m/logistic_regression_15m.json"
DEFAULT_SKLEARN_MODEL_DIR = "models/sklearn_60m_market/hist_gradient_boosting_d3_l2_selective_60m"
DEFAULT_WORLDQUANT_MODEL_DIR = "models/worldquant_live/wq_pca16_base_no_symbol_return_60m"
BALANCE_RUNS_PATH = Path("data/paper_bot_balance_runs.jsonl")


class BotError(RuntimeError):
    pass


LogFn = Callable[[str], None]


def load_dotenv(path: str = DEFAULT_DOTENV_PATH) -> None:
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line_number, raw_line in enumerate(env_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                raise BotError(f"{path}:{line_number} must be KEY=value.")

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if not key:
                raise BotError(f"{path}:{line_number} has an empty key.")
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class AlpacaCredentials:
    key_id: str
    secret_key: str

    @classmethod
    def from_env(cls, *, live_trading: bool = False) -> "AlpacaCredentials":
        if live_trading:
            key_id_name = "ALPACA_LIVE_API_KEY_ID"
            secret_key_name = "ALPACA_LIVE_API_SECRET_KEY"
            key_id = os.getenv(key_id_name)
            secret_key = os.getenv(secret_key_name)
        else:
            key_id_name = "ALPACA_API_KEY_ID"
            secret_key_name = "ALPACA_API_SECRET_KEY"
            key_id = os.getenv("ALPACA_PAPER_API_KEY_ID") or os.getenv(key_id_name)
            secret_key = os.getenv("ALPACA_PAPER_API_SECRET_KEY") or os.getenv(secret_key_name)
        if not key_id or not secret_key:
            raise BotError(
                f"Missing {key_id_name} or {secret_key_name} environment variables."
            )
        return cls(key_id=key_id, secret_key=secret_key)


@dataclass(frozen=True)
class StockBar:
    timestamp: str
    close: Decimal
    volume: int


@dataclass
class SymbolState:
    in_position: bool = False
    last_price: Decimal | None = None
    entry_price: Decimal | None = None
    high_price: Decimal | None = None
    qty: Decimal | None = None
    entry_time: datetime | None = None
    consecutive_up_checks: int = 0


@dataclass(frozen=True)
class MarketClock:
    timestamp: str
    is_open: bool
    next_open: str
    next_close: str


class AlpacaClient:
    def __init__(
        self,
        credentials: AlpacaCredentials,
        *,
        trading_base_url: str = ALPACA_PAPER_TRADING_URL,
        data_base_url: str = ALPACA_DATA_URL,
        timeout_seconds: float = 20,
    ):
        self.credentials = credentials
        self.trading_base_url = trading_base_url.rstrip("/")
        self.data_base_url = data_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.opener = build_opener(
            ProxyHandler({}),
            HTTPSHandler(context=ssl_context),
        )

    def latest_trade_price(self, symbol: str, feed: str = "iex") -> Decimal:
        query = urlencode({"feed": feed})
        data = self._request(
            "GET",
            f"{self.data_base_url}/v2/stocks/{symbol.upper()}/trades/latest?{query}",
        )
        try:
            return parse_decimal(data["trade"]["p"], "latest trade price")
        except KeyError as error:
            raise BotError(f"Alpaca latest trade response did not include trade.p: {data}") from error

    def market_clock(self) -> MarketClock:
        data = self._request("GET", f"{self.trading_base_url}/v2/clock")
        try:
            return MarketClock(
                timestamp=str(data["timestamp"]),
                is_open=bool(data["is_open"]),
                next_open=str(data["next_open"]),
                next_close=str(data["next_close"]),
            )
        except KeyError as error:
            raise BotError(f"Alpaca clock response was missing a field: {data}") from error

    def stock_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        feed: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[StockBar]:
        query_params: dict[str, str | int] = {
            "symbols": symbol.upper(),
            "timeframe": timeframe,
            "feed": feed,
            "limit": limit,
            "sort": "asc",
            "adjustment": "split",
        }
        if start:
            query_params["start"] = start
        if end:
            query_params["end"] = end

        data = self._request(
            "GET",
            f"{self.data_base_url}/v2/stocks/bars?{urlencode(query_params)}",
        )
        raw_bars = data.get("bars", {}).get(symbol.upper(), [])
        bars: list[StockBar] = []
        for raw_bar in raw_bars:
            try:
                bars.append(
                    StockBar(
                        timestamp=str(raw_bar["t"]),
                        close=parse_decimal(raw_bar["c"], "bar close"),
                        volume=int(raw_bar["v"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise BotError(f"Unexpected Alpaca bar response for {symbol}: {raw_bar}") from error
        return bars

    def stock_bar_rows(
        self,
        symbol: str,
        *,
        timeframe: str,
        feed: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, str]]:
        query_params: dict[str, str | int] = {
            "symbols": symbol.upper(),
            "timeframe": timeframe,
            "feed": feed,
            "limit": limit,
            "sort": "asc",
            "adjustment": "split",
        }
        if start:
            query_params["start"] = start
        if end:
            query_params["end"] = end

        data = self._request(
            "GET",
            f"{self.data_base_url}/v2/stocks/bars?{urlencode(query_params)}",
        )
        raw_bars = data.get("bars", {}).get(symbol.upper(), [])
        rows: list[dict[str, str]] = []
        for raw_bar in raw_bars:
            rows.append(
                {
                    "timestamp": str(raw_bar.get("t", "")),
                    "symbol": symbol.upper(),
                    "open": str(raw_bar.get("o", "")),
                    "high": str(raw_bar.get("h", "")),
                    "low": str(raw_bar.get("l", "")),
                    "close": str(raw_bar.get("c", "")),
                    "volume": str(raw_bar.get("v", "")),
                    "trade_count": str(raw_bar.get("n", "")),
                    "vwap": str(raw_bar.get("vw", raw_bar.get("c", ""))),
                }
            )
        return rows

    def stock_bar_rows_multi(
        self,
        symbols: Iterable[str],
        *,
        timeframe: str,
        feed: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, str]]:
        requested_symbols = [symbol.upper() for symbol in symbols]
        query_params: dict[str, str | int] = {
            "symbols": ",".join(requested_symbols),
            "timeframe": timeframe,
            "feed": feed,
            "limit": limit,
            "sort": "asc",
            "adjustment": "split",
        }
        if start:
            query_params["start"] = start
        if end:
            query_params["end"] = end

        data = self._request(
            "GET",
            f"{self.data_base_url}/v2/stocks/bars?{urlencode(query_params)}",
        )
        rows: list[dict[str, str]] = []
        for symbol in requested_symbols:
            for raw_bar in data.get("bars", {}).get(symbol, []):
                rows.append(
                    {
                        "timestamp": str(raw_bar.get("t", "")),
                        "symbol": symbol,
                        "open": str(raw_bar.get("o", "")),
                        "high": str(raw_bar.get("h", "")),
                        "low": str(raw_bar.get("l", "")),
                        "close": str(raw_bar.get("c", "")),
                        "volume": str(raw_bar.get("v", "")),
                        "trade_count": str(raw_bar.get("n", "")),
                        "vwap": str(raw_bar.get("vw", raw_bar.get("c", ""))),
                    }
                )
        return rows

    def positions(self) -> dict[str, dict]:
        data = self._request("GET", f"{self.trading_base_url}/v2/positions")
        if not isinstance(data, list):
            raise BotError(f"Alpaca positions response was not a list: {data}")
        positions = {}
        for position in data:
            symbol = str(position.get("symbol", "")).upper()
            if symbol:
                positions[symbol] = position
        return positions

    def account(self) -> dict:
        return self._request("GET", f"{self.trading_base_url}/v2/account")

    def place_market_buy(
        self,
        *,
        symbol: str,
        qty: Decimal,
        time_in_force: str,
    ) -> dict:
        payload = {
            "symbol": symbol.upper(),
            "qty": decimal_to_api_string(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": time_in_force,
        }
        return self._request(
            "POST",
            f"{self.trading_base_url}/v2/orders",
            payload=payload,
        )

    def place_trailing_stop_sell(
        self,
        *,
        symbol: str,
        qty: Decimal,
        trail: Decimal,
        time_in_force: str,
    ) -> dict:
        payload = {
            "symbol": symbol.upper(),
            "qty": decimal_to_api_string(qty),
            "side": "sell",
            "type": "trailing_stop",
            "time_in_force": time_in_force,
            "trail_price": decimal_to_api_string(trail),
        }
        return self._request(
            "POST",
            f"{self.trading_base_url}/v2/orders",
            payload=payload,
        )

    def place_market_sell(
        self,
        *,
        symbol: str,
        qty: Decimal,
        time_in_force: str,
    ) -> dict:
        payload = {
            "symbol": symbol.upper(),
            "qty": decimal_to_api_string(qty),
            "side": "sell",
            "type": "market",
            "time_in_force": time_in_force,
        }
        return self._request(
            "POST",
            f"{self.trading_base_url}/v2/orders",
            payload=payload,
        )

    def _request(self, method: str, url: str, payload: dict | None = None) -> dict:
        headers = {
            "Accept": "application/json",
            "APCA-API-KEY-ID": self.credentials.key_id,
            "APCA-API-SECRET-KEY": self.credentials.secret_key,
        }
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")

        request = Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body) if response_body else {}
        except HTTPError as error:
            message = error.read().decode("utf-8") or error.reason
            raise BotError(f"{method} {url} returned {error.code}: {message}") from error
        except URLError as error:
            raise BotError(f"Could not reach {url}: {error.reason}") from error


class DryRunBroker:
    def place_market_buy(
        self,
        *,
        symbol: str,
        qty: Decimal,
        time_in_force: str,
    ) -> dict:
        return {
            "dry_run": True,
            "symbol": symbol.upper(),
            "qty": decimal_to_api_string(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": time_in_force,
        }

    def place_trailing_stop_sell(
        self,
        *,
        symbol: str,
        qty: Decimal,
        trail: Decimal,
        time_in_force: str,
    ) -> dict:
        return {
            "dry_run": True,
            "symbol": symbol.upper(),
            "qty": decimal_to_api_string(qty),
            "side": "sell",
            "type": "trailing_stop",
            "time_in_force": time_in_force,
            "trail_price": decimal_to_api_string(trail),
        }

    def place_market_sell(
        self,
        *,
        symbol: str,
        qty: Decimal,
        time_in_force: str,
    ) -> dict:
        return {
            "dry_run": True,
            "symbol": symbol.upper(),
            "qty": decimal_to_api_string(qty),
            "side": "sell",
            "type": "market",
            "time_in_force": time_in_force,
        }


def decimal_to_api_string(value: Decimal) -> str:
    return format(value.normalize(), "f")


def parse_decimal(value: object, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise argparse.ArgumentTypeError(f"{label} must be a valid number.") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{label} must be greater than 0.")
    return parsed


def parse_price_list(raw_prices: str) -> list[Decimal]:
    prices = [
        parse_decimal(raw_price.strip(), "price")
        for raw_price in raw_prices.split(",")
        if raw_price.strip()
    ]
    if not prices:
        raise argparse.ArgumentTypeError("--prices must contain at least one price.")
    return prices


def account_balance_snapshot(account: dict) -> dict[str, str | None]:
    return {
        "equity": string_or_none(account.get("equity")),
        "portfolio_value": string_or_none(account.get("portfolio_value")),
        "cash": string_or_none(account.get("cash")),
        "buying_power": string_or_none(account.get("buying_power")),
        "last_equity": string_or_none(account.get("last_equity")),
    }


def string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def equity_delta(start_account: dict[str, str | None] | None, end_account: dict[str, str | None] | None) -> str | None:
    if not start_account or not end_account:
        return None
    start_equity = start_account.get("equity")
    end_equity = end_account.get("equity")
    if start_equity is None or end_equity is None:
        return None
    try:
        return decimal_to_api_string(Decimal(end_equity) - Decimal(start_equity))
    except InvalidOperation:
        return None


def append_balance_run_event(event: dict) -> None:
    BALANCE_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BALANCE_RUNS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True) + "\n")


def monitor_prices(
    *,
    prices: Iterable[Decimal],
    symbol: str,
    qty: Decimal,
    trail: Decimal,
    broker: AlpacaClient | DryRunBroker,
    time_in_force: str,
    sleep_seconds: float = 0,
) -> bool:
    high: Decimal | None = None

    for price in prices:
        high = price if high is None else max(high, price)
        stop_price = high - trail
        print(
            f"{symbol.upper()} price={price} high={high} trailing_stop={stop_price}",
            flush=True,
        )

        if price <= stop_price:
            order = broker.place_market_sell(
                symbol=symbol,
                qty=qty,
                time_in_force=time_in_force,
            )
            print("Triggered sell order:")
            print(json.dumps(order, indent=2, sort_keys=True))
            return True

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print("No sell trigger reached.")
    return False


def live_price_stream(
    *,
    client: AlpacaClient,
    symbol: str,
    poll_seconds: float,
    feed: str,
) -> Iterable[Decimal]:
    while True:
        yield client.latest_trade_price(symbol, feed=feed)
        time.sleep(poll_seconds)


def calculate_whole_share_qty(trade_amount: Decimal, price: Decimal) -> Decimal:
    if price <= 0:
        raise BotError("price must be greater than 0.")
    qty = (trade_amount / price).to_integral_value(rounding=ROUND_DOWN)
    if qty <= 0:
        raise BotError(f"${trade_amount} is not enough to buy one share at price {price}.")
    return qty


def trailing_stop_price(high_price: Decimal, trail_percent: Decimal) -> Decimal:
    if trail_percent <= 0 or trail_percent >= 100:
        raise BotError("trail percent must be greater than 0 and less than 100.")
    return high_price * (Decimal("1") - (trail_percent / Decimal("100")))


def parse_symbols(raw_symbols: str) -> list[str]:
    symbols = []
    seen = set()
    for raw_symbol in raw_symbols.replace("\n", ",").split(","):
        symbol = raw_symbol.strip().upper()
        if symbol == "SPACEX":
            symbol = "SPCX"
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    if not symbols:
        raise BotError("Provide at least one tradable public ticker symbol.")
    return symbols


def daily_volume_context(
    *,
    client: AlpacaClient,
    symbol: str,
    feed: str,
    average_days: int,
) -> tuple[StockBar, list[StockBar], Decimal]:
    start = (datetime.now(timezone.utc) - timedelta(days=average_days * 3)).date().isoformat()
    bars = client.stock_bars(
        symbol,
        timeframe="1Day",
        feed=feed,
        start=start,
        limit=max(average_days + 5, 20),
    )
    if len(bars) < average_days + 1:
        raise BotError(
            f"Need at least {average_days + 1} daily bars for {symbol}; Alpaca returned {len(bars)}."
        )

    current_bar = bars[-1]
    history = bars[-average_days - 1 : -1]
    average_volume = Decimal(sum(bar.volume for bar in history)) / Decimal(len(history))
    return current_bar, history, average_volume


@dataclass(frozen=True)
class MlPrediction:
    timestamp: str
    expected_return_15m: float
    prob_profitable_15m: float
    features: dict[str, float]


@dataclass(frozen=True)
class MlGate:
    linear_model: dict
    logistic_model: dict
    min_probability: float
    min_expected_return: float


def load_ml_gate(
    *,
    linear_model_path: str,
    logistic_model_path: str,
    min_probability: float,
    min_expected_return: float,
) -> MlGate:
    try:
        return MlGate(
            linear_model=load_model(Path(linear_model_path)),
            logistic_model=load_model(Path(logistic_model_path)),
            min_probability=min_probability,
            min_expected_return=min_expected_return,
        )
    except MlModelError as error:
        raise BotError(str(error)) from error


def predict_symbol_from_recent_bars(
    *,
    client: AlpacaClient,
    symbol: str,
    feed: str,
    linear_model: dict,
    logistic_model: dict,
    lookback_days: int = 10,
) -> MlPrediction:
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = client.stock_bar_rows(
        symbol,
        timeframe="1Min",
        feed=feed,
        start=start,
        limit=10000,
    )
    feature_names = list(linear_model["feature_names"])
    try:
        timestamp, features = build_latest_features(
            symbol=symbol,
            bars=rows,
            feature_names=feature_names,
        )
        expected_return = predict_model(linear_model, features)
        probability = predict_model(logistic_model, features)
    except MlModelError as error:
        raise BotError(str(error)) from error
    return MlPrediction(
        timestamp=timestamp,
        expected_return_15m=expected_return,
        prob_profitable_15m=probability,
        features=features,
    )


def ml_gate_allows_buy(
    *,
    prediction: MlPrediction,
    gate: MlGate,
) -> bool:
    return (
        prediction.prob_profitable_15m >= gate.min_probability
        and prediction.expected_return_15m >= gate.min_expected_return
    )


@dataclass(frozen=True)
class SklearnMlPrediction:
    timestamp: str
    expected_return_60m: float
    prob_profitable_60m: float
    features: dict[str, float]


@dataclass(frozen=True)
class SklearnSelectiveGate:
    classifier: object
    return_model: object
    feature_names: list[str]
    probability_threshold: float
    expected_return_threshold: float
    expected_return_enabled: bool
    model_dir: str


@dataclass(frozen=True)
class WorldQuantMlPrediction:
    timestamp: str
    expected_return_60m: float
    features: dict[str, float]


@dataclass(frozen=True)
class WorldQuantReturnGate:
    model: object
    feature_names: list[str]
    expected_return_threshold: float
    lookback_days: int
    model_dir: str


def load_sklearn_selective_gate(
    model_dir: str,
    *,
    probability_threshold: float | None = None,
    expected_return_threshold: float | None = None,
    expected_return_enabled: bool = True,
) -> SklearnSelectiveGate:
    model_path = Path(model_dir)
    metadata_path = model_path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BotError(f"Missing sklearn model metadata: {metadata_path}") from error
    thresholds = metadata.get("thresholds", {})
    try:
        return SklearnSelectiveGate(
            classifier=load_joblib(model_path / "classifier.joblib"),
            return_model=load_joblib(model_path / "return_model.joblib"),
            feature_names=list(metadata["feature_names"]),
            probability_threshold=(
                probability_threshold
                if probability_threshold is not None
                else float(thresholds["probability_threshold"])
            ),
            expected_return_threshold=(
                expected_return_threshold
                if expected_return_threshold is not None
                else float(thresholds["expected_return_threshold"])
            ),
            expected_return_enabled=expected_return_enabled,
            model_dir=str(model_path),
        )
    except (KeyError, FileNotFoundError) as error:
        raise BotError(f"Could not load sklearn selective model from {model_path}: {error}") from error


def load_worldquant_return_gate(
    model_dir: str,
    *,
    expected_return_threshold: float | None = None,
    lookback_days: int = 30,
) -> WorldQuantReturnGate:
    model_path = Path(model_dir)
    metadata_path = model_path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        thresholds = metadata.get("thresholds", {})
        return WorldQuantReturnGate(
            model=load_joblib(model_path / "model.joblib"),
            feature_names=list(metadata["feature_names"]),
            expected_return_threshold=(
                expected_return_threshold
                if expected_return_threshold is not None
                else float(thresholds["expected_return_threshold"])
            ),
            lookback_days=lookback_days,
            model_dir=str(model_path),
        )
    except (KeyError, FileNotFoundError) as error:
        raise BotError(f"Could not load WorldQuant model from {model_path}: {error}") from error


def bar_rows_to_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_worldquant_feature_frame_from_dataframes(
    bars_df: pd.DataFrame,
    *,
    market_bars_df: pd.DataFrame,
    required_feature_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    base_frame, _ = build_feature_frame_from_dataframes(
        bars_df,
        market_bars_df=market_bars_df,
        required_feature_names=None,
    )
    enhanced_frame, _ = add_enhanced_features(base_frame)
    alpha_frame, _ = add_worldquant_alpha_features(enhanced_frame)
    for feature_name in required_feature_names:
        if feature_name not in alpha_frame.columns:
            alpha_frame[feature_name] = 0.0
    return alpha_frame, required_feature_names


def predict_sklearn_symbol_from_recent_bars(
    *,
    client: AlpacaClient,
    symbol: str,
    feed: str,
    gate: SklearnSelectiveGate,
    lookback_days: int = 10,
) -> SklearnMlPrediction:
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    symbol_rows = client.stock_bar_rows(
        symbol,
        timeframe="1Min",
        feed=feed,
        start=start,
        limit=10000,
    )
    market_rows: list[dict[str, str]] = []
    for market_symbol in ("SPY", "QQQ"):
        market_rows.extend(
            client.stock_bar_rows(
                market_symbol,
                timeframe="1Min",
                feed=feed,
                start=start,
                limit=10000,
            )
        )

    bars_frame = bar_rows_to_frame(symbol_rows)
    market_frame = bar_rows_to_frame(market_rows)
    if bars_frame.empty or market_frame.empty:
        raise BotError(f"Not enough recent bars to predict {symbol}.")

    feature_frame, feature_names = build_feature_frame_from_dataframes(
        bars_frame,
        market_bars_df=market_frame,
        required_feature_names=gate.feature_names,
    )
    usable = feature_frame[feature_frame["symbol"].str.upper() == symbol.upper()].dropna(
        subset=feature_names
    )
    if usable.empty:
        raise BotError(f"Could not compute complete sklearn features for {symbol}.")

    latest = usable.iloc[-1]
    model_input = latest[feature_names].to_frame().T
    probability = float(gate.classifier.predict_proba(model_input)[0, 1])
    expected_return = float(gate.return_model.predict(model_input)[0])
    return SklearnMlPrediction(
        timestamp=str(latest["timestamp"]),
        expected_return_60m=expected_return,
        prob_profitable_60m=probability,
        features={feature_name: float(latest[feature_name]) for feature_name in feature_names},
    )


def sklearn_gate_allows_buy(
    *,
    prediction: SklearnMlPrediction,
    gate: SklearnSelectiveGate,
) -> bool:
    if prediction.prob_profitable_60m < gate.probability_threshold:
        return False
    if gate.expected_return_enabled:
        return prediction.expected_return_60m >= gate.expected_return_threshold
    return True


def predict_worldquant_symbols_from_recent_bars(
    *,
    client: AlpacaClient,
    symbols: list[str],
    feed: str,
    gate: WorldQuantReturnGate,
) -> dict[str, WorldQuantMlPrediction]:
    start = (datetime.now(timezone.utc) - timedelta(days=gate.lookback_days)).isoformat()
    symbol_rows: list[dict[str, str]] = []
    for symbol in symbols:
        symbol_rows.extend(
            client.stock_bar_rows_multi(
                [symbol],
                timeframe="1Min",
                feed=feed,
                start=start,
                limit=10000,
            )
        )
    market_rows: list[dict[str, str]] = []
    for market_symbol in ("SPY", "QQQ"):
        market_rows.extend(
            client.stock_bar_rows_multi(
                [market_symbol],
                timeframe="1Min",
                feed=feed,
                start=start,
                limit=10000,
            )
        )

    bars_frame = bar_rows_to_frame(symbol_rows)
    market_frame = bar_rows_to_frame(market_rows)
    if bars_frame.empty or market_frame.empty:
        raise BotError("Not enough recent bars to compute WorldQuant predictions.")

    feature_frame, feature_names = build_worldquant_feature_frame_from_dataframes(
        bars_frame,
        market_bars_df=market_frame,
        required_feature_names=gate.feature_names,
    )
    usable = feature_frame.dropna(subset=feature_names).copy()
    if usable.empty:
        raise BotError("Could not compute complete WorldQuant features for the watched basket.")

    latest_by_symbol = (
        usable.sort_values(["symbol", "timestamp"])
        .groupby("symbol", as_index=False, group_keys=False)
        .tail(1)
    )
    predictions: dict[str, WorldQuantMlPrediction] = {}
    model_input = latest_by_symbol[feature_names]
    expected_returns = gate.model.predict(model_input)
    for (_, row), expected_return in zip(latest_by_symbol.iterrows(), expected_returns):
        symbol = str(row["symbol"]).upper()
        predictions[symbol] = WorldQuantMlPrediction(
            timestamp=str(row["timestamp"]),
            expected_return_60m=float(expected_return),
            features={feature_name: float(row[feature_name]) for feature_name in feature_names},
        )
    return predictions


def worldquant_gate_allows_buy(
    *,
    prediction: WorldQuantMlPrediction,
    gate: WorldQuantReturnGate,
) -> bool:
    return prediction.expected_return_60m >= gate.expected_return_threshold


def monitor_volume_momentum(
    *,
    client: AlpacaClient,
    broker: AlpacaClient | DryRunBroker,
    symbol: str,
    qty: Decimal,
    trail: Decimal,
    time_in_force: str,
    poll_seconds: float,
    feed: str,
    average_days: int,
    volume_multiplier: Decimal,
    stop_event: threading.Event | None = None,
    log: LogFn | None = None,
) -> None:
    in_position = False
    last_price: Decimal | None = None
    entry_price: Decimal | None = None
    high_price: Decimal | None = None
    emit = log or print

    while stop_event is None or not stop_event.is_set():
        clock = client.market_clock()
        if not clock.is_open:
            emit(
                (
                    f"Market closed at {clock.timestamp}. "
                    f"No paper orders will be submitted. Next open: {clock.next_open}."
                )
            )
            if stop_event is not None:
                if stop_event.wait(max(poll_seconds, 60)):
                    break
            else:
                time.sleep(max(poll_seconds, 60))
            continue

        current_bar, history, average_volume = daily_volume_context(
            client=client,
            symbol=symbol,
            feed=feed,
            average_days=average_days,
        )
        price = client.latest_trade_price(symbol, feed=feed)
        reference_price = last_price if last_price is not None else history[-1].close
        price_increasing = price > reference_price
        volume_threshold = average_volume * volume_multiplier
        high_volume = Decimal(current_bar.volume) > volume_threshold

        if not in_position:
            emit(
                (
                    f"{symbol.upper()} scan price={price} ref={reference_price} "
                    f"volume={current_bar.volume} avg_{average_days}d={average_volume:.0f} "
                    f"threshold={volume_threshold:.0f} high_volume={high_volume} "
                    f"price_up={price_increasing}"
                ),
            )
            if high_volume and price_increasing:
                order = broker.place_market_buy(
                    symbol=symbol,
                    qty=qty,
                    time_in_force=time_in_force,
                )
                entry_price = price
                high_price = price
                in_position = True
                emit("Triggered buy order:")
                emit(json.dumps(order, indent=2, sort_keys=True))
        else:
            assert entry_price is not None
            high_price = price if high_price is None else max(high_price, price)
            stop_price = high_price - trail
            emit(
                (
                    f"{symbol.upper()} holding entry={entry_price} price={price} "
                    f"high={high_price} trailing_stop={stop_price}"
                ),
            )
            if price <= stop_price:
                order = broker.place_market_sell(
                    symbol=symbol,
                    qty=qty,
                    time_in_force=time_in_force,
                )
                emit("Triggered trailing sell order:")
                emit(json.dumps(order, indent=2, sort_keys=True))
                in_position = False
                entry_price = None
                high_price = None

        last_price = price
        if stop_event is not None:
            if stop_event.wait(poll_seconds):
                break
        else:
            time.sleep(poll_seconds)

    emit("Momentum bot stopped.")


def monitor_volume_momentum_basket(
    *,
    client: AlpacaClient,
    broker: AlpacaClient | DryRunBroker,
    symbols: list[str],
    trade_amount: Decimal,
    trail_percent: Decimal,
    time_in_force: str,
    poll_seconds: float,
    feed: str,
    average_days: int,
    volume_multiplier: Decimal,
    max_hold_minutes: float | None,
    ml_gate: MlGate | None = None,
    sklearn_gate: SklearnSelectiveGate | None = None,
    worldquant_gate: WorldQuantReturnGate | None = None,
    stop_event: threading.Event | None = None,
    log: LogFn | None = None,
) -> None:
    states = {symbol: SymbolState() for symbol in symbols}
    emit = log or print
    try:
        positions = client.positions()
        for symbol in symbols:
            position = positions.get(symbol)
            if not position:
                continue
            qty = parse_decimal(position.get("qty"), f"{symbol} position qty")
            entry_price = parse_decimal(
                position.get("avg_entry_price"),
                f"{symbol} average entry price",
            )
            latest_price = client.latest_trade_price(symbol, feed=feed)
            states[symbol] = SymbolState(
                in_position=True,
                last_price=latest_price,
                entry_price=entry_price,
                high_price=max(entry_price, latest_price),
                qty=qty,
                entry_time=datetime.now(timezone.utc),
            )
            emit(
                (
                    f"Recovered existing paper position for {symbol}: "
                    f"qty={qty} avg_entry={entry_price} latest={latest_price}"
                )
            )
    except BotError as error:
        emit(f"Could not recover existing positions before scanning: {error}")

    while stop_event is None or not stop_event.is_set():
        clock = client.market_clock()
        if not clock.is_open:
            emit(
                (
                    f"Market closed at {clock.timestamp}. "
                    f"No paper orders will be submitted. Next open: {clock.next_open}."
                )
            )
            if stop_event is not None:
                if stop_event.wait(max(poll_seconds, 60)):
                    break
            else:
                time.sleep(max(poll_seconds, 60))
            continue

        worldquant_predictions: dict[str, WorldQuantMlPrediction] = {}
        if worldquant_gate is not None:
            try:
                worldquant_predictions = predict_worldquant_symbols_from_recent_bars(
                    client=client,
                    symbols=symbols,
                    feed=feed,
                    gate=worldquant_gate,
                )
            except BotError as error:
                emit(f"WorldQuant basket prediction skipped this cycle: {error}")

        for symbol in symbols:
            if stop_event is not None and stop_event.is_set():
                break
            state = states[symbol]
            try:
                current_bar, history, average_volume = daily_volume_context(
                    client=client,
                    symbol=symbol,
                    feed=feed,
                    average_days=average_days,
                )
                price = client.latest_trade_price(symbol, feed=feed)
                reference_price = state.last_price if state.last_price is not None else history[-1].close
                price_increasing = price > reference_price
                state.consecutive_up_checks = (
                    state.consecutive_up_checks + 1 if price_increasing else 0
                )
                volume_threshold = average_volume * volume_multiplier
                high_volume = Decimal(current_bar.volume) > volume_threshold

                if not state.in_position:
                    emit(
                        (
                            f"{symbol} scan price={price} ref={reference_price} "
                            f"volume={current_bar.volume} avg_{average_days}d={average_volume:.0f} "
                            f"threshold={volume_threshold:.0f} high_volume={high_volume} "
                            f"price_up={price_increasing} up_checks={state.consecutive_up_checks}/2"
                        )
                    )
                    if worldquant_gate is not None:
                        wq_prediction = worldquant_predictions.get(symbol.upper())
                        if wq_prediction is None:
                            should_buy = False
                            emit(f"{symbol} WorldQuant prediction unavailable; buy skipped.")
                        else:
                            should_buy = worldquant_gate_allows_buy(
                                prediction=wq_prediction,
                                gate=worldquant_gate,
                            )
                            emit(
                                (
                                    f"{symbol} worldquant_return ts={wq_prediction.timestamp} "
                                    f"expected_return_60m={wq_prediction.expected_return_60m:.6f} "
                                    f"required_return={worldquant_gate.expected_return_threshold:.6f} "
                                    f"ml_allowed={should_buy}"
                                )
                            )
                    elif sklearn_gate is not None:
                        sklearn_prediction = predict_sklearn_symbol_from_recent_bars(
                            client=client,
                            symbol=symbol,
                            feed=feed,
                            gate=sklearn_gate,
                        )
                        should_buy = sklearn_gate_allows_buy(
                            prediction=sklearn_prediction,
                            gate=sklearn_gate,
                        )
                        required_return = (
                            f"{sklearn_gate.expected_return_threshold:.6f}"
                            if sklearn_gate.expected_return_enabled
                            else "disabled"
                        )
                        emit(
                            (
                                f"{symbol} sklearn_selective ts={sklearn_prediction.timestamp} "
                                f"expected_return_60m={sklearn_prediction.expected_return_60m:.6f} "
                                f"prob_profitable_60m={sklearn_prediction.prob_profitable_60m:.3f} "
                                f"required_prob={sklearn_gate.probability_threshold:.3f} "
                                f"required_return={required_return} "
                                f"ml_allowed={should_buy}"
                            )
                        )
                    else:
                        should_buy = high_volume and state.consecutive_up_checks >= 2
                    if (
                        should_buy
                        and ml_gate is not None
                        and sklearn_gate is None
                        and worldquant_gate is None
                    ):
                        prediction = predict_symbol_from_recent_bars(
                            client=client,
                            symbol=symbol,
                            feed=feed,
                            linear_model=ml_gate.linear_model,
                            logistic_model=ml_gate.logistic_model,
                        )
                        ml_allowed = ml_gate_allows_buy(prediction=prediction, gate=ml_gate)
                        emit(
                            (
                                f"{symbol} ML prediction ts={prediction.timestamp} "
                                f"expected_return_15m={prediction.expected_return_15m:.6f} "
                                f"prob_profitable_15m={prediction.prob_profitable_15m:.3f} "
                                f"required_prob={ml_gate.min_probability:.3f} "
                                f"required_return={ml_gate.min_expected_return:.6f} "
                                f"ml_allowed={ml_allowed}"
                            )
                        )
                        should_buy = ml_allowed

                    if should_buy:
                        qty = calculate_whole_share_qty(trade_amount, price)
                        order = broker.place_market_buy(
                            symbol=symbol,
                            qty=qty,
                            time_in_force=time_in_force,
                        )
                        state.entry_price = price
                        state.high_price = price
                        state.qty = qty
                        state.entry_time = datetime.now(timezone.utc)
                        state.in_position = True
                        emit(f"Triggered buy order for {symbol}: about ${trade_amount} -> {qty} shares")
                        emit(json.dumps(order, indent=2, sort_keys=True))
                else:
                    assert state.entry_price is not None
                    assert state.qty is not None
                    state.high_price = price if state.high_price is None else max(state.high_price, price)
                    stop_price = trailing_stop_price(state.high_price, trail_percent)
                    held_minutes = (
                        (datetime.now(timezone.utc) - state.entry_time).total_seconds() / 60
                        if state.entry_time is not None
                        else 0.0
                    )
                    horizon_exit = (
                        max_hold_minutes is not None
                        and state.entry_time is not None
                        and held_minutes >= max_hold_minutes
                    )
                    emit(
                        (
                            f"{symbol} holding qty={state.qty} entry={state.entry_price} "
                            f"price={price} high={state.high_price} trailing_stop={stop_price:.4f} "
                            f"trail={trail_percent}% held={held_minutes:.1f}m "
                            f"max_hold={max_hold_minutes if max_hold_minutes is not None else 'disabled'}"
                        )
                    )
                    if price <= stop_price or horizon_exit:
                        reason = "horizon" if horizon_exit and price > stop_price else "trailing stop"
                        order = broker.place_market_sell(
                            symbol=symbol,
                            qty=state.qty,
                            time_in_force=time_in_force,
                        )
                        emit(f"Triggered {reason} sell order for {symbol}: {state.qty} shares")
                        emit(json.dumps(order, indent=2, sort_keys=True))
                        state.in_position = False
                        state.entry_price = None
                        state.high_price = None
                        state.qty = None
                        state.entry_time = None
                        state.consecutive_up_checks = 0

                state.last_price = price
            except BotError as error:
                emit(f"{symbol} skipped this cycle: {error}")

        if stop_event is not None:
            if stop_event.wait(poll_seconds):
                break
        else:
            time.sleep(poll_seconds)

    emit("Momentum basket bot stopped.")


@dataclass(frozen=True)
class MomentumSettings:
    symbols: list[str]
    trade_amount: Decimal
    trail_percent: Decimal
    time_in_force: str
    poll_seconds: float
    feed: str
    average_days: int
    volume_multiplier: Decimal
    max_hold_minutes: float | None
    ml_enabled: bool
    linear_model_path: str
    logistic_model_path: str
    ml_min_probability: float
    ml_min_expected_return: float
    sklearn_model_dir: str
    sklearn_selective_enabled: bool
    sklearn_probability_threshold: float | None
    sklearn_expected_return_threshold: float | None
    sklearn_expected_return_enabled: bool
    worldquant_enabled: bool
    worldquant_model_dir: str
    worldquant_expected_return_threshold: float | None
    worldquant_lookback_days: int


def parse_momentum_settings(fields: dict[str, list[str]]) -> MomentumSettings:
    def field(name: str, default: str | None = None) -> str:
        values = fields.get(name, [])
        if values and values[0].strip():
            return values[0].strip()
        if default is not None:
            return default
        raise BotError(f"Missing {name}.")

    def field_bool(name: str, default: bool) -> bool:
        values = [value.strip().lower() for value in fields.get(name, []) if value.strip()]
        if not values:
            values = ["true" if default else "false"]
        if any(value in {"1", "true", "yes", "on"} for value in values):
            return True
        value = values[0]
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise BotError(f"{name} must be true or false.")

    def optional_float(name: str) -> float | None:
        values = fields.get(name, [])
        if not values or not values[0].strip():
            return None
        return float(values[0].strip())

    symbols = parse_symbols(
        field(
            "symbols",
            field("symbol", "NVDA,TSLA,SPCX,GOOGL,AAPL,MSFT,AMZN,META,AVGO,AMD,MU,ORCL,PLTR"),
        )
    )
    trade_amount = parse_decimal(field("trade_amount", "5000"), "trade amount")
    trail_percent = parse_decimal(field("trail_percent", field("trail", "2")), "trail percent")
    time_in_force = field("time_in_force", "day")
    if time_in_force not in {"day", "gtc", "opg", "cls", "ioc", "fok"}:
        raise BotError("time_in_force must be one of day, gtc, opg, cls, ioc, or fok.")

    feed = field("feed", "iex")
    if feed not in {"iex", "sip"}:
        raise BotError("feed must be iex or sip.")

    try:
        poll_seconds = float(field("poll_seconds", "10"))
        average_days = int(field("average_days", "21"))
        ml_min_probability = float(field("ml_min_probability", "0.55"))
        ml_min_expected_return = float(field("ml_min_expected_return", "0.0005"))
        sklearn_probability_threshold = optional_float("sklearn_probability_threshold")
        sklearn_expected_return_threshold = optional_float("sklearn_expected_return_threshold")
        worldquant_expected_return_threshold = optional_float("worldquant_expected_return_threshold")
        worldquant_lookback_days = int(field("worldquant_lookback_days", "30"))
        max_hold_minutes_raw = float(field("max_hold_minutes", "60"))
    except ValueError as error:
        raise BotError("poll_seconds, average_days, lookback days, max hold, and ML thresholds must be numbers.") from error
    if poll_seconds <= 0:
        raise BotError("poll_seconds must be greater than 0.")
    if average_days <= 0:
        raise BotError("average_days must be greater than 0.")
    if worldquant_lookback_days <= 0:
        raise BotError("worldquant_lookback_days must be greater than 0.")
    max_hold_minutes = max_hold_minutes_raw if max_hold_minutes_raw > 0 else None

    return MomentumSettings(
        symbols=symbols,
        trade_amount=trade_amount,
        trail_percent=trail_percent,
        time_in_force=time_in_force,
        poll_seconds=poll_seconds,
        feed=feed,
        average_days=average_days,
        volume_multiplier=parse_decimal(field("volume_multiplier", "1.5"), "volume multiplier"),
        max_hold_minutes=max_hold_minutes,
        ml_enabled=field_bool("ml_enabled", True),
        linear_model_path=field("linear_model_path", DEFAULT_LINEAR_MODEL_PATH),
        logistic_model_path=field("logistic_model_path", DEFAULT_LOGISTIC_MODEL_PATH),
        ml_min_probability=ml_min_probability,
        ml_min_expected_return=ml_min_expected_return,
        sklearn_model_dir=field("sklearn_model_dir", DEFAULT_SKLEARN_MODEL_DIR),
        sklearn_selective_enabled=field_bool("sklearn_selective_enabled", False),
        sklearn_probability_threshold=sklearn_probability_threshold,
        sklearn_expected_return_threshold=sklearn_expected_return_threshold,
        sklearn_expected_return_enabled=field_bool("sklearn_expected_return_enabled", True),
        worldquant_enabled=field_bool("worldquant_enabled", True),
        worldquant_model_dir=field("worldquant_model_dir", DEFAULT_WORLDQUANT_MODEL_DIR),
        worldquant_expected_return_threshold=worldquant_expected_return_threshold,
        worldquant_lookback_days=worldquant_lookback_days,
    )


class BotRunner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.settings: MomentumSettings | None = None
        self.logs: list[str] = []
        self.last_error: str | None = None
        self.run_id: str | None = None
        self.run_started_at: str | None = None
        self.run_ended_at: str | None = None
        self.start_account: dict[str, str | None] | None = None
        self.end_account: dict[str, str | None] | None = None

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-300:]
        try:
            print(line, flush=True)
        except BrokenPipeError:
            pass

    def record_account_balance(self, *, client: AlpacaClient, label: str) -> None:
        try:
            snapshot = account_balance_snapshot(client.account())
        except BotError as error:
            self.log(f"Could not record {label} account balance: {error}")
            return

        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            if label == "start":
                self.run_id = timestamp.replace(":", "").replace("-", "")
                self.run_started_at = timestamp
                self.run_ended_at = None
                self.start_account = snapshot
                self.end_account = None
                delta = None
            else:
                self.run_ended_at = timestamp
                self.end_account = snapshot
                delta = equity_delta(self.start_account, self.end_account)
            event = {
                "event": label,
                "run_id": self.run_id,
                "timestamp": timestamp,
                "account": snapshot,
                "equity_delta": delta,
                "settings": (
                    {
                        "symbols": self.settings.symbols,
                        "trade_amount": decimal_to_api_string(self.settings.trade_amount),
                        "trail_percent": decimal_to_api_string(self.settings.trail_percent),
                        "poll_seconds": self.settings.poll_seconds,
                        "feed": self.settings.feed,
                        "max_hold_minutes": self.settings.max_hold_minutes,
                        "sklearn_selective_enabled": self.settings.sklearn_selective_enabled,
                        "sklearn_model_dir": self.settings.sklearn_model_dir,
                        "sklearn_probability_threshold": self.settings.sklearn_probability_threshold,
                        "sklearn_expected_return_threshold": self.settings.sklearn_expected_return_threshold,
                        "sklearn_expected_return_enabled": self.settings.sklearn_expected_return_enabled,
                        "worldquant_enabled": self.settings.worldquant_enabled,
                        "worldquant_model_dir": self.settings.worldquant_model_dir,
                        "worldquant_expected_return_threshold": self.settings.worldquant_expected_return_threshold,
                        "worldquant_lookback_days": self.settings.worldquant_lookback_days,
                    }
                    if self.settings
                    else None
                ),
            }

        try:
            append_balance_run_event(event)
        except OSError as error:
            self.log(f"Could not write {label} balance record: {error}")

        message = (
            f"Recorded {label} paper balance: "
            f"equity={snapshot.get('equity')} "
            f"portfolio_value={snapshot.get('portfolio_value')} "
            f"cash={snapshot.get('cash')}"
        )
        if delta is not None:
            message += f" equity_delta={delta}"
        self.log(message)

    def start(self, settings: MomentumSettings) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise BotError("Bot is already running.")
            self.stop_event = threading.Event()
            self.settings = settings
            self.last_error = None
            self.run_id = None
            self.run_started_at = None
            self.run_ended_at = None
            self.start_account = None
            self.end_account = None
            stop_event = self.stop_event

        def worker() -> None:
            client: AlpacaClient | None = None
            try:
                credentials = AlpacaCredentials.from_env()
                client = AlpacaClient(credentials, trading_base_url=ALPACA_PAPER_TRADING_URL)
                self.record_account_balance(client=client, label="start")
                worldquant_gate = (
                    load_worldquant_return_gate(
                        settings.worldquant_model_dir,
                        expected_return_threshold=settings.worldquant_expected_return_threshold,
                        lookback_days=settings.worldquant_lookback_days,
                    )
                    if settings.worldquant_enabled
                    else None
                )
                sklearn_gate = (
                    load_sklearn_selective_gate(
                        settings.sklearn_model_dir,
                        probability_threshold=settings.sklearn_probability_threshold,
                        expected_return_threshold=settings.sklearn_expected_return_threshold,
                        expected_return_enabled=settings.sklearn_expected_return_enabled,
                    )
                    if settings.sklearn_selective_enabled and worldquant_gate is None
                    else None
                )
                ml_gate = (
                    load_ml_gate(
                        linear_model_path=settings.linear_model_path,
                        logistic_model_path=settings.logistic_model_path,
                        min_probability=settings.ml_min_probability,
                        min_expected_return=settings.ml_min_expected_return,
                    )
                    if settings.ml_enabled and sklearn_gate is None and worldquant_gate is None
                    else None
                )
                self.log(
                    (
                        "Started paper momentum basket bot "
                        f"symbols={','.join(settings.symbols)} trade_amount={settings.trade_amount} "
                        f"trail_percent={settings.trail_percent} "
                        f"poll={settings.poll_seconds}s feed={settings.feed} "
                        f"max_hold_minutes={settings.max_hold_minutes} "
                        f"ml_enabled={settings.ml_enabled} "
                        f"sklearn_selective_enabled={settings.sklearn_selective_enabled} "
                        f"worldquant_enabled={settings.worldquant_enabled}"
                    )
                )
                monitor_volume_momentum_basket(
                    client=client,
                    broker=client,
                    symbols=settings.symbols,
                    trade_amount=settings.trade_amount,
                    trail_percent=settings.trail_percent,
                    time_in_force=settings.time_in_force,
                    poll_seconds=settings.poll_seconds,
                    feed=settings.feed,
                    average_days=settings.average_days,
                    volume_multiplier=settings.volume_multiplier,
                    max_hold_minutes=settings.max_hold_minutes,
                    ml_gate=ml_gate,
                    sklearn_gate=sklearn_gate,
                    worldquant_gate=worldquant_gate,
                    stop_event=stop_event,
                    log=self.log,
                )
            except Exception as error:
                with self.lock:
                    self.last_error = str(error)
                self.log(f"Bot error: {error}")
            finally:
                if client is not None:
                    self.record_account_balance(client=client, label="end")

        thread = threading.Thread(target=worker, name="momentum-paper-bot", daemon=True)
        with self.lock:
            self.thread = thread
        thread.start()

    def stop(self) -> None:
        with self.lock:
            stop_event = self.stop_event
            running = self.thread is not None and self.thread.is_alive()
        if not running or stop_event is None:
            raise BotError("Bot is not running.")
        self.log("Stop requested.")
        stop_event.set()

    def wait_until_stopped(self, timeout_seconds: float | None = None) -> None:
        with self.lock:
            thread = self.thread
        if thread is not None:
            thread.join(timeout_seconds)

    def snapshot(self) -> dict:
        with self.lock:
            running = self.thread is not None and self.thread.is_alive()
            settings = self.settings
            account_tracking = {
                "run_id": self.run_id,
                "run_started_at": self.run_started_at,
                "run_ended_at": self.run_ended_at,
                "start_account": self.start_account,
                "end_account": self.end_account,
                "equity_delta": equity_delta(self.start_account, self.end_account),
                "record_path": str(BALANCE_RUNS_PATH),
            }
            return {
                "running": running,
                "last_error": self.last_error,
                "account_tracking": account_tracking,
                "settings": {
                    "symbols": settings.symbols,
                    "trade_amount": decimal_to_api_string(settings.trade_amount),
                    "trail_percent": decimal_to_api_string(settings.trail_percent),
                    "time_in_force": settings.time_in_force,
                    "poll_seconds": settings.poll_seconds,
                    "feed": settings.feed,
                    "average_days": settings.average_days,
                    "volume_multiplier": decimal_to_api_string(settings.volume_multiplier),
                    "max_hold_minutes": settings.max_hold_minutes,
                    "ml_enabled": settings.ml_enabled,
                    "linear_model_path": settings.linear_model_path,
                    "logistic_model_path": settings.logistic_model_path,
                    "ml_min_probability": settings.ml_min_probability,
                    "ml_min_expected_return": settings.ml_min_expected_return,
                    "sklearn_model_dir": settings.sklearn_model_dir,
                    "sklearn_selective_enabled": settings.sklearn_selective_enabled,
                    "sklearn_probability_threshold": settings.sklearn_probability_threshold,
                    "sklearn_expected_return_threshold": settings.sklearn_expected_return_threshold,
                    "sklearn_expected_return_enabled": settings.sklearn_expected_return_enabled,
                    "worldquant_enabled": settings.worldquant_enabled,
                    "worldquant_model_dir": settings.worldquant_model_dir,
                    "worldquant_expected_return_threshold": settings.worldquant_expected_return_threshold,
                    "worldquant_lookback_days": settings.worldquant_lookback_days,
                }
                if settings
                else None,
                "logs": list(self.logs[-100:]),
            }


def render_dashboard() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alpaca Paper Bot</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111312;
      --panel: #181b19;
      --panel-2: #20241f;
      --text: #f3f5ef;
      --muted: #aeb7ad;
      --line: #333a33;
      --accent: #ffd426;
      --danger: #ff6f61;
      --ok: #48d17c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: #151714;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      min-height: calc(100vh - 64px);
    }
    form {
      padding: 22px;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    label {
      display: grid;
      gap: 6px;
      margin-bottom: 14px;
      color: var(--muted);
      font-weight: 600;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: #0f1110;
      color: var(--text);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 18px;
    }
    button {
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 10px 13px;
      color: #111312;
      background: var(--accent);
      font-weight: 800;
      cursor: pointer;
    }
    button.secondary {
      color: var(--text);
      background: #242923;
      border-color: var(--line);
    }
    section {
      min-width: 0;
      padding: 22px;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 16px;
      color: var(--muted);
    }
    .account {
      margin-bottom: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #0f1110;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
    }
    .dot.running { background: var(--ok); }
    pre {
      margin: 0;
      height: calc(100vh - 130px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #0c0e0d;
      color: #dce5d9;
      white-space: pre-wrap;
    }
    @media (max-width: 840px) {
      main { grid-template-columns: 1fr; }
      form { border-right: 0; border-bottom: 1px solid var(--line); }
      pre { height: 55vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Alpaca Paper Bot</h1>
    <div class="status"><span id="dot" class="dot"></span><span id="state">Stopped</span></div>
  </header>
  <main>
    <form id="settings">
      <label>Symbols <input name="symbols" value="NVDA, TSLA, SPCX, GOOGL, AAPL, MSFT, AMZN, META, AVGO, AMD, MU, ORCL, PLTR" autocomplete="off"></label>
      <div class="grid">
        <label>Trade Amount $ <input name="trade_amount" value="5000" inputmode="decimal"></label>
        <label>Trail % <input name="trail_percent" value="2" inputmode="decimal"></label>
      </div>
      <div class="grid">
        <label>Poll Seconds <input name="poll_seconds" value="10" inputmode="decimal"></label>
        <label>Average Days <input name="average_days" value="21" inputmode="numeric"></label>
      </div>
      <div class="grid">
        <label>Volume Multiplier <input name="volume_multiplier" value="1.5" inputmode="decimal"></label>
        <label>Max Hold Minutes <input name="max_hold_minutes" value="60" inputmode="decimal"></label>
      </div>
      <div class="grid">
        <label>Feed
          <select name="feed">
            <option value="iex">IEX</option>
            <option value="sip">SIP</option>
          </select>
        </label>
      </div>
      <label>Time In Force
        <select name="time_in_force">
          <option value="day">DAY</option>
          <option value="gtc">GTC</option>
          <option value="ioc">IOC</option>
          <option value="fok">FOK</option>
        </select>
      </label>
      <label>
        <input type="hidden" name="ml_enabled" value="false">
        <input type="checkbox" name="ml_enabled" value="true" checked>
        Use ML buy gate
      </label>
      <label>
        <input type="hidden" name="sklearn_selective_enabled" value="false">
        <input type="checkbox" name="sklearn_selective_enabled" value="true">
        Use sklearn selective model
      </label>
      <label>Sklearn Model Dir <input name="sklearn_model_dir" value="models/sklearn_60m_market/hist_gradient_boosting_d3_l2_selective_60m" autocomplete="off"></label>
      <label>
        <input type="hidden" name="sklearn_expected_return_enabled" value="false">
        <input type="checkbox" name="sklearn_expected_return_enabled" value="true" checked>
        Use sklearn expected-return cutoff
      </label>
      <div class="grid">
        <label>Sklearn Probability Cutoff <input name="sklearn_probability_threshold" value="0.565" inputmode="decimal"></label>
        <label>Sklearn Return Cutoff <input name="sklearn_expected_return_threshold" value="0.004" inputmode="decimal"></label>
      </div>
      <label>
        <input type="hidden" name="worldquant_enabled" value="false">
        <input type="checkbox" name="worldquant_enabled" value="true" checked>
        Use WorldQuant return model
      </label>
      <label>WorldQuant Model Dir <input name="worldquant_model_dir" value="models/worldquant_live/wq_pca16_base_no_symbol_return_60m" autocomplete="off"></label>
      <div class="grid">
        <label>WQ Return Cutoff <input name="worldquant_expected_return_threshold" value="0.003" inputmode="decimal"></label>
        <label>WQ Lookback Days <input name="worldquant_lookback_days" value="30" inputmode="numeric"></label>
      </div>
      <div class="grid">
        <label>Min ML Probability <input name="ml_min_probability" value="0.565" inputmode="decimal"></label>
        <label>Min Expected Return <input name="ml_min_expected_return" value="0.004" inputmode="decimal"></label>
      </div>
      <label>Linear Model <input name="linear_model_path" value="models/year_1m/linear_regression_15m.json" autocomplete="off"></label>
      <label>Logistic Model <input name="logistic_model_path" value="models/year_1m/logistic_regression_15m.json" autocomplete="off"></label>
      <div class="actions">
        <button type="submit">Start Paper Bot</button>
        <button class="secondary" id="stop" type="button">Stop</button>
      </div>
    </form>
    <section>
      <div id="account" class="account">Balance tracking will appear after the bot starts.</div>
      <pre id="logs"></pre>
    </section>
  </main>
  <script>
    const dot = document.querySelector("#dot");
    const state = document.querySelector("#state");
    const logs = document.querySelector("#logs");
    const account = document.querySelector("#account");
    const form = document.querySelector("#settings");

    async function post(path, body) {
      const response = await fetch(path, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    async function refresh() {
      const response = await fetch("/api/status");
      const data = await response.json();
      dot.classList.toggle("running", data.running);
      state.textContent = data.running ? "Running" : "Stopped";
      const tracking = data.account_tracking || {};
      const start = tracking.start_account || {};
      const end = tracking.end_account || {};
      account.textContent = [
        "Balance Tracking",
        "Run ID: " + (tracking.run_id || "not recorded yet"),
        "Started: " + (tracking.run_started_at || "not recorded yet"),
        "Start equity: " + (start.equity || "not recorded yet"),
        "Start portfolio value: " + (start.portfolio_value || "not recorded yet"),
        "Start cash: " + (start.cash || "not recorded yet"),
        "Ended: " + (tracking.run_ended_at || "still running"),
        "End equity: " + (end.equity || "not recorded yet"),
        "Equity delta: " + (tracking.equity_delta || "not recorded yet"),
        "Record file: " + (tracking.record_path || "not recorded yet")
      ].join("\\n");
      const lines = data.logs || [];
      if (data.last_error) lines.push("Last error: " + data.last_error);
      logs.textContent = lines.join("\\n");
      logs.scrollTop = logs.scrollHeight;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        await post("/api/start", new URLSearchParams(new FormData(form)));
        await refresh();
      } catch (error) {
        alert(error.message);
      }
    });

    document.querySelector("#stop").addEventListener("click", async () => {
      try {
        await post("/api/stop", new URLSearchParams());
        await refresh();
      } catch (error) {
        alert(error.message);
      }
    });

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class BotServerHandler(BaseHTTPRequestHandler):
    runner: BotRunner

    def do_GET(self) -> None:
        if self.path == "/":
            self.respond_html(render_dashboard())
            return
        if self.path == "/api/status":
            self.respond_json(self.runner.snapshot())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/start":
                fields = self.read_form_fields()
                settings = parse_momentum_settings(fields)
                self.runner.start(settings)
                self.respond_json(self.runner.snapshot())
                return
            if self.path == "/api/stop":
                self.runner.stop()
                self.respond_json(self.runner.snapshot())
                return
            self.send_error(404)
        except BotError as error:
            self.respond_json({"error": str(error)}, status=400)

    def read_form_fields(self) -> dict[str, list[str]]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        return parse_qs(raw_body, keep_blank_values=True)

    def respond_html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def respond_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve_dashboard(host: str, port: int) -> None:
    runner = BotRunner()

    class Handler(BotServerHandler):
        pass

    Handler.runner = runner
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Alpaca paper bot dashboard running at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.", flush=True)
    finally:
        try:
            runner.stop()
            runner.wait_until_stopped(30)
        except BotError:
            pass
        server.server_close()


def build_broker(args: argparse.Namespace) -> AlpacaClient | DryRunBroker:
    if args.dry_run:
        return DryRunBroker()

    credentials = AlpacaCredentials.from_env(live_trading=args.live_trading)
    trading_url = ALPACA_LIVE_TRADING_URL if args.live_trading else ALPACA_PAPER_TRADING_URL
    return AlpacaClient(credentials, trading_base_url=trading_url)


def add_common_order_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", required=True, help="Ticker symbol, for example AAPL.")
    parser.add_argument("--qty", required=True, type=lambda value: parse_decimal(value, "qty"))
    parser.add_argument(
        "--trail",
        required=True,
        type=lambda value: parse_decimal(value, "trail"),
        help="Dollar trailing amount, for example 1 means sell after a $1 pullback from the highest price seen.",
    )
    parser.add_argument(
        "--time-in-force",
        default="gtc",
        choices=["day", "gtc", "opg", "cls", "ioc", "fok"],
    )
    parser.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Actually submit orders. Without this flag, orders are printed only.",
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument(
        "--live-trading",
        action="store_true",
        help="Use Alpaca live trading instead of paper trading. Requires --live.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place or simulate a trailing stop sell for a long stock position."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    native_parser = subparsers.add_parser(
        "native",
        help="Submit a broker-native trailing stop sell order.",
    )
    add_common_order_args(native_parser)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Monitor Alpaca latest-trade data locally and submit a market sell after the trail is breached.",
    )
    add_common_order_args(monitor_parser)
    monitor_parser.add_argument("--poll-seconds", type=float, default=5)
    monitor_parser.add_argument("--feed", default="iex", choices=["iex", "sip"])

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run the trailing stop logic against a comma-separated price list.",
    )
    add_common_order_args(simulate_parser)
    simulate_parser.add_argument("--prices", required=True, type=parse_price_list)

    momentum_parser = subparsers.add_parser(
        "momentum",
        help="Buy when volume beats its recent average and price is rising, then sell on a trailing loss.",
    )
    add_common_order_args(momentum_parser)
    momentum_parser.add_argument("--poll-seconds", type=float, default=10)
    momentum_parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    momentum_parser.add_argument(
        "--average-days",
        type=int,
        default=21,
        help="Number of prior daily bars used for average volume. 21 is roughly one trading month.",
    )
    momentum_parser.add_argument(
        "--volume-multiplier",
        type=lambda value: parse_decimal(value, "volume multiplier"),
        default=Decimal("1"),
        help="Require current daily volume to exceed average volume times this multiplier.",
    )

    server_parser = subparsers.add_parser(
        "server",
        help="Run a local dashboard for starting and stopping the paper momentum bot.",
    )
    server_parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    server_parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Load trained ML models and predict from recent 1-minute historical bars.",
    )
    predict_parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    predict_parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    predict_parser.add_argument("--lookback-days", type=int, default=10)
    predict_parser.add_argument("--linear-model", default=DEFAULT_LINEAR_MODEL_PATH)
    predict_parser.add_argument("--logistic-model", default=DEFAULT_LOGISTIC_MODEL_PATH)
    predict_parser.add_argument(
        "--features",
        action="store_true",
        help="Include the exact computed feature values in the output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "server":
        serve_dashboard(args.host, args.port)
        return 0

    if args.command == "predict":
        if args.lookback_days <= 0:
            raise BotError("--lookback-days must be greater than 0.")
        credentials = AlpacaCredentials.from_env()
        client = AlpacaClient(credentials, trading_base_url=ALPACA_PAPER_TRADING_URL)
        try:
            linear_model = load_model(Path(args.linear_model))
            logistic_model = load_model(Path(args.logistic_model))
        except MlModelError as error:
            raise BotError(str(error)) from error
        predictions = []
        for symbol in parse_symbols(args.symbols):
            prediction = predict_symbol_from_recent_bars(
                client=client,
                symbol=symbol,
                feed=args.feed,
                linear_model=linear_model,
                logistic_model=logistic_model,
                lookback_days=args.lookback_days,
            )
            payload = {
                "symbol": symbol,
                "timestamp": prediction.timestamp,
                "expected_return_15m": prediction.expected_return_15m,
                "prob_profitable_15m": prediction.prob_profitable_15m,
                "linear_model": args.linear_model,
                "logistic_model": args.logistic_model,
            }
            if args.features:
                payload["features"] = prediction.features
            predictions.append(payload)
        print(json.dumps(predictions, indent=2, sort_keys=True))
        return 0

    if args.live_trading and args.dry_run:
        raise BotError("--live-trading only matters when --live is also passed.")

    broker = build_broker(args)

    if args.command == "native":
        order = broker.place_trailing_stop_sell(
            symbol=args.symbol,
            qty=args.qty,
            trail=args.trail,
            time_in_force=args.time_in_force,
        )
        print(json.dumps(order, indent=2, sort_keys=True))
        return 0

    if args.command == "simulate":
        triggered = monitor_prices(
            prices=args.prices,
            symbol=args.symbol,
            qty=args.qty,
            trail=args.trail,
            broker=broker,
            time_in_force=args.time_in_force,
        )
        return 0 if triggered else 2

    if args.command == "monitor":
        if args.dry_run:
            raise BotError(
                "monitor mode needs Alpaca market data credentials. Use simulate mode for no-credential tests, "
                "or pass --live to enable Alpaca access."
            )
        if args.poll_seconds <= 0:
            raise BotError("--poll-seconds must be greater than 0.")
        if not isinstance(broker, AlpacaClient):
            raise BotError("monitor mode requires an Alpaca broker client.")
        monitor_prices(
            prices=live_price_stream(
                client=broker,
                symbol=args.symbol,
                poll_seconds=args.poll_seconds,
                feed=args.feed,
            ),
            symbol=args.symbol,
            qty=args.qty,
            trail=args.trail,
            broker=broker,
            time_in_force=args.time_in_force,
        )
        return 0

    if args.command == "momentum":
        if args.dry_run:
            raise BotError(
                "momentum mode needs Alpaca market data credentials. Pass --live to use paper trading API access."
            )
        if args.live_trading:
            raise BotError("momentum mode is intentionally paper-only for now; remove --live-trading.")
        if args.poll_seconds <= 0:
            raise BotError("--poll-seconds must be greater than 0.")
        if args.average_days <= 0:
            raise BotError("--average-days must be greater than 0.")
        if not isinstance(broker, AlpacaClient):
            raise BotError("momentum mode requires an Alpaca client.")
        monitor_volume_momentum(
            client=broker,
            broker=broker,
            symbol=args.symbol,
            qty=args.qty,
            trail=args.trail,
            time_in_force=args.time_in_force,
            poll_seconds=args.poll_seconds,
            feed=args.feed,
            average_days=args.average_days,
            volume_multiplier=args.volume_multiplier,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BotError, argparse.ArgumentTypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
