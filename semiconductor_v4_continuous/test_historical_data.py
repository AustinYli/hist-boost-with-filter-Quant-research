#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import historical_data as hd


def write_bundle(stage: Path) -> tuple[dict, dict]:
    dates = [item.date() for item in pd.bdate_range("2020-01-02", periods=90)]
    symbols = ["PERMNO:10001", "PERMNO:10002", "PERMNO:10003"]
    market_prices = np.concatenate(
        [
            np.linspace(100, 115, 25),
            np.linspace(115, 75, 8),
            np.linspace(75, 118, 22),
            np.linspace(118, 125, 35),
        ]
    )

    stocks = []
    for number, symbol in enumerate(symbols):
        end = 74 if symbol == symbols[-1] else len(dates)
        for index, date in enumerate(dates[:end]):
            close = 25.0 + number * 10.0 + index * 0.04
            stocks.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": close - 0.05,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 2_000_000,
                    "trade_count": 10_000,
                    "vwap": close,
                }
            )
    pd.DataFrame(stocks, columns=hd.TABLE_COLUMNS["stock_bars.csv"]).to_csv(
        stage / "stock_bars.csv", index=False
    )

    market = []
    for symbol, multiplier in [("SOXX", 1.0), ("SPY", 1.5), ("QQQ", 1.2)]:
        for date, price in zip(dates, market_prices * multiplier):
            market.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 5_000_000,
                    "trade_count": 20_000,
                    "vwap": price,
                }
            )
    pd.DataFrame(market, columns=hd.TABLE_COLUMNS["market_bars.csv"]).to_csv(
        stage / "market_bars.csv", index=False
    )

    master = []
    membership = []
    for index, symbol in enumerate(symbols):
        delisted = symbol == symbols[-1]
        valid_to = dates[72] if delisted else ""
        master.append(
            {
                "symbol": symbol,
                "identifier_type": "permno",
                "identifier_value": symbol.split(":")[1],
                "ticker": f"S{index}",
                "valid_from": dates[0],
                "valid_to": valid_to,
                "cusip": f"0000000{index}",
                "isin": "",
                "source": "synthetic licensed export",
            }
        )
        membership.append(
            {
                "symbol": symbol,
                "valid_from": dates[0],
                "valid_to": valid_to,
                "industry": "Semiconductors & Semiconductor Equipment",
                "security_type": "common_equity",
                "us_listed": True,
                "survivorship_safe": True,
                "delisted": delisted,
                "delisting_handled": True,
                "source": "synthetic licensed export",
            }
        )
    pd.DataFrame(master, columns=hd.TABLE_COLUMNS["security_master.csv"]).to_csv(
        stage / "security_master.csv", index=False
    )
    membership_frame = pd.DataFrame(
        membership, columns=hd.TABLE_COLUMNS["universe_membership.csv"]
    )
    membership_frame.to_csv(stage / "universe_membership.csv", index=False)

    borrow = []
    for row in stocks:
        date = pd.Timestamp(row["timestamp"]).date()
        symbol = row["symbol"]
        valid_to = dates[72] if symbol == symbols[-1] else dates[-1]
        if dates[5] <= date <= valid_to:
            borrow.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "shortable": symbol != symbols[-1] or date < dates[68],
                    "annual_borrow_rate": 0.03 if symbol != symbols[-1] else 0.40,
                    "source": "synthetic securities finance tape",
                }
            )
    pd.DataFrame(borrow, columns=hd.TABLE_COLUMNS["borrow_costs.csv"]).to_csv(
        stage / "borrow_costs.csv", index=False
    )
    pd.DataFrame(
        [{"date": dates[30], "symbol": symbols[0], "cash_amount": 0.10, "source": "synthetic actions"}],
        columns=hd.TABLE_COLUMNS["dividends.csv"],
    ).to_csv(stage / "dividends.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": dates[40],
                "symbol": symbols[0],
                "action_type": "split",
                "value": 2.0,
                "handled": True,
                "source": "synthetic actions",
            },
            {
                "date": dates[73],
                "symbol": symbols[-1],
                "action_type": "delisting_return",
                "value": -0.60,
                "handled": True,
                "source": "synthetic delisting file",
            },
        ],
        columns=hd.TABLE_COLUMNS["corporate_actions.csv"],
    ).to_csv(stage / "corporate_actions.csv", index=False)

    provenance = {
        "bundle_id": "synthetic-test",
        "price_adjustment": "split_adjusted_dividend_unadjusted",
        "coverage": {
            "point_in_time_membership": True,
            "active_and_inactive_securities": True,
            "delisting_outcomes": True,
            "borrow_availability": True,
            "borrow_fees": True,
            "dividends": True,
            "corporate_actions": True,
        },
        "sources": [{"name": "synthetic"}],
    }
    (stage / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    raw = stage / "raw"
    raw.mkdir()
    (raw / "vendor-export.dat").write_bytes(b"immutable synthetic licensed export\n")
    hd.register_raw_exports(
        stage,
        vendor="synthetic-vendor",
        version="test-v1",
        as_of_date="2026-08-09",
        license_reference="unit-test-license",
    )

    contract = hd.load_json(hd.CONTRACT_PATH)
    contract["challenge_protocol"]["historical_unseen_start"] = str(dates[5])
    contract["acceptance"].update(
        {
            "minimum_history_years": 0.20,
            "minimum_market_sessions": 80,
            "minimum_stock_symbols": 2,
            "minimum_realized_five_session_periods": 8,
            "minimum_negative_soxx_periods": 1,
            "minimum_distinct_twenty_percent_drawdown_episodes": 1,
            "maximum_missing_borrow_fraction": 0.0,
            "minimum_sessions_before_historical_oos": 5,
        }
    )
    spec = hd.load_json(hd.SPEC_PATH)
    spec["data"]["research_label_cutoff"] = str(dates[-7])
    spec["execution"]["decision_grid_anchor_date"] = str(dates[-10])
    spec["universe"].update(
        {
            "minimum_history_sessions": 10,
            "liquidity_window_sessions": 5,
            "minimum_median_dollar_volume": 1_000_000,
            "minimum_names_to_trade": 2,
        }
    )
    return contract, spec


class HistoricalDataAuditTests(unittest.TestCase):
    def test_modified_raw_export_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            contract, spec = write_bundle(stage)
            (stage / "raw" / "vendor-export.dat").write_bytes(b"modified\n")
            report = hd.audit_bundle(stage, contract=contract, spec=spec)

        self.assertFalse(report["source_strict_ready"])
        self.assertTrue(any("raw_export_fingerprint" in item for item in report["blockers"]))

    def test_synthetic_point_in_time_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            contract, spec = write_bundle(stage)
            report = hd.audit_bundle(stage, contract=contract, spec=spec)

        self.assertTrue(report["source_strict_ready"], report["blockers"])
        self.assertTrue(report["research_sample_adequate"], report["blockers"])
        self.assertTrue(report["v4_unchanged_rerun_ready"])
        self.assertEqual(report["metrics"]["delisted_symbols"], 1)
        self.assertEqual(report["metrics"]["borrow_missing_fraction"], 0.0)

    def test_missing_borrow_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            contract, spec = write_bundle(stage)
            pd.DataFrame(columns=hd.TABLE_COLUMNS["borrow_costs.csv"]).to_csv(
                stage / "borrow_costs.csv", index=False
            )
            report = hd.audit_bundle(stage, contract=contract, spec=spec)

        self.assertFalse(report["source_strict_ready"])
        self.assertEqual(report["metrics"]["borrow_missing_fraction"], 1.0)
        self.assertTrue(any("borrow_panel_coverage" in item for item in report["blockers"]))

    def test_normalized_bundle_preserves_strict_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            output = root / "normalized"
            stage.mkdir()
            contract, spec = write_bundle(stage)
            report = hd.audit_bundle(stage, contract=contract, spec=spec)

            hd.write_normalized(stage, output, report)
            normalized = hd.audit_bundle(output, contract=contract, spec=spec)
            manifest_exists = (output / "source_manifest.json").exists()

        self.assertTrue(normalized["source_strict_ready"], normalized["blockers"])
        self.assertTrue(manifest_exists)

    def test_current_exploratory_bundle_remains_blocked(self) -> None:
        report = hd.audit_bundle(hd.DATA_DIR)

        self.assertFalse(report["v4_unchanged_rerun_ready"])
        self.assertEqual(report["metrics"]["realized_periods"], 33)
        self.assertEqual(report["metrics"]["negative_periods"], 9)


class DelistingMaterializationTests(unittest.TestCase):
    def test_delisting_return_creates_terminal_tail_and_ends_signal_eligibility(self) -> None:
        dates = [item.date() for item in pd.bdate_range("2026-01-02", periods=12)]
        stock = pd.DataFrame(
            [
                {
                    "timestamp": date,
                    "symbol": "PERMNO:1",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 100,
                    "trade_count": 1,
                    "vwap": 10.0,
                }
                for date in dates[:4]
            ]
        )
        market = pd.DataFrame({"timestamp": dates})
        membership = pd.DataFrame(
            [
                {
                    "symbol": "PERMNO:1",
                    "valid_from": dates[0],
                    "valid_to": dates[3],
                    "delisted": True,
                }
            ]
        )
        actions = pd.DataFrame(
            [
                {
                    "date": dates[4],
                    "symbol": "PERMNO:1",
                    "action_type": "delisting_return",
                    "value": -1.0,
                    "handled": True,
                    "source": "test",
                }
            ]
        )

        result, adjusted_membership, audit = hd.materialize_delisting_tails(
            stock, market, membership, actions, horizon=5
        )

        synthetic = result[pd.to_datetime(result["timestamp"]).dt.date > dates[3]]
        self.assertEqual(len(synthetic), 6)
        self.assertTrue((synthetic[["open", "high", "low", "close"]] == 0.0).all().all())
        self.assertEqual(str(adjusted_membership.iloc[0]["valid_to"]), str(dates[2]))
        self.assertEqual(audit[0]["terminal_value"], 0.0)


class FreezeTests(unittest.TestCase):
    def test_frozen_v4_manifest_matches(self) -> None:
        self.assertTrue(hd.verify_lock()["valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
