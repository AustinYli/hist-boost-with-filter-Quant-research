#!/usr/bin/env python3

import json
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import research


ROOT = Path(__file__).resolve().parent


def test_spec() -> dict:
    return research.load_spec(ROOT / "spec.json")


def market_bars(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for symbol, offset in [("SOXX", 0.0), ("SPY", 10.0), ("QQQ", 20.0)]:
        for index, timestamp in enumerate(dates):
            value = 100.0 + offset + index
            rows.append(
                {
                    "date": timestamp.date(),
                    "symbol": symbol,
                    "open": value,
                    "high": value + 1,
                    "low": value - 1,
                    "close": value + 0.5,
                    "volume": 1_000_000 + index,
                }
            )
    return pd.DataFrame(rows)


class TimingTests(unittest.TestCase):
    def test_open_to_open_label_uses_t_plus_1_and_t_plus_6(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=8)
        market = research.market_feature_frame(market_bars(dates), horizon=5)
        stocks = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(dates, utc=True),
                "date": [item.date() for item in dates],
                "symbol": ["TEST"] * len(dates),
                "open": np.arange(200.0, 208.0),
                "high": np.arange(201.0, 209.0),
                "low": np.arange(199.0, 207.0),
                "close": np.arange(200.5, 208.5),
                "volume": [2_000_000] * len(dates),
            }
        )
        panel = research.stock_feature_frame(stocks, market, test_spec())

        self.assertAlmostEqual(market.iloc[0]["direction_target"], 106.0 / 101.0 - 1.0)
        self.assertEqual(market.iloc[0]["label_entry_date"], dates[1].date())
        self.assertEqual(market.iloc[0]["label_exit_date"], dates[6].date())
        self.assertAlmostEqual(panel.iloc[0]["future_stock_return_5d"], 206.0 / 201.0 - 1.0)
        self.assertEqual(panel.iloc[0]["stock_label_entry_date"], dates[1].date())
        self.assertEqual(panel.iloc[0]["stock_label_exit_date"], dates[6].date())

    def test_grid_is_phase_aligned_to_august_10(self) -> None:
        sessions = [item.date() for item in pd.bdate_range("2026-08-03", "2026-08-10")]
        grid, anchor_index, in_data = research.make_decision_grid(
            sessions, sessions[0], date(2026, 8, 10), 5
        )

        self.assertTrue(in_data)
        self.assertEqual(anchor_index, 5)
        self.assertEqual(grid, [date(2026, 8, 3), date(2026, 8, 10)])

    def test_retrain_audit_obeys_label_embargo(self) -> None:
        path = ROOT / "artifacts" / "retrain_audit.csv"
        if not path.exists():
            self.skipTest("Run exploratory research first.")
        audit = pd.read_csv(path)
        self.assertTrue(
            (
                pd.to_datetime(audit["direction_last_label_exit"])
                <= pd.to_datetime(audit["retrain_signal_date"])
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(audit["selector_last_label_exit"])
                <= pd.to_datetime(audit["retrain_signal_date"])
            ).all()
        )


class PortfolioTests(unittest.TestCase):
    def prediction_day(self, direction: float, stock_return: float = 0.01) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_date": [date(2026, 1, 2)] * 20,
                "entry_date": [date(2026, 1, 5)] * 20,
                "exit_date": [date(2026, 1, 12)] * 20,
                "symbol": [f"S{index:02d}" for index in range(20)],
                "selection_prediction": np.arange(20, dtype=float),
                "direction_hgb_prediction": [direction] * 20,
                "direction_ridge_prediction": [direction] * 20,
                "future_stock_return_5d": [stock_return] * 20,
                "future_soxx_return_5d": [0.02] * 20,
                "future_spy_return_5d": [0.01] * 20,
                "future_qqq_return_5d": [0.015] * 20,
                "shortable": [True] * 20,
                "annual_borrow_rate": [0.05] * 20,
                "soxx_above_ma_200": [True] * 20,
            }
        )

    def test_direction_sign_controls_long_top_or_short_bottom(self) -> None:
        long_weights, long_side = research.weights_for_strategy(
            self.prediction_day(0.01), "hgb_directional", test_spec()
        )
        short_weights, short_side = research.weights_for_strategy(
            self.prediction_day(-0.01), "hgb_directional", test_spec()
        )

        self.assertEqual(long_side, "long")
        self.assertEqual(set(long_weights), {f"S{index:02d}" for index in range(10, 20)})
        self.assertAlmostEqual(sum(long_weights.values()), 1.0)
        self.assertEqual(short_side, "short")
        self.assertEqual(set(short_weights), {f"S{index:02d}" for index in range(10)})
        self.assertAlmostEqual(sum(short_weights.values()), -1.0)

    def test_each_cross_section_has_total_training_weight_one(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [date(2026, 1, 2)] * 2 + [date(2026, 1, 9)] * 5,
                "symbol": list("ABCDEFG"),
            }
        )
        weights = research.cross_section_sample_weights(frame)
        totals = weights.groupby(frame["date"]).sum()

        self.assertTrue(np.allclose(totals.to_numpy(), np.ones(2)))

    def test_primary_costs_include_entry_exit_and_short_borrow(self) -> None:
        spec = test_spec()
        long_result = research.backtest(self.prediction_day(0.01, 0.10), "hgb_directional", spec)
        short_result = research.backtest(self.prediction_day(-0.01, -0.10), "hgb_directional", spec)

        self.assertAlmostEqual(long_result.iloc[0]["transaction_cost"], 0.002)
        self.assertAlmostEqual(long_result.iloc[0]["net_return"], 0.098)
        expected_borrow = 0.05 * 5 / 252
        self.assertAlmostEqual(short_result.iloc[0]["borrow_cost"], expected_borrow)
        self.assertAlmostEqual(short_result.iloc[0]["net_return"], 0.10 - 0.002 - expected_borrow)


class SafetyTests(unittest.TestCase):
    def test_strict_mode_rejects_current_universe_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Strict V3 research blocked"):
            research.build_research_data(test_spec(), exploratory=False)

    def test_v3_has_no_order_submission_path(self) -> None:
        source = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ["research.py", "prepare_data.py", "forward.py"]
        )
        forbidden = ["place_market_buy", "place_market_sell", "submit_order", "LIVE_ORDERS_ENABLED"]
        self.assertFalse(any(token in source for token in forbidden))

    def test_spec_is_exactly_one_direction_and_one_selector_family(self) -> None:
        spec = test_spec()
        self.assertFalse(spec["hypothesis"]["neutral_state"])
        self.assertEqual(set(spec["models"]), {
            "direction_hgb",
            "direction_ridge",
            "selection_hgb",
            "retrain_every_sessions",
            "minimum_training_decision_dates",
            "fresh_model_each_retrain",
            "selector_cross_section_weight",
        })
        self.assertEqual(len(spec["selection_features"]), 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
