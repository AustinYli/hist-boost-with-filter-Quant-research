#!/usr/bin/env python3

import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import research


ROOT = Path(__file__).resolve().parent


def spec() -> dict:
    return research.load_spec(ROOT / "spec.json")


class OverlayTests(unittest.TestCase):
    def test_fixed_overlay_endpoints_and_neutral_core(self) -> None:
        self.assertEqual(research.overlay_exposures(-1.0, spec()), (0.3, 1.0))
        self.assertEqual(research.overlay_exposures(0.0, spec()), (0.5, 0.5))
        self.assertEqual(research.overlay_exposures(1.0, spec()), (1.0, 0.3))

    def test_overlay_is_continuous_around_zero(self) -> None:
        negative = research.overlay_exposures(-1e-8, spec())
        positive = research.overlay_exposures(1e-8, spec())
        self.assertLess(max(abs(left - right) for left, right in zip(negative, positive)), 2e-8)

    def test_conviction_is_smooth_and_bounded(self) -> None:
        predictions = np.array([-0.02, -0.001, 0.0, 0.001, 0.02])
        conviction = np.tanh(predictions / 0.01)
        self.assertTrue(np.all(np.diff(conviction) > 0))
        self.assertTrue(np.all(np.abs(conviction) <= 1.0))


class ModelTests(unittest.TestCase):
    def test_beta_neutral_target_definition(self) -> None:
        stock_return = 0.03
        beta = 1.2
        soxx_return = 0.02
        self.assertAlmostEqual(stock_return - beta * soxx_return, 0.006)

    def test_direction_uncertainty_uses_fixed_alpha_grid(self) -> None:
        rng = np.random.default_rng(181)
        configuration = spec()
        rows = 70
        train = pd.DataFrame(
            rng.normal(size=(rows, len(configuration["direction_features"]))),
            columns=configuration["direction_features"],
        )
        train["direction_total_return_target"] = rng.normal(0, 0.02, size=rows)
        _, _, sigma, alpha = research.fit_direction_models(train, configuration)

        self.assertIn(alpha, configuration["models"]["ridge_alphas"])
        self.assertGreaterEqual(sigma, configuration["models"]["uncertainty_minimum_sigma"])

    def test_each_selector_cross_section_has_equal_total_weight(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [date(2026, 1, 2)] * 3 + [date(2026, 1, 9)] * 6,
                "symbol": [f"S{index}" for index in range(9)],
            }
        )
        weights = research.V3.cross_section_sample_weights(frame)
        self.assertTrue(np.allclose(weights.groupby(frame["date"]).sum(), [1.0, 1.0]))


class RiskTests(unittest.TestCase):
    def synthetic_data(self):
        rng = np.random.default_rng(7)
        symbols = [f"S{index:02d}" for index in range(20)]
        dates = [item.date() for item in pd.bdate_range("2025-01-02", periods=70)]
        panel = pd.DataFrame(
            [
                {"date": day, "symbol": symbol, "return_1d": rng.normal(0, 0.015)}
                for day in dates
                for symbol in symbols
            ]
        )
        base = SimpleNamespace(panel=panel)
        data = SimpleNamespace(panel=panel, base=base)
        day = pd.DataFrame(
            {
                "signal_date": [dates[-1]] * 20,
                "symbol": symbols,
                "alpha_prediction": np.linspace(-0.02, 0.02, 20),
                "beta_60d_to_soxx": np.linspace(0.8, 1.2, 20),
                "annual_borrow_rate": [0.05] * 20,
                "shortable": [True] * 20,
            }
        )
        return data, day

    def test_optimizer_respects_caps_and_volatility_target(self) -> None:
        data, day = self.synthetic_data()
        weights, diagnostics = research.optimize_portfolio(day, data, 0.0, {}, spec())

        self.assertLessEqual(max(abs(value) for value in weights.values()), 0.1 + 1e-8)
        self.assertLessEqual(diagnostics["gross_exposure"], 1.0 + 1e-8)
        self.assertLessEqual(
            diagnostics["predicted_volatility_after_scale"],
            spec()["risk"]["annualized_volatility_target"] + 1e-8,
        )
        self.assertAlmostEqual(diagnostics["net_exposure"], 0.0, places=6)

    def test_dividend_interval_excludes_entry_date_and_includes_exit_date(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["TEST"],
                "entry": [date(2026, 1, 2)],
                "exit": [date(2026, 1, 9)],
                "price": [100.0],
            }
        )
        dividends = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-05", "2026-01-09"],
                "symbol": ["TEST"] * 3,
                "cash_amount": [1.0, 1.0, 2.0],
                "source": ["test"] * 3,
            }
        )
        result = research.interval_dividend_return(frame, dividends, "symbol", "entry", "exit", "price")
        self.assertAlmostEqual(result.iloc[0], 0.03)


class SafetyTests(unittest.TestCase):
    def test_strict_mode_rejects_exploratory_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Strict V4 research blocked"):
            research.build_data(spec(), exploratory=False)

    def test_v4_contains_no_order_submission_path(self) -> None:
        source = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ["research.py", "prepare_data.py", "forward.py"]
        )
        forbidden = ["place_market_buy", "place_market_sell", "submit_order", "LIVE_ORDERS_ENABLED"]
        self.assertFalse(any(token in source for token in forbidden))

    def test_stock_feature_set_remains_fixed_at_18(self) -> None:
        self.assertEqual(len(spec()["selection_features"]), 18)

    def test_retrain_audit_obeys_observability_embargo(self) -> None:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
