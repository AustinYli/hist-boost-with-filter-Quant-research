#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import confirmation as CF
import historical_data as HD


def period_frame() -> pd.DataFrame:
    rows = []
    for index, (soxx, prediction, net_exposure, net_return) in enumerate(
        [(-0.08, 0.01, 0.20, -0.01), (0.04, -0.02, -0.30, 0.02), (-0.03, -0.01, 0.05, 0.01)]
    ):
        rows.append(
            {
                "signal_date": f"2020-01-{2 + index * 7:02d}",
                "entry_date": f"2020-01-{3 + index * 7:02d}",
                "exit_date": f"2020-01-{8 + index * 7:02d}",
                "strategy": "v4_continuous",
                "gross_return": net_return + 0.001,
                "net_return": net_return,
                "transaction_cost": 0.0008,
                "borrow_cost": 0.0002,
                "turnover": 1.0,
                "direction_prediction": prediction,
                "direction_target": soxx,
                "direction_conviction": np.tanh(prediction / 0.05),
                "future_soxx_return_5d": soxx,
                "future_spy_return_5d": soxx * 0.7,
                "future_qqq_return_5d": soxx * 0.9,
                "weights": json.dumps({"A": 0.1, "B": -0.1}),
                "optimizer_success": True,
                "conviction": prediction,
                "risk_scale": 1.0,
                "predicted_volatility_after_scale": 0.15,
                "gross_exposure": 0.2,
                "net_exposure": net_exposure,
                "estimated_soxx_beta": 0.0,
                "maximum_abs_weight": 0.1,
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(periods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number, period in enumerate(periods.itertuples(index=False)):
        for index in range(20):
            symbol = "A" if index == 0 else "B" if index == 19 else f"S{index:02d}"
            stock_return = 0.01 + index * 0.001
            rows.append(
                {
                    "signal_date": period.signal_date,
                    "symbol": symbol,
                    "direction_prediction": period.direction_prediction,
                    "forecast_error_sigma": 0.05,
                    "direction_conviction": period.direction_conviction,
                    "direction_target": period.direction_target,
                    "alpha_prediction": float(index),
                    "alpha_target": float(index) / 100,
                    "future_stock_return_5d": stock_return,
                }
            )
    return pd.DataFrame(rows)


class DatasetLockTests(unittest.TestCase):
    def test_inadequate_current_data_cannot_be_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "dataset-lock.json"
            with self.assertRaises(ValueError):
                CF.freeze_dataset(lock_path=lock, confirmation=CF.FREEZE_CONFIRMATION)
            self.assertFalse(lock.exists())

    def test_dataset_fingerprint_detects_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in HD.INSTALL_TABLES + ["source_manifest.json"]:
                path = root / name
                path.write_text(f"immutable {name}\n", encoding="utf-8")
                files[name] = {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": HD.sha256(path),
                }
            lock = root / "dataset-lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "data_contract_sha256": HD.sha256(HD.CONTRACT_PATH),
                        "frozen_hypothesis_manifest_sha256": HD.sha256(HD.LOCK_PATH),
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(CF.verify_dataset_lock(lock)["valid"])
            (root / "stock_bars.csv").write_text("changed\n", encoding="utf-8")
            self.assertFalse(CF.verify_dataset_lock(lock)["valid"])

    def test_one_shot_receipt_cannot_be_claimed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            CF.claim_receipt(receipt, {"status": "claimed"})
            with self.assertRaises(FileExistsError):
                CF.claim_receipt(receipt, {"status": "claimed-again"})


class ReportingTests(unittest.TestCase):
    def test_regime_counts_keep_three_questions_separate(self) -> None:
        periods = period_frame()
        spec = CF.V4.load_spec(HD.SPEC_PATH)
        report = CF.regime_report(periods, spec)

        self.assertEqual(report["counts"]["realized_negative_soxx_periods"], 2)
        self.assertEqual(report["counts"]["predicted_negative_soxx_periods"], 2)
        self.assertEqual(report["counts"]["negative_net_v4_periods"], 1)

    def test_books_reconstruct_reported_gross_return(self) -> None:
        periods = period_frame()
        predictions = prediction_frame(periods)
        for index, row in periods.iterrows():
            day = predictions[predictions["signal_date"].eq(row["signal_date"])].set_index("symbol")
            gross = 0.1 * day.at["A", "future_stock_return_5d"] - 0.1 * day.at["B", "future_stock_return_5d"]
            periods.at[index, "gross_return"] = gross
            periods.at[index, "net_return"] = gross - row["transaction_cost"] - row["borrow_cost"]
        books = CF.book_contributions(periods, predictions)

        error = (books["gross_return_reconstructed"] - books["gross_return_reported"]).abs().max()
        self.assertAlmostEqual(float(error), 0.0)

    def test_selector_and_direction_diagnostics_are_populated(self) -> None:
        periods = period_frame()
        predictions = prediction_frame(periods)
        selector = CF.selector_metrics(predictions)
        direction = CF.direction_report(predictions)

        self.assertEqual(selector["periods"], 3)
        self.assertAlmostEqual(selector["mean_rank_ic"], 1.0)
        self.assertEqual(direction["periods"], 3)
        self.assertTrue(direction["calibration_bins"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
