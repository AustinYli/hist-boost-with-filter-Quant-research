#!/usr/bin/env python3

import argparse
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import semiconductor_multi_horizon_paper_bot as bot


def expanding_signal(created_at: str, data_asof_date: str) -> dict:
    return {
        "created_at_utc": created_at,
        "data_asof_date": data_asof_date,
        "model": {"training_mode": "expanding_window"},
    }


class SignalFreshnessTests(unittest.TestCase):
    def args(self, signal_mode: str = bot.SIGNAL_MODE_EXPANDING) -> argparse.Namespace:
        return argparse.Namespace(signal_mode=signal_mode)

    @patch.object(bot, "latest_available_date", return_value=(date(2026, 8, 5), []))
    def test_already_executed_signal_can_remain_until_cadence_is_due(self, _latest) -> None:
        old = expanding_signal("old-id", "2026-07-31")
        state = {"last_executed_signal_ids": {"5": "old-id"}}

        result, latest_date = bot.ensure_pending_signal_is_fresh(self.args(), 5, state, old)

        self.assertIs(result, old)
        self.assertEqual(latest_date, "2026-08-05")

    @patch.object(bot, "signal_log_for_horizon")
    @patch.object(bot, "latest_available_date", return_value=(date(2026, 8, 5), []))
    def test_pending_stale_signal_uses_archived_latest_close(self, _latest, signal_log) -> None:
        old = expanding_signal("old-id", "2026-07-31")
        fresh = expanding_signal("fresh-id", "2026-08-05")
        state = {"last_executed_signal_ids": {"5": "previous-id"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.jsonl"
            path.write_text(
                '{"created_at_utc":"fresh-id","data_asof_date":"2026-08-05",'
                '"model":{"training_mode":"expanding_window"}}\n',
                encoding="utf-8",
            )
            signal_log.return_value = path

            result, latest_date = bot.ensure_pending_signal_is_fresh(self.args(), 5, state, old)

        self.assertEqual(result, fresh)
        self.assertEqual(latest_date, "2026-08-05")

    @patch.object(bot, "latest_available_date", return_value=(date(2026, 8, 5), []))
    def test_pending_stale_fixed_signal_is_rejected(self, _latest) -> None:
        old = {
            "created_at_utc": "old-id",
            "data_asof_date": "2026-07-31",
            "model": {"training_mode": "fixed_weight"},
        }
        state = {"last_executed_signal_ids": {"5": "previous-id"}}

        with self.assertRaisesRegex(bot.BotError, "Refusing stale 5d fixed signal"):
            bot.ensure_pending_signal_is_fresh(self.args(bot.SIGNAL_MODE_FIXED), 5, state, old)

    def test_account_state_selects_its_own_signal_from_shared_log(self) -> None:
        records = [
            expanding_signal("paper-id", "2026-08-05"),
            expanding_signal("live-id", "2026-08-06"),
        ]

        self.assertEqual(bot.record_for_created_at(records, "paper-id"), records[0])
        self.assertIsNone(bot.record_for_created_at(records, "missing-id"))


if __name__ == "__main__":
    unittest.main()
