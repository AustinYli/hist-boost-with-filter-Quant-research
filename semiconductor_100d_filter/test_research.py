from __future__ import annotations

import unittest

from semiconductor_100d_filter.research import ORIGINAL_OOS_END, copied_model_manifest


class IsolatedCopyTests(unittest.TestCase):
    def test_model_copies_are_physical_and_hash_identical(self):
        manifest = copied_model_manifest()
        self.assertEqual(manifest["copy_type"], "physical_independent_files")
        self.assertEqual(
            [(item["moving_average_days"], item["horizon_days"]) for item in manifest["copies"]],
            [(100, 3), (100, 5), (300, 3), (300, 5)],
        )
        for copy_record in manifest["copies"]:
            for file_record in copy_record["files"]:
                self.assertEqual(file_record["source_sha256"], file_record["copy_sha256"])
                self.assertFalse(file_record["copy_is_symlink"])
                self.assertNotEqual(file_record["source"], file_record["copy"])

    def test_original_comparison_endpoints_are_locked(self):
        self.assertEqual(str(ORIGINAL_OOS_END[3]), "2026-07-27")
        self.assertEqual(str(ORIGINAL_OOS_END[5]), "2026-07-23")


if __name__ == "__main__":
    unittest.main()
