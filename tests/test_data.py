import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import assign_row_split_codes, load_config, prepare_dataset  # noqa: E402


class DataPipelineTest(unittest.TestCase):
    def test_row_split_is_deterministic_and_exclusive(self) -> None:
        rows = np.arange(100_000, dtype=np.uint64)
        first = assign_row_split_codes(123, rows, [0.70, 0.15, 0.15], 2026)
        second = assign_row_split_codes(123, rows, [0.70, 0.15, 0.15], 2026)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isin(first, [0, 1, 2]).all())
        self.assertEqual(len(rows), sum(np.count_nonzero(first == code) for code in range(3)))

    def test_prepare_preserves_distribution_and_writes_leakage_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "output"
            dataset.mkdir()
            rng = np.random.default_rng(9)
            for label, rows in (("BENIGN", 900), ("DrDoS_DNS", 600), ("Syn", 300)):
                frame = pd.DataFrame({
                    "Flow ID": [f"{label}-{index}" for index in range(rows)],
                    "Timestamp": pd.date_range("2019-01-01", periods=rows, freq="s").astype(str),
                    "Feature A": rng.normal(size=rows),
                    "Feature B": rng.normal(size=rows),
                    "Label": label,
                })
                frame.loc[::37, "Feature A"] = np.nan
                frame.loc[::53, "Feature B"] = np.inf
                frame.to_parquet(dataset / f"{label}.parquet", index=False, row_group_size=113)

            config = load_config(PROJECT_ROOT / "config" / "data.json")
            config["dataset"]["data_dir"] = str(dataset)
            config["output"]["rows_per_part"] = 127
            manifest = prepare_dataset(config, output)

            self.assertEqual(sum(manifest["split"]["sizes"].values()), 1800)
            self.assertTrue(manifest["split"]["group_aware"])
            self.assertTrue(manifest["leakage_audit"]["passed"])
            self.assertTrue(manifest["leakage_audit"]["sample_id_assertion_passed"])
            self.assertTrue(manifest["leakage_audit"]["group_assertion_passed"])
            self.assertTrue(all(not values for values in manifest["split"]["classes_missing_from_split"].values()))

            with (output / "preprocessing.json").open(encoding="utf-8") as handle:
                preprocessing = json.load(handle)
            self.assertEqual(preprocessing["scaling"], "none")
            self.assertEqual(preprocessing["imbalance_handling"], "none")
            self.assertEqual(preprocessing["categorical_features"], [])
            dropped = {item["column"] for item in preprocessing["dropped_columns"]}
            self.assertIn("Flow ID", dropped)
            self.assertIn("Timestamp", dropped)

            split_frames = {}
            for split, parts in manifest["parts"].items():
                split_frames[split] = pd.concat(
                    [pd.read_parquet(output / item["path"]) for item in parts], ignore_index=True
                )
                self.assertIn("_label", split_frames[split])
                self.assertNotIn("_label_name", split_frames[split])
            identities = {
                split: set(zip(frame["_sample_file_id"], frame["_sample_row_id"]))
                for split, frame in split_frames.items()
            }
            self.assertFalse(identities["train"] & identities["validation"])
            self.assertFalse(identities["train"] & identities["test"])
            self.assertFalse(identities["validation"] & identities["test"])

            with (output / "data_profile.json").open(encoding="utf-8") as handle:
                profile = json.load(handle)
            self.assertEqual(profile["total_selected_rows"], 1800)
            self.assertEqual(profile["feature_count"], 2)


if __name__ == "__main__":
    unittest.main()
