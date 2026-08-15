import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import IterationRecorder, TrainingPauseRequested, build_datasets, macro_f1_metric  # noqa: E402


class LightGBMResumeIntegrationTest(unittest.TestCase):
    def test_parquet_sequence_uses_bounded_memory_cache_and_does_not_build_test_dataset(self) -> None:
        try:
            import lightgbm  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("LightGBM and PyArrow are required")
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            parts = {name: [] for name in ("train", "validation", "test")}
            sizes = {"train": 37, "validation": 13, "test": 11}
            for split, rows in sizes.items():
                split_dir = prepared / "splits" / split
                split_dir.mkdir(parents=True)
                for number, (start, stop) in enumerate(((0, rows // 2), (rows // 2, rows))):
                    count = stop - start
                    frame = pd.DataFrame({
                        "f[0]": np.arange(start, stop, dtype=np.float32),
                        "f\"1:rate": np.linspace(0, 1, count, dtype=np.float32),
                        "_sample_file_id": np.zeros(count, dtype=np.uint64),
                        "_sample_row_id": np.arange(start, stop, dtype=np.uint64),
                        "_label": np.arange(start, stop, dtype=np.int32) % 3,
                    })
                    path = split_dir / f"part-{number:06d}.parquet"
                    frame.to_parquet(path, index=False)
                    parts[split].append({
                        "path": path.relative_to(prepared).as_posix(), "rows": count, "bytes": path.stat().st_size
                    })
            artifacts = {
                "sample_manifest.json": {"parts": parts, "split": {"sizes": sizes}},
                "preprocessing.json": {
                    "feature_columns_in_order": ["f[0]", "f\"1:rate"],
                    "feature_dtypes": {"f[0]": "float32", "f\"1:rate": "float32"},
                    "categorical_features": [], "scaling": "none", "imbalance_handling": "none",
                },
                "label_mapping.json": {"a": 0, "b": 1, "c": 2},
                "data_profile.json": {"safe_to_materialize_for_lightgbm": False},
            }
            for name, payload in artifacts.items():
                (prepared / name).write_text(json.dumps(payload), encoding="utf-8")
            config = json.loads((PROJECT_ROOT / "config" / "train.json").read_text(encoding="utf-8"))
            config["dataset"]["sequence_batch_rows"] = 7
            config["dataset"]["sequence_row_group_cache_mb"] = 1
            bundle = build_datasets(prepared, config)
            booster = lightgbm.train(
                bundle.params, bundle.train_dataset, num_boost_round=2,
                valid_sets=[bundle.validation_dataset], valid_names=["validation"],
            )
            self.assertEqual(bundle.train_dataset.num_data(), sizes["train"])
            self.assertEqual(bundle.validation_dataset.num_data(), sizes["validation"])
            self.assertFalse(hasattr(bundle, "test_dataset"))
            self.assertFalse((prepared / ".lightgbm_sequence_cache").exists())
            self.assertEqual(bundle.features["test"].shape, (sizes["test"], 2))
            self.assertEqual(bundle.features["test"].iloc[2:5].shape, (3, 2))
            self.assertEqual(booster.feature_name(), ["f0000_f_0", "f0001_f_1_rate"])
            prediction = booster.predict(bundle.features["test"].iloc[:4])
            self.assertEqual(prediction.shape, (4, 3))

    def test_native_booster_resumes_without_repeating_iterations(self) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest("LightGBM is not installed in this Python environment")

        rng = np.random.default_rng(2026)
        features = rng.normal(size=(240, 6)).astype(np.float32)
        labels = np.repeat(np.arange(3, dtype=np.int32), 80)
        features[:, 0] += labels * 0.8
        train_set = lgb.Dataset(features[:180], label=labels[:180], free_raw_data=False)
        validation_set = lgb.Dataset(
            features[180:], label=labels[180:], reference=train_set, free_raw_data=False
        )
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.001,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "metric": ["multi_logloss", "multi_error"],
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        target_iteration = 100
        history = []
        saved_model = {}
        checkpoint_events = []

        def checkpoint(booster, records, status):
            checkpoint_events.append((booster.current_iteration(), status))
            saved_model["text"] = booster.model_to_string(num_iteration=booster.current_iteration())
            return 0.01

        first = IterationRecorder(
            history, "session_one", target_iteration, 0.001, 10, checkpoint,
            None, 20, 0, 12.0, 20.0,
        )
        with self.assertRaises(TrainingPauseRequested):
            lgb.train(
                params,
                train_set,
                num_boost_round=target_iteration,
                valid_sets=[train_set, validation_set],
                valid_names=["train", "validation"],
                feval=macro_f1_metric(3),
                keep_training_booster=True,
                callbacks=[first],
            )
        self.assertEqual([record["iteration"] for record in history], list(range(1, 21)))
        self.assertEqual(checkpoint_events, [(10, "running"), (20, "paused")])

        resumed_model = lgb.Booster(model_str=saved_model["text"])
        second = IterationRecorder(
            history, "session_two", target_iteration, 0.001, 10,
            checkpoint, None, None, 20, 12.0, 20.0,
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=target_iteration - 20,
            valid_sets=[train_set, validation_set],
            valid_names=["train", "validation"],
            feval=macro_f1_metric(3),
            init_model=resumed_model,
            keep_training_booster=True,
            callbacks=[second],
        )
        self.assertEqual(booster.current_iteration(), target_iteration)
        self.assertEqual(
            [record["iteration"] for record in history], list(range(1, target_iteration + 1))
        )
        self.assertEqual({record["session_id"] for record in history[:20]}, {"session_one"})
        self.assertEqual({record["session_id"] for record in history[20:]}, {"session_two"})
        self.assertEqual(len({record["iteration"] for record in history}), target_iteration)
        self.assertTrue(history[-1]["is_final_round"])
        self.assertEqual(checkpoint_events[-1], (target_iteration, "ready_for_report"))


if __name__ == "__main__":
    unittest.main()
