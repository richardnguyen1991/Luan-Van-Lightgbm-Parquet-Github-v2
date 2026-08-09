import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from make_report import evaluate_final_model, generate_report  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FakeBooster:
    def __init__(self, feature_names):
        self._features = list(feature_names)
        self.params = {"num_threads": 1}

    def current_iteration(self):
        return 100

    def num_model_per_iteration(self):
        return 3

    def num_trees(self):
        return 300

    def feature_name(self):
        return list(self._features)

    def model_to_string(self, num_iteration=100):
        return "fake-lightgbm-model-round-100"

    def feature_importance(self, importance_type, iteration=100):
        if importance_type == "gain":
            return np.array([9.0, 4.0, 1.0, 0.0])
        return np.array([12, 7, 2, 0])

    def predict(self, values, num_iteration=100, pred_contrib=False):
        array = np.asarray(values, dtype=np.float32)
        if pred_contrib:
            blocks = []
            for class_index in range(3):
                contribution = array * np.float32((class_index + 1) * 0.01)
                blocks.append(np.column_stack((contribution, np.zeros(len(array), dtype=np.float32))))
            return np.concatenate(blocks, axis=1)
        logits = np.column_stack((
            0.8 * array[:, 0] - 0.2 * array[:, 1],
            0.7 * array[:, 1] + 0.1 * array[:, 2],
            0.6 * array[:, 2] - 0.1 * array[:, 0],
        ))
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class ReportPipelineTest(unittest.TestCase):
    def test_full_report_is_idempotent_and_writes_all_figure_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "lightgbm_20260809-1200"
            for directory in ("config", "metrics", "checkpoints", "raw", "figures", "explainability"):
                (run_dir / directory).mkdir(parents=True, exist_ok=True)
            feature_names = ["f0", "f1", "f2", "f3"]
            class_names = ["BENIGN", "DNS", "SYN"]
            mapping = {name: index for index, name in enumerate(class_names)}
            write_json(run_dir / "config" / "label_mapping.json", mapping)
            write_json(run_dir / "config" / "preprocessing.json", {
                "feature_columns_in_order": feature_names,
                "feature_dtypes": {name: "float32" for name in feature_names},
                "categorical_features": [],
                "scaling": "none",
                "imbalance_handling": "none",
            })
            counts = {name: 30 for name in class_names}
            write_json(run_dir / "config" / "sample_manifest.json", {
                "split": {"class_counts": {"train": counts, "validation": counts, "test": counts}}
            })
            write_json(run_dir / "config" / "run_config.json", {
                "run_id": run_dir.name,
                "model_name": "lightgbm",
                "seed": 2026,
                "session": {"maximum_hours": 12.0, "stop_before_minutes": 20.0},
            })
            history = []
            for iteration in range(1, 101):
                history.append({
                    "iteration": iteration,
                    "session_id": "session_one" if iteration <= 50 else "session_two",
                    "timestamp_start": "2026-08-09T00:00:00+00:00",
                    "timestamp_end": "2026-08-09T00:00:01+00:00",
                    "learning_rate": 0.001,
                    "train_multi_logloss": 1.2 - iteration * 0.002,
                    "val_multi_logloss": 1.25 - iteration * 0.0018,
                    "train_multi_error": 0.5 - iteration * 0.001,
                    "val_multi_error": 0.52 - iteration * 0.0009,
                    "train_macro_f1": 0.4 + iteration * 0.002,
                    "val_macro_f1": 0.38 + iteration * 0.0018,
                    "iteration_seconds": 0.1,
                    "checkpoint_seconds": 0.2 if iteration % 10 == 0 else 0.0,
                    "is_final_round": iteration == 100,
                })
            write_json(run_dir / "metrics" / "history.json", history)
            (run_dir / "checkpoints" / "final_model_round_100.txt").write_text("fake", encoding="utf-8")

            rng = np.random.default_rng(2026)
            labels = np.repeat(np.arange(3, dtype=np.int32), 30)
            features = pd.DataFrame(rng.normal(size=(90, 4)).astype(np.float32), columns=feature_names)
            features["f0"] += labels * 0.4
            booster = FakeBooster(feature_names)
            original_module = sys.modules.get("lightgbm")
            sys.modules["lightgbm"] = SimpleNamespace(
                Booster=lambda model_file: booster,
                __version__="4.6.0-test",
            )
            try:
                generated = evaluate_final_model(run_dir, booster, features, labels)
                first_summary = (run_dir / "metrics" / "summary_metrics.csv").read_bytes()
                generate_report(run_dir)
                second_summary = (run_dir / "metrics" / "summary_metrics.csv").read_bytes()
            finally:
                if original_module is None:
                    sys.modules.pop("lightgbm", None)
                else:
                    sys.modules["lightgbm"] = original_module

            self.assertTrue(generated)
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(np.load(run_dir / "raw" / "y_true.npy", mmap_mode="r").dtype, np.int16)
            self.assertEqual(np.load(run_dir / "raw" / "y_prob.npy", mmap_mode="r").dtype, np.float32)
            expected_figures = {
                "learning_curves", "lr_schedule", "confusion_matrix", "confusion_matrix_raw",
                "roc_curves", "pr_curves", "per_class_f1", "class_distribution", "iteration_time",
                "feature_importance_gain", "feature_importance_split", "permutation_importance",
                "shap_feature_importance",
            }
            for name in expected_figures:
                self.assertTrue((run_dir / "figures" / f"{name}.png").exists(), name)
                self.assertTrue((run_dir / "figures" / f"{name}.pdf").exists(), name)
            self.assertTrue((run_dir / "metrics" / "confusion_matrix.csv").exists())
            self.assertTrue((run_dir / "metrics" / "confusion_matrix_normalized.csv").exists())
            for name in (
                "feature_importance_gain", "feature_importance_split",
                "permutation_importance", "shap_feature_importance",
                "feature_importance_comparison",
            ):
                self.assertTrue((run_dir / "explainability" / f"{name}.csv").exists(), name)
            summary = pd.read_csv(run_dir / "metrics" / "summary_metrics.csv")
            for column in ("Balanced Accuracy", "MCC", "Minority Class F1", "AUC-ROC macro-OVR", "PR-AUC macro"):
                self.assertIn(column, summary.columns)


if __name__ == "__main__":
    unittest.main()
