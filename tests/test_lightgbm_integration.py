import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import IterationRecorder, TrainingPauseRequested, macro_f1_metric  # noqa: E402


class LightGBMResumeIntegrationTest(unittest.TestCase):
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
