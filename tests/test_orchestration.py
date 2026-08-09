import base64
import json
import sys
import time
import unittest
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_kaggle_notebook import EMBEDDED_FILES, build_notebook  # noqa: E402
from generate_presigned_config import run_keys  # noqa: E402
from kaggle_orchestrator import decide_next_session  # noqa: E402


class NotebookBundleTest(unittest.TestCase):
    def test_checked_in_notebooks_are_self_contained_cpu_bundles_without_secrets(self):
        for name, profile in (("kaggle_notebook.ipynb", "production"), ("kaggle_smoke_test.ipynb", "smoke")):
            checked_in = json.loads((PROJECT_ROOT / name).read_text(encoding="utf-8"))
            expected = build_notebook(profile)
            self.assertEqual(checked_in, expected)
            self.assertEqual(checked_in["metadata"]["kaggle"]["accelerator"], "none")
            source = "\n".join("".join(cell.get("source", [])) for cell in checked_in["cells"])
            self.assertIn("PRESIGNED_CONFIG_B64 = ''", source)
            self.assertNotIn("AKIA", source)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY=", source)
            encoded_literal = source.split("encoded_files = json.loads(", 1)[1].split(")\n", 1)[0]
            encoded = json.loads(json.loads(encoded_literal))
            self.assertEqual(set(encoded), set(EMBEDDED_FILES))
            for relative, value in encoded.items():
                decoded = zlib.decompress(base64.b64decode(value))
                self.assertEqual(decoded, (PROJECT_ROOT / relative).read_bytes())

    def test_metadata_and_workflow_force_cpu_and_expected_dataset(self):
        metadata = json.loads((PROJECT_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], "dungnguyen28101991/luan-van-lightgbm-parquet-github-v2")
        self.assertEqual(metadata["enable_gpu"], "false")
        self.assertEqual(metadata["enable_tpu"], "false")
        self.assertEqual(metadata["dataset_sources"], ["dungnguyen28101991/cicddos2019-parquet-per-classes"])
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "run-kaggle.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/kaggle_orchestrator.py decide", workflow)
        self.assertIn("scripts/generate_presigned_config.py", workflow)
        self.assertIn('KAGGLE_API_TOKEN_SECRET: ${{ secrets.KAGGLE_API_TOKEN }}', workflow)
        self.assertIn('kaggle_dir / "access_token"', workflow)
        self.assertIn('handle.write(f"KAGGLE_API_TOKEN={token}', workflow)
        self.assertIn("uses: actions/checkout@v4", workflow)
        self.assertIn("scripts/kaggle_http.py push", workflow)
        self.assertNotIn("kaggle kernels push --path .kaggle_bundle", workflow)
        self.assertIn("AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}", workflow)

    def test_presigned_key_manifest_covers_checkpoint_report_and_explainability(self):
        keys = run_keys("project", "lightgbm_20260809-1200")
        expected = {
            "project/active_run.json",
            "project/lightgbm_20260809-1200/checkpoints/last_model.txt",
            "project/lightgbm_20260809-1200/checkpoints/training_state.json",
            "project/lightgbm_20260809-1200/checkpoints/final_model_round_100.txt",
            "project/lightgbm_20260809-1200/metrics/summary_metrics.csv",
            "project/lightgbm_20260809-1200/raw/y_prob.npy",
            "project/lightgbm_20260809-1200/explainability/shap_feature_importance.csv",
            "project/lightgbm_20260809-1200/figures/roc_curves.pdf",
        }
        self.assertTrue(expected.issubset(keys))


class WatchdogDecisionTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT_ROOT / "config" / "orchestration.json").read_text(encoding="utf-8"))
        self.now = time.time()

    def test_complete_run_never_pushes(self):
        decision = decide_next_session(
            {"status": "complete", "current_iteration": 100}, {}, "complete", self.config, self.now
        )
        self.assertFalse(decision.should_push)

    def test_running_kernel_never_pushes_duplicate_session(self):
        decision = decide_next_session(
            {"status": "paused", "current_iteration": 20}, {}, "running", self.config, self.now
        )
        self.assertFalse(decision.should_push)

    def test_paused_cancelled_and_ready_for_report_runs_continue(self):
        for active, kernel in (
            ({"status": "paused", "current_iteration": 20}, "complete"),
            ({"status": "running", "current_iteration": 30}, "cancelled"),
            ({"status": "ready_for_report", "current_iteration": 100}, "complete"),
        ):
            with self.subTest(active=active, kernel=kernel):
                decision = decide_next_session(active, {}, kernel, self.config, self.now)
                self.assertTrue(decision.should_push)

    def test_recent_push_guard_and_stagnation_limit_prevent_loops(self):
        recent = {"last_push_at": "2099-01-01T00:00:00+00:00"}
        self.assertFalse(decide_next_session({}, recent, "complete", self.config, self.now).should_push)
        stagnant = {"stagnant_restarts": self.config["maximum_stagnant_restarts"]}
        self.assertFalse(decide_next_session({}, stagnant, "complete", self.config, self.now).should_push)


if __name__ == "__main__":
    unittest.main()
