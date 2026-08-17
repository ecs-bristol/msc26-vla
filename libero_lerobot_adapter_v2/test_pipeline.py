import argparse
import tempfile
import unittest
from pathlib import Path

import libero_pipeline as pipeline
from interactive_grasp import (
    accuracy_strategies, append_history, build_candidates, infer_object,
    normalize, select_target, success_rate,
)
from inspect_libero_dataset import validate_sample
from visual_detector import normalize_label, target_detected


class FakeTensor:
    def __init__(self, shape): self.shape = shape


class PipelineTests(unittest.TestCase):
    def test_task_ids_are_normalized(self):
        self.assertEqual(pipeline.task_ids("[0, 2]"), "[0,2]")
        with self.assertRaises(argparse.ArgumentTypeError): pipeline.task_ids("[-1]")

    def test_identifiers_reject_traversal(self):
        with self.assertRaises(argparse.ArgumentTypeError): pipeline.safe_identifier("../secret")

    def test_root_output_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError): pipeline.output_path(Path.cwd().anchor)

    def test_preview_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "new"
            result = pipeline.execute(["missing-command"], destination, False, "egl")
            self.assertFalse(result.executed)
            self.assertFalse(destination.exists())

    def test_sample_contract(self):
        sample = {
            "observation.state": FakeTensor((8,)), "observation.images.image": FakeTensor((3, 224, 224)),
            "observation.images.image2": FakeTensor((3, 224, 224)), "action": FakeTensor((7,)), "task": "pick",
        }
        self.assertEqual(validate_sample(sample)["action"]["shape"], [7])
        sample["action"] = FakeTensor((6,))
        with self.assertRaises(ValueError): validate_sample(sample)

    def test_interactive_object_matching(self):
        self.assertEqual(infer_object("pick up the milk and place it in the basket"), "milk")
        self.assertEqual(normalize("牛奶"), "milk")
        candidates = [(0, "pick up the milk and place it in the basket", "milk")]
        self.assertEqual(select_target(candidates, "牛奶")[0], 0)

    def test_reliable_eval_options_are_forwarded(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["eval", "--episode-length", "500", "--seed", "7",
                                  "--policy-num-steps", "10", "--policy-n-action-steps", "1"])
        command = pipeline.eval_command(args)
        self.assertIn("--env.episode_length=500", command)
        self.assertIn("--seed=7", command)
        self.assertIn("--policy.n_action_steps=1", command)

    def test_explicit_target_is_always_in_candidates(self):
        tasks = [
            (0, "pick up the milk and place it in the basket", "milk"),
            (1, "pick up the butter and place it in the basket", "butter"),
            (2, "pick up the orange juice and place it in the basket", "orange juice"),
        ]
        candidates = build_candidates(tasks, 2, 12, "牛奶")
        self.assertEqual(len(candidates), 2)
        self.assertIn("milk", [item[2] for item in candidates])

    def test_accuracy_profile_uses_diverse_fallbacks(self):
        strategies = accuracy_strategies()
        self.assertEqual([name for name, _ in strategies], ["native", "smooth", "responsive"])
        self.assertEqual(strategies[0][1], [])
        self.assertIn("10", strategies[1][1])

    def test_success_rate_and_history(self):
        self.assertEqual(success_rate([True, True, False]), 66.67)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.csv"
            summary = {
                "timestamp": "2026-08-17T20:00:00", "target": "milk", "task_id": 7,
                "recognition_method": "libero_task_metadata", "mode": "accurate", "seed": 7,
                "best_strategy": "smooth", "successes": 3, "attempts": 3,
                "success_rate": 100.0, "run_directory": "run",
            }
            append_history(path, summary)
            append_history(path, summary)
            rows = path.read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(rows), 3)

    def test_visual_target_confirmation(self):
        detections = [{"label": "orange juice", "score": 0.81, "box": [1, 2, 3, 4]}]
        self.assertTrue(target_detected("Orange Juice", detections))
        self.assertFalse(target_detected("milk", detections))
        self.assertEqual(normalize_label("BBQ-sauce"), "bbqsauce")


if __name__ == "__main__": unittest.main()
