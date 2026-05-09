import os
import sys
import unittest

_TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TEST_FILE_DIR, "..", "..", ".."))
_SRC_MAIN = os.path.join(_PROJECT_ROOT, "src", "main")
if _SRC_MAIN not in sys.path:
    sys.path.insert(0, _SRC_MAIN)

from base.llm_base_module import compute_warmup_steps  # noqa: E402


class TestComputeWarmupSteps(unittest.TestCase):
    """Pin the warmup_ratio -> warmup_steps conversion. transformers >=5 emits a
    deprecation for `warmup_ratio`, so SFTConfig is now built with `warmup_steps`
    derived here."""

    def test_simple_case_matches_steps_per_epoch_times_epochs_times_ratio(self):
        # 6 examples, batch=1, grad_accum=1, epochs=2 -> 12 total steps, 0.5 ratio -> 6 warmup.
        self.assertEqual(compute_warmup_steps(6, 1, 1, 2, 0.5), 6)

    def test_gradient_accumulation_divides_steps_per_epoch(self):
        # 100 examples, batch=4, grad_accum=2 -> effective 8/step,
        # ceil(100/8)=13 steps/epoch * 3 epochs = 39 steps, *0.1=3.9 -> rounds to 4.
        self.assertEqual(compute_warmup_steps(100, 4, 2, 3, 0.1), 4)

    def test_none_gradient_accumulation_treated_as_one(self):
        self.assertEqual(
            compute_warmup_steps(6, 1, None, 2, 0.5),
            compute_warmup_steps(6, 1, 1, 2, 0.5),
        )

    def test_zero_examples_returns_zero(self):
        self.assertEqual(compute_warmup_steps(0, 1, 1, 1, 0.5), 0)

    def test_zero_epochs_returns_zero(self):
        self.assertEqual(compute_warmup_steps(100, 1, 1, 0, 0.5), 0)

    def test_zero_ratio_returns_zero(self):
        self.assertEqual(compute_warmup_steps(100, 1, 1, 1, 0.0), 0)

    def test_negative_inputs_clamp_to_zero(self):
        self.assertEqual(compute_warmup_steps(-5, 1, 1, 1, 0.5), 0)
        self.assertEqual(compute_warmup_steps(100, 1, 1, -1, 0.5), 0)
        self.assertEqual(compute_warmup_steps(100, 1, 1, 1, -0.5), 0)

    def test_default_3_percent_ratio_for_typical_run(self):
        # 1000 examples, batch=4, grad_accum=1, 3 epochs -> 250 steps/epoch * 3 = 750
        # 0.03 * 750 = 22.5 -> rounds to 22 (banker's rounding) or 23. Python round() uses
        # banker's rounding so 22.5 -> 22; assert that without locking the project to a
        # specific rounding mode.
        self.assertIn(compute_warmup_steps(1000, 4, 1, 3, 0.03), {22, 23})

    def test_returns_python_int(self):
        result = compute_warmup_steps(6, 1, 1, 2, 0.5)
        self.assertIsInstance(result, int)


if __name__ == "__main__":
    unittest.main()
