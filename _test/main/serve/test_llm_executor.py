import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

_TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TEST_FILE_DIR, "..", "..", ".."))
_SRC_MAIN = os.path.join(_PROJECT_ROOT, "src", "main")
if _SRC_MAIN not in sys.path:
    sys.path.insert(0, _SRC_MAIN)

from serve.llm_executor import LlmExecutor  # noqa: E402


def _make_stub_tokenizer():
    """Tokenizer stub good enough to exercise LlmExecutor without HF model loading."""
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    tok.padding_side = "right"

    def call(prompts, padding=False, return_tensors="pt"):
        # Return a simple BatchEncoding-like object with a .to() and .input_ids.
        encoding = MagicMock()
        encoding.input_ids = torch.tensor([[1, 2, 3]])
        encoding.attention_mask = torch.tensor([[1, 1, 1]])
        encoding.to = MagicMock(return_value=encoding)
        # Allow **encoding unpacking by giving it a keys()/__getitem__ contract.
        encoding.keys = MagicMock(return_value=["input_ids", "attention_mask"])
        encoding.__getitem__ = lambda self, k: getattr(self, k)
        return encoding

    tok.side_effect = call
    tok.batch_decode = MagicMock(return_value=["decoded"])
    tok.__len__ = MagicMock(return_value=32)
    return tok


def _make_stub_model():
    model = MagicMock()
    model.generation_config = MagicMock()
    model.resize_token_embeddings = MagicMock()
    model.eval = MagicMock()
    # Return a tensor longer than the input so slicing works.
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 9, 9, 9]]))
    return model


class TestLlmExecutorMaxTokensCap(unittest.TestCase):
    """Server-side cap on max_new_tokens prevents a single client from OOM'ing the GPU."""

    def test_when_client_requests_more_than_cap_the_cap_is_enforced(self):
        model = _make_stub_model()
        tokenizer = _make_stub_tokenizer()
        executor = LlmExecutor(model, tokenizer, padding_side="right", cpu_only=True, max_new_tokens_cap=64)

        executor.completion("hello", max_tokens=10_000, temperature=0.5)

        model.generate.assert_called_once()
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs["max_new_tokens"], 64,
                         "executor must clamp client-supplied max_tokens to the configured cap")

    def test_when_client_requests_under_cap_the_request_value_is_used(self):
        model = _make_stub_model()
        tokenizer = _make_stub_tokenizer()
        executor = LlmExecutor(model, tokenizer, padding_side="right", cpu_only=True, max_new_tokens_cap=512)

        executor.completion("hello", max_tokens=32, temperature=0.5)

        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs["max_new_tokens"], 32)


class TestLlmExecutorDeviceRouting(unittest.TestCase):
    """`cpu_only=True` must keep tensors off CUDA — previously hardcoded to .to('cuda')."""

    def test_inputs_are_moved_to_cpu_when_cpu_only_is_true(self):
        model = _make_stub_model()
        tokenizer = _make_stub_tokenizer()
        executor = LlmExecutor(model, tokenizer, padding_side="right", cpu_only=True)

        executor.completion("hello", max_tokens=4)

        # The encoding's .to() must have been called with the cpu device.
        # Recover the encoding the tokenizer returned and inspect its .to call.
        encoding_call = tokenizer.call_args
        self.assertIsNotNone(encoding_call)
        # The encoding is constructed inside the side_effect; its `.to` mock receives
        # the device. Because side_effect produces a fresh encoding per call, capture
        # the device via a wrapped tokenizer on a new executor.
        captured = {}

        def call(prompts, padding=False, return_tensors="pt"):
            encoding = MagicMock()
            encoding.input_ids = torch.tensor([[1, 2, 3]])
            encoding.attention_mask = torch.tensor([[1, 1, 1]])
            def to(device):
                captured["device"] = device
                return encoding
            encoding.to = to
            encoding.keys = MagicMock(return_value=["input_ids", "attention_mask"])
            encoding.__getitem__ = lambda self, k: getattr(self, k)
            return encoding

        tok2 = MagicMock()
        tok2.pad_token = "<pad>"; tok2.eos_token = "<eos>"; tok2.pad_token_id = 0; tok2.padding_side = "right"
        tok2.__len__ = MagicMock(return_value=32)
        tok2.batch_decode = MagicMock(return_value=["decoded"])
        tok2.side_effect = call
        executor2 = LlmExecutor(_make_stub_model(), tok2, padding_side="right", cpu_only=True)
        executor2.completion("hello", max_tokens=4)
        self.assertEqual(captured.get("device"), "cpu")

    def test_model_eval_is_called_at_executor_construction(self):
        model = _make_stub_model()
        tokenizer = _make_stub_tokenizer()
        LlmExecutor(model, tokenizer, padding_side="right", cpu_only=True)
        model.eval.assert_called_once()


class TestLlmExecutorApplyChatTemplate(unittest.TestCase):
    """Chat rendering must use the served model's tokenizer.chat_template, not a hand-built string."""

    def test_uses_tokenizer_apply_chat_template_when_template_is_present(self):
        tokenizer = _make_stub_tokenizer()
        tokenizer.chat_template = "<some-jinja-template>"
        tokenizer.apply_chat_template = MagicMock(return_value="<rendered-prompt>")
        executor = LlmExecutor(_make_stub_model(), tokenizer, padding_side="right", cpu_only=True)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi!"},
        ]
        result = executor.apply_chat_template(messages)

        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.assertEqual(result, "<rendered-prompt>")

    def test_falls_back_to_role_content_join_when_no_chat_template(self):
        tokenizer = _make_stub_tokenizer()
        tokenizer.chat_template = None
        # Should not be called when there's no template configured.
        tokenizer.apply_chat_template = MagicMock()
        executor = LlmExecutor(_make_stub_model(), tokenizer, padding_side="right", cpu_only=True)

        result = executor.apply_chat_template([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ])

        self.assertEqual(result, "system: S\nuser: U\n")
        tokenizer.apply_chat_template.assert_not_called()


class TestLlmExecutorCountTokens(unittest.TestCase):
    """Token counts in `usage` must come from the served model's tokenizer."""

    def test_count_tokens_uses_served_tokenizer_encode(self):
        tokenizer = _make_stub_tokenizer()
        tokenizer.encode = MagicMock(return_value=[10, 11, 12, 13, 14])
        executor = LlmExecutor(_make_stub_model(), tokenizer, padding_side="right", cpu_only=True)

        self.assertEqual(executor.count_tokens("anything"), 5)
        tokenizer.encode.assert_called_once_with("anything", add_special_tokens=False)

    def test_count_tokens_returns_zero_for_empty_string(self):
        tokenizer = _make_stub_tokenizer()
        tokenizer.encode = MagicMock(return_value=[])
        executor = LlmExecutor(_make_stub_model(), tokenizer, padding_side="right", cpu_only=True)

        self.assertEqual(executor.count_tokens(""), 0)
        tokenizer.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
