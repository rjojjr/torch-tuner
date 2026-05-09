import json
import os
import sys
import tempfile
import unittest

_TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TEST_FILE_DIR, "..", "..", ".."))
_SRC_MAIN = os.path.join(_PROJECT_ROOT, "src", "main")
if _SRC_MAIN not in sys.path:
    sys.path.insert(0, _SRC_MAIN)

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

from utils.model_utils import (  # noqa: E402
    setup_chat_format,
    _CHATML_TEMPLATE,
    _CHATML_BOS,
    _CHATML_EOS,
    _TOOL_CALL_OPEN,
    _TOOL_CALL_CLOSE,
)


_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class _StubModel:
    """Stand-in for a transformers model: just enough surface for setup_chat_format."""
    def __init__(self):
        self.config = type("C", (), {})()
        self.generation_config = type("G", (), {})()
    def resize_token_embeddings(self, n, pad_to_multiple_of=None):
        pass


def _make_tokenizer_with_chat_template():
    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    setup_chat_format(_StubModel(), tok)
    return tok


class TestChatTemplateRegression(unittest.TestCase):
    """Plain user/assistant chat must still render correctly after the tool-call extension."""

    def test_simple_user_assistant_renders_chatml(self):
        tok = _make_tokenizer_with_chat_template()
        out = tok.apply_chat_template(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            tokenize=False,
        )
        self.assertIn(f"{_CHATML_BOS}user\nHi{_CHATML_EOS}", out)
        self.assertIn(f"{_CHATML_BOS}assistant\nHello{_CHATML_EOS}", out)

    def test_system_message_renders_with_role_system(self):
        tok = _make_tokenizer_with_chat_template()
        out = tok.apply_chat_template(
            [
                {"role": "system", "content": "You are X."},
                {"role": "user", "content": "Hi"},
            ],
            tokenize=False,
        )
        self.assertIn(f"{_CHATML_BOS}system\nYou are X.{_CHATML_EOS}", out)

    def test_add_generation_prompt_appends_assistant_header(self):
        tok = _make_tokenizer_with_chat_template()
        out = tok.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertTrue(out.rstrip().endswith(f"{_CHATML_BOS}assistant"),
                        f"add_generation_prompt should leave an open assistant header; got tail: {out[-60:]!r}")


class TestChatTemplateToolCalls(unittest.TestCase):
    """LangGraph / OpenAI tool-using messages must render with <tool_call> blocks
    and a JSON-shaped tool-response message."""

    def test_assistant_tool_calls_render_inside_tool_call_tags(self):
        tok = _make_tokenizer_with_chat_template()
        out = tok.apply_chat_template(
            [
                {"role": "user", "content": "div"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_020", "type": "function",
                     "function": {"name": "divide", "arguments": "{\"input\": \"100,8\"}"}},
                ]},
            ],
            tokenize=False,
        )
        self.assertIn(_TOOL_CALL_OPEN, out)
        self.assertIn(_TOOL_CALL_CLOSE, out)
        # The id, name, and arguments are all present in the rendered tool_call block.
        self.assertIn('"id": "call_020"', out)
        self.assertIn('"name": "divide"', out)
        self.assertIn('"arguments": {"input": "100,8"}', out,
                      "arguments string should be embedded as raw JSON, not double-quoted")

    def test_tool_role_message_renders_as_json_with_tool_call_id(self):
        tok = _make_tokenizer_with_chat_template()
        out = tok.apply_chat_template(
            [
                {"role": "user", "content": "?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            ],
            tokenize=False,
        )
        # Tool response gets its own ChatML block under the `tool` role with a JSON body.
        self.assertIn(f"{_CHATML_BOS}tool\n", out)
        self.assertIn('"tool_call_id": "call_1"', out)
        self.assertIn('"content": "42"', out)

    def test_full_langgraph_react_example_renders_end_to_end(self):
        tok = _make_tokenizer_with_chat_template()
        messages = [
            {"role": "system", "content": "You are Newton AI."},
            {"role": "user", "content": "What is 100 divided by 8?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_020", "type": "function",
                 "function": {"name": "divide", "arguments": "{\"input\": \"100,8\"}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_020", "content": "12.5"},
            {"role": "assistant", "content": "100 divided by 8 is 12.5."},
        ]
        out = tok.apply_chat_template(messages, tokenize=False)

        # Each turn appears in order.
        for needle in [
            "You are Newton AI.",
            "What is 100 divided by 8?",
            '"name": "divide"',
            "12.5",
            "100 divided by 8 is 12.5.",
        ]:
            self.assertIn(needle, out, f"missing expected fragment: {needle!r}")

        # There are exactly two assistant turns and the order user → assistant(tool_call) →
        # tool → assistant(text) is preserved.
        order_indices = [
            out.index("What is 100 divided by 8?"),
            out.index(_TOOL_CALL_OPEN),
            out.index('"tool_call_id": "call_020"'),
            out.index("100 divided by 8 is 12.5."),
        ]
        self.assertEqual(order_indices, sorted(order_indices),
                         "rendered turns should appear in conversation order")

    def test_assistant_with_both_content_and_tool_calls_renders_both(self):
        tok = _make_tokenizer_with_chat_template()
        out = tok.apply_chat_template(
            [
                {"role": "user", "content": "?"},
                {"role": "assistant", "content": "Let me check.", "tool_calls": [
                    {"id": "c", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}},
                ]},
            ],
            tokenize=False,
        )
        self.assertIn("Let me check.", out)
        self.assertIn(_TOOL_CALL_OPEN, out)


class TestSpecialTokensRegistered(unittest.TestCase):
    """Tool-call delimiters must be added to the tokenizer's vocab as single tokens."""

    def test_tool_call_delimiters_are_special_tokens(self):
        tok = _make_tokenizer_with_chat_template()
        added = tok.get_added_vocab()
        self.assertIn(_TOOL_CALL_OPEN, added,
                      "<tool_call> must be registered as an added/special token after setup_chat_format")
        self.assertIn(_TOOL_CALL_CLOSE, added,
                      "</tool_call> must be registered as an added/special token after setup_chat_format")

    def test_chatml_delimiters_are_still_registered(self):
        tok = _make_tokenizer_with_chat_template()
        added = tok.get_added_vocab()
        self.assertIn(_CHATML_BOS, added)
        self.assertIn(_CHATML_EOS, added)


class TestToolCallTrainingEndToEnd(unittest.TestCase):
    """Full SFTTrainer training pass on a tool-using conversation. Guards the
    integration with TRL's `_tokenize` path that previously failed because
    `processing_class` was not being forwarded."""

    def test_sft_trainer_tokenizes_and_trains_on_langgraph_tool_jsonl(self):
        from trl import SFTConfig, SFTTrainer

        with tempfile.TemporaryDirectory() as tmp:
            sample_path = os.path.join(tmp, "tools.jsonl")
            with open(sample_path, "w") as f:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": "You are Newton AI."},
                    {"role": "user", "content": "What is 100 divided by 8?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call_020", "type": "function",
                         "function": {"name": "divide", "arguments": "{\"input\": \"100,8\"}"}},
                    ]},
                    {"role": "tool", "tool_call_id": "call_020", "content": "12.5"},
                    {"role": "assistant", "content": "100 divided by 8 is 12.5."},
                ]}) + "\n")

            tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
            model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model, tok = setup_chat_format(model, tok)
            ds = load_dataset("json", data_files={"train": sample_path})

            cfg = SFTConfig(
                output_dir=tmp, do_train=True, num_train_epochs=1,
                per_device_train_batch_size=1, max_length=256, learning_rate=1e-4,
                save_strategy="no", logging_strategy="no", report_to="none",
                use_cpu=True, do_eval=False, eval_strategy="no",
            )
            trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds["train"])
            result = trainer.train()
            self.assertIsNotNone(result)
            # If we reached here, the chat template successfully rendered the tool-using
            # conversation and the resulting tokens fed through one training step.


if __name__ == "__main__":
    unittest.main()
