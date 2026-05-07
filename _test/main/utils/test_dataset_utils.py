import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Setup paths so `from utils.dataset_utils import load_dataset` resolves to src/main.
_TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TEST_FILE_DIR, "..", "..", ".."))
_SRC_MAIN = os.path.join(_PROJECT_ROOT, "src", "main")
if _SRC_MAIN not in sys.path:
    sys.path.insert(0, _SRC_MAIN)

from datasets import Dataset, DatasetDict  # noqa: E402

from arguments.arguments import TuneArguments  # noqa: E402
from exception.exceptions import ArgumentValidationException  # noqa: E402
from utils.dataset_utils import load_dataset as project_load_dataset  # noqa: E402


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(c) for c in row) + "\n")


def _make_tune_args(**overrides):
    """Build TuneArguments with safe defaults for parsing tests."""
    defaults = dict(
        new_model="dataset-utils-test",
        training_data_dir=overrides.pop("training_data_dir", "/tmp"),
        train_file=overrides.pop("train_file", None),
        epochs=1,
        do_train=True,
        do_eval=False,
        eval_dataset=None,
        hf_training_dataset_id=None,
    )
    defaults.update(overrides)
    return TuneArguments(**defaults)


class TestLoadDatasetPromptCompletionJsonl(unittest.TestCase):
    """`load_dataset` for prompt/completion JSONL training files."""

    def test_when_train_file_is_prompt_completion_jsonl_it_loads_a_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "train.jsonl")
            _write_jsonl(jsonl_path, [
                {"prompt": "hello", "completion": "world"},
                {"prompt": "ping", "completion": "pong"},
            ])
            args = _make_tune_args(
                training_data_dir=tmp,
                train_file="train.jsonl",
            )

            ds = project_load_dataset(args)

            self.assertIn("train", ds)
            self.assertEqual(len(ds["train"]), 2)
            self.assertIn("prompt", ds["train"].column_names)
            self.assertIn("completion", ds["train"].column_names)
            self.assertNotIn("messages", ds["train"].column_names)

    def test_when_training_dir_lacks_trailing_separator_load_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "train.jsonl")
            _write_jsonl(jsonl_path, [{"prompt": "a", "completion": "b"}])

            # Confirm both with-and-without trailing separator paths work,
            # since dataset_utils builds the path manually for jsonl.
            for dir_arg in (tmp, tmp + os.sep):
                args = _make_tune_args(training_data_dir=dir_arg, train_file="train.jsonl")
                ds = project_load_dataset(args)
                self.assertEqual(len(ds["train"]), 1)
                self.assertEqual(ds["train"][0]["prompt"], "a")


class TestLoadDatasetChatJsonl(unittest.TestCase):
    """`load_dataset` for OpenAI chat-format JSONL training files."""

    def test_when_train_file_is_openai_chat_jsonl_it_loads_messages_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "chat.jsonl")
            _write_jsonl(jsonl_path, [
                {"messages": [
                    {"role": "system", "content": "You are a helper."},
                    {"role": "user", "content": "The car is a red 2022 Tesla Model 3."},
                    {"role": "assistant", "content": "{\"make\": \"Tesla\"}"},
                ]},
                {"messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]},
            ])
            args = _make_tune_args(training_data_dir=tmp, train_file="chat.jsonl")

            ds = project_load_dataset(args)

            self.assertIn("train", ds)
            self.assertEqual(len(ds["train"]), 2)
            self.assertEqual(ds["train"].column_names, ["messages"])
            first_row = ds["train"][0]["messages"]
            self.assertEqual([m["role"] for m in first_row], ["system", "user", "assistant"])

    def test_chat_jsonl_messages_column_is_what_drives_chat_format_detection(self):
        # The detection logic in llm_base_module is `messages` in train.column_names
        # and the file path ends with `jsonl`. We assert the contract directly.
        with tempfile.TemporaryDirectory() as tmp:
            chat_path = os.path.join(tmp, "chat.jsonl")
            _write_jsonl(chat_path, [{"messages": [{"role": "user", "content": "hi"}]}])
            pc_path = os.path.join(tmp, "pc.jsonl")
            _write_jsonl(pc_path, [{"prompt": "x", "completion": "y"}])

            chat_ds = project_load_dataset(_make_tune_args(training_data_dir=tmp, train_file="chat.jsonl"))
            pc_ds = project_load_dataset(_make_tune_args(training_data_dir=tmp, train_file="pc.jsonl"))

            self.assertTrue("messages" in chat_ds["train"].column_names)
            self.assertFalse("messages" in pc_ds["train"].column_names)


class TestLoadDatasetNonJsonlTrainingFile(unittest.TestCase):
    """`load_dataset` for non-JSONL training files (CSV/etc.)."""

    def test_when_train_file_is_csv_it_loads_via_directory_data_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "train.csv")
            _write_csv(csv_path, ["text"], [["row a"], ["row b"], ["row c"]])
            args = _make_tune_args(training_data_dir=tmp, train_file="train.csv")

            ds = project_load_dataset(args)

            self.assertIn("train", ds)
            self.assertEqual(len(ds["train"]), 3)
            self.assertEqual(ds["train"][0]["text"], "row a")


class TestLoadDatasetHuggingfaceDatasetId(unittest.TestCase):
    """`load_dataset` for the Hugging Face training-dataset-id path (no local file)."""

    def test_when_hf_training_dataset_id_is_set_it_loads_with_split_train(self):
        # We patch the underlying datasets.load_dataset import inside dataset_utils
        # so the test does not require network access.
        sentinel = Dataset.from_dict({"text": ["one", "two"]})
        with patch("utils.dataset_utils.load_data_set", return_value=sentinel) as mocked:
            args = _make_tune_args(
                training_data_dir="/tmp",
                train_file=None,
                hf_training_dataset_id="some/repo",
            )
            result = project_load_dataset(args)

            mocked.assert_called_once_with("some/repo", split="train")
            self.assertIs(result, sentinel)


class TestLoadDatasetEvalRouting(unittest.TestCase):
    """Eval-dataset routing logic in `load_dataset` and `_load_eval_ds`."""

    def test_when_do_train_and_do_eval_with_no_eval_dataset_it_uses_train_as_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "train.jsonl")
            _write_jsonl(jsonl_path, [
                {"prompt": "a", "completion": "b"},
                {"prompt": "c", "completion": "d"},
            ])
            args = _make_tune_args(
                training_data_dir=tmp,
                train_file="train.jsonl",
                do_eval=True,
                eval_dataset=None,
            )

            ds = project_load_dataset(args)

            self.assertIn("train", ds)
            self.assertIn("eval", ds)
            self.assertEqual(len(ds["eval"]), 2)
            # _load_eval_ds aliases train as eval — same column schema.
            self.assertEqual(ds["train"].column_names, ds["eval"].column_names)

    def test_when_eval_dataset_is_jsonl_file_it_loads_into_eval_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.jsonl")
            eval_path = os.path.join(tmp, "eval.jsonl")
            _write_jsonl(train_path, [{"prompt": "a", "completion": "b"}])
            _write_jsonl(eval_path, [
                {"prompt": "e1", "completion": "v1"},
                {"prompt": "e2", "completion": "v2"},
            ])
            args = _make_tune_args(
                training_data_dir=tmp,
                train_file="train.jsonl",
                do_eval=True,
                eval_dataset=eval_path,
            )

            ds = project_load_dataset(args)

            self.assertIn("eval", ds)
            self.assertEqual(len(ds["eval"]), 2)
            self.assertEqual(ds["eval"][0]["prompt"], "e1")

    def test_when_eval_dataset_is_csv_file_it_loads_into_eval_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.jsonl")
            _write_jsonl(train_path, [{"prompt": "a", "completion": "b"}])
            eval_path = os.path.join(tmp, "eval.csv")
            _write_csv(eval_path, ["text"], [["row1"], ["row2"]])
            args = _make_tune_args(
                training_data_dir=tmp,
                train_file="train.jsonl",
                do_eval=True,
                eval_dataset=eval_path,
            )

            ds = project_load_dataset(args)

            self.assertIn("eval", ds)
            self.assertEqual(len(ds["eval"]), 2)
            self.assertEqual(ds["eval"][0]["text"], "row1")

    def test_when_eval_only_mode_with_no_train_and_no_eval_dataset_it_raises(self):
        # do_train=False with no eval_dataset and no pre-existing train split should fail.
        args = _make_tune_args(
            training_data_dir="/tmp",
            train_file=None,
            do_train=False,
            do_eval=True,
            eval_dataset=None,
        )

        with self.assertRaises(ArgumentValidationException):
            project_load_dataset(args)

    def test_when_eval_only_mode_with_eval_dataset_jsonl_it_loads_eval_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_path = os.path.join(tmp, "eval.jsonl")
            _write_jsonl(eval_path, [{"prompt": "p", "completion": "c"}])
            args = _make_tune_args(
                training_data_dir="/tmp",
                train_file=None,
                do_train=False,
                do_eval=True,
                eval_dataset=eval_path,
            )

            ds = project_load_dataset(args)

            self.assertIn("eval", ds)
            self.assertEqual(len(ds["eval"]), 1)
            self.assertEqual(ds["eval"][0]["prompt"], "p")

    def test_when_eval_dataset_is_huggingface_id_string_it_calls_with_split_eval(self):
        # When eval_dataset is not a local file path (no os.path.isfile match), it is
        # treated as an HF dataset id and loaded with split="eval".
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.jsonl")
            _write_jsonl(train_path, [{"prompt": "a", "completion": "b"}])

            # Real load_data_set is called for the train file; mock the eval call only
            # by intercepting `load_data_set` after the train load. Easiest is to drive
            # eval-only mode, where _load_eval_ds is the only call site.
            args = _make_tune_args(
                training_data_dir="/tmp",
                train_file=None,
                do_train=False,
                do_eval=True,
                eval_dataset="some/eval-repo",
            )

            with patch(
                "utils.dataset_utils.load_data_set",
                return_value=DatasetDict({"eval": Dataset.from_dict({"text": ["x"]})}),
            ) as mocked:
                ds = project_load_dataset(args)

                mocked.assert_called_once_with("some/eval-repo", split="eval")
                self.assertIn("eval", ds)


class TestChatFormatDetectionContract(unittest.TestCase):
    """Replicates the in-line chat-format detection logic used by llm_base_module
    and asserts it agrees with the dataset returned by `load_dataset`."""

    @staticmethod
    def _detect(train_file, ds):
        is_jsonl = train_file is not None and train_file.endswith("jsonl")
        is_chat = is_jsonl and "train" in ds and "messages" in ds["train"].column_names
        is_pc = is_jsonl and not is_chat
        return is_jsonl, is_chat, is_pc

    def test_chat_jsonl_is_classified_as_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chat.jsonl")
            _write_jsonl(path, [{"messages": [{"role": "user", "content": "hi"}]}])
            ds = project_load_dataset(_make_tune_args(training_data_dir=tmp, train_file="chat.jsonl"))
            is_jsonl, is_chat, is_pc = self._detect("chat.jsonl", ds)
            self.assertTrue(is_jsonl)
            self.assertTrue(is_chat)
            self.assertFalse(is_pc)

    def test_prompt_completion_jsonl_is_classified_as_prompt_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pc.jsonl")
            _write_jsonl(path, [{"prompt": "p", "completion": "c"}])
            ds = project_load_dataset(_make_tune_args(training_data_dir=tmp, train_file="pc.jsonl"))
            is_jsonl, is_chat, is_pc = self._detect("pc.jsonl", ds)
            self.assertTrue(is_jsonl)
            self.assertFalse(is_chat)
            self.assertTrue(is_pc)

    def test_non_jsonl_file_is_neither_chat_nor_prompt_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.csv")
            _write_csv(path, ["text"], [["row"]])
            ds = project_load_dataset(_make_tune_args(training_data_dir=tmp, train_file="train.csv"))
            is_jsonl, is_chat, is_pc = self._detect("train.csv", ds)
            self.assertFalse(is_jsonl)
            self.assertFalse(is_chat)
            self.assertFalse(is_pc)

    def test_no_train_file_is_neither_chat_nor_prompt_completion(self):
        is_jsonl, is_chat, is_pc = self._detect(None, DatasetDict({}))
        self.assertFalse(is_jsonl)
        self.assertFalse(is_chat)
        self.assertFalse(is_pc)


if __name__ == "__main__":
    unittest.main()
