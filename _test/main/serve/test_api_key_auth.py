import os
import sys
import unittest
from unittest.mock import MagicMock

_TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TEST_FILE_DIR, "..", "..", ".."))
_SRC_MAIN = os.path.join(_PROJECT_ROOT, "src", "main")
if _SRC_MAIN not in sys.path:
    sys.path.insert(0, _SRC_MAIN)

from flask import Flask  # noqa: E402

from arguments.arguments import ServerArguments  # noqa: E402
from exception.exceptions import ArgumentValidationException  # noqa: E402
from serve.routers import open_ai_router  # noqa: E402
from utils.argument_utils import do_initial_arg_validation  # noqa: E402


def _stub_llm():
    """LlmExecutor stub that produces a deterministic completion without loading a model."""
    llm = MagicMock()
    llm.completion = MagicMock(return_value="ok")
    llm.apply_chat_template = MagicMock(return_value="rendered prompt")
    llm.count_tokens = MagicMock(return_value=1)
    return llm


def _make_app(accepted_api_key):
    app = Flask(__name__)
    open_ai_router.build_routes(app, _stub_llm(), accepted_api_key=accepted_api_key)
    return app


def _make_cli_args(**overrides):
    defaults = {
        "accepted_api_key": None,
        "serve": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "fine_tune": False,
        "merge": False,
        "push": False,
        "epochs": 1,
        "hf_training_dataset_id": "x",
        "training_data_dir": None,
        "training_data_file": None,
        "new_model": "x",
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestServerArgumentsNormalization(unittest.TestCase):
    """ServerArguments must normalize empty/None into a no-auth state."""

    def test_none_accepted_api_key_disables_auth(self):
        args = ServerArguments(port=8080, accepted_api_key=None)
        self.assertIsNone(args.accepted_api_key)

    def test_empty_accepted_api_key_disables_auth(self):
        args = ServerArguments(port=8080, accepted_api_key="")
        self.assertIsNone(args.accepted_api_key,
                         "empty string must be normalized to None so auth is not enforced")

    def test_non_empty_accepted_api_key_enables_auth(self):
        args = ServerArguments(port=8080, accepted_api_key="secret-123")
        self.assertEqual(args.accepted_api_key, "secret-123")


class TestCliArgValidation(unittest.TestCase):
    """`--accepted-api-key` is only valid in serve mode."""

    def test_accepted_api_key_without_serve_raises(self):
        args = _make_cli_args(accepted_api_key="secret", serve=False)
        with self.assertRaises(ArgumentValidationException) as ctx:
            do_initial_arg_validation(args)
        self.assertIn("--accepted-api-key", str(ctx.exception))

    def test_accepted_api_key_with_serve_does_not_raise(self):
        args = _make_cli_args(accepted_api_key="secret", serve=True)
        # Should not raise; serve mode skips most other validations once we get past this check.
        try:
            do_initial_arg_validation(args)
        except ArgumentValidationException as e:
            # Other validations may legitimately trip on these stub args; the only
            # failure mode this test forbids is one mentioning --accepted-api-key.
            self.assertNotIn("--accepted-api-key", str(e))

    def test_no_accepted_api_key_does_not_raise_in_non_serve_mode(self):
        args = _make_cli_args(accepted_api_key=None, serve=False)
        try:
            do_initial_arg_validation(args)
        except ArgumentValidationException as e:
            self.assertNotIn("--accepted-api-key", str(e))

    def test_empty_accepted_api_key_is_treated_as_unset(self):
        args = _make_cli_args(accepted_api_key="", serve=False)
        try:
            do_initial_arg_validation(args)
        except ArgumentValidationException as e:
            self.assertNotIn("--accepted-api-key", str(e),
                             "an empty key should be ignored, not flagged as a serve-only conflict")


class TestRouterAuthEnforced(unittest.TestCase):
    """When a non-empty key is configured, /v1/* requires `Authorization: Bearer <key>`."""

    def setUp(self):
        self.client = _make_app("secret-key").test_client()

    def test_chat_completions_returns_403_when_no_authorization_header(self):
        resp = self.client.post("/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(resp.status_code, 403)

    def test_chat_completions_returns_403_when_token_is_wrong(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer not-the-secret"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_chat_completions_returns_403_when_scheme_is_not_bearer(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Basic secret-key"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_chat_completions_succeeds_with_correct_bearer_token(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
        self.assertEqual(resp.status_code, 200)

    def test_completions_returns_403_without_auth_header(self):
        resp = self.client.post("/v1/completions", json={"model": "tiny", "prompt": "hi"})
        self.assertEqual(resp.status_code, 403)

    def test_completions_succeeds_with_correct_bearer_token(self):
        resp = self.client.post(
            "/v1/completions",
            json={"model": "tiny", "prompt": "hi"},
            headers={"Authorization": "Bearer secret-key"},
        )
        self.assertEqual(resp.status_code, 200)


class TestRouterAuthIgnoredWhenKeyNotConfigured(unittest.TestCase):
    """When the key is None or empty, the endpoints don't enforce auth at all."""

    def test_no_key_means_request_succeeds_without_auth_header(self):
        client = _make_app(None).test_client()
        resp = client.post("/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(resp.status_code, 200)

    def test_empty_key_means_request_succeeds_without_auth_header(self):
        client = _make_app("").test_client()
        resp = client.post("/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(resp.status_code, 200,
                         "empty accepted_api_key should disable auth, same as None")

    def test_no_key_means_a_bogus_authorization_header_is_ignored(self):
        client = _make_app(None).test_client()
        resp = client.post(
            "/v1/completions",
            json={"model": "tiny", "prompt": "hi"},
            headers={"Authorization": "Bearer anything"},
        )
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
