from flask import Flask, request, jsonify
import uuid

from utils import time_utils
from utils.serve_utils import parse_temp
from serve.llm_executor import LlmExecutor


_FORBIDDEN_BODY = {"error": {"message": "Forbidden", "type": "invalid_request_error", "code": 403}}


def _extract_bearer_token(req) -> str | None:
    header = req.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer "):].strip() or None
    return None


def build_routes(app: Flask, llm: LlmExecutor, accepted_api_key: str | None = None) -> None:
    # Normalize: empty string disables auth, same as not providing the arg.
    expected_key = accepted_api_key if accepted_api_key else None

    def _check_auth():
        if expected_key is None:
            return None
        if _extract_bearer_token(request) != expected_key:
            return jsonify(_FORBIDDEN_BODY), 403
        return None

    @app.route("/v1/chat/completions", methods=['POST'])
    def chat_completions_endpoint():
        unauthorized = _check_auth()
        if unauthorized is not None:
            return unauthorized
        # TODO - implement other body properties that configure how response is generated
        body = request.get_json(force=True)
        # TODO - implement other body properties that configure how response is generated
        body = request.get_json(force=True)

        prompt = llm.apply_chat_template(body['messages'])
        max_tokens = int(body['max_tokens']) if 'max_tokens' in body else 100
        completion = llm.completion(prompt, max_tokens, parse_temp(float(body['temperature']) if 'temperature' in body else 0), stops=body['stop'] if 'stop' in body else None, repetition_penalty=body['frequency_penalty'] if 'frequency_penalty' in body else None)
        prompt_tokens = llm.count_tokens(prompt)
        completion_tokens = llm.count_tokens(completion)
        chat_response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": time_utils.current_milli_time(),
            "model": body['model'],
            "system_fingerprint": "fp_torch_tuner",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"{completion}",
                },
                "logprobs": None,
                "finish_reason": _get_finish_reason(body, completion)
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        return jsonify(chat_response)

    @app.route("/v1/completions", methods=['POST'])
    def completions_endpoint():
        unauthorized = _check_auth()
        if unauthorized is not None:
            return unauthorized
        body = request.get_json(force=True)
        max_tokens = int(body['max_tokens']) if 'max_tokens' in body else 100
        completion = llm.completion(body['prompt'], max_tokens, parse_temp(float(body['temperature']) if 'temperature' in body else 0), stops=body['stop'] if 'stop' in body else None, repetition_penalty=body['frequency_penalty'] if 'frequency_penalty' in body else None)
        prompt_tokens = llm.count_tokens(body['prompt'])
        completion_tokens = llm.count_tokens(completion)

        completion_response = {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": time_utils.current_milli_time(),
            "model": body['model'],
            "system_fingerprint": "fp_torch_tuner",
            "choices": [
                {
                    "text": completion,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": _get_finish_reason(body, completion)
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        return jsonify(completion_response)


def _get_finish_reason(body: dict, completion: str) -> str:
    if 'stop' in body:
        for stop in body['stop']:
            if completion.endswith(stop):
                return "stop"
    return "length"