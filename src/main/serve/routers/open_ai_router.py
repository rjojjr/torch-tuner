from flask import Flask, request, jsonify
import uuid

from openai import completions

from utils import time_utils
from utils.serve_utils import parse_temp
import tiktoken
from serve.llm_executor import LlmExecutor

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def build_routes(app: Flask, llm: LlmExecutor) -> None:

    # TODO - How does Open AI parse chat messages?
    def _construct_chat_prompt(body: dict) -> str:
        prompt = ""
        for msg in body['messages']:
            prompt = f"{prompt}{msg['role']}: {msg['content']}\n"
        return prompt

    @app.route("/v1/chat/completions", methods=['POST'])
    def chat_completions_endpoint():
        # TODO - implement other body properties that configure how response is generated
        body = request.get_json(force=True)

        prompt = _construct_chat_prompt(body)

        max_tokens = int(body['max_tokens']) if body['max_tokens'] is not None else 100
        completion = llm.completion(prompt, max_tokens, parse_temp(float(body['temperature'])), stops=body['stop'], repetition_penalty=body['frequency_penalty'])
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(encoding.encode(completion))
        chat_response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": time_utils.current_milli_time(),
            "model": body['model'],
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"{completion}",
                },
                "logprobs": None,
                "finish_reason": "length" if body['stop'] is None or not completion.endswith(body['stop']) else "stop"
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
        body = request.get_json(force=True)
        max_tokens = int(body['max_tokens']) if body['max_tokens'] is not None else 100
        completion = llm.completion(body['prompt'], max_tokens, parse_temp(float(body['temperature'])), stops=body['stop'], repetition_penalty=body['frequency_penalty'])
        prompt_tokens = len(encoding.encode(body['prompt']))
        completion_tokens = len(encoding.encode(completion))

        completion_response = {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": time_utils.current_milli_time(),
            "model": body['model'],
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "text": completion,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length" if body['stop'] is None or not completion.endswith(body['stop']) else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        return jsonify(completion_response)
