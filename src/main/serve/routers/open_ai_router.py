from flask import Flask, request, jsonify
import uuid
from utils import time_utils
import tiktoken
from serve.llm_executor import LlmExecutor

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def build_routes(app: Flask, llm: LlmExecutor) -> None:

    @app.route("/v1/chat/completions", methods=['POST'])
    def chat_completions_endpoint():
        body = request.get_json(force=True)

        # TODO - How does Open AI parse messages?
        prompt = ""
        for msg in body['messages']:
            if prompt == "":
                prompt = f"{msg['role']}: {msg['content']}"
            else:
                # TODO - Probably should replace `\n\n` with stop sequence(?)
                prompt = f"\n\n{msg['role']}: {msg['content']}"

        completion = llm.completion(prompt, int(body['max_tokens']), float(body['temperature']))
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
                "finish_reason": "stop"
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
        completion = llm.completion(body['prompt'], int(body['max_tokens']), float(body['temperature']))
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
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        return jsonify(completion_response)
