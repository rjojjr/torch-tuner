from serve.llm_executor import LlmExecutor
from arguments.arguments import ServerArguments
from flask import Flask, request, jsonify
import uuid
import time
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


class LlmServer:
    def __init__(self, llm: LlmExecutor):
        self._llm = llm

    def start_server(self, arguments: ServerArguments):
        pass


class OpenAiServer(LlmServer):
    def __init__(self, llm: LlmExecutor):
        super(OpenAiServer, self).__init__(llm)

    def start_server(self, arguments: ServerArguments):
        app = Flask(__name__)

        _build_open_ai_routes(app, self._llm)

        app.run(host='0.0.0.0', port=arguments.port, debug=arguments.debug)


def _build_open_ai_routes(app, llm):
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

        completion = llm.completion(prompt, int(body['max_tokens']))
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(encoding.encode(completion))
        chat_response = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": _current_milli_time(),
            "model": body['model'],
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    # TODO - Probably should replace `\n\n` with stop sequence(?)
                    "content": f"\n\n{completion}",
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
        completion = llm.completion(body['prompt'], int(body['max_tokens']))
        prompt_tokens = len(encoding.encode(body['prompt']))
        completion_tokens = len(encoding.encode(completion))

        completion_response = {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": _current_milli_time(),
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


def _current_milli_time():
    return round(time.time() * 1000)