from serve.llm_executor import LlmExecutor
from arguments.arguments import ServerArguments
from flask import Flask
from serve.routers import open_ai_router


class LlmServer:
    """Serve LLM as REST API."""

    def __init__(self, llm: LlmExecutor):
        self._llm = llm

    def start_server(self, arguments: ServerArguments) -> None:
        """Start LLM API server."""
        pass


class OpenAiLlmServer(LlmServer):
    """Mimics OpenAI completion endpoints."""

    def __init__(self, llm: LlmExecutor):
        super(OpenAiLlmServer, self).__init__(llm)

    def start_server(self, arguments: ServerArguments) -> None:
        app = Flask(__name__)

        open_ai_router.build_routes(app, self._llm)

        app.run(host='0.0.0.0', port=arguments.port, debug=arguments.debug)
