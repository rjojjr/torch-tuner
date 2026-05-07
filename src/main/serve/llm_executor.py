from typing import Callable
from arguments.arguments import LlmExecutorFactoryArguments
from hf.hf_auth import resolve_hf_token
from transformers import AutoTokenizer, AutoModelForCausalLM, StopStringCriteria, StoppingCriteriaList

from serve.concurrency import ConcurrencyGateKeeper
from utils.torch_utils import get_bnb_config_and_dtype
from exception.exceptions import LlmServerException
import torch
import time
import gc

max_attempts = 10
retry_interval = 1


# TODO - use base model & apply LoRA adapters
# TODO - Set context size?
class LlmExecutor:
    """Manage served LLM instance."""

    # TODO - Another instance of a constructor to that needs to be made "private"
    def __init__(self, model, tokenizer, padding_side: str | None, cpu_only: bool = False, max_parallel_requests: int = 1, max_new_tokens_cap: int = 4096):
        self._padding_side = padding_side
        self._cpu_only = cpu_only
        self._device = "cpu" if cpu_only else "cuda"
        self._max_new_tokens_cap = max_new_tokens_cap
        if padding_side is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = padding_side

        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        self._gate_keeper = ConcurrencyGateKeeper(max_parallel_requests)

        self._model = model
        self._tokenizer = tokenizer

    def completion(self, prompt: str, max_tokens: int = 150, temperature: float = 1, attempt: int = 1, stops: list | None = None, repetition_penalty: float | None = None) -> str:
        """Predict what text should follow the provided prompt."""
        return self._gate_keeper.execute(lambda: self._execute_completion(prompt, max_tokens, temperature, attempt, stops, repetition_penalty))

    def apply_chat_template(self, messages: list) -> str:
        """Render a list of OpenAI-style chat messages using the served model's
        chat template. Falls back to a `role: content` join if the tokenizer has
        no chat_template configured."""
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages)

    def count_tokens(self, text: str) -> int:
        """Token count using the served model's tokenizer (so usage reporting
        matches what the model actually sees)."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _execute_completion(self, prompt: str, max_tokens: int = 150, temperature: float = 1, attempt: int = 1, stops: list | None = None, repetition_penalty: float | None = None) -> str:
        if stops is None:
            stops = []
        # Cap max_new_tokens server-side so a single client request can't OOM the GPU.
        bounded_max_tokens = min(max_tokens, self._max_new_tokens_cap)
        try:
            stopping_criteria = StoppingCriteriaList(
                [StopStringCriteria(stop_strings=stops, tokenizer=self._tokenizer)]
            ) if stops else None
            model_inputs = self._tokenizer(
                [prompt],
                padding=True if self._padding_side is not None else False,
                return_tensors="pt",
            ).to(self._device)
            input_length = model_inputs.input_ids.shape[1]
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=bounded_max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    stopping_criteria=stopping_criteria,
                    repetition_penalty=repetition_penalty,
                )
            response = self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
            return response
        except torch.OutOfMemoryError:
            gc.collect()
            if not self._cpu_only:
                torch.cuda.empty_cache()
            if max_attempts is None or attempt <= max_attempts:
                print(f"CUDA OOM: retrying (attempt {attempt}/{max_attempts})")
                time.sleep(retry_interval * attempt)
                # NOTE: bypass the gate-keeper here — we already hold the slot.
                return self._execute_completion(prompt, max_tokens, temperature, attempt + 1, stops, repetition_penalty)
            print("CUDA OOM: raising exception")
            raise LlmServerException(message="CUDA OOM, exceeded max_attempts")


# Only use this function to construct LLM executors
def build_llm_executor_factory(arguments: LlmExecutorFactoryArguments) -> Callable[[], LlmExecutor]:
    """Construct configured LLM executor factory function."""
    arguments.validate()

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    return lambda: LlmExecutor(
        AutoModelForCausalLM.from_pretrained(
            arguments.model,
            device_map={"": 0} if not arguments.use_cpu_only else "cpu",
            low_cpu_mem_usage=True,
            quantization_config=None if arguments.use_cpu_only else bnb_config,
            torch_dtype="auto",
            token=resolve_hf_token(arguments.huggingface_auth_token),
        ),
        AutoTokenizer.from_pretrained(arguments.model, token=resolve_hf_token(arguments.huggingface_auth_token)),
        padding_side=arguments.padding_side,
        cpu_only=arguments.use_cpu_only,
        max_parallel_requests=arguments.max_parallel_requests,
    )


