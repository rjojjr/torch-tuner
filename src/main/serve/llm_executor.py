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

max_attempts = 5
retry_interval = 1


# TODO - use base model & apply LoRA adapters
# TODO - max concurrent requests
# TODO - Set context size?
class LlmExecutor:
    """Manage served LLM instance."""

    # TODO - Another instance of a constructor to that needs to be made "private"
    def __init__(self, model, tokenizer, padding_side: str | None,  max_parallel_requests: int = 1):
        # TODO - fix this
        # model.generation_config.cache_implementation = "static"
        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        self._padding_side = padding_side
        if padding_side is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            tokenizer.padding_side = padding_side

        model.resize_token_embeddings(len(tokenizer))
        self.gate_keeper = ConcurrencyGateKeeper(max_parallel_requests)

        self._model = model
        self._tokenizer = tokenizer

    # TODO - FIXME - multiple calls results in GPU memory overload(may be caused bnb?)
    def completion(self, prompt: str, max_tokens: int = 150, temperature: float = 1, attempt: int = 1, stops: list | None = None, repetition_penalty: float | None = None) -> str:
        """Predict what text should follow the provided input."""
        return self.gate_keeper.execute(lambda : self._execute_completion(prompt, max_tokens, temperature, attempt, stops, repetition_penalty))

    def _execute_completion(self, prompt: str, max_tokens: int = 150, temperature: float = 1, attempt: int = 1, stops: list | None = None, repetition_penalty: float | None = None) -> str:
        if stops is None:
            stops = []
        try:
            stopping_criteria = StoppingCriteriaList([StopStringCriteria(stop_strings=stops, tokenizer=self._tokenizer)])
            model_inputs = self._tokenizer([prompt], padding=True if self._padding_side is not None else False, return_tensors="pt").to("cuda")
            input_length = model_inputs.input_ids.shape[1]
            generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, stopping_criteria=stopping_criteria, repetition_penalty=repetition_penalty)
            response = self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
            # TODO - FIXME - big hack to stop OOM
            gc.collect()
            torch.cuda.empty_cache()

            return response
        except torch.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            if max_attempts is None or attempt <= max_attempts:
                print("CUDA OOM: retrying")
                time.sleep(retry_interval * attempt)
                return self.completion(prompt, max_tokens, attempt + 1)
            print("CUDA OOM: raising exception")
            raise LlmServerException(message="CUDA OOM, exceeded max_attempts")


# Only use this function to construct LLM executors
def build_llm_executor_factory(arguments: LlmExecutorFactoryArguments) -> Callable[[], LlmExecutor]:
    """Construct configured LLM executor factory function."""
    arguments.validate()

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    return lambda: LlmExecutor(AutoModelForCausalLM.from_pretrained(
        arguments.model,
        # TODO - use CPU
        device_map={"":0},
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        torch_dtype="auto",
        token=resolve_hf_token(arguments.huggingface_auth_token)
        # TODO - investigate if this is effective
        # attn_implementation="flash_attention_2"
    ), AutoTokenizer.from_pretrained(arguments.model, token=resolve_hf_token(arguments.huggingface_auth_token)), padding_side=arguments.padding_side)


