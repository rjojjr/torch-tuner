from typing import Callable
from arguments.arguments import LlmExecutorFactoryArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.torch_utils import get_bnb_config_and_dtype
from exception.exceptions import TunerException
import torch
import time

max_attempts = 5
retry_interval = 0.5


# TODO - use base model & apply LoRA adapters
# TODO - max concurrent requests
# TODO - Set context size?
# TODO - Set temperature?
class LlmExecutor:
    # TODO - Another instance of a constructor to that needs to be made "private"
    def __init__(self, model, tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        self._model = model
        self._tokenizer = tokenizer

    # TODO - FIXME - multiple calls results in GPU memory overload
    # TODO - Stop sequences
    def completion(self, input: str, max_tokens: int = 150, attempt: int = 1):
        try:
            model_inputs = self._tokenizer([input], padding=True, return_tensors="pt").to("cuda")
            input_length = model_inputs.input_ids.shape[1]
            generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=True)
            response = self._tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
            return response
        except torch.OutOfMemoryError as e:
            if max_attempts is None or attempt <= max_attempts:
                print("CUDA OOM: retrying")
                time.sleep(retry_interval)
                return self.completion(input, max_tokens, attempt + 1)
            print("CUDA OOM: raising exception")
            raise TunerException(message="CUDA OOM, exceeded max_attempts")


# Only use this function to construct LLM executors
def llm_executor_factory(arguments: LlmExecutorFactoryArguments) -> Callable[[], LlmExecutor]:
    arguments.validate()
    bnb_config, dtype = get_bnb_config_and_dtype(arguments)
    return lambda: LlmExecutor(AutoModelForCausalLM.from_pretrained(
        arguments.model,
        low_cpu_mem_usage=False,
        quantization_config=bnb_config
    ), AutoTokenizer.from_pretrained(arguments.model, padding_side="right"))


