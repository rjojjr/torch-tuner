from typing import Callable
from main.arguments.arguments import LlmExecutorFactoryArguments
from transformers import AutoModelForCausalLM, AutoTokenizer


# TODO - Set context size?
# TODO - Set temperature?
class LlmExecutor:
    # TODO - Another instance of a constructor to that needs to be made "private"
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    # TODO - FIXME - multiple calls results in GPU memory overload
    # TODO - Stop sequences
    def completion(self, input: str, max_tokens: int = 150):
        model_inputs = self._tokenizer([input], return_tensors="pt").to("cuda")
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_tokens)
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


# Only use this function to construct LLM executors
def llm_executor_factory(arguments: LlmExecutorFactoryArguments) -> Callable[[], LlmExecutor]:
    # TODO - Use bnb config
    return lambda: LlmExecutor(AutoModelForCausalLM.from_pretrained(
        arguments.model,
        device_map="auto",
        load_in_4bit=arguments.use_4bit,
        load_in_8bit=arguments.use_8bit
    ), AutoTokenizer.from_pretrained(arguments.model, padding_side="right"))


