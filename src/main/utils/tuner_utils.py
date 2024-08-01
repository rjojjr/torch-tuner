# This is how I imagine modules for different LLM types should be imported and used to construct a 'Tuner' instance.
import main.llama.functions as llama
from main.tuner import Tuner, LLM_TYPES
from main.exception.exceptions import ArgumentValidationException


def get_tuner(prog_args) -> Tuner:
    tuner = Tuner(llama.fine_tune, llama.merge, llama.push, LLM_TYPES['llama'])
    _evaluate_supported_llm_type(prog_args.llm_type)

    return tuner


def _evaluate_supported_llm_type(llm_type):
    is_supported = False
    for key in LLM_TYPES:
        if llm_type == key:
            is_supported = True
            break

    if not is_supported:
        raise ArgumentValidationException(f'LLM type "{llm_type}" not supported yet, ONLY "llama" is supported currently')