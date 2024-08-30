# This is how I imagine modules for different LLM types should be imported and used to construct a 'Tuner' instance.
import modules.llama as llama
from base.tuner import Tuner, LLM_TYPES
from exception.exceptions import ArgumentValidationException
from typing import Callable


# This will probably be useful for the future API impl.
def build_llm_tuner_factory(prog_args) -> Callable[[], Tuner]:
    """Returns configured LLM factory function."""
    return lambda: _construct_tuner(prog_args)


def parse_temp(temp: float) -> float:
    """Handle invalid temperature values."""
    if temp > 1:
        return 1
    if temp < 0:
        return 0
    return temp


def _construct_tuner(prog_args) -> Tuner:
    _evaluate_supported_llm_type(prog_args.llm_type)
    match prog_args.llm_type:
        # Since Llama is the only type currently
        case _:
            tuner = Tuner(llama.fine_tune, llama.merge, llama.push, LLM_TYPES['llama'])

    return tuner


def _evaluate_supported_llm_type(llm_type):
    is_supported = False
    for key in LLM_TYPES:
        if llm_type == key:
            is_supported = True
            break

    if not is_supported:
        raise ArgumentValidationException(f'LLM type "{llm_type}" not supported yet, ONLY "llama" is supported currently')
