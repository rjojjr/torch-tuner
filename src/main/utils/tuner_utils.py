import main.llama.functions as llama
from main.tuner import Tuner
from main.exception.exceptions import ArgumentValidationException


def get_tuner(prog_args) -> Tuner:
    tuner = Tuner(llama.fine_tune, llama.merge, llama.push)
    if prog_args.llm_type != 'llama':
        raise ArgumentValidationException(f'LLM type "{prog_args.llm_type}" not supported yet, ONLY "llama" is supported currently')

    return tuner
