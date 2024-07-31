import main.llama.functions as llama
from main.tuner import Tuner
import torch
from main.exception.exceptions import ArgumentValidationException
from main.arguments.arguments import TunerFunctionArguments


def get_tuner(prog_args) -> Tuner:
    tuner = Tuner(llama.fine_tune, llama.merge, llama.push)
    if prog_args.llm_type != 'llama':
        raise ArgumentValidationException(f'LLM type "{prog_args.llm_type}" not supported yet, ONLY "llama" is supported currently')

    return tuner


def get_dtype(arguments: TunerFunctionArguments) -> torch.dtype:
    dtype = torch.float32
    if arguments.is_fp16:
        dtype = torch.float16
    if arguments.is_bf16:
        dtype = torch.bfloat16

    return dtype


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'auto'
