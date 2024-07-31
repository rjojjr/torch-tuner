from main.arguments.arguments import TunerFunctionArguments
import torch


def get_dtype(arguments: TunerFunctionArguments) -> torch.dtype:
    dtype = torch.float32
    if arguments.is_fp16:
        dtype = torch.float16
    if arguments.is_bf16:
        dtype = torch.bfloat16

    return dtype


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'auto'
