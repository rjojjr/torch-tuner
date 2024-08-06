from arguments.arguments import TunerFunctionArguments, LlmExecutorFactoryArguments
import torch
from transformers import BitsAndBytesConfig


def get_dtype(arguments: TunerFunctionArguments | LlmExecutorFactoryArguments) -> torch.dtype:
    dtype = torch.float32
    if arguments.is_fp16:
        dtype = torch.float16
    elif arguments.is_bf16:
        dtype = torch.bfloat16

    return dtype


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'auto'


def get_bnb_config_and_dtype(arguments: TunerFunctionArguments | LlmExecutorFactoryArguments) -> tuple[BitsAndBytesConfig, torch.dtype]:
    dtype = get_dtype(arguments)
    bnb_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload
    )
    if arguments.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload
        )
    elif arguments.use_4bit and isinstance(arguments, TunerFunctionArguments):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif arguments.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True
        )
    return bnb_config, dtype
