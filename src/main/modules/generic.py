from utils.torch_utils import get_bnb_config_and_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

from arguments.arguments import TuneArguments, MergeArguments, PushArguments
import base.llm_base_module as base_module
import os


def merge(arguments: MergeArguments) -> None:
    """Generic LLM type specific merge function."""
    lora_dir = f"{arguments.output_dir}{os.sep}adapters{os.sep}{arguments.new_model}"
    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    base_model = AutoModelForCausalLM.from_pretrained(
        arguments.base_model,
        low_cpu_mem_usage=False,
        return_dict=True
    )

    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    base_module.merge_base(arguments, tokenizer, base_model, bnb_config)


def push(arguments: PushArguments) -> None:
    """Generic LLM type specific push function."""

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    if not arguments.use_8bit and not arguments.use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            arguments.model_dir,
            low_cpu_mem_usage=False,
            return_dict=True,
            torch_dtype=dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            arguments.model_dir,
            low_cpu_mem_usage=True,
            return_dict=True,
            quantization_config=bnb_config,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(arguments.model_dir)
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    base_module.push_base(arguments, tokenizer, model)


def fine_tune(arguments: TuneArguments) -> None:
    """Generic LLM type specific fine-tune function."""
    model_to_use = arguments.base_model if arguments.do_train else arguments.output_directory + os.sep + arguments.new_model

    tokenizer = AutoTokenizer.from_pretrained(model_to_use)
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    model = AutoModelForCausalLM.from_pretrained(model_to_use, quantization_config=bnb_config, device_map="auto")

    base_module.fine_tune_eval_base(arguments, tokenizer, model)