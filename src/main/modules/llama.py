from hf.hf_auth import resolve_hf_token
from utils.torch_utils import get_bnb_config_and_dtype

from transformers import LlamaForCausalLM, AutoTokenizer

from arguments.arguments import TuneArguments, MergeArguments, PushArguments
import base.llm_base_module as base_module
import os


def merge(arguments: MergeArguments) -> None:
    """Llama specific merge function."""
    lora_dir = f"{arguments.output_dir}{os.sep}adapters{os.sep}{arguments.new_model}"
    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    base_model = LlamaForCausalLM.from_pretrained(
        arguments.base_model,
        low_cpu_mem_usage=False,
        return_dict=True,
        torch_dtype=dtype,
        token=resolve_hf_token(arguments.huggingface_auth_token)
    )

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, token=resolve_hf_token(arguments.huggingface_auth_token))
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    base_module.merge_base(arguments, tokenizer, base_model, bnb_config)


def push(arguments: PushArguments) -> None:
    """Llama specific push function."""

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    if not arguments.use_8bit and not arguments.use_4bit:
        model = LlamaForCausalLM.from_pretrained(
            arguments.model_dir,
            low_cpu_mem_usage=False,
            return_dict=True,
            torch_dtype=dtype,
            token=resolve_hf_token(arguments.huggingface_auth_token)
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            arguments.model_dir,
            low_cpu_mem_usage=True,
            return_dict=True,
            quantization_config=bnb_config,
            device_map="auto",
            token=resolve_hf_token(arguments.huggingface_auth_token)
        )

    tokenizer = AutoTokenizer.from_pretrained(arguments.model_dir, token=resolve_hf_token(arguments.huggingface_auth_token))
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    base_module.push_base(arguments, tokenizer, model)


def fine_tune(arguments: TuneArguments) -> None:
    """Llama specific fine-tune function."""
    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model if arguments.do_train else arguments.new_model, token=resolve_hf_token(arguments.huggingface_auth_token))
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    model = LlamaForCausalLM.from_pretrained(arguments.base_model if arguments.do_train else arguments.new_model, quantization_config=bnb_config, device_map="auto", token=resolve_hf_token(arguments.huggingface_auth_token))

    base_module.fine_tune_eval_base(arguments, tokenizer, model)


