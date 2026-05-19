from utils.torch_utils import get_bnb_config_and_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from arguments.arguments import TuneArguments, MergeArguments, PushArguments
import base.llm_base_module as base_module
import os
from utils.debugging_utils import debugging_wrapper


def merge(arguments: MergeArguments) -> None:
    """Generic LLM type specific merge function."""
    with debugging_wrapper(arguments.is_debug_mode):
        lora_dir = f"{arguments.output_dir}{os.sep}adapters{os.sep}{arguments.new_model}"
        bnb_config, dtype = get_bnb_config_and_dtype(arguments)

        # Merge always happens on CPU at the configured dtype. The 4/8-bit
        # quantized base is the wrong starting point for merge_and_unload
        # (PEFT would dequantize on-the-fly), and the GPU usually can't fit
        # the full unquantized model anyway -- e.g. a 9B fp32 is ~36GB.
        base_model = AutoModelForCausalLM.from_pretrained(
            arguments.base_model,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=dtype,
            device_map="cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        if arguments.padding_side is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = arguments.padding_side

        base_module.merge_base(arguments, tokenizer, base_model, bnb_config)


def push(arguments: PushArguments) -> None:
    """Generic LLM type specific push function."""
    with debugging_wrapper(arguments.is_debug_mode):
        # The merged model and tokenizer are already on disk at model_dir
        # from the merge phase; push_base uploads that folder directly via
        # HfApi.upload_folder, so we don't materialize the model at all.
        base_module.push_base(arguments)


def fine_tune(arguments: TuneArguments) -> None:
    """Generic LLM type specific fine-tune function."""
    with debugging_wrapper(arguments.is_debug_mode):
        model_to_use = arguments.base_model if arguments.do_train else arguments.output_directory + os.sep + 'merged-models' + os.sep + arguments.new_model

        tokenizer = AutoTokenizer.from_pretrained(model_to_use)
        if arguments.padding_side is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = arguments.padding_side

        bnb_config, dtype = get_bnb_config_and_dtype(arguments)

        model = AutoModelForCausalLM.from_pretrained(model_to_use, quantization_config=bnb_config, torch_dtype=dtype, device_map="cpu" if arguments.cpu_only_tuning else ("mps" if torch.backends.mps.is_available() else "auto"), attn_implementation=arguments.flash_attention_impl if arguments.use_flash_attention else None)

        base_module.fine_tune_eval_base(arguments, tokenizer, model)