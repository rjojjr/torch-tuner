from utils.torch_utils import get_bnb_config_and_dtype

from transformers import LlamaForCausalLM, AutoTokenizer

from arguments.arguments import TuneArguments, MergeArguments, PushArguments
import base.llm_base_module as base_module


def merge(arguments: MergeArguments) -> None:
    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    base_model = LlamaForCausalLM.from_pretrained(
        arguments.base_model,
        low_cpu_mem_usage=False,
        return_dict=True,
        torch_dtype=dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model)
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    base_module.merge_base(arguments, tokenizer, base_model, bnb_config)


def push(arguments: PushArguments) -> None:

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    if not arguments.use_8bit and not arguments.use_4bit:
        model = LlamaForCausalLM.from_pretrained(
            arguments.model_dir,
            low_cpu_mem_usage=False,
            return_dict=True,
            torch_dtype=dtype
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
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
    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model, add_eos_token=True, add_bos_token=True)
    if arguments.padding_side is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = arguments.padding_side

    bnb_config, dtype = get_bnb_config_and_dtype(arguments)

    model = LlamaForCausalLM.from_pretrained(arguments.base_model, quantization_config=bnb_config, device_map="auto")

    base_module.fine_tune_base(arguments, tokenizer, model)


