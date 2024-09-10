from utils.argument_utils import parse_arguments, do_initial_arg_validation
from utils.tuner_utils import build_llm_tuner_factory
from exception.exceptions import main_exception_handler
from hf.hf_auth import authenticate_with_hf
from utils.argument_utils import build_and_validate_push_args, build_and_validate_tune_args, build_and_validate_merge_args
from utils.config_utils import print_serve_mode_config, print_tune_mode_config, print_fine_tune_config
from serve.llm_executor import build_llm_executor_factory
from serve.serve import OpenAiLlmServer
from arguments.arguments import ServerArguments, LlmExecutorFactoryArguments
import os

# TODO - Automate this
version = '2.1.1'

title = f'Torch-Tuner CLI v{version}'
description = 'This app is a simple CLI to automate the Supervised Fine-Tuning(SFT)(and testing of) of AI Large Language Model(LLM)s with simple text and jsonl on Nvidia GPUs(and Intel/AMD CPUs) using LoRA, Torch and Transformers.'

args = parse_arguments(title, description)

# For better performance with less GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,expandable_segments:True"


def main() -> None:
    tuner_factory = build_llm_tuner_factory(args)

    print(title)
    print('---------------------------------------------')
    print(description)
    print('---------------------------------------------')
    print('Run with --help flag for a list of available arguments.')
    print('')
    if args.debug:
        print("Is Debug Mode: True")
        print('')
    if args.serve:
        print_serve_mode_config(args)

        authenticate_with_hf(args.huggingface_auth_token)
        model_path = os.path.expanduser(f"{args.output_directory}{os.sep}{args.serve_model}" if (not '/' in args.serve_model and not os.sep in args.serve_model) else args.serve_model)
        llm_factory_args = LlmExecutorFactoryArguments(model=model_path, use_4bit=args.use_4bit, use_8bit=args.use_8bit, is_fp16=args.use_fp_16, is_bf16=args.use_bf_16, padding_side=args.padding_side)
        llm_executor_factory = build_llm_executor_factory(llm_factory_args)
        server = OpenAiLlmServer(llm_executor_factory())
        server.start_server(ServerArguments(port=args.serve_port, debug=args.debug))
        return

    # Do all validations before printing configuration values
    do_initial_arg_validation(args)

    tuner = tuner_factory()

    lora_scale = round(args.lora_alpha / args.lora_r, 1)
    model_dir = os.path.expanduser(f'{args.output_directory}{os.sep}merged-models{os.sep}{args.new_model}')

    authenticate_with_hf(args.huggingface_auth_token)

    tune_arguments = build_and_validate_tune_args(args)
    merge_arguments = build_and_validate_merge_args(args)
    push_arguments = build_and_validate_push_args(args, model_dir)

    print_tune_mode_config(args, model_dir, tuner)

    if args.fine_tune:
        print_fine_tune_config(args, lora_scale, tune_arguments)

    if args.fine_tune:
        print('')
        print(f'Tuning LoRA adapter for model {args.new_model} on base model {args.base_model} with {args.training_data_file} to {args.epochs} epochs')
        print('')
        tuner.fine_tune(tune_arguments)
        print('')
        print(f'Tuned LoRA adapter for model {args.base_model} on base model {args.base_model} with {args.training_data_file} to {args.epochs} epochs')

    if args.merge:
        print('')
        print(f'Merging LoRA Adapter for {args.new_model} with base model {args.base_model}')
        print('')
        tuner.merge(merge_arguments)
        print('')
        print(f'Merged LoRA Adapter for {args.new_model} with base model {args.base_model}')

    if args.push:
        print('')
        print(f'Pushing {args.new_model} to Huggingface')
        print('')
        tuner.push(push_arguments)
        print('')
        print(f'Pushed {args.new_model} to Huggingface')

    print('')
    print('---------------------------------------------')
    print(f'{title} COMPLETED')





main_exception_handler(main, title, args.debug)
