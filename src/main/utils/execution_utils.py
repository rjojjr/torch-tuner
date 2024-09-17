import os

from utils.argument_utils import do_initial_arg_validation
from hf.hf_auth import authenticate_with_hf, resolve_hf_token
from utils.argument_utils import build_and_validate_push_args, build_and_validate_tune_args, build_and_validate_merge_args
from utils.config_utils import print_global_config, print_serve_mode_config, print_fine_tune_merge_common_config, print_tuner_mode_config, print_fine_tune_config
from serve.llm_executor import build_llm_executor_factory
from serve.serve import OpenAiLlmServer
from arguments.arguments import ServerArguments, LlmExecutorFactoryArguments
from utils.tuner_utils import build_llm_tuner_factory


def execute_command(args) -> None:
    print_global_config(args)
    _execute_hf_auth(args)
    if args.serve:
        _execute_serve_mode(args)
        return

    _execute_tuner_mode(args)


def _execute_hf_auth(args):
    hf_token = resolve_hf_token(args.huggingface_auth_token)
    if hf_token is not None:
        authenticate_with_hf(hf_token)


def _execute_tuner_mode(args) -> None:
    # Do all validations before printing configuration values
    do_initial_arg_validation(args)

    tuner_factory = build_llm_tuner_factory(args)
    tuner = tuner_factory()
    lora_scale = round(args.lora_alpha / args.lora_r, 1)
    model_dir = os.path.expanduser(f'{args.output_directory}{os.sep}merged-models{os.sep}{args.new_model}')
    tune_arguments = build_and_validate_tune_args(args)
    merge_arguments = build_and_validate_merge_args(args)
    push_arguments = build_and_validate_push_args(args, model_dir)
    print_tuner_mode_config(args, tuner)
    if args.fine_tune or args.merge or args.do_eval:
        print_fine_tune_merge_common_config(args, model_dir)
    if args.fine_tune:
        print_fine_tune_config(args, lora_scale, tune_arguments)
        print('')
        print(
            f'Tuning LoRA adapter for model {args.new_model} on base model {args.base_model} with {args.training_data_file} to {args.epochs} epochs')
        print('')
        tuner.fine_tune(tune_arguments)
        print('')
        print(
            f'Tuned LoRA adapter for model {args.new_model} on base model {args.base_model} with {args.training_data_file} to {args.epochs} epochs')

    # TODO - create new base module function for full eval
    if not args.fine_tune and args.do_eval:
        print('')
        print(f'Running full evaluation against model {args.new_model}')
        print('')
        tuner.fine_tune(tune_arguments)
        print('')
        print(f'Ran full evaluation against model {args.new_model}')
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


def _execute_serve_mode(args):
    print_serve_mode_config(args)
    authenticate_with_hf(args.huggingface_auth_token)
    model_path = os.path.expanduser(f"{args.output_directory}{os.sep}{args.serve_model}" if (
            not '/' in args.serve_model and not os.sep in args.serve_model) else args.serve_model)
    llm_factory_args = LlmExecutorFactoryArguments(model=model_path, use_4bit=args.use_4bit, use_8bit=args.use_8bit,
                                                   is_fp16=args.use_fp_16, is_bf16=args.use_bf_16,
                                                   padding_side=args.padding_side, max_parallel_requests=args.max_parallel_requests)
    llm_executor_factory = build_llm_executor_factory(llm_factory_args)
    server = OpenAiLlmServer(llm_executor_factory())
    server.start_server(ServerArguments(port=args.serve_port, debug=args.debug))