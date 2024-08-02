from utils.argument_utils import parse_arguments, do_initial_arg_validation
from utils.tuner_utils import llm_tuner_factory
from exception.exceptions import main_exception_handler
from hf.hf_auth import authenticate_with_hf
from utils.argument_utils import build_and_validate_push_args, build_and_validate_tune_args, build_and_validate_merge_args

# Bump with every PR
# TODO - Automate this? Probably need to get this hosted in a repo first.
version = '1.1.1'

# TODO - Change this once support for more LLMs is added
title = f'Llama AI LLM LoRA Torch Text Fine-Tuner v{version}'
description = 'Fine-Tune Llama LLM models with simple text on Nvidia GPUs using Torch and LoRA.'


def main() -> None:
    args = parse_arguments(title, description)
    tuner_factory = llm_tuner_factory(args)

    print(title)
    print('---------------------------------------------')
    print(description)
    print('---------------------------------------------')
    print('Run with --help flag for a list of available arguments.')
    print('')

    # Do all validations before printing configuration values
    do_initial_arg_validation(args)

    tuner = tuner_factory()

    lora_scale = round(args.lora_alpha / args.lora_r, 1)
    model_dir = f'{args.output_directory}/{args.new_model}'

    tune_arguments = build_and_validate_tune_args(args)
    merge_arguments = build_and_validate_merge_args(args)
    push_arguments = build_and_validate_push_args(args, model_dir)

    authenticate_with_hf()

    print('')
    print(f'Using LLM Type: {tuner.llm_type}')

    print('')
    print(f'Output Directory: {args.output_directory}')
    print(f'Base Model: {args.base_model}')
    print(f'Model Save Directory: {model_dir}')
    print(f'Training File: {args.training_data_file}')
    print('')
    print(f'Epochs: {str(args.epochs)}')
    print(f'Using LoRA R: {str(args.lora_r)}')
    print(f'Using LoRA Alpha: {str(args.lora_alpha)}')
    print(f'LoRA Adapter Scale(alpha/r): {str(lora_scale)}')
    print(f'Using Base Learning Rate: {str(args.learning_rate_base)}')
    print(f'Using LoRA Dropout: {str(args.lora_dropout)}')
    print('')
    print(f'Using tf32: {str(args.use_tf_32)}')
    print(f'Using bf16: {str(args.use_bf_16)}')
    print(f'Using fp16: {str(args.use_fp_16)}')
    print(f'Using 8bit: {str(args.use_8bit)}')
    print(f'Using 4bit: {str(args.use_4bit)}')
    print(f'Using fp32 CPU Offload: {str(args.fp32_cpu_offload)}')
    print('')
    print(f'Is Fine-Tuning: {str(args.fine_tune)}')
    print(f'Is Merging: {str(args.merge)}')
    print(f'Is Pushing: {str(args.push)}')
    print('')
    print(f'Using Checkpointing: {str(not args.no_checkpoint)}')
    print(f'Using Max Saves: {str(args.max_saved)}')
    print(f'Using Batch Size: {str(args.batch_size)}')
    print(f'Using Optimizer: {args.optimizer_type}')
    print(f'Using Save Strategy: {args.save_strategy}')
    print(f'Using Save Steps: {str(args.save_steps)}')
    print(f'Using Save Embeddings: {str(args.save_embeddings)}')

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
    exit(0)


main_exception_handler(main, title, False)
