from utils.argument_utils import parse_arguments, do_initial_arg_validation
from utils.tuner_utils import build_llm_tuner_factory
from exception.exceptions import main_exception_handler
from hf.hf_auth import authenticate_with_hf
from utils.argument_utils import build_and_validate_push_args, build_and_validate_tune_args, build_and_validate_merge_args
from serve.llm_executor import build_llm_executor_factory
from serve.serve import OpenAiLlmServer
from arguments.arguments import ServerArguments, LlmExecutorFactoryArguments
import os

# TODO - Automate this
version = '2.0.2'

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
        print("Running in serve mode")
        print()
        print("WARNING - Serve mode is currently EXPERIMENTAL and should NEVER be used in a production environment!")
        print()
        print(f'Using bf16: {str(args.use_bf_16)}')
        print(f'Using fp16: {str(args.use_fp_16)}')
        print(f'Using 8bit: {str(args.use_8bit)}')
        print(f'Using 4bit: {str(args.use_4bit)}')
        print(f'Using fp32 CPU Offload: {str(args.fp32_cpu_offload)}')
        print()
        print(f"Serving {args.serve_model} on port {args.serve_port}")
        llm_factory_args = LlmExecutorFactoryArguments(model=args.serve_model, use_4bit=args.use_4bit, use_8bit=args.use_8bit, is_fp16=args.use_fp_16, is_bf16=args.use_bf_16, padding_side=args.padding_side)
        llm_executor_factory = build_llm_executor_factory(llm_factory_args)
        server = OpenAiLlmServer(llm_executor_factory())
        server.start_server(ServerArguments(port=args.serve_port, debug=args.debug))
        return

    # Do all validations before printing configuration values
    do_initial_arg_validation(args)

    tuner = tuner_factory()

    lora_scale = round(args.lora_alpha / args.lora_r, 1)
    model_dir = f'{args.output_directory}/merged-models/{args.new_model}'

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
    print(f'Using CPU Only Tuning: {str(args.cpu_only_tuning)}')
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
    print(f'Is Chat Model: {args.is_chat_model}')
    print(f'Is Instruct Model: {args.is_instruct_model}')
    print(f'Using Additional Vocab Tokens: {args.additional_vocabulary_tokens}')
    print(f'Is LangChain Agent Model: {args.use_agent_tokens}')

    if args.fine_tune:
        print('')
        if args.torch_empty_cache_steps is not None:
            print(f'Empty Torch Cache After {args.torch_empty_cache_steps} Steps')

        print(f'Using Checkpointing: {str(not args.no_checkpoint)}')
        print(f'Using Max Saves: {str(args.max_saved)}')
        print(f'Using Batch Size: {str(args.batch_size)}')
        print(f'Using Save Strategy: {args.save_strategy}')
        print(f'Using Save Steps: {str(args.save_steps)}')
        print(f'Using Save Embeddings: {str(args.save_embeddings)}')

        print('')
        print(f'Epochs: {str(args.epochs)}')
        print(f'Using LoRA R: {str(args.lora_r)}')
        print(f'Using LoRA Alpha: {str(args.lora_alpha)}')
        print(f'LoRA Adapter Scale(alpha/r): {str(lora_scale)}')
        print(f'Using Optimizer: {args.optimizer_type}')
        if 'adamw' in args.optimizer_type:
            print(f'Using Base Learning Rate: {str(args.base_learning_rate)}')
            print(f'Using Actual(Base Learning Rate * Batch Size) Learning Rate: {str(args.base_learning_rate * args.batch_size)}')
            print(f'Learning Rate Scheduler Type: {str(args.lr_scheduler_type)}')
        print(f'Using LoRA Dropout: {str(args.lora_dropout)}')
        print(f'Using Warmup Ratio: {args.warmup_ratio}')
        print(f'Using Max Sequence Length: {args.max_seq_length}')


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
