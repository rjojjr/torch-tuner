from base.tuner import Tuner


def print_global_config(args) -> None:
    print('')
    print(f'Is Low GPU Memory Mode: {args.use_low_gpu_memory}')
    if args.debug:
        print("Is Debug Mode: True")
        print('')


def print_serve_mode_config(args) -> None:
    print()
    print("Running in serve mode")
    print()
    print("WARNING - Serve mode is currently EXPERIMENTAL and should NEVER be used in a production environment!")
    _print_dtype_config(args)
    print()
    print(f"Serving {args.serve_model} on port {args.serve_port}")


def print_tuner_mode_config(args, tuner: Tuner) -> None:
    print()
    print(f'Using LLM Type: {tuner.llm_type}')

    _print_dtype_config(args)

    print()
    print(f'Is Fine-Tuning: {str(args.fine_tune)}')
    print(f'Is Merging: {str(args.merge)}')
    print(f'Is Pushing: {str(args.push)}')


def print_fine_tune_merge_common_config(args, model_dir) -> None:
    print()
    print(f'Creating Model/Adapter: {args.new_model}')
    print(f'Output Directory: {args.output_directory}')
    print(f'Base Model: {args.base_model}')
    print(f'Model Save Directory: {model_dir}')
    print()
    print(f'Is Chat Model: {args.is_chat_model}')
    print(f'Is Instruct Model: {args.is_instruct_model}')
    print(f'Using LangChain Agent Tokens: {args.use_agent_tokens}')
    print(f'Using Additional Vocabulary Tokens: {args.additional_vocabulary_tokens}')


def print_fine_tune_config(args, lora_scale, tune_arguments) -> None:
    print()
    print(f'Epochs: {str(args.epochs)}')
    print(f'Using Tuning Dataset: {args.hf_training_dataset_id if args.hf_training_dataset_id is not None else args.training_data_file}')

    if args.torch_empty_cache_steps is not None:
        print(f'Empty Torch Cache After {args.torch_empty_cache_steps} Steps')
    print(f'Using Checkpointing: {str(not args.no_checkpoint)}')
    if not args.no_checkpoint:
        print(f'Using Max Saved Checkpoints: {args.max_saved}')
    print(f'Using Batch Size: {str(args.batch_size)}')
    print(f'Using Save Strategy: {args.save_strategy}')
    print(f'Using Save Steps: {str(args.save_steps)}')
    print(f'Using Save Embeddings: {str(args.save_embeddings)}')
    if args.target_modules is not None:
        print(f'Targeting Modules: {args.target_modules}')
    elif args.target_all_modules:
        print(f'Targeting Modules: ALL')
    else:
        print(f'Targeting Modules: LINEAR')
    print()
    print(f'Using LoRA R: {str(args.lora_r)}')
    print(f'Using LoRA Alpha: {str(args.lora_alpha)}')
    print(f'LoRA Adapter Scale(alpha/r): {str(lora_scale)}')
    print(f'Using LoRA Dropout: {str(args.lora_dropout)}')
    print(f'Using LoRA Bias: {str(args.bias)}')
    print()
    print(f'Using Optimizer: {args.optimizer_type}')
    if 'adamw' in args.optimizer_type:
        print(f'Using Base Learning Rate: {str(args.base_learning_rate)}')
        print(
            f'Using Actual Learning Rate(LR)(Base LR * Batch Size): {str(args.base_learning_rate * args.batch_size)}')
        print(f'Learning Rate Scheduler Type: {str(args.lr_scheduler_type)}')

    print(f'Using Warmup Ratio: {args.warmup_ratio}')
    print(f'Using Max Sequence Length: {args.max_seq_length}')
    print(f'Using Group By Length: {args.group_by_length}')
    print(f'Using Weight Decay: {args.weight_decay}')
    print(f'Using Max Gradient Norm: {args.max_gradient_norm}')
    print(f'Using Gradient Accumulation Steps: {args.gradient_accumulation_steps}')
    print()
    print(f'Using Do Eval: {args.do_eval}')
    if args.do_eval is not None and args.do_eval:
        print(f'Using Eval Strategy: {tune_arguments.eval_strategy}')
        if tune_arguments.eval_strategy == 'steps':
            print(f'Using Eval Steps: {tune_arguments.eval_steps}')
        if args.eval_dataset is None:
            print(f'Using Eval Dataset: {args.hf_training_dataset_id if args.hf_training_dataset_id is not None else args.training_data_file}')
        else:
            print(f'Using Eval Dataset: {args.eval_dataset }')
    print()
    if args.is_instruct_model:
        print(f'Using NEFTune Noise Alpha: {args.neftune_noise_alpha}')


def _print_dtype_config(args) -> None:
    print()
    print(f'Using CPU Only: {str(args.cpu_only_tuning)}')
    print(f'Using tf32: {str(args.use_tf_32)}')
    print(f'Using bf16: {str(args.use_bf_16)}')
    print(f'Using fp16: {str(args.use_fp_16)}')
    print(f'Using 8bit: {str(args.use_8bit)}')
    print(f'Using 4bit: {str(args.use_4bit)}')
    print(f'Using fp32 CPU Offload: {str(args.fp32_cpu_offload)}')


