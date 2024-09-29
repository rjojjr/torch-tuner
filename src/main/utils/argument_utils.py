import sys, os
from argparse import ArgumentParser

from exception.exceptions import ArgumentValidationException
from arguments.arguments import PushArguments, MergeArguments, TuneArguments


def build_and_validate_push_args(prog_args, model_dir: str):
    """Construct/validate push arguments."""
    if prog_args.push:
        push_arguments = PushArguments(
            new_model=prog_args.new_model,
            model_dir=os.path.expanduser(model_dir),
            use_4bit=prog_args.use_4bit,
            use_8bit=prog_args.use_8bit,
            is_bf16=prog_args.use_bf_16,
            is_fp16=prog_args.use_fp_16,
            is_debug_mode=prog_args.debug,
            public_push=prog_args.public_push,
            padding_side=prog_args.padding_side,
            use_agent_tokens=prog_args.use_agent_tokens,
            additional_vocabulary_tokens=prog_args.additional_vocabulary_tokens,
            huggingface_auth_token=prog_args.huggingface_auth_token
        )
        push_arguments.validate()
        return push_arguments

    return PushArguments(
        new_model=prog_args.new_model,
        model_dir=model_dir
    )


def build_and_validate_merge_args(prog_args) -> MergeArguments:
    """Construct/validate merge arguments."""
    if prog_args.merge:
        merge_arguments = MergeArguments(
            new_model=prog_args.new_model,
            base_model=prog_args.base_model,
            use_4bit=prog_args.use_4bit,
            use_8bit=prog_args.use_8bit,
            is_bf16=prog_args.use_bf_16,
            is_fp16=prog_args.use_fp_16,
            is_debug_mode=prog_args.debug,
            output_dir=os.path.expanduser(prog_args.output_directory) if prog_args.output_directory is not None else None,
            padding_side=prog_args.padding_side,
            use_agent_tokens=prog_args.use_agent_tokens,
            additional_vocabulary_tokens=prog_args.additional_vocabulary_tokens,
            is_chat_model=prog_args.is_chat_model or (prog_args.training_data_file is not None and prog_args.training_data_file.endswith(".jsonl")),
            overwrite_output=prog_args.overwrite_output,
            huggingface_auth_token=prog_args.huggingface_auth_token
        )
        merge_arguments.validate()
        return merge_arguments

    return MergeArguments(new_model=prog_args.new_model)


def build_and_validate_tune_args(prog_args) -> TuneArguments:
    """Construct/validate tune arguments."""
    if prog_args.fine_tune or prog_args.do_eval:
        tune_arguments = TuneArguments(
            base_model=prog_args.base_model,
            new_model=prog_args.new_model,
            show_token_metrics=prog_args.show_token_metrics,
            training_data_dir=os.path.expanduser(prog_args.training_data_dir) if prog_args.training_data_dir is not None else None,
            train_file=prog_args.training_data_file,
            r=prog_args.lora_r,
            alpha=prog_args.lora_alpha,
            epochs=prog_args.epochs,
            batch_size=prog_args.batch_size,
            load_best_before_save=prog_args.load_best_before_save,
            is_fp16=prog_args.use_fp_16,
            is_bf16=prog_args.use_bf_16,
            base_learning_rate=prog_args.base_learning_rate,
            lora_dropout=prog_args.lora_dropout,
            no_checkpoint=prog_args.no_checkpoint,
            bias=prog_args.bias,
            optimizer_type=prog_args.optimizer_type,
            gradient_accumulation_steps=prog_args.gradient_accumulation_steps,
            weight_decay=prog_args.weight_decay,
            max_gradient_norm=prog_args.max_gradient_norm,
            is_tf32=prog_args.use_tf_32,
            save_strategy=prog_args.save_strategy,
            save_steps=prog_args.save_steps,
            do_eval=prog_args.do_eval,
            max_checkpoints=prog_args.max_saved,
            use_8bit=prog_args.use_8bit,
            use_4bit=prog_args.use_4bit,
            save_embeddings=prog_args.save_embeddings,
            output_directory=os.path.expanduser(prog_args.output_directory),
            fp32_cpu_offload=prog_args.fp32_cpu_offload,
            is_chat_model=prog_args.is_chat_model,
            padding_side=prog_args.padding_side,
            use_agent_tokens=prog_args.use_agent_tokens,
            lr_scheduler_type=prog_args.lr_scheduler_type,
            target_modules=prog_args.target_modules,
            torch_empty_cache_steps=prog_args.torch_empty_cache_steps if not prog_args.use_low_gpu_memory else (1 if prog_args.torch_empty_cache_steps is None else prog_args.torch_empty_cache_steps),
            warmup_ratio=prog_args.warmup_ratio,
            additional_vocabulary_tokens=prog_args.additional_vocabulary_tokens,
            cpu_only_tuning=prog_args.cpu_only_tuning,
            is_instruct_model=prog_args.is_instruct_model,
            group_by_length=prog_args.group_by_length,
            hf_training_dataset_id=prog_args.hf_training_dataset_id,
            max_seq_length=prog_args.max_seq_length,
            overwrite_output=prog_args.overwrite_output,
            neftune_noise_alpha=prog_args.neftune_noise_alpha,
            huggingface_auth_token=prog_args.huggingface_auth_token,
            eval_dataset=prog_args.eval_dataset,
            do_train=prog_args.fine_tune,
            is_debug_mode=prog_args.debug,
            eval_strategy=prog_args.eval_strategy if prog_args.eval_strategy is not None else prog_args.save_strategy,
            eval_steps=prog_args.eval_steps if prog_args.eval_steps is not None else prog_args.save_steps
        )
        tune_arguments.validate()
        return tune_arguments

    return TuneArguments(
        new_model=prog_args.new_model,
        training_data_dir=prog_args.training_data_dir,
        train_file=prog_args.training_data_file
    )


def do_initial_arg_validation(args):
    """Do initial argument validations."""
    # TODO - FIXME - Some of these validations are unaware of the mode being ran, but they should be
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ArgumentValidationException("'lora-r' and 'lora-alpha' must both be greater than zero")
    if not args.fine_tune and not args.merge and not args.push:
        raise ArgumentValidationException("'merge-only' cannot be used when both 'merge' and 'push' are set to 'false'")
    if args.fine_tune and args.epochs <= 0:
        raise ArgumentValidationException("cannot tune when epochs is set to <= 0")
    if args.fine_tune and (args.hf_training_dataset_id is None) and (args.training_data_dir is None or args.training_data_file is None or not os.path.exists(args.training_data_dir) or not os.path.exists(
            f'{args.training_data_dir}/{args.training_data_file}')):
        raise ArgumentValidationException('training data directory or file not found')
    if args.new_model is None:
        raise ArgumentValidationException("'--new-mode' CLI argument must be provided")


def parse_arguments(title: str, description: str):
    """Parse CLI arguments."""
    parser = _build_program_argument_parser(title, description)
    return _parse_arguments(parser)


def _parse_arguments(arg_parser):
    a_args = sys.argv
    a_args.pop(0)
    return arg_parser.parse_args(a_args)


def _parse_bool_arg(arg: str | None) -> bool:
    return arg is not None and arg.lower().strip() == 'true'


def _parse_nullable_arg(arg: str | None) -> str | None:
    if arg is None or arg.strip() == '' or arg.lower().strip() == 'none' or arg.lower().strip() == 'null':
        return None
    return arg

def _parse_nullable_int_arg(arg: str | None) -> int | None:
    if arg is None or arg.strip() == '' or arg.lower().strip() == 'none' or arg.lower().strip() == 'null':
        return None
    return int(arg)

def _parse_nullable_float_arg(arg: str | None) -> float | None:
    if arg is None or arg.strip() == '' or arg.lower().strip() == 'none' or arg.lower().strip() == 'null':
        return None
    return float(arg)

def _parse_nullable_list_arg(arg: str | None) -> list | None:
    if arg is None or arg.strip() == '' or arg.lower().strip() == 'none' or arg.lower().strip() == 'null':
        return None
    return arg.split(',')

def _build_program_argument_parser(title: str, description: str) -> ArgumentParser:
    parser = ArgumentParser(
        prog=title,
        description=description)
    parser.add_argument('-nm', '--new-model', help="Name of the new fine-tuned model/adapter(REQUIRED[for fine-tune, merge & push only])")
    parser.add_argument('-hftdi', '--hf-training-dataset-id', help="HF dataset identifier(NOTE - overrides any defined tuning data file)(REQUIRED[for fine-tune without file only])")
    parser.add_argument('-tdd', '--training-data-dir', help="Training data directory(REQUIRED[for fine-tune from only])")
    parser.add_argument('-tdf', '--training-data-file', help="Training dataset filename(txt or jsonl)(REQUIRED[for fine-tune from file only])")
    parser.add_argument('-bm', '--base-model', help="Base model to tune(can be either HF model identifier or path to local model)(default: meta-llama/Meta-Llama-3-8B-Instruct)", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('-od', '--output-directory', help=f"Directory path to store output state(default: ~{os.sep}torch-tuner)", default=f"~{os.sep}torch-tuner")
    parser.add_argument('-owo', '--overwrite-output', help="Overwrite previous model output(default: true)", default="true", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-debug', '--debug', help="Debug mode(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-tam', '--target-all-modules', help="Target all tunable modules(targets all linear modules when false)(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-tm', '--target-modules', help="Modules to target(CSV List: 'q,k')(OVERRIDES '--target-all-modules' when not None)(default: None)", type=lambda x: _parse_nullable_list_arg(x), default="None")
    parser.add_argument('-tecs', '--torch-empty-cache-steps', help="Empty torch cache after x steps(NEVER empties cache when set to None)(USEFUL to prevent OOM issues)(default: 1)", type=lambda x: _parse_nullable_int_arg(x), default="1")
    parser.add_argument('-cft', '--cpu-only-tuning', default="false", help="Run a fine-tune job on CPU ONLY(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-hfat', '--huggingface-auth-token', default="None", help="Huggingface auth token(default: None)", type=lambda x: _parse_nullable_arg(x))

    parser.add_argument('-lgpumem', '--use-low-gpu-memory', default="true", help="Use low GPU memory optimizations(default: true)", type=lambda x: _parse_bool_arg(x))

    parser.add_argument('-ft', '--fine-tune', default="true", help="Run a fine-tune job(default: true)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-lbbs', '--load-best-before-save', default="false", help="Load best checkpoint before saving LoRA adapter(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-stm', '--show-token-metrics', default="false", help="Print token metrics during fine-tuning(WARNING - slows down tuning)(default: false)", type=lambda x: _parse_bool_arg(x))


    parser.add_argument('-m', '--merge', default="true",
                        help="Merge the tuned LoRA adapter with the base model(default: true)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-p', '--push', help="Push merged model to Huggingface(default: true)", default="true", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-pp', '--public-push', help="Push to public HF repo(push is private if false)(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-de', '--do-eval', help="Do evaluation on each configured step(does full evaluation when `--fine-tune` argument is set to false)(default: false)", default="false", type=lambda x: _parse_bool_arg(x))

    parser.add_argument('-serve', '--serve', help="Serve model(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-sm', '--serve-model', help="Huggingface repo or full path of the model to serve(REQUIRED[for serve only)")
    parser.add_argument('-sp', '--serve-port', help="Port to serve model on(default: 8080)", type=int, default=8080)
    parser.add_argument('-tmlm', '--train-masked-language-model', help="Train masked language model(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-mlmp', '--mlm-probability', help="MLM probability(default: 0.15)", type=lambda x: _parse_nullable_float_arg(x), default=0.15)
    parser.add_argument('-mt', '--mask-token', help="Mask token(default: \nObservation)", default="\nObservation")
    parser.add_argument('-mpr', '--max-parallel-requests', help="Maximum nuber of requests to execute against LLM in parallel(for serve only)(default: 1)", type=int, default=1)


    parser.add_argument('-cm', '--is-chat-model', help="Tune your new model for chat(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-avt', '--additional-vocabulary-tokens', help="Add additional tokens to model vocabulary(This should be a comma separated list[ex: USER:,AI:])(default: None)", type=lambda x: _parse_nullable_list_arg(x), default="None")
    parser.add_argument('-uat', '--use-agent-tokens', default="false", help="Tune with LangChain agent tokens(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-iim', '--is-instruct-model', help="Is the model being tuned intended for instruct(when set to true, enables several enhancements for instruct models)(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-nna', '--neftune-noise-alpha', help="NEFTune Noise Alpha(ONLY applies when '--is-instruct-model' argument is set to true)(default 5.0)", type=lambda x: _parse_nullable_float_arg(x), default="5.0")

    parser.add_argument('-ps', '--padding-side', help="Padding side(when set to 'None' disables padding)(default: right)", type=lambda x: _parse_nullable_arg(x), default="right")

    # TODO - FIXME - Handle situation when user selects multiple quant./precision options(Which options take highest priority?)
    parser.add_argument('-4bit', '--use-4bit', help="Use 4bit quantization(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-8bit', '--use-8bit', help="Use 8bit quantization(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-fp16', '--use-fp-16', help="Use fp-16 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bf16', '--use-bf-16', help="Use bf-16 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-tf32', '--use-tf-32', help="Use tf-32(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-f32cpu', '--fp32-cpu-offload', default="false", help="Offload fp32 to CPU(default: false)", type=lambda x: _parse_bool_arg(x))


    parser.add_argument('-bs', '--batch-size', help="Per-device training/eval batch size(default 4)", type=int, default=4)
    parser.add_argument('-gbl', '--group-by-length', help="Group training samples of similar lengths together(default true)", type=lambda x: _parse_bool_arg(x), default="true")
    parser.add_argument('-wur', '--warmup-ratio', help="Linear warmup over warmup_ratio fraction of total steps(default 0.03)", type=float, default=0.03)
    parser.add_argument('-r', '--lora-r', type=int, help="LoRA Rank(R) value(default: 8)", default=8)
    parser.add_argument('-a', '--lora-alpha', type=int, help="LoRA Alpha value(determines LoRA Scale[scale = alpha/R])(NOTE - high LoRA scale can lead to over-fitting)(default: 16)", default=16)
    parser.add_argument('-e', '--epochs', type=int, help="Number of iterations over of the entire dataset(default: 10)", default=10)
    parser.add_argument('-se', '--save-embeddings', default="false", help="Save embeddings layers(NOTE - consumes a lot of memory when set to true)(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-lrb', '--base-learning-rate', help="Base learning rate(actual rate = batch-size * learning-base-rate)(This value CHANGES if --lr-scheduler-type is not set to 'constant')(ONLY applies to AdamW optimizers)(default: 2e-5)", type=float, default=2e-5)
    parser.add_argument('-lrst', '--lr-scheduler-type', default="linear", help="Learning rate scheduler type(determines the learning rate decrease as tuning progresses[helps stabilize tuning and prevent over-fitting])(ONLY applies to AdamW optimizers)(default: linear)")
    parser.add_argument('-do', '--lora-dropout', help="LoRA dropout(this helps to prevent over-fitting)(default: 0.02)", type=float, default=0.02)
    parser.add_argument('-ncp', '--no-checkpoint', help="Don't use checkpointing(does not save trainer state until tuning is complete and creating the LoRA adapter when set to true)(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bias', '--bias', help="Bias(default: none)", default="none")
    parser.add_argument('-ot', '--optimizer-type', help="Optimizer type(default: adamw_torch_fused)", default="adamw_torch_fused")
    parser.add_argument('-gas', '--gradient-accumulation-steps', help="Gradient accumulation steps(default: 4)", type=int, default=4)
    parser.add_argument('-wd', '--weight-decay', help="Weight decay(default: 0.01)", type=float, default=0.01)
    parser.add_argument('-mgn', '--max-gradient-norm', help="Max gradient norm(default: 0.0)", type=float, default=0.0)
    parser.add_argument('-ss', '--save-strategy', help="Save strategy(default: epoch)", default="epoch")
    parser.add_argument('-msl', '--max-seq-length', help="The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset(default: the smaller of the `tokenizer.model_max_length` and `1024`)", type=lambda x: _parse_nullable_int_arg(x), default="None")
    parser.add_argument('-ssteps', '--save-steps', help="Save after each --save-steps steps(ignored when SAVE_STRATEGY='epoch')(default: 50)", default=50, type=int)
    parser.add_argument('-ms', '--max-saved', help="Maximum number of checkpoint saves to keep(this helps prevent filling up disk while tuning)(default: 5)(USE None to keep all checkpoint saves)", default="5", type=lambda x: _parse_nullable_int_arg(x))
    parser.add_argument('-eds', '--eval-dataset', help="Path or HF id of evaluation dataset(defaults to training dataset when set to None)(default: None)", default="None", type=lambda x: _parse_nullable_arg(x))
    parser.add_argument('-llm', '--llm-type', help="LLM Type(default: generic[options: generic, llama])", default="generic")
    parser.add_argument('-evalstrat', '--eval-strategy', help="Eval strategy('None', 'epoch' or 'steps')(Defaults to SAVE_STRATEGY when set to None)(default: None)", default="None", type=lambda x: _parse_nullable_arg(x))
    parser.add_argument('-evalsteps', '--eval-steps', help="Steps between evaluations(Ignored when EVAL_STRATEGY is set to 'epoch')(Defaults to SAVE_STEPS when set to None)(default: None)", default="None", type=lambda x: _parse_nullable_int_arg(x))

    return parser
