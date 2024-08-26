import sys, os
from argparse import ArgumentParser
from exception.exceptions import ArgumentValidationException
from arguments.arguments import PushArguments, MergeArguments, TuneArguments


def build_and_validate_push_args(prog_args, model_dir: str):
    """Construct/validate push arguments."""
    if prog_args.push:
        push_arguments = PushArguments(
            new_model=prog_args.new_model,
            model_dir=model_dir,
            use_4bit=prog_args.use_4bit,
            use_8bit=prog_args.use_8bit,
            is_bf16=prog_args.use_bf_16,
            is_fp16=prog_args.use_fp_16,
            public_push=prog_args.public_push,
            padding_side=prog_args.padding_side,
            use_agent_tokens=prog_args.use_agent_tokens,
            additional_vocabulary_tokens=prog_args.additional_vocabulary_tokens
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
            output_dir=prog_args.output_directory,
            padding_side=prog_args.padding_side,
            use_agent_tokens=prog_args.use_agent_tokens,
            additional_vocabulary_tokens=prog_args.additional_vocabulary_tokens
        )
        merge_arguments.validate()
        return merge_arguments

    return MergeArguments(new_model=prog_args.new_model)


def build_and_validate_tune_args(prog_args) -> TuneArguments:
    """Construct/validate tune arguments."""
    if prog_args.fine_tune:
        tune_arguments = TuneArguments(
            base_model=prog_args.base_model,
            new_model=prog_args.new_model,
            training_data_dir=prog_args.training_data_dir,
            train_file=prog_args.training_data_file,
            r=prog_args.lora_r,
            alpha=prog_args.lora_alpha,
            epochs=prog_args.epochs,
            batch_size=prog_args.batch_size,
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
            output_directory=prog_args.output_directory,
            fp32_cpu_offload=prog_args.fp32_cpu_offload,
            is_chat_model=prog_args.is_chat_model,
            padding_side=prog_args.padding_side,
            use_agent_tokens=prog_args.use_agent_tokens,
            lr_scheduler_type=prog_args.lr_scheduler_type,
            target_modules=prog_args.target_modules,
            torch_empty_cache_steps=prog_args.torch_empty_cache_steps,
            warmup_ratio=prog_args.warmup_ratio,
            additional_vocabulary_tokens=prog_args.additional_vocabulary_tokens,
            cpu_only_tuning=prog_args.cpu_only_tuning,
            is_instruct_model=prog_args.is_instruct_model
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
    if args.fine_tune and (not os.path.exists(args.training_data_dir) or not os.path.exists(
            f'{args.training_data_dir}/{args.training_data_file}')):
        raise ArgumentValidationException('training data directory or file not found')


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

def _parse_nullable_list_arg(arg: str | None) -> list | None:
    if arg is None or arg.strip() == '' or arg.lower().strip() == 'none' or arg.lower().strip() == 'null':
        return None
    return arg.split(',')

def _build_program_argument_parser(title: str, description: str) -> ArgumentParser:
    parser = ArgumentParser(
        prog=title,
        description=description)
    parser.add_argument('-nm', '--new-model', help="Name of the new fine-tuned model/adapter(REQUIRED[for fine-tune, merge & push only])")
    parser.add_argument('-tdd', '--training-data-dir', help="Training data directory or HF dataset name(REQUIRED[for fine-tune only])")
    parser.add_argument('-tdf', '--training-data-file', help="Training dataset filename(txt or jsonl)(REQUIRED[for fine-tune from file only])")
    parser.add_argument('-bm', '--base-model', help="Base model to tune(can be either HF model identifier or path to local model)(default: meta-llama/Meta-Llama-3-8B-Instruct)", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('-od', '--output-directory', help="Directory path to store output state(default: ./models)", default="./models")
    parser.add_argument('-debug', '--debug', help="Debug mode(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-cm', '--is-chat-model', help="Tune your new model for chat(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-tam', '--target-all-modules', help="Target all tunable modules(targets all linear modules when false)(default: false)", type=lambda x: _parse_bool_arg(x), default="false")
    parser.add_argument('-tm', '--target-modules', help="Modules to target(CSV List: 'q,k')(OVERRIDES '--target-all-modules' when not None)(default: None)", type=lambda x: _parse_nullable_list_arg(x), default="None")
    parser.add_argument('-tecs', '--torch-empty-cache-steps', help="Empty torch cache after x steps(NEVER empties cache when set to None)(USEFUL to prevent OOM issues)(default: 1)", type=lambda x: _parse_nullable_int_arg(x), default="1")
    parser.add_argument('-avt', '--additional-vocabulary-tokens', help="Add additional tokens to model vocabulary(This should be a comma separated list[ex: USER:,AI:])(default: None)", type=lambda x: _parse_nullable_list_arg(x), default="None")

    parser.add_argument('-ps', '--padding-side', help="Padding side(when set to 'None' disables padding)(default: right)", type=lambda x: _parse_nullable_arg(x), default="right")
    parser.add_argument('-iim', '--is-instruct-model', help="Is the model being tuned intended for instruct(when set to true, enables several enhancements for instruct models)(default: false)", type=lambda x: _parse_bool_arg(x), default="false")

    parser.add_argument('-serve', '--serve', help="Serve model(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-sm', '--serve-model', help="Huggingface repo or full path of the model to serve(REQUIRED[for serve only)")
    parser.add_argument('-sp', '--serve-port', help="Port to serve model on(default: 8080)", type=int, default=8080)

    parser.add_argument('-ft', '--fine-tune', default="true", help="Run a fine-tune job(default: true)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-cft', '--cpu-only-tuning', default="false", help="Run a fine-tune job on CPU ONLY(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-m', '--merge', default="true",
                        help="Merge the tuned LoRA adapter with the base model(default: true)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-p', '--push', help="Push merged model to Huggingface(default: true)", default="true", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-pp', '--public-push', help="Push to public HF repo(push is private if false)(default: false)", default="false", type=lambda x: _parse_bool_arg(x))

    # TODO - FIXME - Handle situation when user selects multiple quant./precision options(Which options take highest priority?)
    parser.add_argument('-4bit', '--use-4bit', help="Use 4bit quantization(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-8bit', '--use-8bit', help="Use 8bit quantization(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-fp16', '--use-fp-16', help="Use fp-16 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bf16', '--use-bf-16', help="Use bf-16 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-tf32', '--use-tf-32', help="Use tf-32(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-f32cpu', '--fp32-cpu-offload', default="false", help="Offload fp32 to CPU(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-uat', '--use-agent-tokens', default="false", help="Tune with LangChain agent tokens(default: false)", type=lambda x: _parse_bool_arg(x))

    parser.add_argument('-bs', '--batch-size', help="Per-device training/eval batch size(default 4)", type=int, default=4)
    parser.add_argument('-wur', '--warmup-ratio', help="Linear warmup over warmup_ratio fraction of total steps(default 0.03)", type=float, default=0.03)
    parser.add_argument('-r', '--lora-r', type=int, help="LoRA rank(R) value(default: 8)", default=8)
    parser.add_argument('-a', '--lora-alpha', type=int, help="LoRA Alpha value(determines LoRA Scale[scale = alpha/R])(NOTE - high LoRA scale can lead to over-fitting)(default: 16)", default=16)
    parser.add_argument('-e', '--epochs', type=int, help="Number of iterations over of the entire dataset(default: 10)", default=10)
    parser.add_argument('-se', '--save-embeddings', default="false", help="Save embeddings layers(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-lrb', '--base-learning-rate', help="Base learning rate(actual rate = batch-size * learning-base-rate)(This value CHANGES if --lr-scheduler-type is not set to 'constant')(ONLY applies to AdamW optimizers)(default: 2e-5)", type=float, default=2e-5)
    parser.add_argument('-lrst', '--lr-scheduler-type', default="linear", help="Learning rate scheduler type(determines the learning rate decrease as tuning progresses[helps stabilize tuning and prevent over-fitting])(ONLY applies to AdamW optimizers)(default: linear)")
    parser.add_argument('-do', '--lora-dropout', help="LoRA dropout(this helps to prevent over-fitting)(default: 0.05)", type=float, default=0.05)
    parser.add_argument('-ncp', '--no-checkpoint', help="Don't use checkpointing(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bias', '--bias', help="Bias(default: none)", default="none")
    parser.add_argument('-ot', '--optimizer-type', help="Optimizer type(default: adamw_8bit)", default="adamw_8bit")
    parser.add_argument('-gas', '--gradient-accumulation-steps', help="Gradient accumulation steps(default: 4)", type=int, default=4)
    parser.add_argument('-wd', '--weight-decay', help="Weight decay(default: 0.01)", type=float, default=0.01)
    parser.add_argument('-mgn', '--max-gradient-norm', help="Max gradient norm(default: 0.0)", type=float, default=0.0)
    parser.add_argument('-ss', '--save-strategy', help="Save strategy(default: epoch)", default="epoch")
    parser.add_argument('-ssteps', '--save-steps', help="Save after each --save-steps steps(ignored when --save-strategy='epoch')(default: 50)", default=50, type=int)
    parser.add_argument('-ms', '--max-saved', help="Maximum number of checkpoint saves to keep(this helps prevent filling up disk while tuning)(default: 5)", default=5, type=int)
    parser.add_argument('-de', '--do-eval', help="Do evaluation on each save step(default: true)", default="true", type=lambda x: _parse_bool_arg(x))

    parser.add_argument('-llm', '--llm-type', help="LLM Type(default: llama)", default="llama")

    return parser
