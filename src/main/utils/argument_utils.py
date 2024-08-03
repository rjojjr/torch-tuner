import sys, os
from argparse import ArgumentParser
from main.exception.exceptions import ArgumentValidationException
from main.arguments.arguments import PushArguments, MergeArguments, TuneArguments


def build_and_validate_push_args(prog_args, model_dir: str):
    push_arguments = PushArguments(
        new_model=prog_args.new_model,
        model_dir=model_dir
    )

    if prog_args.push:
        push_arguments = PushArguments(
            new_model=prog_args.new_model,
            model_dir=model_dir,
            use_4bit=prog_args.use_4bit,
            use_8bit=prog_args.use_8bit,
            is_bf16=prog_args.use_bf_16,
            is_fp16=prog_args.use_fp_16,
            public_push=prog_args.public_push
        )
        push_arguments.validate()

    return push_arguments


def build_and_validate_merge_args(prog_args):
    merge_arguments = MergeArguments(new_model=prog_args.new_model)
    if prog_args.merge:
        merge_arguments = MergeArguments(
            new_model=prog_args.new_model,
            base_model=prog_args.base_model,
            use_4bit=prog_args.use_4bit,
            use_8bit=prog_args.use_8bit,
            is_bf16=prog_args.use_bf_16,
            is_fp16=prog_args.use_fp_16,
            output_dir=prog_args.output_directory
        )
        merge_arguments.validate()

    return merge_arguments


def build_and_validate_tune_args(prog_args):
    tune_arguments = TuneArguments(
        new_model=prog_args.new_model,
        training_data_dir=prog_args.training_data_dir,
        train_file=prog_args.training_data_file
    )
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
            learning_rate_base=prog_args.learning_rate_base,
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
            fp32_cpu_offload=prog_args.fp32_cpu_offload
        )
        tune_arguments.validate()

    return tune_arguments


def do_initial_arg_validation(args):
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
    parser = _build_program_argument_parser(title, description)
    return _parse_arguments(parser)


def _parse_arguments(arg_parser):
    a_args = sys.argv
    a_args.pop(0)
    return arg_parser.parse_args(a_args)


def _parse_bool_arg(arg: str | None) -> bool:
    return arg is not None and arg.lower().strip() == 'true'


def _build_program_argument_parser(title: str, description: str) -> ArgumentParser:
    parser = ArgumentParser(
        prog=title,
        description=description)
    parser.add_argument('-n', '--new-model', required=True, help="Name of the new fine-tuned model(REQUIRED)")
    parser.add_argument('-tdd', '--training-data-dir', help="Training data directory(REQUIRED)")
    parser.add_argument('-tf', '--training-data-file', help="Training dataset filename(REQUIRED)")
    parser.add_argument('-b', '--base-model', help="Base model(from HF) to tune(default: meta-llama/Meta-Llama-3-8B-Instruct)", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('-p', '--push', help="Push merged model to Huggingface(default: true)", default="true", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-m', '--merge', default="true",
                        help="Merge the tuned LoRA adapter with the base model(default: true)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-ft', '--fine-tune', default="true", help="Run a fine-tune job(default: true)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bs', '--batch-size', help="Samples per iteration(default 4)", type=int, default=4)
    parser.add_argument('-r', '--lora-r', type=int, help="LoRA R value(default: 8)", default=8)
    parser.add_argument('-a', '--lora-alpha', type=int, help="LoRA Alpha value(default: 32)", default=32)
    parser.add_argument('-od', '--output-directory', help="Directory path to store output state(default: ./models)", default="./models")

    parser.add_argument('-llm', '--llm-type', help="LLM Type(default: llama)", default="llama")
    parser.add_argument('-e', '--epochs', type=int, help="Number of iterations of the entire dataset(default: 10)", default=10)
    parser.add_argument('-sel', '--save-embeddings', default="false", help="Save embeddings(default: false)", type=lambda x: _parse_bool_arg(x))

    parser.add_argument('-pp', '--public-push', help="Push to public repo(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-4bit', '--use-4bit', help="Use 4bit quantization(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-8bit', '--use-8bit', help="Use 8bit quantization(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-fp16', '--use-fp-16', help="Use fp-16 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bf16', '--use-bf-16', help="Use bf-16 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-tf32', '--use-tf-32', help="Use tf-32 precision(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-f32cpu', '--fp32-cpu-offload', default="false", help="Offload fp32 to CPU(default: false)", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-lrb', '--learning-rate-base', help="Base learning rate(actual rate = batch-size * learning-base-rate)(default: 2e-5)", type=float, default=2e-5)
    parser.add_argument('-ld', '--lora-dropout', help="LoRA dropout(default: 0.05)", type=float, default=0.05)
    parser.add_argument('-ncp', '--no-checkpoint', help="Don't use checkpoint(default: false)", default="false", type=lambda x: _parse_bool_arg(x))
    parser.add_argument('-bias', '--bias', help="Bias(default: none)", default="none")
    parser.add_argument('-ot', '--optimizer-type', help="Optimizer type(default: paged_adamw_32bit)", default="paged_adamw_32bit")
    parser.add_argument('-gas', '--gradient-accumulation-steps', help="Gradient accumulation steps(default: 4)", type=int, default=4)
    parser.add_argument('-wd', '--weight-decay', help="Weight decay(default: 0.01)", type=float, default=0.01)
    parser.add_argument('-mgn', '--max-gradient-norm', help="Max gradient norm(default: 0.0)", type=float, default=0.0)
    parser.add_argument('-ss', '--save-strategy', help="Save strategy(default: epoch)", default="epoch")
    parser.add_argument('-ssp', '--save-steps', help="Save after each --save-steps steps(ignored when --save-strategy='epoch')(default: 50)", default=50, type=int)
    parser.add_argument('-ms', '--max-saved', help="Maximum number of checkpoint saves to keep(default: 3)", default=3, type=int)
    parser.add_argument('-de', '--do-eval', help="Do eval(default: true)", default="true", type=lambda x: _parse_bool_arg(x))

    return parser
