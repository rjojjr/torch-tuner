import sys, os
from argparse import ArgumentParser
from main.exception.exceptions import ArgumentValidationException


def do_initial_arg_validation(args, merge_model, merge_only, push_model):
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ArgumentValidationException("'lora-r' and 'lora-alpha' must both be greater than zero")

    if merge_only and not merge_model and not push_model:
        raise ArgumentValidationException("'merge-only' cannot be used when both 'merge' and 'push' are set to 'false'")
    if not merge_only and args.epochs <= 0:
        raise ArgumentValidationException("cannot tune when epochs is set to <= 0")
    if not merge_only and (not os.path.exists(args.training_data_dir) or not os.path.exists(
            f'{args.training_data_dir}/{args.training_data_file}')):
        raise ArgumentValidationException('training data directory or file not found')


def parse_arguments(title: str, description: str):
    return _parse_arguments(_build_program_argument_parser(title, description))


def parse_boolean_args(args):
    merge_only = False
    push_model = False
    merge_model = False
    use_fp_16 = False
    use_bf_16 = False
    do_eval = False
    no_checkpoint = False
    use_tf_32 = False
    use_8bit = False
    use_4bit = False
    fp32_cpu_offload = False
    save_embeddings = False
    public_push = False
    if args.fp32_cpu_offload is not None and args.fp32_cpu_offload.lower().strip() == 'true':
        fp32_cpu_offload = True
    if args.merge_only is not None and args.merge_only.lower().strip() == 'true':
        merge_only = True
    if args.use_8bit is not None and args.use_8bit.lower().strip() == 'true':
        use_8bit = True
    if args.use_4bit is not None and args.use_4bit.lower().strip() == 'true':
        use_4bit = True
        use_8bit = False
    if args.do_eval is not None and args.do_eval.lower().strip() == 'true':
        do_eval = True
    if args.save_embeddings_layer is not None and args.save_embeddings_layer.lower().strip() == 'true':
        save_embeddings = True
    if args.push is not None and args.push.lower().strip() == 'true':
        push_model = True
    if args.public_push is not None and args.public_push.lower().strip() == 'true':
        public_push = True
    if args.merge is not None and args.merge.lower().strip() == 'true':
        merge_model = True
    if args.fp_16 is not None and args.fp_16.lower().strip() == 'true':
        use_bf_16 = False
        use_fp_16 = True
        use_tf_32 = False
    if args.bf_16 is not None and args.bf_16.lower().strip() == 'true':
        use_bf_16 = True
        use_fp_16 = False
        use_tf_32 = False
    if args.tf_32 is not None and args.tf_32.lower().strip() == 'true':
        use_tf_32 = True
        use_bf_16 = False
        use_fp_16 = False
    if args.no_checkpoint is not None and args.no_checkpoint.lower().strip() == 'true':
        no_checkpoint = True

    return merge_only, push_model, merge_model, use_fp_16, use_bf_16, do_eval, no_checkpoint, use_tf_32, use_8bit, use_4bit, fp32_cpu_offload, save_embeddings, public_push


def _parse_arguments(arg_parser):
    a_args = sys.argv
    a_args.pop(0)
    return arg_parser.parse_args(a_args)


def _build_program_argument_parser(title: str, description: str) -> ArgumentParser:
    parser = ArgumentParser(
        prog=title,
        description=description)
    parser.add_argument('-b', '--base-model', help="Base model(from HF) to tune(default: meta-llama/Meta-Llama-3-8B-Instruct)", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('-n', '--new-model', help="Name of the new fine-tuned model(REQUIRED)")
    parser.add_argument('-tdd', '--training-data-dir', help="Training data directory(REQUIRED)")
    parser.add_argument('-tf', '--training-data-file', help="Training dataset filename(REQUIRED)")
    parser.add_argument('-p', '--push', help="Push merged model to Huggingface(default: false)", default="false")
    parser.add_argument('-bs', '--batch-size', help="Samples per iteration(default 4)", type=int, default=4)
    parser.add_argument('-r', '--lora-r', type=int, help="LoRA R value(default: 8)", default=8)
    parser.add_argument('-a', '--lora-alpha', type=int, help="LoRA Alpha value(default: 32)", default=32)
    parser.add_argument('-od', '--output-directory', help="Directory path to store output state(default: ../../models)", default="../../models")

    parser.add_argument('-e', '--epochs', type=int, help="Number of iterations of the entire dataset(default: 10)", default=10)
    parser.add_argument('-mo', '--merge-only', default="false", help="Only merge/push model(no tuning)(default: false)")
    parser.add_argument('-sel', '--save-embeddings-layer', default="false", help="Save embeddings(default: false)")
    parser.add_argument('-m', '--merge', default="false",
                        help="Merge the tuned LoRA adapter with the base model(default: false)")

    parser.add_argument('-f32cpu', '--fp32-cpu-offload', default="false", help="Offload fp32 to CPU(default: false)")

    parser.add_argument('-4bit', '--use-4bit', help="Use 4bit quantization(default: false)", default="false")
    parser.add_argument('-pp', '--public-push', help="Push to public repo(default: false)", default="false")
    parser.add_argument('-8bit', '--use-8bit', help="Use 8bit quantization(default: false)", default="false")
    parser.add_argument('-fp16', '--fp-16', help="Use fp-16 precision(default: false)", default="false")
    parser.add_argument('-bf16', '--bf-16', help="Use bf-16 precision(default: false)", default="false")
    parser.add_argument('-tf32', '--tf-32', help="Use tf-32 precision(default: false)", default="false")
    parser.add_argument('-lrb', '--learning-rate-base', help="Base learning rate(actual rate = batch-size * learning-base-rate)(default: 2e-5)", type=float, default=2e-5)
    parser.add_argument('-ld', '--lora-dropout', help="LoRA dropout(default: 0.05)", type=float, default=0.05)
    parser.add_argument('-ncp', '--no-checkpoint', help="Don't use checkpoint(default: false)", default="false")
    parser.add_argument('-bias', '--bias', help="Bias(default: none)", default="none")
    parser.add_argument('-ot', '--optimizer-type', help="Optimizer type(default: paged_adamw_32bit)", default="paged_adamw_32bit")
    parser.add_argument('-gas', '--gradient-accumulation-steps', help="Gradient accumulation steps(default: 4)", type=int, default=4)
    parser.add_argument('-wd', '--weight-decay', help="Weight decay(default: 0.01)", type=float, default=0.01)
    parser.add_argument('-mgn', '--max-gradient-norm', help="Max gradient norm(default: 0.0)", type=float, default=0.0)
    parser.add_argument('-ss', '--save-strategy', help="Save strategy(default: epoch)", default="epoch")
    parser.add_argument('-ssp', '--save-steps', help="Save after each --save-steps steps(ignored when --save-strategy='epoch')(default: 50)", default=50, type=int)
    parser.add_argument('-ms', '--max-saved', help="Maximum number of checkpoint saves to keep(default: 3)", default=3, type=int)
    parser.add_argument('-de', '--do-eval', help="Do eval(default: true)", default="true")

    return parser
