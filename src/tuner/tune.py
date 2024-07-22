import sys
from text.methods import fine_tune, push, merge

from argparse import ArgumentParser


def parse_arguments(arg_parser):
    a_args = sys.argv
    a_args.pop(0)
    return arg_parser.parse_args(a_args)


parser = ArgumentParser(
    prog='LoRA AI LLM Text Fine-Tuner',
    description='Fine-Tune AI LLM models with text using Torch and LoRA.')

parser.add_argument('-b', '--base-model', help="Base model(from HF) to tune(default: meta-llama/Meta-Llama-3-8B-Instruct)", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('-n', '--new-model', help="Name of the fine-tuned model")
parser.add_argument('-r', '--lora-r', type=int, help="LoRA R value(default: 8)", default=8)
parser.add_argument('-a', '--lora-alpha', type=int, help="LoRA Alpha value(default: 32)", default=32)

parser.add_argument('-e', '--epochs', type=int, help="Number of iterations of the entire dataset(default: 10)", default=10)
parser.add_argument('-mo', '--merge-only', default="false", help="Only merge/push model(no tuning)(default: false)")
parser.add_argument('-m', '--merge', default="false",
                    help="Merge the tuned LoRA adapter with the base model(default: false)")

parser.add_argument('-tdd', '--training-data-dir', help="Training data directory")
parser.add_argument('-tf', '--training-data-file', help="Training dataset filename")
parser.add_argument('-p', '--push', help="Push merged model to Huggingface(default: false)", default="false")
parser.add_argument('-bs', '--batch-size', help="Base model to tune", type=int, default=4)
parser.add_argument('-fp16', '--fp-16', help="Use fp-16(default: false)", default="false")
parser.add_argument('-bf16', '--bf-16', help="Use bf-16(default: false)", default="false")
parser.add_argument('-lrb', '--learning-rate-base', help="Base learning rate(default: 2e-5)", type=float, default=2e-5)
parser.add_argument('-ld', '--lora-dropout', help="LoRA dropout(default: 0.05)", type=float, default=0.05)
parser.add_argument('-ncp', '--no-checkpoint', help="Don't use checkpoint(default: false)", default="false")
parser.add_argument('-bias', '--bias', help="Bias(default: none)", default="none")
parser.add_argument('-ot', '--optimizer-type', help="Optimizer type(default: paged_adamw_32bit)", default="paged_adamw_32bit")
parser.add_argument('-gas', '--gradient-accumulation-steps', help="Gradient accumulation steps(default: 4)", type=int, default=4)
parser.add_argument('-wd', '--weight-decay', help="Weight decay(default: 0.01)", type=float, default=0.01)
parser.add_argument('-mgn', '--max-gradient-norm', help="Max gradient norm(default: 0.0)", type=float, default=0.0)
parser.add_argument('-tf32', '--tf-32', help="Use TF32(default: true)", default="true")
parser.add_argument('-ss', '--save-strategy', help="Save strategy(default: epoch)", default="epoch")
parser.add_argument('-ssp', '--save-steps', help="Save steps(default: 1)", default=1, type=int)


args = parse_arguments(parser)

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

new_model = ""
training_data_dir = "./"
logs_output_dir = "../../logs"
epochs = 10
train_file = ""
r = 8
a = 32
merge_only = False
push_model = False
merge_model = False
use_fp_16 = False
use_bf_16 = False
no_checkpoint = False
use_tf_32 = False
if args.base_model is not None and args.base_model.strip() != '':
    base_model = args.base_model.strip()
if args.new_model is not None and args.new_model.strip() != '':
    new_model = args.new_model.strip()
if args.lora_r is not None:
    r = args.lora_r
if args.lora_alpha is not None:
    a = args.lora_alpha
if args.epochs is not None:
    epochs = args.epochs

# TODO - impl. proper way to parse bool
if args.merge_only is not None and args.merge_only.lower().strip() == 'true':
    merge_only = True
if args.training_data_dir is not None and args.training_data_dir.strip() != '':
    training_data_dir = args.training_data_dir
if args.training_data_file is not None and args.training_data_file.strip() != '':
    train_file = args.training_data_file
if args.push is not None and args.push.lower().strip() == 'true':
    push_model = True

if args.merge is not None and args.merge.lower().strip() == 'true':
    merge_model = True

if args.fp_16 is not None and args.fp_16.lower().strip() == 'true':
    use_fp_16 = True

if args.bf_16 is not None and args.bf_16.lower().strip() == 'true':
    use_bf_16 = True

if args.tf_32 is not None and args.tf_32.lower().strip() == 'true':
    use_tf_32 = True

if args.no_checkpoint is not None and args.no_checkpoint.lower().strip() == 'true':
    no_checkpoint = True

if not merge_only:
    fine_tune(r, a, epochs, base_model, new_model, training_data_dir, train_file, args.batch_size, use_fp_16, use_bf_16, args.learning_rate_base, args.lora_dropout, no_checkpoint, args.bias, args.optimizer_type, args.gradient_accumulation_steps, args.weight_decay, args.max_gradient_norm, use_tf_32, args.save_strategy, args.save_steps)

if merge_model:
    merge(base_model, new_model)

if push_model:
    push(new_model)
