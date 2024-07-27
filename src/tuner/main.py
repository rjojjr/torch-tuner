import sys
from text.functions import fine_tune, push, merge

from argparse import ArgumentParser

version = '1.0.0'

parser = ArgumentParser(
    prog=f'AI LLM LoRA Torch Text Fine-Tuner v{version}',
    description='Fine-Tune LLM models with text using Torch and LoRA.')
parser.add_argument('-b', '--base-model', help="Base model(from HF) to tune(default: meta-llama/Meta-Llama-3-8B-Instruct)", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('-n', '--new-model', help="Name of the new fine-tuned model")
parser.add_argument('-tdd', '--training-data-dir', help="Training data directory")
parser.add_argument('-tf', '--training-data-file', help="Training dataset filename")
parser.add_argument('-p', '--push', help="Push merged model to Huggingface(default: false)", default="false")
parser.add_argument('-bs', '--batch-size', help="Samples per iteration(default 4)", type=int, default=4)
parser.add_argument('-r', '--lora-r', type=int, help="LoRA R value(default: 8)", default=8)
parser.add_argument('-a', '--lora-alpha', type=int, help="LoRA Alpha value(default: 32)", default=32)

parser.add_argument('-e', '--epochs', type=int, help="Number of iterations of the entire dataset(default: 10)", default=10)
parser.add_argument('-mo', '--merge-only', default="false", help="Only merge/push model(no tuning)(default: false)")
parser.add_argument('-sel', '--save-embeddings-layer', default="false", help="Save embeddings(default: false)")
parser.add_argument('-m', '--merge', default="false",
                    help="Merge the tuned LoRA adapter with the base model(default: false)")

parser.add_argument('-f32cpu', '--fp32-cpu-offload', default="false", help="Offload fp32 to CPU(default: false)")

parser.add_argument('-4bit', '--use-4bit', help="Use 4bit quantization(default: false)", default="false")
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


def parse_arguments(arg_parser):
    a_args = sys.argv
    a_args.pop(0)
    return arg_parser.parse_args(a_args)


args = parse_arguments(parser)

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_id="google/flan-t5-xl"
#model_id="google/flan-t5-xxl"
new_model = "newton_dv_11"
# data_train="/home/dev/ollama/tuning/data/newton-beta-0-10.txt.tmp"
# logs_output_dir="/home/dev/ollama/tuning/logs"
training_data_dir = "/home/robert/IdeaProjects/ai/openai-chat-module/lora/training-data/newton"
logs_output_dir = "../../logs"
epochs = 10
train_file = "newton-beta-0-11.txt.tmp"
r = 8
a = 32
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
if args.fp32_cpu_offload is not None and args.fp32_cpu_offload.lower().strip() == 'true':
    fp32_cpu_offload = True
if args.merge_only is not None and args.merge_only.lower().strip() == 'true':
    merge_only = True
if args.use_8bit is not None and args.use_8bit.lower().strip() == 'true':
    use_8bit = True
if args.use_4bit is not None and args.use_4bit.lower().strip() == 'true':
    use_4bit = True
if args.do_eval is not None and args.do_eval.lower().strip() == 'true':
    do_eval = True
if args.save_embeddings_layer is not None and args.save_embeddings_layer.lower().strip() == 'true':
    save_embeddings = True
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

print(f'AI LLM LoRA Torch Text Fine-Tuner v{version}')
print('---------------------------------------------')
print('Run with --help flag for a list of available arguments.')
print('')
print(f'Epochs: {epochs}')
print(f'Using LoRA Alpha: {a}')
print(f'Using LoRA R: {r}')
print(f'Using tf32: {use_tf_32}')
print(f'Using bf16: {use_bf_16}')
print(f'Using pf16: {use_fp_16}')
print(f'Using 8bit: {use_8bit}')
print(f'Using 4bit: {use_4bit}')

print(f'Is Merging: {merge_model}')
print(f'Is Pushing: {push_model}')
print(f'Is Merge/Push Only: {merge_only}')

print(f'Using Batch Size: {args.batch_size}')
print(f'Using Optimizer: {args.optimizer_type}')
print(f'Using Save Strategy: {args.save_strategy}')


if not merge_only:
    print('')
    print(f'Tuning model {new_model} with base model {base_model} to {epochs} epochs')
    fine_tune(r, a, epochs, base_model, new_model, training_data_dir, train_file, args.batch_size, use_fp_16, use_bf_16, args.learning_rate_base, args.lora_dropout, no_checkpoint, args.bias, args.optimizer_type, args.gradient_accumulation_steps, args.weight_decay, args.max_gradient_norm, use_tf_32, args.save_strategy, args.save_steps, do_eval, args.max_saved, use_8bit, use_4bit, save_embeddings, fp32_cpu_offload)
    print(f'Tuned model {new_model} with base model {base_model} to {epochs} epochs')

if merge_model:
    print('')
    print(f'Merging LoRA Adapter for {new_model} with base model {base_model}')
    merge(base_model, new_model, use_fp_16, use_bf_16, use_4bit, use_8bit)
    print(f'Merged LoRA Adapter for {new_model} with base model {base_model}')

if push_model:
    print('')
    print(f'Pushing {new_model} to Huggingface')
    push(new_model, use_fp_16, use_bf_16, use_4bit, use_8bit)
    print(f'Pushed {new_model} to Huggingface')

print('')
print('---------------------------------------------')
print('AI LLM LoRA Torch Text Fine-Tuner COMPLETED')