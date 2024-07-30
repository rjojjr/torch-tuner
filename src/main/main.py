import sys
from llama.functions import fine_tune, push, merge
from llama.arguments import TuneArguments, MergeArguments, PushArguments

from argparse import ArgumentParser

from utils.argument_utils import parse_arguments

build = 4

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
parser.add_argument('-od', '--output-directory', help="Directory path to store output state(default: ../../models)", default="../../models")

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

args = parse_arguments(parser)

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
    
lora_scale = round(args.lora_alpha / args.lora_r, 1)
model_dir = f'{args.output_directory}/{args.new_model}'

print(f'AI LLM LoRA Torch Text Fine-Tuner v{version}')
print(f'Build: {str(build)}')
print('---------------------------------------------')
print('Run with --help flag for a list of available arguments.')
print('')
print(f'Output Directory: {args.output_directory}')
print(f'Base Model: {args.base_model}')
print(f'Model Save Directory: {model_dir}')
print('')
print(f'Epochs: {str(args.epochs)}')
print(f'Using LoRA R: {str(args.lora_r)}')
print(f'Using LoRA Alpha: {str(args.lora_alpha)}')
print(f'LoRA Adapter Scale(alpha/r): {str(lora_scale)}')
print(f'Using Base Learning Rate: {str(args.learning_rate_base)}')
print(f'Using LoRA Dropout: {str(args.lora_dropout)}')
print('')
print(f'Using tf32: {str(use_tf_32)}')
print(f'Using bf16: {str(use_bf_16)}')
print(f'Using pf16: {str(use_fp_16)}')
print(f'Using 8bit: {str(use_8bit)}')
print(f'Using 4bit: {str(use_4bit)}')
print(f'Using 4bit: {str(use_4bit)}')
print(f'Using fp32 CPU Offload: {str(fp32_cpu_offload)}')
print('')
print(f'Is Merging: {merge_model}')
print(f'Is Pushing: {push_model}')
print(f'Is Merge/Push Only: {str(merge_only)}')

print('')
print(f'Using Checkpointing: {str(not no_checkpoint)}')
print(f'Using Max Saves: {str(args.max_saved)}')
print(f'Using Batch Size: {str(args.batch_size)}')
print(f'Using Optimizer: {args.optimizer_type}')
print(f'Using Save Strategy: {args.save_strategy}')
print(f'Using Save Steps: {args.save_steps}')
print(f'Using Save Embeddings: {str(save_embeddings)}')


if not merge_only:
    tune_arguments = TuneArguments(
        base_model=args.base_model,
        new_model=args.new_model,
        training_data_dir=args.training_data_dir,
        train_file=args.training_data_file,
        r=args.lora_r,
        alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_fp_16=use_fp_16,
        use_bf_16=use_bf_16,
        learning_rate_base=args.learning_rate_base,
        lora_dropout=args.lora_dropout,
        no_checkpoint=no_checkpoint,
        bias=args.bias,
        optimizer_type=args.optimizer_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_gradient_norm=args.max_gradient_norm,
        tf_32=use_tf_32,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        do_eval=do_eval,
        max_checkpoints=args.max_saved,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        save_embeddings=save_embeddings,
        output_directory=args.output_directory,
        fp32_cpu_offload=fp32_cpu_offload
    )
    print('')
    print(f'Tuning model {args.new_model} on base model {args.base_model} with {args.training_data_file} to {args.epochs} epochs')
    fine_tune(tune_arguments)
    print(f'Tuned model {args.base_model} on base model {args.base_model} with {args.training_data_file} to {args.epochs} epochs')

if merge_model:
    merge_arguments = MergeArguments(
        new_model_name=args.new_model,
        model_base=args.base_mode,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        is_bf16=use_bf_16,
        is_fp16=use_fp_16,
        output_dir=args.output_directory
    )
    print('')
    print(f'Merging LoRA Adapter for {args.new_model} with base model {args.base_model}')
    merge(merge_arguments)
    print(f'Merged LoRA Adapter for {args.new_model} with base model {args.base_model}')

if push_model:
    push_arguments = PushArguments(
        new_model=args.new_model,
        model_dir=model_dir,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        is_bf16=use_bf_16,
        is_fp16=use_fp_16
    )
    print('')
    print(f'Pushing {args.new_model} to Huggingface')
    push(push_arguments)
    print(f'Pushed {args.new_model} to Huggingface')

print('')
print('---------------------------------------------')
print('AI LLM LoRA Torch Text Fine-Tuner COMPLETED')