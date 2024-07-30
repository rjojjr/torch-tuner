from llama.functions import fine_tune, push, merge
from arguments.arguments import TuneArguments, MergeArguments, PushArguments



from utils.argument_utils import parse_arguments, parse_boolean_args

version = '1.0.1'

args = parse_arguments(version)

merge_only, push_model, merge_model, use_fp_16, use_bf_16, do_eval, no_checkpoint, use_tf_32, use_8bit, use_4bit, fp32_cpu_offload, save_embeddings, public_push = parse_boolean_args(args)

lora_scale = round(args.lora_alpha / args.lora_r, 1)
model_dir = f'{args.output_directory}/{args.new_model}'

print(f'Llama AI LLM LoRA Torch Text Fine-Tuner v{version}')
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