from main.arguments.arguments import TuneArguments, MergeArguments, PushArguments
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
import os
import shutil

# LLM independent base functions


# TODO - Tune/extract an embeddings only model
def fine_tune_base(arguments: TuneArguments, tokenizer, base_model) -> None:
    print(f"Starting fine-tuning of base model {arguments.base_model} for {arguments.new_model}")
    print('')
    output_dir = f"{arguments.output_directory}/in-progress/{arguments.new_model}"
    lora_dir = f"{arguments.output_directory}/in-progress/{arguments.new_model}/adapter"
    if not arguments.no_checkpoint:
        print(f'Checkpointing to {output_dir}')
        print('')

    # TODO - Need to figure out to best configure the tuner for non-plaintext training data
    ds = load_dataset(arguments.training_data_dir, data_files={"train": arguments.train_file})

    target_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj", "lm-head"]
    if arguments.save_embeddings:
        target_modules.append("embed_tokens")
    lora_config = LoraConfig(
        r=arguments.r,
        lora_alpha=arguments.alpha,
        target_modules=target_modules,
        lora_dropout=arguments.lora_dropout,
        bias=arguments.bias,
        task_type=TaskType.CAUSAL_LM
    )
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    learning_rate = arguments.batch_size * arguments.learning_rate_base

    train_params = SFTConfig(
        output_dir=output_dir,
        include_tokens_per_second=False,
        include_num_input_tokens_seen=False,
        num_train_epochs=arguments.epochs,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        overwrite_output_dir=True,
        optim=arguments.optimizer_type,
        save_strategy=arguments.save_strategy,
        save_steps=arguments.save_steps,
        logging_strategy=arguments.save_strategy,
        logging_steps=arguments.save_steps,
        save_total_limit=arguments.max_checkpoints,
        learning_rate=learning_rate,
        weight_decay=arguments.weight_decay,
        fp16=arguments.is_fp16,
        tf32=arguments.is_tf32,
        bf16=arguments.is_bf16,
        max_grad_norm=arguments.max_gradient_norm,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        do_eval=arguments.do_eval,
        max_seq_length=10240,
        dataset_text_field="text",
    )

    train = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        tokenizer=tokenizer,
        args=train_params
    )

    model.config.use_cache = False

    # TODO - FIXME - There is a warning from checkpointing I believe is do to underlying torch impl.
    if os.path.exists(output_dir) and not arguments.no_checkpoint:
        model.gradient_checkpointing_enable()
        last_checkpoint = get_last_checkpoint(output_dir)
        train.train(resume_from_checkpoint=last_checkpoint)
    else:
        train.train()

    print('')
    print(f'Saving LoRA adapter to {lora_dir}')
    if os.path.exists(lora_dir):
        shutil.rmtree(lora_dir)

    train.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)


def merge_base(arguments: MergeArguments, tokenizer, base_model, bnb_config) -> None:
    # TODO - Refactor how output dirs are constructed
    lora_dir = f"{arguments.output_dir}/in-progress/{arguments.new_model}/adapter"
    model_dir = f'{arguments.output_dir}/{arguments.new_model}'
    print(f"merging {arguments.base_model} with LoRA into {arguments.new_model}")
    print('')

    model = PeftModel.from_pretrained(base_model, lora_dir, quantization_config=bnb_config)
    model = model.merge_and_unload(progressbar=True)
    print('')

    print(f'Saving model to {model_dir}')
    print('')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def push_base(arguments: PushArguments, tokenizer, model) -> None:
    print(f"pushing {arguments.new_model} to HF")
    print('')

    is_private = not arguments.public_push
    model.push_to_hub(arguments.new_model, private=is_private)
    tokenizer.push_to_hub(arguments.new_model, private=is_private)
