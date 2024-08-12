from arguments.arguments import TuneArguments, MergeArguments, PushArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, AutoPeftModelForCausalLM, PeftModel
from trl import SFTTrainer, SFTConfig, setup_chat_format
from transformers.trainer_utils import get_last_checkpoint
from utils.model_utils import get_all_layers, get_all_linear_layers
import os
import shutil


# LLM independent base functions


def _add_agent_tokens(tokenizer, model):
    agent_tokens = ["\nThought:", "\nAction:", "\nAction Input:", "\nObservation:", "\nFinal Answer:"]
    agent_tokens = set(agent_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(agent_tokens))
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))


# TODO - Tune/extract an embeddings only model
def fine_tune_base(arguments: TuneArguments, tokenizer, base_model) -> None:
    if arguments.is_chat_model:
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    if arguments.use_agent_tokens:
        _add_agent_tokens(tokenizer, base_model)
    print(f"Starting fine-tuning of base model {arguments.base_model} for {arguments.new_model}")
    print('')
    output_dir = f"{arguments.output_directory}/checkpoints/{arguments.new_model}"
    lora_dir = f"{arguments.output_directory}/checkpoints/{arguments.new_model}/adapter"
    if not arguments.no_checkpoint:
        print(f'Checkpointing to {output_dir}')
        print('')

    if arguments.train_file is not None:
        ds = load_dataset(arguments.training_data_dir, data_files={"train": arguments.train_file})
    else:
        ds = load_dataset(arguments.training_data_dir, split='train')

    if arguments.target_modules is None or len(arguments.target_modules) == 0:
        target_modules = get_all_layers(base_model) if arguments.target_all_modules else get_all_linear_layers(base_model)
    else:
        target_modules = arguments.target_modules

    if arguments.use_agent_tokens or arguments.is_chat_model:
        target_modules.append("embed_tokens")
        target_modules.append("lm_head")
        target_modules = list(set(target_modules))

    modules_to_save=["embed_tokens"] if arguments.save_embeddings else []


    lora_config = LoraConfig(
        r=arguments.r,
        lora_alpha=arguments.alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=arguments.lora_dropout,
        bias=arguments.bias,
        task_type=TaskType.CAUSAL_LM
    )
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    learning_rate = arguments.batch_size * arguments.base_learning_rate

    train_params = SFTConfig(
        output_dir=output_dir,
        include_tokens_per_second=False,
        include_num_input_tokens_seen=False,
        num_train_epochs=arguments.epochs,
        torch_empty_cache_steps=arguments.torch_empty_cache_steps,
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
        # TODO - CPU Tuning
        use_cpu=False,
        fp16=arguments.is_fp16,
        tf32=arguments.is_tf32,
        bf16=arguments.is_bf16,
        max_grad_norm=arguments.max_gradient_norm,
        max_steps=-1,
        warmup_ratio=arguments.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=arguments.lr_scheduler_type,
        report_to="tensorboard",
        do_eval=arguments.do_eval,
        # TODO - is this ignored bt SFTTrainer?
        max_seq_length=4096,
        dataset_text_field="text"
        # TODO - investigate for instruction training
        #neftune_noise_alpha
    )

    train = SFTTrainer(
        model=model,
        train_dataset=ds['train'] if arguments.train_file is not None else ds,
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
    train.model.config.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    del model
    del base_model
    del tokenizer


def merge_base(arguments: MergeArguments, tokenizer, base_model, bnb_config) -> None:
    if arguments.is_chat_model:
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    if arguments.use_agent_tokens:
        _add_agent_tokens(tokenizer, base_model)
    lora_dir = f"{arguments.output_dir}/checkpoints/{arguments.new_model}/adapter"
    model_dir = f'{arguments.output_dir}/{arguments.new_model}'
    print(f"merging {arguments.base_model} with LoRA into {arguments.new_model}")

    if arguments.use_agent_tokens:
        model = AutoPeftModelForCausalLM.from_pretrained(lora_dir)
    else:
        model = PeftModel.from_pretrained(base_model, lora_dir, quantization_config=bnb_config)

    model = model.merge_and_unload(progressbar=True)
    print('')

    print(f'Saving model to {model_dir}')
    print('')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    del model
    del base_model
    del tokenizer


def push_base(arguments: PushArguments, tokenizer, model) -> None:
    print(f"pushing {arguments.new_model} to HF")
    print('')

    is_private = not arguments.public_push
    model.push_to_hub(arguments.new_model, private=is_private)
    tokenizer.push_to_hub(arguments.new_model, private=is_private)
    del model
    del tokenizer
