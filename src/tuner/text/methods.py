from transformers import TrainingArguments
import os.path, shutil

import torch
from trl import SFTTrainer

from transformers import LlamaTokenizer, LlamaForCausalLM

from datasets import load_dataset, Features, Value
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType

from huggingface_hub import login
from transformers.trainer_utils import get_last_checkpoint

def merge(model_base, new_model_name):
    lora_dir = f"../../models/in-progress/{new_model_name}/adapter"
    model_dir = f'../../models/{new_model_name}'
    print(f"merging {model_base} with LoRA into {new_model_name}")
    base_model = LlamaForCausalLM.from_pretrained(
        model_base,
        low_cpu_mem_usage=False,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload(progressbar=True)

    tokenizer = LlamaTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def push(new_model):
    # TODO - change default dirs
    model_dir = f'../../models/{new_model}'

    print(f"pushing {new_model} to HF")
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=False,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map="auto"
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Huggingface auth token should be set in 'HUGGING_FACE_TOKEN' evv. var.
    login(os.environ.get('HUGGING_FACE_TOKEN'))

    model.push_to_hub(new_model, private=True)
    tokenizer.push_to_hub(new_model, private=True)


def fine_tune(r, alpha, epochs, base_model, new_model, data_train, train_file, batch_size, use_fp_16, use_bf_16, learning_rate_base, lora_dropout, no_checkpoint, bias, optimizer_type, gradient_accumulation_steps, weight_decay, max_gradient_norm, tf_32, save_strategy, save_steps):
    print(f"Starting fresh tuning of {new_model}")
    output_dir = "../../models/in-progress/" + new_model
    lora_dir = "../../models/in-progress/" + new_model + "/adapter"

    login(os.environ.get('HUGGING_FACE_TOKEN'))

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_dataset(data_train, data_files={"train": train_file})

    model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=True, device_map="auto")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        # Text
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj", "embed_tokens", "lm-head"],
        # Text To Image
        #target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    learning_rate = batch_size * learning_rate_base

    train_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        overwrite_output_dir=True,
        optim=optimizer_type,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=50,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=use_fp_16,
        tf32=tf_32,
        bf16=use_bf_16,
        max_grad_norm=max_gradient_norm,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        do_eval=True
    )

    # TODO - customize training data format
    train = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params,
        max_seq_length=10240
    )

    model.config.use_cache = False

    if os.path.exists(output_dir) and not no_checkpoint:
        last_checkpoint = get_last_checkpoint(output_dir)
        train.train(resume_from_checkpoint=last_checkpoint)
    else:
        train.train()

    if os.path.exists(lora_dir):
        shutil.rmtree(lora_dir)
    train.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
