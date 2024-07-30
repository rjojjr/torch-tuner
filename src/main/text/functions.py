from transformers import TrainingArguments
import os.path
import shutil

import torch
from trl import SFTTrainer

from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig

from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType

from huggingface_hub import login
from transformers.trainer_utils import get_last_checkpoint
from text.arguments import TuneArguments


def merge(model_base, new_model_name, is_fp16, is_bf16, use_4bit, use_8bit):
    lora_dir = f"../../models/in-progress/{new_model_name}/adapter"
    model_dir = f'../../models/{new_model_name}'
    print(f"merging {model_base} with LoRA into {new_model_name}")

    login(os.environ.get('HUGGING_FACE_TOKEN'))

    dtype = torch.float32
    if is_fp16:
        dtype = torch.float16
    if is_bf16:
        dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig()
    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    base_model = LlamaForCausalLM.from_pretrained(
        model_base,
        low_cpu_mem_usage=False,
        return_dict=True,
        torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base_model, lora_dir, quantization_config=bnb_config)
    model = model.merge_and_unload(progressbar=True)

    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def push(new_model, is_fp16, is_bf16, use_4bit, use_8bit):
    model_dir = f'../../models/{new_model}'

    print(f"pushing {new_model} to HF")
    dtype = torch.float32
    if is_fp16:
        dtype = torch.float16
    if is_bf16:
        dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig()
    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=False,
        return_dict=True,
        torch_dtype=dtype,
        quantization_config=bnb_config
        # device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Huggingface auth token should be set in 'HUGGING_FACE_TOKEN' evv. var.
    login(os.environ.get('HUGGING_FACE_TOKEN'))

    model.push_to_hub(new_model, private=True)
    tokenizer.push_to_hub(new_model, private=True)


# TODO - create args class for cleaner and more flexible signatures
def fine_tune(arguments: TuneArguments):
    print(f"Starting fresh tuning of {arguments.new_model}")
    output_dir = "../../models/in-progress/" + arguments.new_model
    lora_dir = "../../models/in-progress/" + arguments.new_model + "/adapter"

    login(os.environ.get('HUGGING_FACE_TOKEN'))

    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_dataset(arguments.training_data_dir, data_files={"train": arguments.train_file})

    dtype = torch.float32
    if arguments.use_fp_16:
        dtype = torch.float16
    if arguments.use_bf_16:
        dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload
    )
    if arguments.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
            llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload
        )
    if arguments.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload
        )

    model = LlamaForCausalLM.from_pretrained(arguments.base_model, quantization_config=bnb_config, device_map="auto")

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
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    learning_rate = arguments.batch_size * arguments.learning_rate_base

    train_params = TrainingArguments(
        output_dir=output_dir,
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
        fp16=arguments.use_fp_16,
        tf32=arguments.tf_32,
        bf16=arguments.use_bf_16,
        max_grad_norm=arguments.max_gradient_norm,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        do_eval=arguments.do_eval
    )

    train = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params,
        max_seq_length=10240
    )

    model.config.use_cache = False

    if os.path.exists(output_dir) and not arguments.no_checkpoint:
        last_checkpoint = get_last_checkpoint(output_dir)
        train.train(resume_from_checkpoint=last_checkpoint)
    else:
        train.train()

    if os.path.exists(lora_dir):
        shutil.rmtree(lora_dir)
    train.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)


def _get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'auto'