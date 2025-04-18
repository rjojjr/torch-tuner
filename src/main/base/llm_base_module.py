from transformers import DataCollatorForLanguageModeling

from exception.exceptions import TuningModuleFunctionException

from arguments.arguments import TuneArguments, MergeArguments, PushArguments
from utils.debugging_utils import debugging_wrapper

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, AutoPeftModelForCausalLM, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
from utils.model_utils import get_all_layers, get_all_linear_layers, prepare_model_vocabulary
from utils.dataset_utils import load_dataset
import os
import shutil


# LLM independent base functions


def fine_tune_eval_base(arguments: TuneArguments, tokenizer, base_model) -> None:
    with debugging_wrapper(arguments.is_debug_mode):
        if tokenizer.chat_template is not None:
            tokenizer.chat_template = None

        if arguments.do_train:
            print(f"Starting fine-tuning of base model {arguments.base_model} for {arguments.new_model}")
            print('')
        else:
            print(f"Starting evaluation of {arguments.new_model}")
            print('')
        output_dir = f"{arguments.output_directory}{os.sep}checkpoints{os.sep}{arguments.new_model}"
        lora_dir = f"{arguments.output_directory}{os.sep}adapters{os.sep}{arguments.new_model}"
        if arguments.do_train and not arguments.no_checkpoint:
            print(f'Checkpointing to {output_dir}')
            print('')

        if arguments.do_train and os.path.exists(lora_dir) and not arguments.overwrite_output:
            raise TuningModuleFunctionException(f'cannot overwrite existing LoRA directory({lora_dir}) when `--overwrite-output` CLI argument is not set to "true"', 'FINE_TUNE')

        if arguments.do_train:
            base_model, tokenizer = prepare_model_vocabulary(arguments, base_model, tokenizer)

        ds = load_dataset(arguments)
        if arguments.target_modules is None or len(arguments.target_modules) == 0:
            target_modules = get_all_layers(base_model) if arguments.target_all_modules else get_all_linear_layers(base_model)
        else:
            target_modules = arguments.target_modules

        modules_to_save=["embed_tokens"] if arguments.do_train and arguments.save_embeddings else []

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
        if arguments.train_masked_language_model:
            tokenizer._mask_token = arguments.mask_token
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=arguments.mlm_probability)

        train_params = SFTConfig(
            output_dir=output_dir,
            load_best_model_at_end=arguments.load_best_before_save,
            do_train=arguments.do_train,
            include_tokens_per_second=arguments.show_token_metrics,
            include_num_input_tokens_seen=arguments.show_token_metrics,
            num_train_epochs=arguments.epochs,
            torch_empty_cache_steps=arguments.torch_empty_cache_steps,
            per_device_train_batch_size=arguments.batch_size,
            per_device_eval_batch_size=arguments.batch_size,
            gradient_accumulation_steps=int(arguments.gradient_accumulation_steps) if arguments.gradient_accumulation_steps is not None else 1,
            gradient_checkpointing=True,
            eval_accumulation_steps=arguments.gradient_accumulation_steps,
            overwrite_output_dir=arguments.overwrite_output,
            optim=arguments.optimizer_type,
            save_strategy=arguments.save_strategy,
            save_steps=arguments.save_steps,
            eval_steps=arguments.eval_steps,
            logging_strategy=arguments.save_strategy,
            logging_steps=arguments.save_steps,
            save_total_limit=arguments.max_checkpoints,
            learning_rate=learning_rate,
            weight_decay=arguments.weight_decay,
            use_cpu=arguments.cpu_only_tuning,
            fp16=arguments.is_fp16,
            tf32=arguments.is_tf32,
            bf16=arguments.is_bf16,
            max_grad_norm=arguments.max_gradient_norm,
            max_steps=-1,
            warmup_ratio=arguments.warmup_ratio,
            group_by_length=arguments.group_by_length,
            lr_scheduler_type=arguments.lr_scheduler_type,
            report_to="tensorboard",
            do_eval=arguments.do_eval,
            eval_strategy=arguments.eval_strategy if arguments.do_eval else 'no',
            eval_on_start=arguments.do_eval,
            max_seq_length=arguments.max_seq_length,
            neftune_noise_alpha=arguments.neftune_noise_alpha if arguments.is_instruct_model else None,
            dataset_text_field="text" ,
            label_names=["completions"] if arguments.train_file.endswith("jsonl") else ['labels'],
            use_ipex=arguments.cpu_only_tuning,
        )

        # TODO - custom dataset formatting
        def formatting_prompts_func(sample):
            output_texts = []
            for i in range(len(sample['prompt'])):
                text = f"### Prompt: {sample['prompt'][i - 1]}\n ### Completion: {sample['completion'][i - 1]}"
                output_texts.append(text)
            return output_texts

        def tokenize_jsonl_dataset_factory(prediction_target = 'completion', template_func = None):
            def tokenize_jsonl_dataset(samples):
                prompts = [prompt
                           for prompt in samples["prompt"]] if template_func is None else template_func(samples)
                return tokenizer(prompts,
                                 text_target=samples[prediction_target],
                                 truncation=True, padding='do_not_pad',
                                 max_length=arguments.max_seq_length if arguments.max_seq_length is not None else (1024 if 1024 <= tokenizer.model_max_length else tokenizer.model_max_length),
                                 return_overflowing_tokens=True
                                 )


            return tokenize_jsonl_dataset

        processed_tuning_dataset = ds['train'].map(
            tokenize_jsonl_dataset_factory(),
            batched=True,
            desc="Tokenized dataset") if arguments.train_file.endswith("jsonl") else ds['train']

        processed_eval_dataset = (ds['eval'].map(
            tokenize_jsonl_dataset_factory(),
            batched=True,
            desc="Tokenized eval dataset") if arguments.train_file.endswith("jsonl") else ds['eval']) if 'eval' in ds else processed_tuning_dataset

        train = SFTTrainer(
            model=model,
            # formatting_func=formatting_prompts_func,
            train_dataset=processed_tuning_dataset if arguments.do_train else processed_eval_dataset,
            args=train_params,
            # TODO - FIXME - eval_loss stat is not printed
            eval_dataset=processed_eval_dataset if arguments.do_eval else None,
            data_collator=data_collator if arguments.train_masked_language_model else None,
        )

        model.config.use_cache = False
        if arguments.do_train:
            if os.path.exists(output_dir) and not arguments.no_checkpoint:
                print()
                print('Loading checkpoint')
                last_checkpoint = get_last_checkpoint(output_dir)
                print()
                print('Executing fine-tune job')
                print()
                train.train(resume_from_checkpoint=last_checkpoint)
            else:
                print()
                print('Executing fine-tune job')
                print()
                train.train()

            print('')
            print(f'Saving LoRA adapter to {lora_dir}')
            if os.path.exists(lora_dir) and arguments.overwrite_output:
                shutil.rmtree(lora_dir)

            train.model.save_pretrained(lora_dir)
            train.model.config.save_pretrained(lora_dir)
            tokenizer.save_pretrained(lora_dir)

            with open(f"{lora_dir}/tune_config.json", "w") as file:
                file.write(arguments.to_json())
                file.close()

            if arguments.push_adapter:
                print()
                print('Pushing LoRA adapter to huggingface')
                # TODO - pass private push argument to here
                # TODO - should this be async?
                train.model.push_to_hub(repo_id=f'{arguments.new_model}-lora-adaptor', commit_message=f"Tuned LORA adapter for {arguments.new_model}.", commit_description=f"Tuning Config: {arguments.to_json()}", private=True)
                tokenizer.push_to_hub(repo_id=f'{arguments.new_model}-lora-adaptor', commit_message=f"Add tokenizer for tuned LORA adapter {arguments.new_model}.", commit_description=f"Add tokenizer", private=True)
                print()

        else:
            print()
            print('Executing evaluation job')
            print()

            metrics = train.evaluate()
            print()
            print(f'Evaluation Results: {str(metrics)}')
            print()

        del model
        del base_model
        del tokenizer


def merge_base(arguments: MergeArguments, tokenizer, base_model, bnb_config) -> None:
    with debugging_wrapper(arguments.is_debug_mode):
        if tokenizer.chat_template is not None:
            tokenizer.chat_template = None
        if arguments.train_masked_language_model:
            tokenizer._mask_token = arguments.mask_token
        lora_dir = f"{arguments.output_dir}{os.sep}adapters{os.sep}{arguments.new_model}"
        model_dir = f'{arguments.output_dir}{os.sep}merged-models{os.sep}{arguments.new_model}'
        print(f"merging {arguments.base_model} with LoRA into {arguments.new_model}")

        if not os.path.exists(lora_dir):
            raise TuningModuleFunctionException(f'cannot merge model because LoRA adapter @ {lora_dir} is missing', 'MERGE')

        if os.path.exists(model_dir) and not arguments.overwrite_output:
            raise TuningModuleFunctionException(f'cannot overwrite existing model directory({model_dir}) when `--overwrite-output` CLI argument is not set to "true"', 'MERGE')

        prepare_model_vocabulary(arguments, base_model, tokenizer)

        if arguments.use_agent_tokens or arguments.additional_vocabulary_tokens is not None:
            model = AutoPeftModelForCausalLM.from_pretrained(lora_dir)
        else:
            model = PeftModel.from_pretrained(base_model, lora_dir, quantization_config=bnb_config)

        model = model.merge_and_unload(progressbar=True)
        print('')

        print(f'Saving model to {model_dir}')
        print('')
        if os.path.exists(model_dir) and arguments.overwrite_output:
            shutil.rmtree(model_dir)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        del model
        del base_model
        del tokenizer
        with open(f"{model_dir}/merge_config.json", "w") as file:
            file.write(arguments.to_json())
            file.close()
        tune_config_path = f'{lora_dir}/tune_config.json'
        if os.path.exists(tune_config_path):
            shutil.copyfile(tune_config_path, f'{model_dir}/tune_config.json')


def push_base(arguments: PushArguments, tokenizer, model) -> None:
    with debugging_wrapper(arguments.is_debug_mode):
        print(f"pushing {arguments.new_model} to HF")
        print('')

        is_private = not arguments.public_push
        model.push_to_hub(arguments.new_model, private=is_private, commit_message=f"Merge {arguments.new_model} LoRA adapter with base model")
        tokenizer.push_to_hub(arguments.new_model, private=is_private, commit_message=f"Add {arguments.new_model} tokenizer")
        del model
        del tokenizer



