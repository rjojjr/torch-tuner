import json

from exception.exceptions import ArgumentValidationException


class CliArguments:
    """Base CLI arguments class"""

    def validate(self) -> None:
        """Raise TunerException if arguments are invalid."""
        pass


class ServerArguments(CliArguments):
    """LLM REST API server arguments."""

    def __init__(self, port: int = 8080, debug: bool = False):
        self.port = port
        self.debug = debug

    def validate(self) -> None:
        if self.port <= 0:
            raise ArgumentValidationException("`port` must be positive non-zero integer value")


class LlmArguments(CliArguments):
    """Base LLM load parameters."""

    def __init__(self, model: str, use_4bit: bool = False, use_8bit: bool = False, is_fp16: bool = False, is_bf16: bool = False, fp32_cpu_offload: bool = False, padding_side: str | None = 'right', huggingface_auth_token: str | None = None, max_parallel_requests: int = 1):
        self.model = model
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.fp32_cpu_offload = fp32_cpu_offload
        self.padding_side = padding_side
        self.huggingface_auth_token = huggingface_auth_token
        self.max_parallel_requests = max_parallel_requests

    def validate(self) -> None:
        dt_type_count = 0
        dt_args = [self.use_4bit, self.use_8bit, self.is_bf16, self.is_fp16]
        for dt_arg in dt_args:
            if dt_arg:
                dt_type_count += 1
        if dt_type_count > 1:
            raise ArgumentValidationException("only one of `use-4bit`, `use-8bit`, `is-bf16` or `is-fp16` data type options can be enabled at any instance in time")

        if self.padding_side is not None and not (self.padding_side == 'right' or self.padding_side == 'left'):
            raise ArgumentValidationException("`padding-side` must be one of either 'None', 'left' or 'right'")


class TunerFunctionArguments(CliArguments):
    """Base tuning related function arguments."""

    def __init__(self, new_model: str, is_fp16: bool = False, is_bf16: bool = False, use_4bit: bool = False, use_8bit: bool = False,
                 fp32_cpu_offload: bool = False, is_chat_model: bool = True,
                 padding_side: str | None = 'right', use_agent_tokens: bool = False, additional_vocabulary_tokens: list | None = None, huggingface_auth_token: str | None = None,
                 is_debug_mode: bool = False):
        self.new_model = new_model
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.fp32_cpu_offload = fp32_cpu_offload
        self.is_chat_model = is_chat_model
        self.padding_side = padding_side
        self.use_agent_tokens = use_agent_tokens
        self.additional_vocabulary_tokens = additional_vocabulary_tokens
        self.huggingface_auth_token = huggingface_auth_token
        self.is_debug_mode = is_debug_mode

    def validate(self) -> None:
        """Validate members."""
        is_valid_dtype = self.is_fp16 is not None and self.is_bf16 is not None
        is_valid_dtype = is_valid_dtype and self.use_8bit is not None and self.use_4bit is not None
        is_valid_dtype = is_valid_dtype and self.fp32_cpu_offload is not None

        if not is_valid_dtype:
            raise ArgumentValidationException("'Tuner Arguments' are missing required data-type properties")

        is_valid = self.new_model is not None and self.is_chat_model is not None
        is_valid = is_valid and self.use_agent_tokens is not None

        if not is_valid:
            raise ArgumentValidationException("'Tuner Arguments' are missing required properties")

        dt_type_count = 0
        dt_args = [self.use_4bit, self.use_8bit, self.is_bf16, self.is_fp16]
        for dt_arg in dt_args:
            if dt_arg:
                dt_type_count += 1
        if dt_type_count > 1:
            raise ArgumentValidationException("only one of `use-4bit`, `use-8bit`, `is-bf16` or `is-fp16` data type options can be enabled at any instance in time")

        if self.padding_side is not None and not (self.padding_side == 'right' or self.padding_side == 'left'):
            raise ArgumentValidationException("`padding-side` must be one of either 'None', 'left' or 'right'")

        if self.additional_vocabulary_tokens is not None and len(self.additional_vocabulary_tokens) == 0:
            raise ArgumentValidationException("`additional-vocabulary-tokens` must be one of either 'None' or a CSV list, it must never be empty")


class LlmExecutorFactoryArguments(LlmArguments):
    """Init LLM Executor factory."""
    def __init__(self, model: str, use_4bit: bool = False, use_8bit: bool = False, is_fp16: bool = False, is_bf16: bool = False, fp32_cpu_offload: bool = False, padding_side: str | None = 'right', max_parallel_requests: int = 1, use_cpu_only: bool = False):
        super(LlmExecutorFactoryArguments, self).__init__(model, use_4bit, use_8bit, is_fp16, is_bf16, fp32_cpu_offload, padding_side, max_parallel_requests=max_parallel_requests)
        self.use_cpu_only = use_cpu_only

    def validate(self) -> None:
        dt_type_count = 0
        dt_args = [self.use_4bit, self.use_8bit, self.is_bf16, self.is_fp16]
        for dt_arg in dt_args:
            if dt_arg:
                dt_type_count += 1
        if dt_type_count > 1:
            raise ArgumentValidationException("only one of `use-4bit`, `use-8bit`, `is-bf16` or `is-fp16` data type options can be enabled at any instance in time")

        if self.padding_side is not None and not (self.padding_side == 'right' or self.padding_side == 'left'):
            raise ArgumentValidationException("`padding-side` must be one of either 'None', 'left' or 'right'")


class TuneArguments(TunerFunctionArguments):
    """'fine-tune' function arguments."""

    def __init__(self,
                 new_model: str,
                 training_data_dir: str,
                 train_file: str | None,
                 base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
                 r: int = 8,
                 alpha: int = 16,
                 epochs: int = 10,
                 batch_size: int = 4,
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 base_learning_rate: float = 2e-5,
                 lora_dropout: float = 0.05,
                 no_checkpoint: bool = False,
                 bias: str = "none",
                 optimizer_type: str = 'adamw_torch_fused',
                 gradient_accumulation_steps: int | None = None,
                 weight_decay: float = 0.01,
                 max_gradient_norm: float = 0.0,
                 is_tf32: bool = False,
                 save_strategy: str = "epoch",
                 save_steps: int = 50,
                 do_eval: bool = False,
                 max_checkpoints: int = 3,
                 use_8bit: bool = False,
                 use_4bit: bool = False,
                 save_embeddings: bool = False,
                 output_directory: str = "./models",
                 fp32_cpu_offload: bool = True,
                 is_chat_model: bool = True,
                 target_all_modules: bool = False,
                 padding_side: str | None = 'right',
                 use_agent_tokens: bool = False,
                 lr_scheduler_type: str = 'linear',
                 target_modules: list | None = None,
                 torch_empty_cache_steps: int | None = 1,
                 warmup_ratio: float = 0.03,
                 additional_vocabulary_tokens: list | None = None,
                 cpu_only_tuning: bool = False,
                 is_instruct_model: bool = False,
                 group_by_length: bool = True,
                 hf_training_dataset_id: str | None = None,
                 max_seq_length: int | None = None,
                 overwrite_output: bool = True,
                 neftune_noise_alpha: float | None = 5.0,
                 huggingface_auth_token: str | None = None,
                 eval_dataset: str | None = None,
                 eval_strategy: str | None = None,
                 eval_steps: int | None = None,
                 do_train: bool = True,
                 is_debug_mode: bool = True,
                 load_best_before_save: bool = False,
                 show_token_metrics: bool = False,
                 train_masked_language_model: bool = False,
                 mask_token: str = '\nObservation',
                 mlm_probability: float = 0.15,
                 use_flash_attention: bool = False,
                 flash_attention_impl: str = 'flash_attention_2',
                 push_adapter: bool = True
                 ):
        super(TuneArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, fp32_cpu_offload, is_chat_model, padding_side, use_agent_tokens, additional_vocabulary_tokens, huggingface_auth_token, is_debug_mode=is_debug_mode)
        self.r = r
        self.alpha = alpha
        self.epochs = epochs
        self.base_model = base_model
        self.training_data_dir = training_data_dir
        self.train_file = train_file
        self.batch_size = batch_size
        self.base_learning_rate = base_learning_rate
        self.lora_dropout = lora_dropout
        self.no_checkpoint = no_checkpoint
        self.bias = bias
        self.optimizer_type = optimizer_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_gradient_norm = max_gradient_norm
        self.is_tf32 = is_tf32
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.do_eval = do_eval
        self.max_checkpoints = max_checkpoints
        self.save_embeddings = save_embeddings
        self.output_directory = output_directory
        self.target_all_modules = target_all_modules
        self.lr_scheduler_type = lr_scheduler_type
        self.target_modules = target_modules
        self.torch_empty_cache_steps = torch_empty_cache_steps
        self.warmup_ratio = warmup_ratio
        self.cpu_only_tuning = cpu_only_tuning
        self.is_instruct_model = is_instruct_model
        self.group_by_length = group_by_length
        self.hf_training_dataset_id = hf_training_dataset_id
        self.max_seq_length = max_seq_length
        self.overwrite_output = overwrite_output
        self.neftune_noise_alpha = neftune_noise_alpha
        self.eval_dataset = eval_dataset
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.do_train = do_train
        self.load_best_before_save = load_best_before_save
        self.show_token_metrics = show_token_metrics
        self.train_masked_language_model = train_masked_language_model
        self.mask_token = mask_token
        self.mlm_probability = mlm_probability
        self.use_flash_attention = use_flash_attention
        self.flash_attention_impl = flash_attention_impl
        self.push_adapter = push_adapter

    def validate(self) -> None:
        # I know it's bad, I will clean it up eventually
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.r is not None and self.alpha is not None
        is_valid = is_valid and self.epochs is not None and self.training_data_dir is not None
        is_valid = is_valid and self.batch_size is not None
        is_valid = is_valid and self.base_learning_rate is not None and self.lora_dropout is not None
        is_valid = is_valid and self.no_checkpoint is not None and self.bias is not None
        is_valid = is_valid and self.optimizer_type is not None
        is_valid = is_valid and self.weight_decay is not None and self.max_gradient_norm is not None
        is_valid = is_valid and self.save_strategy is not None and self.save_steps is not None
        is_valid = is_valid and self.do_eval is not None and self.max_checkpoints is not None
        is_valid = is_valid and self.save_embeddings is not None
        is_valid = is_valid and self.output_directory is not None
        is_valid = is_valid and not self.base_model.strip() == '' and not self.new_model.strip() == ''
        is_valid = is_valid and not self.optimizer_type.strip() == '' and not self.bias.strip() == ''
        is_valid = is_valid and not self.save_strategy.strip() == '' and not self.output_directory.strip() == ''

        if not is_valid:
            raise ArgumentValidationException("'Tune Arguments' are missing required properties")

        if self.gradient_accumulation_steps is not None and int(self.gradient_accumulation_steps) == 0:
            raise ArgumentValidationException("'--gradient-accumulation-steps' argument must not be zero")

        if self.hf_training_dataset_id is not None and self.hf_training_dataset_id.strip() == '':
            raise ArgumentValidationException("'--hf-training-dataset-id' argument value is invalid")

        if self.hf_training_dataset_id is None and (self.training_data_dir is None or self.train_file is None or self.train_file.strip() == ''):
            raise ArgumentValidationException("training data arguments are not configured properly")

        if self.alpha <= 0 or self.r <= 0:
            raise ArgumentValidationException("'--lora-r' and '--lora-alpha' arguments must both be greater than zero")

        if self.train_masked_language_model and self.mask_token == '':
            raise ArgumentValidationException('`--mask-token` argument must be non-empty string')

        super(TuneArguments, self).validate()

    def to_json(self):
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=True,
                          indent=4)


class MergeArguments(TunerFunctionArguments):
    """'merge' function arguments."""

    def __init__(self,
                 new_model: str,
                 base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 output_dir: str = '../../models',
                 is_chat_model: bool = True,
                 padding_side: str | None = 'right',
                 use_agent_tokens: bool = False,
                 additional_vocabulary_tokens: list | None = None,
                 overwrite_output: bool = True,
                 huggingface_auth_token: str | None = None,
                 is_debug_mode: bool = False,
                 train_masked_language_model: bool = False,
                 mask_token: str = '\nObservation'):
        super(MergeArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, is_chat_model=is_chat_model, padding_side=padding_side, use_agent_tokens=use_agent_tokens, additional_vocabulary_tokens=additional_vocabulary_tokens, huggingface_auth_token=huggingface_auth_token, is_debug_mode=is_debug_mode)
        self.base_model = base_model
        self.output_dir = output_dir
        self.overwrite_output = overwrite_output
        self.train_masked_language_model = train_masked_language_model
        self.mask_token = mask_token

    def validate(self) -> None:
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None
        is_valid = is_valid and self.output_dir is not None and not self.output_dir.strip() == ''
        is_valid = is_valid and not self.base_model.strip() == '' and not self.new_model.strip() == ''

        if not is_valid:
            raise ArgumentValidationException("'Merge Arguments' are missing required properties")

        if self.train_masked_language_model and self.mask_token == '':
            raise ArgumentValidationException('`--mask-token` argument must be non-empty string')

        super(MergeArguments, self).validate()

    def to_json(self):
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=True,
                          indent=4)


class PushArguments(TunerFunctionArguments):
    """'push' function arguments."""

    def __init__(self,
                 new_model: str,
                 model_dir: str,
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 public_push: bool = False,
                 is_chat_model: bool = True,
                 padding_side: str | None = 'right',
                 use_agent_tokens: bool = False,
                 additional_vocabulary_tokens: list | None = None,
                 huggingface_auth_token: str | None = None,
                 is_debug_mode: bool = False):
        super(PushArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, is_chat_model=is_chat_model, padding_side=padding_side, use_agent_tokens=use_agent_tokens, additional_vocabulary_tokens=additional_vocabulary_tokens, huggingface_auth_token=huggingface_auth_token, is_debug_mode=is_debug_mode)
        self.model_dir = model_dir
        self.public_push = public_push

    def validate(self) -> None:
        is_valid = self.new_model is not None and self.model_dir is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None
        is_valid = is_valid and self.public_push is not None
        is_valid = is_valid and not self.model_dir.strip() == '' and not self.new_model.strip() == ''

        if not is_valid:
            raise ArgumentValidationException("'Push Arguments' are missing required properties")

        super(PushArguments, self).validate()

    def to_json(self):
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=True,
                          indent=4)


class ArgumentsConfig:

    def __init__(self, tune_args: TuneArguments | None, merge_args: MergeArguments | None, push_args: PushArguments | None):
        self.tune_arguments = tune_args
        self.merge_arguments = merge_args
        self.push_arguments = push_args

    def to_json(self):
        return json.dumps(self,
                          default=lambda o: o.__dict__,
                          sort_keys=True,
                          indent=4)

    def from_json(self, json_string: str):
        loaded = json.loads(json_string)
        self.push_arguments = loaded['push_arguments']
        self.merge_arguments = loaded['merge_arguments']
        self.tune_arguments = loaded['tune_arguments']