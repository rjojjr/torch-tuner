from sympy.plotting.textplot import is_valid

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

    def __init__(self, model: str, use_4bit: bool = False, use_8bit: bool = False, is_fp16: bool = False, is_bf16: bool = False, fp32_cpu_offload: bool = False, padding_side: str | None = 'right', huggingface_auth_token: str | None = None):
        self.model = model
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.fp32_cpu_offload = fp32_cpu_offload
        self.padding_side = padding_side
        self.huggingface_auth_token = huggingface_auth_token

    def validate(self) -> None:
        if self.use_4bit and self.use_8bit:
            raise ArgumentValidationException("`use-4bit` and `use-8bit` cannot be enabled at the same time")

        if self.is_bf16 and self.is_fp16:
            raise ArgumentValidationException("`is-bf16` and `is-fp16` cannot be enabled at the same time")

        if self.padding_side is not None and not (self.padding_side == 'right' or self.padding_side == 'left'):
            raise ArgumentValidationException("`padding-side` must be one of either 'None', 'left' or 'right'")


class TunerFunctionArguments(CliArguments):
    """Base tuning related function arguments."""

    def __init__(self, new_model: str, is_fp16: bool = False, is_bf16: bool = False, use_4bit: bool = False, use_8bit: bool = False,
                 fp32_cpu_offload: bool = False, is_chat_model: bool = True,
                 padding_side: str | None = 'right', use_agent_tokens: bool = False, additional_vocabulary_tokens: list | None = None, huggingface_auth_token: str | None = None):
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

        if self.use_4bit and self.use_8bit:
            raise ArgumentValidationException("`use-4bit` and `use-8bit` cannot be enabled at the same time")

        if self.is_bf16 and self.is_fp16:
            raise ArgumentValidationException("`is-bf16` and `is-fp16` cannot be enabled at the same time")

        if self.padding_side is not None and not (self.padding_side == 'right' or self.padding_side == 'left'):
            raise ArgumentValidationException("`padding-side` must be one of either 'None', 'left' or 'right'")

        if self.additional_vocabulary_tokens is not None and len(self.additional_vocabulary_tokens) == 0:
            raise ArgumentValidationException("`additional-vocabulary-tokens` must be one of either 'None' or a CSV list, it must never be empty")


class LlmExecutorFactoryArguments(LlmArguments):
    """Init LLM Executor factory."""
    def __init__(self, model: str, use_4bit: bool = False, use_8bit: bool = False, is_fp16: bool = False, is_bf16: bool = False, fp32_cpu_offload: bool = False, padding_side: str | None = 'right'):
        super(LlmExecutorFactoryArguments, self).__init__(model, use_4bit, use_8bit, is_fp16, is_bf16, fp32_cpu_offload, padding_side)

    def validate(self) -> None:
        if self.use_4bit and self.use_8bit:
            raise ArgumentValidationException("`use-4bit` and `use-8bit` cannot be enabled at the same time")

        if self.is_bf16 and self.is_fp16:
            raise ArgumentValidationException("`is-bf16` and `is-fp16` cannot be enabled at the same time")

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
                 optimizer_type: str = 'adamw_8bit',
                 gradient_accumulation_steps: int = 4,
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
                 do_train: bool = True):
        super(TuneArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, fp32_cpu_offload, is_chat_model, padding_side, use_agent_tokens, additional_vocabulary_tokens, huggingface_auth_token)
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

    def validate(self) -> None:
        # I know it's bad, I will clean it up eventually
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.r is not None and self.alpha is not None
        is_valid = is_valid and self.epochs is not None and self.training_data_dir is not None
        is_valid = is_valid and self.batch_size is not None
        is_valid = is_valid and self.base_learning_rate is not None and self.lora_dropout is not None
        is_valid = is_valid and self.no_checkpoint is not None and self.bias is not None
        is_valid = is_valid and self.optimizer_type is not None and self.gradient_accumulation_steps is not None
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

        if self.hf_training_dataset_id is not None and self.hf_training_dataset_id.strip() == '':
            raise ArgumentValidationException("'--hf-training-dataset-id' argument value is invalid")

        if self.hf_training_dataset_id is None and (self.training_data_dir is None or self.train_file is None or self.train_file.strip() == ''):
            raise ArgumentValidationException("training data arguments are not configured properly")

        if self.alpha <= 0 or self.r <= 0:
            raise ArgumentValidationException("'--lora-r' and '--lora-alpha' arguments must both be greater than zero")

        super(TuneArguments, self).validate()


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
                 huggingface_auth_token: str | None = None):
        super(MergeArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, is_chat_model=is_chat_model, padding_side=padding_side, use_agent_tokens=use_agent_tokens, additional_vocabulary_tokens=additional_vocabulary_tokens, huggingface_auth_token=huggingface_auth_token)
        self.base_model = base_model
        self.output_dir = output_dir
        self.overwrite_output = overwrite_output

    def validate(self) -> None:
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None
        is_valid = is_valid and self.output_dir is not None and not self.output_dir.strip() == ''
        is_valid = is_valid and not self.base_model.strip() == '' and not self.new_model.strip() == ''

        if not is_valid:
            raise ArgumentValidationException("'Merge Arguments' are missing required properties")

        super(MergeArguments, self).validate()


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
                 huggingface_auth_token: str | None = None):
        super(PushArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, is_chat_model=is_chat_model, padding_side=padding_side, use_agent_tokens=use_agent_tokens, additional_vocabulary_tokens=additional_vocabulary_tokens, huggingface_auth_token=huggingface_auth_token)
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
