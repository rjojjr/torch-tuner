from exception.exceptions import ArgumentValidationException


class ServerArguments:
    def __init__(self, port: int = 8080, debug: bool = False):
        self.port = port
        self.debug = debug


class LlmExecutorFactoryArguments:
    def __init__(self, model: str, use_4bit: bool = False, use_8bit: bool = False, is_fp16: bool = False, is_bf16: bool = False, fp32_cpu_offload: bool = False):
        self.model = model
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.fp32_cpu_offload = fp32_cpu_offload

    def validate(self) -> None:
        if self.use_4bit and self.use_8bit:
            raise ArgumentValidationException("`use-4bit` and `use-8bit` cannot be enabled at the same time")
        if self.is_bf16 and self.is_fp16:
            raise ArgumentValidationException("`is-bf16` and `is-fp16` cannot be enabled at the same time")



class TunerFunctionArguments:
    def __init__(self, new_model: str, is_fp16: bool = False, is_bf16: bool = False, use_4bit: bool = False, use_8bit: bool = False, fp32_cpu_offload: bool = False):
        self.new_model = new_model
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.fp32_cpu_offload = fp32_cpu_offload

    def validate(self) -> None:
        pass


class TuneArguments(TunerFunctionArguments):
    def __init__(self,
                 new_model: str,
                 training_data_dir: str,
                 train_file: str,
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
                 optimizer_type: str = 'paged_adamw_32bit',
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
                 output_directory: str = "../../models",
                 fp32_cpu_offload: bool = True):
        super(TuneArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit, fp32_cpu_offload)
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

    def validate(self) -> None:
        # I know it's bad, I will clean it up eventually
        # TODO - validate some fields individually so that ArgumentExceptions are more useful
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.r is not None and self.alpha is not None
        is_valid = is_valid and self.epochs is not None and self.training_data_dir is not None
        is_valid = is_valid and self.train_file is not None and self.batch_size is not None
        is_valid = is_valid and self.base_learning_rate is not None and self.lora_dropout is not None
        is_valid = is_valid and self.no_checkpoint is not None and self.bias is not None
        is_valid = is_valid and self.optimizer_type is not None and self.gradient_accumulation_steps is not None
        is_valid = is_valid and self.weight_decay is not None and self.max_gradient_norm is not None
        is_valid = is_valid and self.save_strategy is not None and self.save_steps is not None
        is_valid = is_valid and self.do_eval is not None and self.max_checkpoints is not None
        is_valid = is_valid and self.save_embeddings is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None and self.is_tf32 is not None
        is_valid = is_valid and self.output_directory is not None and self.fp32_cpu_offload is not None
        is_valid = is_valid and not self.base_model.strip() == '' and not self.new_model.strip() == ''
        is_valid = is_valid and not self.training_data_dir.strip() == '' and not self.train_file.strip() == ''
        is_valid = is_valid and not self.optimizer_type.strip() == '' and not self.bias.strip() == ''
        is_valid = is_valid and not self.save_strategy.strip() == '' and not self.output_directory.strip() == ''

        if not is_valid:
            raise ArgumentValidationException("'Tune Arguments' are missing required properties")


class MergeArguments(TunerFunctionArguments):
    def __init__(self,
                 new_model: str,
                 base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 output_dir: str = '../../models'):
        super(MergeArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit)
        self.base_model = base_model
        self.output_dir = output_dir

    def validate(self) -> None:
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None
        is_valid = is_valid and self.output_dir is not None and not self.output_dir.strip() == ''
        is_valid = is_valid and not self.base_model.strip() == '' and not self.new_model.strip() == ''

        if not is_valid:
            raise ArgumentValidationException("'Merge Arguments' are missing required properties")


class PushArguments(TunerFunctionArguments):
    def __init__(self,
                 new_model: str,
                 model_dir: str,
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 public_push: bool = False):
        super(PushArguments, self).__init__(new_model, is_fp16, is_bf16, use_4bit, use_8bit)
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
