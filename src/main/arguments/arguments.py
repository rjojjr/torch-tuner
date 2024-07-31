from main.exception.exceptions import ArgumentValidationException


class TuneArguments:
    def __init__(self,
                 new_model: str,
                 training_data_dir: str,
                 train_file: str,
                 base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
                 r: int = 8,
                 alpha: int = 16,
                 epochs: int = 10,
                 batch_size: int = 4,
                 use_fp_16: bool = False,
                 use_bf_16: bool = False,
                 learning_rate_base: float = 2e-5,
                 lora_dropout: float = 0.05,
                 no_checkpoint: bool = False,
                 bias: str = "none",
                 optimizer_type: str = 'paged_adamw_32bit',
                 gradient_accumulation_steps: int = 4,
                 weight_decay: float = 0.01,
                 max_gradient_norm: float = 0.0,
                 use_tf_32: bool = False,
                 save_strategy: str = "epoch",
                 save_steps: int = 50,
                 do_eval: bool = False,
                 max_checkpoints: int = 3,
                 use_8bit: bool = False,
                 use_4bit: bool = False,
                 save_embeddings: bool = False,
                 output_directory: str = "../../models",
                 fp32_cpu_offload: bool = True):
        self.r = r
        self.alpha = alpha
        self.epochs = epochs
        self.base_model = base_model
        self.new_model = new_model
        self.training_data_dir = training_data_dir
        self.train_file = train_file
        self.batch_size = batch_size
        self.is_fp16 = use_fp_16
        self.is_bf16 = use_bf_16
        self.learning_rate_base = learning_rate_base
        self.lora_dropout = lora_dropout
        self.no_checkpoint = no_checkpoint
        self.bias = bias
        self.optimizer_type = optimizer_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_gradient_norm = max_gradient_norm
        self.is_tf32 = use_tf_32
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.do_eval = do_eval
        self.max_checkpoints = max_checkpoints
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.save_embeddings = save_embeddings
        self.output_directory = output_directory
        self.fp32_cpu_offload = fp32_cpu_offload

    def validate(self) -> None:
        is_valid = self.new_model is not None and self.base_model is not None
        is_valid = is_valid and self.r is not None and self.alpha is not None
        is_valid = is_valid and self.epochs is not None and self.training_data_dir is not None
        is_valid = is_valid and self.train_file is not None and self.batch_size is not None
        is_valid = is_valid and self.learning_rate_base is not None and self.lora_dropout is not None
        is_valid = is_valid and self.no_checkpoint is not None and self.bias is not None
        is_valid = is_valid and self.optimizer_type is not None and self.gradient_accumulation_steps is not None
        is_valid = is_valid and self.weight_decay is not None and self.max_gradient_norm is not None
        is_valid = is_valid and self.save_strategy is not None and self.save_steps is not None
        is_valid = is_valid and self.do_eval is not None and self.max_checkpoints is not None
        is_valid = is_valid and self.save_embeddings is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None and self.is_tf32 is not None
        is_valid = is_valid and self.output_directory is not None and self.fp32_cpu_offload is not None
        if not is_valid:
            raise ArgumentValidationException("'Tune Arguments' are missing required properties")


class MergeArguments:
    def __init__(self,
                 new_model_name: str,
                 model_base: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 output_dir: str = '../../models'):
        self.new_model_name = new_model_name
        self.model_base = model_base
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.output_dir = output_dir

    def validate(self) -> None:
        is_valid = self.new_model_name is not None and self.model_base is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None
        is_valid = is_valid and self.output_dir is not None
        if not is_valid:
            raise ArgumentValidationException("'Merge Arguments' are missing required properties")


class PushArguments:
    def __init__(self,
                 new_model: str,
                 model_dir: str,
                 is_fp16: bool = False,
                 is_bf16: bool = False,
                 use_4bit: bool = False,
                 use_8bit: bool = False,
                 public_push: bool = False):
        self.new_model = new_model
        self.model_dir = model_dir
        self.is_fp16 = is_fp16
        self.is_bf16 = is_bf16
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.public_push = public_push

    def validate(self) -> None:
        is_valid = self.new_model is not None and self.model_dir is not None
        is_valid = is_valid and self.is_fp16 is not None and self.is_bf16 is not None
        is_valid = is_valid and self.use_8bit is not None and self.use_4bit is not None
        is_valid = is_valid and self.public_push is not None
        if not is_valid:
            raise ArgumentValidationException("'Push Arguments' are missing required properties")


def build_and_validate_push_args(push_model: bool, prog_args, model_dir: str, use_4bit: bool, use_8bit: bool, use_bf_16: bool, use_fp_16: bool):
    push_arguments = PushArguments(
        new_model=prog_args.new_model,
        model_dir=model_dir
    )

    if push_model:
        push_arguments = PushArguments(
            new_model=prog_args.new_model,
            model_dir=model_dir,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            is_bf16=use_bf_16,
            is_fp16=use_fp_16
        )
        push_arguments.validate()

    return push_arguments


def build_and_validate_merge_args(merge_model: bool, prog_args, use_4bit: bool, use_8bit: bool, use_bf_16: bool, use_fp_16: bool):
    merge_arguments = MergeArguments(new_model_name=prog_args.new_model)
    if merge_model:
        merge_arguments = MergeArguments(
            new_model_name=prog_args.new_model,
            model_base=prog_args.base_model,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            is_bf16=use_bf_16,
            is_fp16=use_fp_16,
            output_dir=prog_args.output_directory
        )
        merge_arguments.validate()

    return merge_arguments


def build_and_validate_tune_args(merge_only: bool, prog_args, do_eval: bool, fp32_cpu_offload: bool, no_checkpoint: bool, save_embeddings: bool, use_4bit: bool, use_8bit: bool,
                                 use_bf_16: bool, use_fp_16: bool, use_tf_32: bool):
    tune_arguments = TuneArguments(
        new_model=prog_args.new_model,
        training_data_dir=prog_args.training_data_dir,
        train_file=prog_args.training_data_file
    )
    if not merge_only:
        tune_arguments = TuneArguments(
            base_model=prog_args.base_model,
            new_model=prog_args.new_model,
            training_data_dir=prog_args.training_data_dir,
            train_file=prog_args.training_data_file,
            r=prog_args.lora_r,
            alpha=prog_args.lora_alpha,
            epochs=prog_args.epochs,
            batch_size=prog_args.batch_size,
            use_fp_16=use_fp_16,
            use_bf_16=use_bf_16,
            learning_rate_base=prog_args.learning_rate_base,
            lora_dropout=prog_args.lora_dropout,
            no_checkpoint=no_checkpoint,
            bias=prog_args.bias,
            optimizer_type=prog_args.optimizer_type,
            gradient_accumulation_steps=prog_args.gradient_accumulation_steps,
            weight_decay=prog_args.weight_decay,
            max_gradient_norm=prog_args.max_gradient_norm,
            use_tf_32=use_tf_32,
            save_strategy=prog_args.save_strategy,
            save_steps=prog_args.save_steps,
            do_eval=do_eval,
            max_checkpoints=prog_args.max_saved,
            use_8bit=use_8bit,
            use_4bit=use_4bit,
            save_embeddings=save_embeddings,
            output_directory=prog_args.output_directory,
            fp32_cpu_offload=fp32_cpu_offload
        )
        tune_arguments.validate()

    return tune_arguments
