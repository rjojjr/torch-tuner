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
                 tf_32: bool = False,
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
        self.use_fp_16 = use_fp_16
        self.use_bf_16 = use_bf_16
        self.learning_rate_base = learning_rate_base
        self.lora_dropout = lora_dropout
        self.no_checkpoint = no_checkpoint
        self.bias = bias
        self.optimizer_type = optimizer_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_gradient_norm = max_gradient_norm
        self.tf_32 = tf_32
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.do_eval = do_eval
        self.max_checkpoints = max_checkpoints
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.save_embeddings = save_embeddings
        self.output_directory = output_directory
        self.fp32_cpu_offload = fp32_cpu_offload


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