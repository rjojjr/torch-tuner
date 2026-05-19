from arguments.arguments import TunerFunctionArguments, LlmExecutorFactoryArguments
import importlib.util
import torch

# `BitsAndBytesConfig` is exported by transformers even when the `bitsandbytes`
# package itself is not installed (e.g., on macOS, where our installer strips
# bnb from the requirements). Detect the actual package, not the config class.
_bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # type: ignore

# Optimizers in transformers' OptimizerNames that require the `bitsandbytes`
# package. Selecting any of these without bnb installed raises:
#   "You need to install `bitsandbytes` in order to use bitsandbytes optimizers"
_BNB_OPTIMIZERS = frozenset({
    "adamw_bnb_8bit",
    "adamw_8bit",  # legacy alias
    "paged_adamw_8bit",
    "paged_adamw_32bit",
    "lion_8bit",
    "paged_lion_8bit",
    "paged_lion_32bit",
    "rmsprop_bnb",
    "rmsprop_bnb_8bit",
    "rmsprop_bnb_32bit",
    "ademamix_8bit",
    "paged_ademamix_8bit",
    "paged_ademamix_32bit",
})


def resolve_optim(optimizer_type: str) -> str:
    """Coerce the requested optimizer to one that actually works on this host.

    - On MPS, `adamw_torch_fused` is not supported -> `adamw_torch`.
    - When `bitsandbytes` is not installed (e.g. Apple Silicon, where our
      installer drops the dep), any bnb-backed optimizer -> `adamw_torch`.
    Prints a one-line warning on fallback so the user understands the swap.
    """
    if torch.backends.mps.is_available() and optimizer_type == "adamw_torch_fused":
        print(
            "WARNING - `adamw_torch_fused` is not supported on Apple Silicon (MPS); "
            "falling back to `adamw_torch`."
        )
        return "adamw_torch"
    if optimizer_type in _BNB_OPTIMIZERS and not _bitsandbytes_available:
        print(
            f"WARNING - optimizer `{optimizer_type}` requires the `bitsandbytes` "
            f"package, which is not installed (bitsandbytes has no Apple Silicon "
            f"build); falling back to `adamw_torch`."
        )
        return "adamw_torch"
    return optimizer_type


def get_dtype(arguments: TunerFunctionArguments | LlmExecutorFactoryArguments) -> torch.dtype:
    """Get configured torch data type."""
    dtype = torch.float32
    if arguments.is_fp16:
        dtype = torch.float16
    elif arguments.is_bf16:
        dtype = torch.bfloat16

    return dtype


def get_bnb_config_and_dtype(arguments: TunerFunctionArguments | LlmExecutorFactoryArguments) -> tuple[object | None, torch.dtype]:
    """Construct configured BitsAndBytesConfig, or None on MPS/CPU."""
    if torch.backends.mps.is_available():
        return None, torch.float16

    dtype = get_dtype(arguments)
    bnb_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload,
        bnb_4bit_compute_dtype=dtype
    )
    if arguments.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload,
            bnb_4bit_compute_dtype=dtype
        )
    elif arguments.use_4bit and isinstance(arguments, TunerFunctionArguments):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload,
        )
    elif arguments.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=arguments.fp32_cpu_offload,
        )
    return bnb_config, dtype
