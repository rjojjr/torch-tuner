from arguments.arguments import TunerFunctionArguments, LlmExecutorFactoryArguments
import gc
import importlib.util
import torch


def release_memory() -> None:
    """Force a full GC pass and drain the active device's caching allocator.

    Between tune -> merge -> push phases, each phase loads a fresh copy of the
    base / merged model. Python's cyclic GC is lazy and transformers models
    carry cycles, so unreferenced models linger; PyTorch's MPS / CUDA caching
    allocators then hold those blocks. On unified-memory Macs this stacks
    quickly and OOMs the next phase. Run explicitly between phases.
    """
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# `BitsAndBytesConfig` is exported by transformers even when the `bitsandbytes`
# package itself is not installed (e.g., on macOS, where our installer strips
# bnb from the requirements). Detect the actual package, not the config class.
# Guard object for the Apple Silicon warning printed once on first MPS detection.
_APPLE_SILICON_WARNING_SHOWN = type("_Guard", (), {"val": False})()
_bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
# `optimum-quanto` is our MPS-compatible 4/8-bit quantizer (bitsandbytes has no
# Apple Silicon build, and transformers 5.x has gutted its HQQ integration).
# `find_spec` raises ModuleNotFoundError when the parent package `optimum` is
# itself missing, so guard the lookup.
try:
    _optimum_quanto_available = importlib.util.find_spec("optimum.quanto") is not None
except ModuleNotFoundError:
    _optimum_quanto_available = False
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # type: ignore
try:
    from transformers import QuantoConfig
except ImportError:
    QuantoConfig = None  # type: ignore

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
    """Construct quantization config + compute dtype for the current host.

    - CUDA: returns BitsAndBytesConfig per --use-4bit/--use-8bit.
    - MPS: returns QuantoConfig per --use-4bit/--use-8bit (bitsandbytes has no
      Apple Silicon build; transformers 5.x disabled its HQQ integration;
      optimum-quanto is the working HF-integrated quantizer on MPS).
    - Neither flag set, or optimum-quanto not installed: returns (None, dtype).
    """
    if torch.backends.mps.is_available() and not getattr(_APPLE_SILICON_WARNING_SHOWN, 'val', False):
        _APPLE_SILICON_WARNING_SHOWN.val = True
        print('\x1b[33mWARNING - Support for Apple Silicon is currently EXPERIMENTAL!\x1b[0m')
        # Respect the user's explicit dtype choice; otherwise default to fp16
        # on MPS, which has broader kernel coverage than fp32 / bf16 fallbacks.
        if arguments.is_bf16:
            dtype = torch.bfloat16
        elif arguments.is_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float16
        if arguments.use_8bit or arguments.use_4bit:
            if QuantoConfig is None or not _optimum_quanto_available:
                print(
                    "WARNING - --use-4bit/--use-8bit was requested on Apple Silicon "
                    "but the `optimum-quanto` package is not installed. Re-run the "
                    "installer or `pip install optimum-quanto` inside the torch-tuner "
                    "venv. Falling back to unquantized weights."
                )
                return None, dtype
            weights = "int8" if arguments.use_8bit else "int4"
            return QuantoConfig(weights=weights), dtype
        return None, dtype

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
