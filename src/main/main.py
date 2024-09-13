from utils.argument_utils import parse_arguments
from exception.exceptions import main_exception_handler

from utils.execution_utils import execute_command
import os

# TODO - Automate this
version = '2.1.5'

title = f'Torch-Tuner CLI v{version}'
description = 'This app is a simple CLI to automate the Supervised Fine-Tuning(SFT)(and testing of) of AI Large Language Model(LLM)s with simple text and jsonl on Nvidia GPUs(and Intel/AMD CPUs) using LoRA, Torch and Transformers.'

args = parse_arguments(title, description)

# For better performance with less GPU memory
if args.use_low_gpu_memory:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,expandable_segments:True"


def main() -> None:
    print(title)
    print('---------------------------------------------')
    print(description)
    print('---------------------------------------------')
    print('Run with --help flag for a list of available arguments.')
    execute_command(args)
    print('')
    print('---------------------------------------------')
    print(f'{title} COMPLETED')


if __name__ == "__main__":
    main_exception_handler(main, title, args.debug)
