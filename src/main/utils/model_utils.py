import torch
from transformers import Conv1D
from utils.tokenizer_utils import add_agent_tokens, add_additional_tokens

from arguments.arguments import TuneArguments, MergeArguments
from trl import setup_chat_format

all_modules = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)


def get_all_layers(model):
    """Get all available model layer names."""
    return _get_layer_names(model, False)


def get_all_linear_layers(model):
    """Get all available linear model layer names."""
    return _get_layer_names(model, True)


def _get_layer_names(model, is_linear_only: bool = False):
    layers = []

    for name, module in model.named_modules():
        if isinstance(module, all_modules if not is_linear_only else torch.nn.Linear):
            module_name = '.'.join(name.split('.')[4:]).split('.')[0]
            if module_name.strip() != '':
                layers.append(module_name)

    return list(set(layers))


def prepare_model_vocabulary(arguments: TuneArguments | MergeArguments, model, tokenizer):
    if arguments.additional_vocabulary_tokens is not None:
        add_additional_tokens(tokenizer, model, arguments.additional_vocabulary_tokens)
    if arguments.use_agent_tokens:
        add_agent_tokens(tokenizer, model)
    if arguments.is_chat_model or (arguments.train_file is not None and arguments.train_file.endswith(".jsonl")):
        model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer