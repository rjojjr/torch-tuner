import torch
from transformers import Conv1D

all_modules = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)


def get_all_layers(model):
    """Get all available model layer names."""
    return _get_layer_names(model, True)


def get_all_linear_layers(model):
    """Get all available linear model layer names."""
    return _get_layer_names(model)


def _get_layer_names(model, is_all: bool = False):
    layers = []

    for name, module in model.named_modules():
        if isinstance(module, all_modules if is_all else torch.nn.Linear):
            module_name = '.'.join(name.split('.')[4:]).split('.')[0]
            if module_name.strip() != '':
                layers.append(module_name)

    return list(set(layers))