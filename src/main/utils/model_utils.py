import torch
from transformers import Conv1D


def get_all_layers(model):
    layers = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            module_name = '.'.join(name.split('.')[4:]).split('.')[0]
            if module_name.strip() != '':
                layers.append(module_name)

    return list(set(layers))


def get_all_linear_layers(model):
    layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = '.'.join(name.split('.')[4:]).split('.')[0]
            if module_name.strip() != '':
                layers.append(module_name)

    return list(set(layers))