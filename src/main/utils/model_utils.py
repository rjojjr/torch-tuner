import torch
from transformers import Conv1D
from utils.tokenizer_utils import add_agent_tokens, add_additional_tokens

from arguments.arguments import TuneArguments, MergeArguments

all_modules = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)

_CHATML_BOS = "<|im_start|>"
_CHATML_EOS = "<|im_end|>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
# Tool-aware ChatML template. Handles three cases the basic template does not:
#   1. assistant turns with `tool_calls` (LangGraph / OpenAI format) — each call
#      is wrapped in <tool_call>...</tool_call> as a JSON object.
#   2. role='tool' messages — rendered as a JSON {tool_call_id, content} blob.
#   3. assistant turns with empty content but tool_calls (the common pattern
#      where the model only emits tool calls).
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'tool' %}"
    "<|im_start|>tool\n"
    "{\"tool_call_id\": {{ message['tool_call_id'] | tojson }}, \"content\": {{ message['content'] | tojson }}}"
    "<|im_end|>\n"
    "{% else %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] %}{{ message['content'] }}{% endif %}"
    "{% if message.get('tool_calls') %}"
    "{% for tc in message['tool_calls'] %}"
    "{% if message['content'] or not loop.first %}\n{% endif %}"
    "<tool_call>\n"
    "{\"id\": {{ tc['id'] | tojson }}, \"name\": {{ tc['function']['name'] | tojson }}, \"arguments\": {{ tc['function']['arguments'] }}}"
    "\n</tool_call>"
    "{% endfor %}"
    "{% endif %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def setup_chat_format(model, tokenizer, resize_to_multiple_of: int | None = None):
    # Inlined replacement for trl.setup_chat_format (removed in trl 1.0+).
    # Extended to register tool-call delimiters as additional special tokens so
    # they tokenize cleanly when training on LangGraph / ReAct tool-using data.
    tokenizer.eos_token = _CHATML_EOS
    tokenizer.pad_token = _CHATML_EOS
    tokenizer.bos_token = _CHATML_BOS
    tokenizer.add_special_tokens({
        "additional_special_tokens": [_CHATML_BOS, _CHATML_EOS, _TOOL_CALL_OPEN, _TOOL_CALL_CLOSE]
    })
    tokenizer.chat_template = _CHATML_TEMPLATE

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=resize_to_multiple_of)

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


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