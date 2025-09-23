import json
from jsonschema import validate, ValidationError

def load_json_config(file_path):
    """Load and parse JSON configuration file."""
    with open(file_path, 'r') as file:
        config_data = json.load(file)
    return config_data

def validate_json_config(config_data, schema):
    """Validate JSON configuration data against a schema."""
    try:
        validate(instance=config_data, schema=schema)
    except ValidationError as ve:
        raise ValueError(f"Invalid JSON configuration: {ve}")

# Example JSON schema for validation
json_config_schema = {
    "type": "object",
    "properties": {
        "base_model": {"type": "string"},
        "new_model": {"type": "string"},
        "training_data_dir": {"type": "string"},
        "training_data_file": {"type": "string"},
        "epochs": {"type": "integer"},
        "batch_size": {"type": "integer"},
        "lora_r": {"type": "integer"},
        "lora_alpha": {"type": "integer"},
        "use_4bit": {"type": "boolean"},
        "use_8bit": {"type": "boolean"},
        "use_fp_16": {"type": "boolean"},
        "use_bf_16": {"type": "boolean"},
        "use_tf_32": {"type": "boolean"},
        "fp32_cpu_offload": {"type": "boolean"},
        "learning_rate": {"type": "number"},
        "gradient_accumulation_steps": {"type": ["integer", "null"]},
        "weight_decay": {"type": "number"},
        "max_gradient_norm": {"type": "number"},
        "target_modules": {"type": ["array", "null"], "items": {"type": "string"}},
        "torch_empty_cache_steps": {"type": ["integer", "null"]},
        "cpu_only_tuning": {"type": "boolean"},
        "neftune_noise_alpha": {"type": ["number", "null"]},
        "hf_training_dataset_id": {"type": ["string", "null"]}
    },
    "required": ["base_model", "new_model"],
    "additionalProperties": False
}