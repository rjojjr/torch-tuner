# Unit Tests for Torch Tuner

This directory contains unit tests for the torch-tuner project, organized to mirror the `./src/main` package structure.

## Directory Structure

```
_test/
├── main/
│   ├── arguments/
│   │   └── __init__.py
│   ├── exception/
│   │   ├── __init__.py
│   │   └── test_exceptions.py
│   ├── serve/
│   │   ├── __init__.py
│   │   └── test_atomic_integer.py
│   └── utils/
│       ├── __init__.py
│       ├── test_cli_commands.py
│       └── test_time_utils.py
├── conftest.py
└── README.md
```

## Running Tests

### Run all tests:
```bash
python3 -m unittest discover -s _test -p "test_*.py" -v
```

### Run tests in a specific package:
```bash
python3 -m unittest discover -s _test/main/exception -p "test_*.py" -v
python3 -m unittest discover -s _test/main/serve -p "test_*.py" -v
python3 -m unittest discover -s _test/main/utils -p "test_*.py" -v
```

### Run specific test file:
```bash
python3 -m unittest _test.main.exception.test_exceptions -v
```

### Run specific test class:
```bash
python3 -m unittest _test.main.exception.test_exceptions.TestTunerException -v
```

### Run specific test method:
```bash
python3 -m unittest _test.main.exception.test_exceptions.TestTunerException.test_basic_exception_creation -v
```

## Test Coverage

### Exception Tests (`main/exception/`)
- `test_exceptions.py` - Tests for exception classes and hierarchy (16 tests)
  - TunerException base class and subclasses
  - Exception type and sub-type handling
  - Exception string formatting

### Serve Tests (`main/serve/`)
- `test_atomic_integer.py` - Tests for atomic integer operations (12 tests)
  - Thread-safe increment/decrement operations
  - Type coercion and initialization
  - Value property getter/setter

### Utils Tests (`main/utils/`)
- `test_time_utils.py` - Tests for time utility functions (5 tests)
  - Current millisecond time retrieval
  - Time monotonicity and accuracy
  
- `test_cli_commands.py` - Tests for training CLI command validation (34 tests)
  - Required arguments (--new-model)
  - LoRA parameters validation (R, alpha)
  - Epochs validation for fine-tuning
  - Operation flags (fine-tune, merge, push)
  - Quantization options (4bit, 8bit, fp16, bf16)
  - Padding side validation
  - Batch size and learning rate validation
  - Training data file formats (.txt, .jsonl)
  - HuggingFace dataset support
  - Flash attention and target modules
  - Debug mode and output settings

- `test_training_integration.py` - Tests for CLI command behavior and function call layers
  - Fine-tune command execution and isolation
  - Merge command execution and isolation
  - Push command execution and isolation
  - Tuner factory with different LLM types
  - Argument passing through layers

- `test_dataset_utils.py` - Tests for dataset parsing logic (16 tests)
  - Prompt/completion JSONL loading
  - OpenAI chat-format JSONL loading (`{"messages": [...]}`)
  - Non-JSONL training files (CSV)
  - Hugging Face training-dataset-id routing (mocked)
  - Eval routing: train-as-eval fallback, eval JSONL/CSV, eval-only mode, HF eval id
  - Missing eval-dataset raises `ArgumentValidationException`
  - Chat-format detection contract (what `llm_base_module` relies on)

## Adding New Tests

1. Create a new file named `test_*.py` in the appropriate subdirectory under `_test/main/`
2. Import the module you want to test from `src.main`
3. Create test classes inheriting from `unittest.TestCase`
4. Write test methods prefixed with `test_`
5. Run tests to verify they pass

Example:
```python
# _test/main/arguments/test_arguments.py
import unittest
from src.main.arguments.arguments import PushArguments

class TestPushArguments(unittest.TestCase):
    def test_push_arguments_creation(self):
        args = PushArguments(new_model="test-model", model_dir="/tmp")
        self.assertEqual(args.new_model, "test-model")
```
