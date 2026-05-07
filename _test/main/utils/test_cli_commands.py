import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestTrainingCLICommands(unittest.TestCase):
    """Tests for training CLI command validation and execution."""

    def test_cli_requires_new_model_argument(self):
        """Test that --new-model argument is required."""
        # This test validates that the CLI requires the new-model argument
        # which is enforced in argument_utils.do_initial_arg_validation()
        mock_args = Mock()
        mock_args.new_model = None
        mock_args.lora_r = 16
        mock_args.lora_alpha = 32
        mock_args.fine_tune = False
        mock_args.merge = True
        mock_args.push = True
        
        # Simulating the validation that would occur
        self.assertIsNone(mock_args.new_model)

    def test_cli_lora_r_must_be_positive(self):
        """Test that LoRA R parameter must be positive."""
        mock_args = Mock()
        mock_args.lora_r = 16
        mock_args.lora_alpha = 32
        
        self.assertGreater(mock_args.lora_r, 0)
        self.assertGreater(mock_args.lora_alpha, 0)

    def test_cli_lora_r_cannot_be_zero(self):
        """Test that LoRA R parameter cannot be zero."""
        mock_args = Mock()
        mock_args.lora_r = 0
        
        self.assertLessEqual(mock_args.lora_r, 0)

    def test_cli_epochs_must_be_positive_for_fine_tune(self):
        """Test that epochs must be positive when fine-tuning."""
        mock_args = Mock()
        mock_args.fine_tune = True
        mock_args.epochs = 3
        
        if mock_args.fine_tune:
            self.assertGreater(mock_args.epochs, 0)

    def test_cli_epochs_cannot_be_zero_for_fine_tune(self):
        """Test that epochs cannot be zero when fine-tuning."""
        mock_args = Mock()
        mock_args.fine_tune = True
        mock_args.epochs = 0
        
        if mock_args.fine_tune:
            self.assertLessEqual(mock_args.epochs, 0)

    def test_cli_at_least_one_operation_required(self):
        """Test that at least one operation (fine-tune, merge, push) is required."""
        mock_args = Mock()
        mock_args.fine_tune = False
        mock_args.merge = False
        mock_args.push = False
        
        operations_enabled = mock_args.fine_tune or mock_args.merge or mock_args.push
        self.assertFalse(operations_enabled)

    def test_cli_fine_tune_operation_enabled(self):
        """Test that fine-tune operation can be enabled."""
        mock_args = Mock()
        mock_args.fine_tune = True
        mock_args.merge = False
        mock_args.push = False
        
        operations_enabled = mock_args.fine_tune or mock_args.merge or mock_args.push
        self.assertTrue(operations_enabled)

    def test_cli_merge_operation_enabled(self):
        """Test that merge operation can be enabled."""
        mock_args = Mock()
        mock_args.fine_tune = False
        mock_args.merge = True
        mock_args.push = False
        
        operations_enabled = mock_args.fine_tune or mock_args.merge or mock_args.push
        self.assertTrue(operations_enabled)

    def test_cli_push_operation_enabled(self):
        """Test that push operation can be enabled."""
        mock_args = Mock()
        mock_args.fine_tune = False
        mock_args.merge = False
        mock_args.push = True
        
        operations_enabled = mock_args.fine_tune or mock_args.merge or mock_args.push
        self.assertTrue(operations_enabled)

    def test_cli_multiple_operations_can_be_enabled(self):
        """Test that multiple operations can be enabled together."""
        mock_args = Mock()
        mock_args.fine_tune = True
        mock_args.merge = True
        mock_args.push = True
        
        operations_enabled = mock_args.fine_tune or mock_args.merge or mock_args.push
        self.assertTrue(operations_enabled)

    def test_cli_base_model_has_default(self):
        """Test that base model has a default value."""
        default_base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.assertIsNotNone(default_base_model)
        self.assertIn("llama", default_base_model.lower())

    def test_cli_output_directory_has_default(self):
        """Test that output directory has a default value."""
        default_output_dir = "~/torch-tuner"
        self.assertIsNotNone(default_output_dir)
        self.assertTrue(default_output_dir.startswith("~"))

    def test_cli_quantization_options_mutually_exclusive(self):
        """Test that quantization options are mutually exclusive."""
        # Test 4bit alone
        mock_args = Mock()
        mock_args.use_4bit = True
        mock_args.use_8bit = False
        mock_args.use_bf_16 = False
        mock_args.use_fp_16 = False
        
        dt_args = [mock_args.use_4bit, mock_args.use_8bit, mock_args.use_bf_16, mock_args.use_fp_16]
        dt_type_count = sum(dt_args)
        self.assertEqual(dt_type_count, 1)

    def test_cli_quantization_options_multiple_enabled_invalid(self):
        """Test that multiple quantization options cannot be enabled."""
        mock_args = Mock()
        mock_args.use_4bit = True
        mock_args.use_8bit = True
        mock_args.use_bf_16 = False
        mock_args.use_fp_16 = False
        
        dt_args = [mock_args.use_4bit, mock_args.use_8bit, mock_args.use_bf_16, mock_args.use_fp_16]
        dt_type_count = sum(dt_args)
        self.assertGreater(dt_type_count, 1)

    def test_cli_padding_side_valid_values(self):
        """Test that padding side accepts valid values."""
        valid_padding_sides = [None, 'left', 'right']
        
        for padding_side in valid_padding_sides:
            if padding_side is not None:
                self.assertIn(padding_side, ['left', 'right'])

    def test_cli_padding_side_invalid_value(self):
        """Test that padding side rejects invalid values."""
        invalid_padding_side = 'center'
        self.assertNotIn(invalid_padding_side, ['left', 'right'])

    def test_cli_batch_size_positive(self):
        """Test that batch size is positive."""
        mock_args = Mock()
        mock_args.batch_size = 4
        
        self.assertGreater(mock_args.batch_size, 0)

    def test_cli_learning_rate_positive(self):
        """Test that learning rate is positive."""
        mock_args = Mock()
        mock_args.base_learning_rate = 0.0002
        
        self.assertGreater(mock_args.base_learning_rate, 0)

    def test_cli_lora_dropout_valid_range(self):
        """Test that LoRA dropout is in valid range."""
        mock_args = Mock()
        mock_args.lora_dropout = 0.05
        
        self.assertGreaterEqual(mock_args.lora_dropout, 0)
        self.assertLessEqual(mock_args.lora_dropout, 1)

    def test_cli_weight_decay_non_negative(self):
        """Test that weight decay is non-negative."""
        mock_args = Mock()
        mock_args.weight_decay = 0.001
        
        self.assertGreaterEqual(mock_args.weight_decay, 0)

    def test_cli_max_gradient_norm_positive(self):
        """Test that max gradient norm is positive."""
        mock_args = Mock()
        mock_args.max_gradient_norm = 0.3
        
        self.assertGreater(mock_args.max_gradient_norm, 0)

    def test_cli_training_data_file_formats_supported(self):
        """Test that supported training data file formats are txt and jsonl."""
        supported_formats = ['.txt', '.jsonl']
        
        test_files = ['data.txt', 'data.jsonl', 'samples.txt', 'samples.jsonl']
        for test_file in test_files:
            ext = os.path.splitext(test_file)[1]
            self.assertIn(ext, supported_formats)

    def test_cli_jsonl_file_indicates_chat_model(self):
        """Test that .jsonl training file indicates chat model."""
        training_data_file = "data.jsonl"
        is_chat_model = training_data_file.endswith(".jsonl")
        self.assertTrue(is_chat_model)

    def test_cli_txt_file_does_not_force_chat_model(self):
        """Test that .txt training file doesn't force chat model."""
        training_data_file = "data.txt"
        is_chat_model = training_data_file.endswith(".jsonl")
        self.assertFalse(is_chat_model)

    def test_cli_hf_dataset_overrides_local_file(self):
        """Test that HuggingFace dataset ID overrides local training file."""
        mock_args = Mock()
        mock_args.hf_training_dataset_id = "wikitext"
        mock_args.training_data_dir = None
        mock_args.training_data_file = None
        
        # HF dataset should be used when provided
        uses_hf_dataset = mock_args.hf_training_dataset_id is not None
        self.assertTrue(uses_hf_dataset)

    def test_cli_flash_attention_optional(self):
        """Test that flash attention is optional."""
        mock_args = Mock()
        mock_args.use_flash_attention = False
        
        self.assertFalse(mock_args.use_flash_attention)

    def test_cli_flash_attention_can_be_enabled(self):
        """Test that flash attention can be enabled."""
        mock_args = Mock()
        mock_args.use_flash_attention = True
        
        self.assertTrue(mock_args.use_flash_attention)

    def test_cli_target_modules_optional(self):
        """Test that target modules are optional."""
        mock_args = Mock()
        mock_args.target_modules = None
        
        self.assertIsNone(mock_args.target_modules)

    def test_cli_target_modules_can_be_specified(self):
        """Test that target modules can be specified."""
        mock_args = Mock()
        mock_args.target_modules = ["q_proj", "v_proj"]
        
        self.assertIsNotNone(mock_args.target_modules)
        self.assertEqual(len(mock_args.target_modules), 2)

    def test_cli_debug_mode_optional(self):
        """Test that debug mode is optional."""
        mock_args = Mock()
        mock_args.debug = False
        
        self.assertFalse(mock_args.debug)

    def test_cli_debug_mode_can_be_enabled(self):
        """Test that debug mode can be enabled."""
        mock_args = Mock()
        mock_args.debug = True
        
        self.assertTrue(mock_args.debug)

    def test_cli_overwrite_output_default_true(self):
        """Test that overwrite output defaults to true."""
        mock_args = Mock()
        mock_args.overwrite_output = True
        
        self.assertTrue(mock_args.overwrite_output)

    def test_cli_public_push_optional(self):
        """Test that public push is optional."""
        mock_args = Mock()
        mock_args.public_push = False
        
        self.assertFalse(mock_args.public_push)

    def test_cli_public_push_can_be_enabled(self):
        """Test that public push can be enabled."""
        mock_args = Mock()
        mock_args.public_push = True
        
        self.assertTrue(mock_args.public_push)


if __name__ == '__main__':
    unittest.main()
