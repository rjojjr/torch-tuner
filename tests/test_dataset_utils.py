import unittest
from datasets import DatasetDict, Dataset
from arguments.arguments import TuneArguments
from utils.dataset_utils import load_dataset
class TestDatasetUtils(unittest.TestCase):
    def setUp(self):
        self.prompt_completion_file = "/code/torch-tuner/tests/sample_train_prompt_completion.jsonl"
        self.messages_file = "/code/torch-tuner/tests/sample_train_messages.jsonl"

        self.args_prompt_completion = TuneArguments(
            do_train=True,
            train_file=self.prompt_completion_file,
            training_data_dir="/code/torch-tuner/tests",
            eval_dataset=None
        )
        
        self.args_messages = TuneArguments(
            do_train=True,
            train_file=self.messages_file,
            training_data_dir="/code/torch-tuner/tests",
            eval_dataset=None
        )

    def test_load_prompt_completion_format(self):
        dataset = load_dataset(self.args_prompt_completion)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertEqual(len(dataset['train']), 2)
        
        # Check first entry
        self.assertIn('prompt', dataset['train'][0])
        self.assertIn('completion', dataset['train'][0])
        self.assertEqual(dataset['train'][0]['prompt'], "Hello, how are you?")
        self.assertEqual(dataset['train'][0]['completion'], "I'm fine, thank you!")
        
        # Check second entry
        self.assertIn('prompt', dataset['train'][1])
        self.assertIn('completion', dataset['train'][1])
        self.assertEqual(dataset['train'][1]['prompt'], "What is your name?")
        self.assertEqual(dataset['train'][1]['completion'], "My name is Assistant.")

    def test_load_messages_format(self):
        dataset = load_dataset(self.args_messages)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertEqual(len(dataset['train']), 2)
        
        # Check first entry
        self.assertIn('prompt', dataset['train'][0])
        self.assertIn('completion', dataset['train'][0])
        self.assertEqual(dataset['train'][0]['prompt'], "System: You are a helpful assistant.\nUser: Hello, how are you?")
        self.assertEqual(dataset['train'][0]['completion'], "I'm fine, thank you!")
        
        # Check second entry
        self.assertIn('prompt', dataset['train'][1])
        self.assertIn('completion', dataset['train'][1])
        self.assertEqual(dataset['train'][1]['prompt'], "System: You are a helpful assistant.\nUser: What is your name?")
        self.assertEqual(dataset['train'][1]['completion'], "My name is Assistant.")

if __name__ == "__main__":
    unittest.main()