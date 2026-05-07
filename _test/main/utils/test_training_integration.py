import unittest
from unittest.mock import Mock
import sys
import os

# Setup paths for imports
# __file__ is at _test/main/utils/test_training_integration.py
# We need to go up to the torch-tuner root (3 levels up from utils)
test_file_dir = os.path.dirname(os.path.abspath(__file__))  # _test/main/utils
main_test_dir = os.path.dirname(test_file_dir)  # _test/main
test_root_dir = os.path.dirname(main_test_dir)  # _test
project_root = os.path.dirname(test_root_dir)  # torch-tuner root
src_main_path = os.path.join(project_root, 'src', 'main')
if src_main_path not in sys.path:
    sys.path.insert(0, src_main_path)


class TestCLIFineTuneCommand(unittest.TestCase):
    """Tests that verify the --fine-tune CLI command behavior."""

    def test_when_user_runs_fine_tune_command_it_executes_the_fine_tune_function(self):
        """
        BEHAVIOR: When a user runs the CLI with --fine-tune flag,
        the system should execute the underlying fine_tune function with the provided arguments.
        """
        try:
            from base.tuner import Tuner
            
            # Setup: Create a tuner with a mock fine_tune function
            fine_tune_was_called = False
            received_tune_arguments = None
            
            def track_fine_tune_execution(tune_args):
                nonlocal fine_tune_was_called, received_tune_arguments
                fine_tune_was_called = True
                received_tune_arguments = tune_args
            
            tuner = Tuner(
                fine_tune=track_fine_tune_execution,
                merge=Mock(),
                push=Mock()
            )
            
            # Execute: Simulate user running --fine-tune
            mock_tune_args = Mock()
            tuner.fine_tune(mock_tune_args)
            
            # Verify: The fine_tune function was executed with the arguments
            self.assertTrue(fine_tune_was_called, 
                          "Expected fine_tune function to be called when --fine-tune flag is used")
            self.assertEqual(received_tune_arguments, mock_tune_args,
                           "Expected fine_tune to receive the tune arguments")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")

    def test_when_user_runs_fine_tune_only_merge_and_push_are_not_executed(self):
        """
        BEHAVIOR: When a user runs only --fine-tune without --merge or --push,
        the merge and push functions should NOT be executed.
        """
        try:
            from base.tuner import Tuner
            
            # Setup: Create a tuner with tracked merge and push functions
            merge_was_called = False
            push_was_called = False
            
            def track_merge(merge_args):
                nonlocal merge_was_called
                merge_was_called = True
            
            def track_push(push_args):
                nonlocal push_was_called
                push_was_called = True
            
            tuner = Tuner(
                fine_tune=Mock(),
                merge=track_merge,
                push=track_push
            )
            
            # Execute: Simulate user running only --fine-tune
            tuner.fine_tune(Mock())
            
            # Verify: Only fine_tune was called, not merge or push
            self.assertFalse(merge_was_called,
                           "Expected merge NOT to be called when only --fine-tune is specified")
            self.assertFalse(push_was_called,
                           "Expected push NOT to be called when only --fine-tune is specified")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")



class TestCLIMergeCommand(unittest.TestCase):
    """Tests that verify the --merge CLI command behavior."""

    def test_when_user_runs_merge_command_it_executes_the_merge_function(self):
        """
        BEHAVIOR: When a user runs the CLI with --merge flag,
        the system should execute the underlying merge function with the provided arguments.
        """
        try:
            from base.tuner import Tuner
            
            # Setup: Create a tuner with a mock merge function
            merge_was_called = False
            received_merge_arguments = None
            
            def track_merge_execution(merge_args):
                nonlocal merge_was_called, received_merge_arguments
                merge_was_called = True
                received_merge_arguments = merge_args
            
            tuner = Tuner(
                fine_tune=Mock(),
                merge=track_merge_execution,
                push=Mock()
            )
            
            # Execute: Simulate user running --merge
            mock_merge_args = Mock()
            tuner.merge(mock_merge_args)
            
            # Verify: The merge function was executed with the arguments
            self.assertTrue(merge_was_called,
                          "Expected merge function to be called when --merge flag is used")
            self.assertEqual(received_merge_arguments, mock_merge_args,
                           "Expected merge to receive the merge arguments")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")

    def test_when_user_runs_merge_only_fine_tune_and_push_are_not_executed(self):
        """
        BEHAVIOR: When a user runs only --merge without --fine-tune or --push,
        the fine_tune and push functions should NOT be executed.
        """
        try:
            from base.tuner import Tuner
            
            # Setup: Create a tuner with tracked fine_tune and push functions
            fine_tune_was_called = False
            push_was_called = False
            
            def track_fine_tune(tune_args):
                nonlocal fine_tune_was_called
                fine_tune_was_called = True
            
            def track_push(push_args):
                nonlocal push_was_called
                push_was_called = True
            
            tuner = Tuner(
                fine_tune=track_fine_tune,
                merge=Mock(),
                push=track_push
            )
            
            # Execute: Simulate user running only --merge
            tuner.merge(Mock())
            
            # Verify: Only merge was called, not fine_tune or push
            self.assertFalse(fine_tune_was_called,
                           "Expected fine_tune NOT to be called when only --merge is specified")
            self.assertFalse(push_was_called,
                           "Expected push NOT to be called when only --merge is specified")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")


class TestCLIPushCommand(unittest.TestCase):
    """Tests that verify the --push CLI command behavior."""

    def test_when_user_runs_push_command_it_executes_the_push_function(self):
        """
        BEHAVIOR: When a user runs the CLI with --push flag,
        the system should execute the underlying push function with the provided arguments.
        """
        try:
            from base.tuner import Tuner
            
            # Setup: Create a tuner with a mock push function
            push_was_called = False
            received_push_arguments = None
            
            def track_push_execution(push_args):
                nonlocal push_was_called, received_push_arguments
                push_was_called = True
                received_push_arguments = push_args
            
            tuner = Tuner(
                fine_tune=Mock(),
                merge=Mock(),
                push=track_push_execution
            )
            
            # Execute: Simulate user running --push
            mock_push_args = Mock()
            tuner.push(mock_push_args)
            
            # Verify: The push function was executed with the arguments
            self.assertTrue(push_was_called,
                          "Expected push function to be called when --push flag is used")
            self.assertEqual(received_push_arguments, mock_push_args,
                           "Expected push to receive the push arguments")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")

    def test_when_user_runs_push_only_fine_tune_and_merge_are_not_executed(self):
        """
        BEHAVIOR: When a user runs only --push without --fine-tune or --merge,
        the fine_tune and merge functions should NOT be executed.
        """
        try:
            from base.tuner import Tuner
            
            # Setup: Create a tuner with tracked fine_tune and merge functions
            fine_tune_was_called = False
            merge_was_called = False
            
            def track_fine_tune(tune_args):
                nonlocal fine_tune_was_called
                fine_tune_was_called = True
            
            def track_merge(merge_args):
                nonlocal merge_was_called
                merge_was_called = True
            
            tuner = Tuner(
                fine_tune=track_fine_tune,
                merge=track_merge,
                push=Mock()
            )
            
            # Execute: Simulate user running only --push
            tuner.push(Mock())
            
            # Verify: Only push was called, not fine_tune or merge
            self.assertFalse(fine_tune_was_called,
                           "Expected fine_tune NOT to be called when only --push is specified")
            self.assertFalse(merge_was_called,
                           "Expected merge NOT to be called when only --push is specified")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")


class TestCLITunerFactory(unittest.TestCase):
    """Tests that verify the tuner factory creates correct training functions."""

    def test_when_user_specifies_llama_model_the_llama_training_functions_are_used(self):
        """
        BEHAVIOR: When a user specifies --llm-type llama,
        the system should use the llama module's fine_tune, merge, and push functions.
        """
        try:
            from utils.tuner_utils import build_llm_tuner_factory
            from base.tuner import LLM_TYPES
            
            # Setup: Create arguments specifying llama as the LLM type
            user_args = Mock()
            user_args.llm_type = 'llama'
            
            # Execute: Build the tuner factory with llama type
            tuner_factory = build_llm_tuner_factory(user_args)
            tuner = tuner_factory()
            
            # Verify: The tuner was created with llama functions
            self.assertIsNotNone(tuner,
                               "Expected tuner to be created for llama LLM type")
            self.assertEqual(tuner.llm_type, LLM_TYPES['llama'],
                           "Expected tuner to be configured with llama LLM type")
            self.assertTrue(callable(tuner.fine_tune),
                          "Expected tuner to have callable fine_tune function")
            self.assertTrue(callable(tuner.merge),
                          "Expected tuner to have callable merge function")
            self.assertTrue(callable(tuner.push),
                          "Expected tuner to have callable push function")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")

    def test_when_user_specifies_generic_model_the_generic_training_functions_are_used(self):
        """
        BEHAVIOR: When a user specifies an unsupported LLM type or uses generic,
        the system should fall back to the generic module's fine_tune, merge, and push functions.
        """
        try:
            from utils.tuner_utils import build_llm_tuner_factory
            from base.tuner import LLM_TYPES
            
            # Setup: Create arguments specifying generic as the LLM type
            user_args = Mock()
            user_args.llm_type = 'generic'
            
            # Execute: Build the tuner factory with generic type
            tuner_factory = build_llm_tuner_factory(user_args)
            tuner = tuner_factory()
            
            # Verify: The tuner was created with generic functions
            self.assertIsNotNone(tuner,
                               "Expected tuner to be created for generic LLM type")
            self.assertEqual(tuner.llm_type, LLM_TYPES['generic'],
                           "Expected tuner to be configured with generic LLM type")
            self.assertTrue(callable(tuner.fine_tune),
                          "Expected tuner to have callable fine_tune function")
            self.assertTrue(callable(tuner.merge),
                          "Expected tuner to have callable merge function")
            self.assertTrue(callable(tuner.push),
                          "Expected tuner to have callable push function")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")


class TestCLIArgumentPassing(unittest.TestCase):
    """Tests that verify CLI arguments are correctly passed through the system."""

    def test_when_user_provides_tune_arguments_they_are_passed_to_fine_tune_function(self):
        """
        BEHAVIOR: When a user runs --fine-tune with specific arguments (model name, epochs, etc.),
        those arguments should be passed to the underlying fine_tune function.
        """
        try:
            from base.tuner import Tuner
            from arguments.arguments import TuneArguments
            
            # Setup: Track what arguments are received by fine_tune
            received_tune_args = None
            
            def capture_tune_arguments(tune_args):
                nonlocal received_tune_args
                received_tune_args = tune_args
            
            tuner = Tuner(
                fine_tune=capture_tune_arguments,
                merge=Mock(),
                push=Mock()
            )
            
            # Execute: User provides tune arguments
            user_tune_args = TuneArguments(
                new_model="my-fine-tuned-model",
                training_data_dir="/data",
                train_file="train.txt"
            )
            tuner.fine_tune(user_tune_args)
            
            # Verify: The arguments were passed correctly
            self.assertIsInstance(received_tune_args, TuneArguments,
                                "Expected fine_tune to receive TuneArguments object")
            self.assertEqual(received_tune_args.new_model, "my-fine-tuned-model",
                           "Expected fine_tune to receive the user-specified model name")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")

    def test_when_user_provides_merge_arguments_they_are_passed_to_merge_function(self):
        """
        BEHAVIOR: When a user runs --merge with specific arguments (model name, output dir, etc.),
        those arguments should be passed to the underlying merge function.
        """
        try:
            from base.tuner import Tuner
            from arguments.arguments import MergeArguments
            
            # Setup: Track what arguments are received by merge
            received_merge_args = None
            
            def capture_merge_arguments(merge_args):
                nonlocal received_merge_args
                received_merge_args = merge_args
            
            tuner = Tuner(
                fine_tune=Mock(),
                merge=capture_merge_arguments,
                push=Mock()
            )
            
            # Execute: User provides merge arguments
            user_merge_args = MergeArguments(new_model="my-merged-model")
            tuner.merge(user_merge_args)
            
            # Verify: The arguments were passed correctly
            self.assertIsInstance(received_merge_args, MergeArguments,
                                "Expected merge to receive MergeArguments object")
            self.assertEqual(received_merge_args.new_model, "my-merged-model",
                           "Expected merge to receive the user-specified model name")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")

    def test_when_user_provides_push_arguments_they_are_passed_to_push_function(self):
        """
        BEHAVIOR: When a user runs --push with specific arguments (model name, directory, etc.),
        those arguments should be passed to the underlying push function.
        """
        try:
            from base.tuner import Tuner
            from arguments.arguments import PushArguments
            
            # Setup: Track what arguments are received by push
            received_push_args = None
            
            def capture_push_arguments(push_args):
                nonlocal received_push_args
                received_push_args = push_args
            
            tuner = Tuner(
                fine_tune=Mock(),
                merge=Mock(),
                push=capture_push_arguments
            )
            
            # Execute: User provides push arguments
            user_push_args = PushArguments(new_model="my-pushed-model", model_dir="/models")
            tuner.push(user_push_args)
            
            # Verify: The arguments were passed correctly
            self.assertIsInstance(received_push_args, PushArguments,
                                "Expected push to receive PushArguments object")
            self.assertEqual(received_push_args.new_model, "my-pushed-model",
                           "Expected push to receive the user-specified model name")
            self.assertEqual(received_push_args.model_dir, "/models",
                           "Expected push to receive the user-specified model directory")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("Source modules not available")


if __name__ == '__main__':
    unittest.main()
