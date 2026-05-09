import unittest
from src.main.exception.exceptions import (
    TunerException,
    ServeModeException,
    LlmServerException,
    TuningModuleFunctionException,
    ValidationException,
    ArgumentValidationException,
    HuggingfaceException,
    HuggingfaceAuthException,
)


class TestTunerException(unittest.TestCase):
    def test_basic_exception_creation(self):
        exc = TunerException("Test message")
        self.assertEqual(exc.message, "Test message")
        self.assertEqual(exc.exception_type, "GENERIC")
        self.assertIsNone(exc.sub_type)

    def test_exception_with_type(self):
        exc = TunerException("Test message", exception_type="CUSTOM")
        self.assertEqual(exc.exception_type, "CUSTOM")

    def test_exception_with_sub_type(self):
        exc = TunerException("Test message", sub_type="CUSTOM_SUB")
        self.assertEqual(exc.sub_type, "CUSTOM_SUB")

    def test_to_string_without_sub_type(self):
        exc = TunerException("Test message", exception_type="CUSTOM")
        result = exc.to_string()
        self.assertIn("Test message", result)
        self.assertIn("TYPE: CUSTOM", result)
        self.assertNotIn("SUB TYPE", result)

    def test_to_string_with_sub_type(self):
        exc = TunerException("Test message", exception_type="CUSTOM", sub_type="CUSTOM_SUB")
        result = exc.to_string()
        self.assertIn("Test message", result)
        self.assertIn("TYPE: CUSTOM", result)
        self.assertIn("SUB TYPE: CUSTOM_SUB", result)


class TestServeModeException(unittest.TestCase):
    def test_serve_mode_exception_creation(self):
        exc = ServeModeException("Serve error")
        self.assertEqual(exc.message, "Serve error")
        self.assertEqual(exc.exception_type, "SERVE_MODE")

    def test_serve_mode_exception_with_sub_type(self):
        exc = ServeModeException("Serve error", sub_type="CUSTOM")
        self.assertEqual(exc.sub_type, "CUSTOM")


class TestLlmServerException(unittest.TestCase):
    def test_llm_server_exception_creation(self):
        exc = LlmServerException("Server error")
        self.assertEqual(exc.message, "Server error")
        self.assertEqual(exc.exception_type, "SERVE_MODE")
        self.assertEqual(exc.sub_type, "LLM_SERVER")


class TestTuningModuleFunctionException(unittest.TestCase):
    def test_tuning_module_function_exception_creation(self):
        exc = TuningModuleFunctionException("Tuning error")
        self.assertEqual(exc.message, "Tuning error")
        self.assertEqual(exc.exception_type, "TUNING_FUNCTION")

    def test_tuning_module_function_exception_with_sub_type(self):
        exc = TuningModuleFunctionException("Tuning error", sub_type="CUSTOM")
        self.assertEqual(exc.sub_type, "CUSTOM")


class TestValidationException(unittest.TestCase):
    def test_validation_exception_creation(self):
        exc = ValidationException("Validation error")
        self.assertEqual(exc.message, "Validation error")
        self.assertEqual(exc.exception_type, "VALIDATION")

    def test_validation_exception_with_sub_type(self):
        exc = ValidationException("Validation error", sub_type="CUSTOM")
        self.assertEqual(exc.sub_type, "CUSTOM")


class TestArgumentValidationException(unittest.TestCase):
    def test_argument_validation_exception_creation(self):
        exc = ArgumentValidationException("Invalid argument")
        self.assertEqual(exc.message, "Invalid argument")
        self.assertEqual(exc.exception_type, "VALIDATION")
        self.assertEqual(exc.sub_type, "ARGUMENT_VALIDATION")


class TestHuggingfaceException(unittest.TestCase):
    def test_huggingface_exception_creation(self):
        exc = HuggingfaceException("HF error")
        self.assertEqual(exc.message, "HF error")
        self.assertEqual(exc.exception_type, "HUGGINGFACE")

    def test_huggingface_exception_with_sub_type(self):
        exc = HuggingfaceException("HF error", sub_type="CUSTOM")
        self.assertEqual(exc.sub_type, "CUSTOM")


class TestHuggingfaceAuthException(unittest.TestCase):
    def test_huggingface_auth_exception_creation(self):
        exc = HuggingfaceAuthException("Auth failed")
        self.assertEqual(exc.message, "Auth failed")
        self.assertEqual(exc.exception_type, "HUGGINGFACE")
        self.assertEqual(exc.sub_type, "HUGGINGFACE_AUTH")


if __name__ == '__main__':
    unittest.main()
