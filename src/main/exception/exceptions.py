from typing import Callable


class TunerException(Exception):
    """Application base exception."""

    def __init__(self, message: str, exception_type: str = "GENERIC", sub_type: str | None = None):
        super(TunerException, self).__init__(message)
        self.message = message
        self.exception_type = exception_type
        self.sub_type = sub_type

    def to_string(self) -> str:
        """Return exception as a formatted string."""
        sub_type = f" SUB TYPE: {self.sub_type}" if self.sub_type is not None else ""
        return f'{self.message} TYPE: {self.exception_type}{sub_type}'


class ServeModeException(TunerException):
    """General serve mode exception"""
    def __init__(self, message: str, sub_type: str | None = None):
        super(ServeModeException, self).__init__(message, 'SERVE_MODE', sub_type)


class LlmServerException(ServeModeException):
    """LLM server exception"""
    def __init__(self, message: str):
        super(LlmServerException, self).__init__(message, 'LLM_SERVER')


class TuningModuleFunctionException(TunerException):
    """General tuning module function execution exception"""
    def __init__(self, message: str, sub_type: str | None = None):
        super(TuningModuleFunctionException, self).__init__(message, 'TUNING_FUNCTION', sub_type)


class ValidationException(TunerException):
    """Application validation exception."""

    def __init__(self, message: str, sub_type: str | None = None):
        super(ValidationException, self).__init__(message, 'VALIDATION', sub_type)


class ArgumentValidationException(ValidationException):
    """Application argument validation exception."""

    def __init__(self, message: str):
        super(ArgumentValidationException, self).__init__(message, 'ARGUMENT_VALIDATION')


class HuggingfaceException(TunerException):
    """Application Huggingface exception."""

    def __init__(self, message: str, sub_type: str | None = None):
        super(HuggingfaceException, self).__init__(message, 'HUGGINGFACE', sub_type)


class HuggingfaceAuthException(HuggingfaceException):
    """Application Huggingface auth exception."""

    def __init__(self, message: str):
        super(HuggingfaceAuthException, self).__init__(message, 'HUGGINGFACE_AUTH')


def main_exception_handler(main: Callable, title: str, is_debug: bool = False) -> None:
    """Handle exceptions raised in the main function."""
    try:
        main()
    except TunerException as e:
        print('')
        print(f"A TunerException has been caught: {e.to_string()}")
        if e.sub_type is not None and e.sub_type == 'ARGUMENT_VALIDATION':
            print('')
            print("Please verify that the provided program arguments are valid")
        if e.sub_type is not None and e.sub_type == 'HUGGINGFACE_AUTH':
            print('')
            print("Please verify that the `HUGGING_FACE_TOKEN` environment variable is set to a valid Huggingface auth token")
        print('')
        print(f"{title} is being terminated!")
        exit(100)
    except Exception as e:
        print('')
        print(f"An unexpected Exception has been caught: {str(e)}")
        print('')
        if is_debug:
            print(f"Rethrowing exception")
            raise e
        else:
            print(f"{title} is being terminated!")
            exit(101)


