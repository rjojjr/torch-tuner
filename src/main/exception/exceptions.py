from typing import Callable


class TunerException(Exception):
    def __init__(self, message: str, exception_type: str = "GENERIC", sub_type: str | None = None):
        super(TunerException, self).__init__(message)
        self.exception_type = exception_type
        self.sub_type = sub_type


class ValidationException(TunerException):
    def __init__(self, message: str, sub_type: str | None = None):
        super(ValidationException, self).__init__(message, 'VALIDATION', sub_type)


class ArgumentValidationException(ValidationException):
    def __init__(self, message: str):
        super(ArgumentValidationException, self).__init__(message, 'ARGUMENT_VALIDATION')


class HuggingfaceException(TunerException):
    def __init__(self, message: str, sub_type: str | None = None):
        super(HuggingfaceException, self).__init__(message, 'HUGGINGFACE', sub_type)


class HuggingfaceAuthException(HuggingfaceException):
    def __init__(self, message: str):
        super(HuggingfaceAuthException, self).__init__(message, 'HUGGINGFACE_AUTH')


def main_exception_handler(work: Callable, title: str, is_debug: bool = False) -> None:
    try:
        work()
    except TunerException as e:
        print('')
        print(f"A TunerException has been caught: {str(e)}")
        if e.exception_type == 'VALIDATION' and e.sub_type == 'ARGUMENT_VALIDATION':
            print('')
            print("Please verify that your arguments are valid")
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


