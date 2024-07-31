from typing import Callable


class TunerException(Exception):
    def __init__(self, message: str):
        super(TunerException, self).__init__(message)


class ValidationException(TunerException):
    def __init__(self, message: str):
        super(ValidationException, self).__init__(message)
        self.type = 'VALIDATION'


class ArgumentValidationException(ValidationException):
    def __init__(self, message: str):
        super(ArgumentValidationException, self).__init__(message)
        self.sub_type = 'ARGUMENT_VALIDATION'


class HuggingfaceException(TunerException):
    def __init__(self, message: str):
        super(HuggingfaceException, self).__init__(message)
        self.type = 'HUGGINGFACE'


class HuggingfaceAuthException(HuggingfaceException):
    def __init__(self, message: str):
        super(HuggingfaceAuthException, self).__init__(message)
        self.sub_type = 'HUGGINGFACE_AUTH'


def exception_handler(work: Callable, title: str) -> None:
    try:
        work()
    except TunerException as e:
        print(f"A TunerException has happened: {str(e)}")
        print(f"{title} is being terminated!")
        exit(1)
    except Exception as e:
        print(f"An unexpected Exception has been caught: {str(e)}")
        print(f"Rethrowing exception")
        raise e


