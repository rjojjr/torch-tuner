class TunerException(Exception):
    def __init__(self, message: str):
        super(TunerException, self).__init__(message)

class ArgumentValidationException(TunerException):
    def __init__(self, message: str):
        super(ArgumentValidationException, self).__init__(message)
        self.type = 'ARGUMENT_VALIDATION'