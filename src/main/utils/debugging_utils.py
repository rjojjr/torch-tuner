import contextlib
import warnings

@contextlib.contextmanager
def debugging_wrapper(is_debug_mode):
    if not is_debug_mode:
        warnings.filterwarnings("ignore")
    yield