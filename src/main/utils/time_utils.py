import time


def current_milli_time():
    """Get current time in milliseconds."""
    return round(time.time() * 1000)