from collections.abc import Callable
from threading import Condition


class ConcurrencyGateKeeper:
    """Bounds the number of concurrent generations to avoid GPU OOM.

    Threads are blocked on a Condition rather than recursing, so waiting
    requests don't grow the call stack under load.
    """

    def __init__(self, max_parallel_requests: int = 1):
        self._max_parallel_requests = max_parallel_requests
        self._current_active = 0
        self._cv = Condition()

    def execute(self, request: Callable[[], str]) -> str:
        with self._cv:
            while self._current_active >= self._max_parallel_requests:
                self._cv.wait()
            self._current_active += 1
        try:
            return request()
        finally:
            with self._cv:
                self._current_active -= 1
                self._cv.notify()
