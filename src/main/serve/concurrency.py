from collections.abc import Callable
from threading import Lock

import time

from serve.atomic_integer import AtomicInteger


class ConcurrencyGateKeeper:

    def __init__(self, max_parallel_requests: int = 1, retry_interval: float = 0.1):
        self.max_parallel_requests = max_parallel_requests
        self.current_active = AtomicInteger(0)
        self._retry_interval=retry_interval
        self._mutex = Lock()

    def _get_lock(self) -> bool:
        with self._mutex:
            if self.current_active.value < self.max_parallel_requests:
                self.current_active.increment()
                return True
            return False

    def execute(self, request: Callable[[], str]) -> str:
        if self._get_lock():
            response = request()
            self.current_active.decrement()
            return response
        time.sleep(self._retry_interval)
        return self.execute(request)
