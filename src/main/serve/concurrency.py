from collections.abc import Callable
from threading import Lock
from collections import deque

import time

from serve.atomic_integer import AtomicInteger


class ConcurrencyGateKeeper:

    def __init__(self, max_parallel_requests: int = 1):
        self.max_parallel_requests = max_parallel_requests
        self.current_active = AtomicInteger(0)

    def execute(self, request: Callable):
        if self.current_active.value < self.max_parallel_requests:
            self.current_active.increment()
            response = request()
            self.current_active.decrement()
            return response
        time.sleep(0.5)
        return self.execute(request)
