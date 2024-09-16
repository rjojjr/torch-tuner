import threading
class AtomicInteger():
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()

    def increment(self, d=1):
        with self._lock:
            self._value += int(d)
            return self._value

    def decrement(self, d=1):
        return self.increment(-d)

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = int(v)
            return self._value