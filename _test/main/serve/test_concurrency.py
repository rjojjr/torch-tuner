import os
import sys
import threading
import time
import unittest

_TEST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TEST_FILE_DIR, "..", "..", ".."))
_SRC_MAIN = os.path.join(_PROJECT_ROOT, "src", "main")
if _SRC_MAIN not in sys.path:
    sys.path.insert(0, _SRC_MAIN)

from serve.concurrency import ConcurrencyGateKeeper  # noqa: E402


class TestConcurrencyGateKeeper(unittest.TestCase):
    """The gate keeper exists to bound concurrent generations so multiple in-flight
    requests don't collectively OOM the GPU. These tests pin its memory-relevant
    contract: capacity is enforced, slots are released even on exception, and
    waiting threads are unblocked as slots free up (no recursion -> no stack growth)."""

    def test_concurrent_executions_are_capped_at_max_parallel_requests(self):
        gate = ConcurrencyGateKeeper(max_parallel_requests=3)

        in_flight = 0
        in_flight_max = 0
        bookkeeping = threading.Lock()

        def work():
            nonlocal in_flight, in_flight_max
            with bookkeeping:
                in_flight += 1
                in_flight_max = max(in_flight_max, in_flight)
            time.sleep(0.02)
            with bookkeeping:
                in_flight -= 1
            return "ok"

        threads = [threading.Thread(target=lambda: gate.execute(work)) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(in_flight_max, 3, "gate keeper should hold concurrency at exactly the configured cap under load")

    def test_execute_releases_slot_even_when_request_raises(self):
        gate = ConcurrencyGateKeeper(max_parallel_requests=1)

        with self.assertRaises(RuntimeError):
            gate.execute(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        # Slot must be free again — the next call should not deadlock.
        completed = threading.Event()

        def follow_up():
            gate.execute(lambda: "ok")
            completed.set()

        threading.Thread(target=follow_up, daemon=True).start()
        self.assertTrue(completed.wait(timeout=2.0), "gate keeper failed to release slot after exception")

    def test_waiting_threads_unblock_as_slots_free_without_busy_recursion(self):
        # The previous recursive implementation grew the call stack per retry; this
        # test schedules far more workers than the gate's capacity to confirm the
        # new wait-on-condition path doesn't recurse and completes promptly.
        gate = ConcurrencyGateKeeper(max_parallel_requests=2)
        n = 200
        completed = []
        completed_lock = threading.Lock()

        def work(i):
            time.sleep(0.001)
            with completed_lock:
                completed.append(i)
            return i

        threads = [threading.Thread(target=lambda i=i: gate.execute(lambda: work(i))) for i in range(n)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.monotonic() - start

        self.assertEqual(len(completed), n)
        self.assertLess(elapsed, 5.0, "waiting threads should unblock promptly (no busy recursion)")


if __name__ == "__main__":
    unittest.main()
