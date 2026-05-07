import unittest
import time
from src.main.utils.time_utils import current_milli_time


class TestTimeUtils(unittest.TestCase):
    def test_current_milli_time_returns_integer(self):
        result = current_milli_time()
        self.assertIsInstance(result, int)

    def test_current_milli_time_is_positive(self):
        result = current_milli_time()
        self.assertGreater(result, 0)

    def test_current_milli_time_increases(self):
        time1 = current_milli_time()
        time.sleep(0.01)
        time2 = current_milli_time()
        self.assertGreater(time2, time1)

    def test_current_milli_time_reasonable_range(self):
        result = current_milli_time()
        current_seconds = time.time()
        expected_millis = int(current_seconds * 1000)
        self.assertAlmostEqual(result, expected_millis, delta=100)

    def test_current_milli_time_multiple_calls(self):
        times = [current_milli_time() for _ in range(5)]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i-1])


if __name__ == '__main__':
    unittest.main()
