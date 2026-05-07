import unittest
from src.main.serve.atomic_integer import AtomicInteger


class TestAtomicInteger(unittest.TestCase):
    def test_atomic_integer_initialization(self):
        ai = AtomicInteger(0)
        self.assertEqual(ai.value, 0)

    def test_atomic_integer_initialization_with_value(self):
        ai = AtomicInteger(42)
        self.assertEqual(ai.value, 42)

    def test_atomic_integer_increment(self):
        ai = AtomicInteger(0)
        result = ai.increment()
        self.assertEqual(result, 1)
        self.assertEqual(ai.value, 1)

    def test_atomic_integer_increment_with_delta(self):
        ai = AtomicInteger(0)
        result = ai.increment(5)
        self.assertEqual(result, 5)
        self.assertEqual(ai.value, 5)

    def test_atomic_integer_multiple_increments(self):
        ai = AtomicInteger(0)
        for _ in range(5):
            ai.increment()
        self.assertEqual(ai.value, 5)

    def test_atomic_integer_decrement(self):
        ai = AtomicInteger(5)
        result = ai.decrement()
        self.assertEqual(result, 4)
        self.assertEqual(ai.value, 4)

    def test_atomic_integer_decrement_with_delta(self):
        ai = AtomicInteger(10)
        result = ai.decrement(3)
        self.assertEqual(result, 7)
        self.assertEqual(ai.value, 7)

    def test_atomic_integer_multiple_decrements(self):
        ai = AtomicInteger(5)
        for _ in range(3):
            ai.decrement()
        self.assertEqual(ai.value, 2)

    def test_atomic_integer_negative_values(self):
        ai = AtomicInteger(-5)
        self.assertEqual(ai.value, -5)
        result = ai.increment()
        self.assertEqual(result, -4)
        self.assertEqual(ai.value, -4)

    def test_atomic_integer_set_value(self):
        ai = AtomicInteger(0)
        ai.value = 100
        self.assertEqual(ai.value, 100)

    def test_atomic_integer_string_initialization(self):
        ai = AtomicInteger("42")
        self.assertEqual(ai.value, 42)
        self.assertIsInstance(ai.value, int)

    def test_atomic_integer_float_initialization(self):
        ai = AtomicInteger(3.7)
        self.assertEqual(ai.value, 3)
        self.assertIsInstance(ai.value, int)


if __name__ == '__main__':
    unittest.main()
