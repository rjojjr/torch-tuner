from typing import Callable
import arguments.arguments as arguments


class Tuner:
    def __init__(self, fine_tune: Callable[[arguments.TuneArguments], None], merge: Callable[[arguments.MergeArguments], None], push: Callable[[arguments.PushArguments], None]):
        self.fine_tune = fine_tune
        self.merge = merge
        self.push = push
