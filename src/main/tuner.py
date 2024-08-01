from typing import Callable
import arguments.arguments as arguments


LLM_TYPES = {'llama': 'LLAMA'}


class Tuner:
    def __init__(self, fine_tune: Callable[[arguments.TuneArguments], None], merge: Callable[[arguments.MergeArguments], None], push: Callable[[arguments.PushArguments], None], llm_type: str = LLM_TYPES['llama']):
        self.fine_tune = fine_tune
        self.merge = merge
        self.push = push
        self.llm_type = llm_type
