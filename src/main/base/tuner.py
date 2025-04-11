from typing import Callable
import arguments.arguments as arguments

from arguments.arguments import ArgumentsConfig

LLM_TYPES = {'generic': 'GENERIC', 'llama': 'LLAMA'}


# This should be constructed with the tuning functions for the LLM type you intend to tune on
class Tuner:
    """Tuning module wrapper."""

    # TODO - Is there some way to make this constructor private or hide it so that only the tuner factor can call it?
    def __init__(self, fine_tune: Callable[[arguments.TuneArguments], None], merge: Callable[[arguments.MergeArguments], None], push: Callable[[arguments.PushArguments], None], llm_type: str = LLM_TYPES['generic'], arguments_config: ArgumentsConfig | None = None):
        self.fine_tune = fine_tune
        self.merge = merge
        self.push = push
        self.llm_type = llm_type
        self.arguments_config = arguments_config
