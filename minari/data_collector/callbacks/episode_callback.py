from typing import Any, Dict, Optional

import gymnasium as gym

from minari.dataset.step_data import StepData
from minari.data_collector.episode_buffer import EpisodeBuffer

from typing import (
    Callable,
)


class EpisodeDataCallback:
    def __init__(self, function_hook: Callable):
        """Initialize the episode data callback."""
        self.function_hook = function_hook

    def __call__(self, episode_buffer: EpisodeBuffer) -> None:
        pass
