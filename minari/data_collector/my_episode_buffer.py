from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from minari.dataset.my_step_data import MyStepData


@dataclass(frozen=False)
class MyEpisodeBuffer:
    """Contains the data of a single episode."""

    id: Optional[int] = None
    seed: Optional[int] = None
    options: Optional[dict] = None
    observations: Union[None, list, dict, tuple] = None
    masks: Union[None, list, dict, tuple] = None
    actions: Union[None, list, dict, tuple] = None
    rewards: list = field(default_factory=list)
    terminations: list = field(default_factory=list)
    truncations: list = field(default_factory=list)
    infos: Optional[dict] = None

    def add_step_data(self, step_data: MyStepData) -> MyEpisodeBuffer:
        """Add step data dictionary to episode buffer.

        Args:
            step_data (StepData): dictionary with data for a single step

        Returns:
            MyEpisodeBuffer: episode buffer with appended data
        """
        try:
            import jax.tree_util as jtu
        except ImportError:
            raise ImportError(
                'jax is not installed. Please install it using `pip install "minari[create]"`'
            )

        def _append(data, buffer):
            if isinstance(buffer, list):
                buffer.append(data)
                return buffer
            else:
                return [buffer, data]

        observations = step_data["observation"]
        masks = step_data["masks"]
        if self.observations is not None:
            observations = jtu.tree_map(
                _append, step_data["observation"], self.observations
            )
        if self.masks is not None:
            masks = jtu.tree_map(_append, step_data["masks"], self.masks)

        if self.actions is None:
            actions = jtu.tree_map(lambda x: [x], step_data["action"])
        else:
            actions = jtu.tree_map(_append, step_data["action"], self.actions)

        if self.infos is None:
            infos = jtu.tree_map(lambda x: [x], step_data["info"])
        else:
            infos = jtu.tree_map(_append, step_data["info"], self.infos)

        self.rewards.append(step_data["reward"])
        self.terminations.append(step_data["terminated"])
        self.truncations.append(step_data["truncated"])

        return MyEpisodeBuffer(
            id=self.id,
            seed=self.seed,
            options=self.options,
            observations=observations,
            masks=masks,
            actions=actions,
            rewards=self.rewards,
            terminations=self.terminations,
            truncations=self.truncations,
            infos=infos,
        )

    def __len__(self) -> int:
        """Buffer length."""
        return len(self.rewards)
