from __future__ import annotations

import io
import pathlib
from collections import OrderedDict
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from PIL import Image

from minari.data_collector import EpisodeBuffer
from minari.data_collector.my_episode_buffer import MyEpisodeBuffer

from minari.dataset.minari_storage import MinariStorage, is_image_space


from minari.dataset._storages.hdf5_storage import (
    HDF5Storage,
    _decode_info,
    flatten_dict,
    unflatten_dict,
    _get_from_h5py,
    _add_episode_to_group,
)

try:
    import h5py
except ImportError:
    raise ImportError(
        'h5py is not installed. Please install it using `pip install "minari[hdf5]"`'
    )

_MAIN_FILE_NAME = "main_data.hdf5"


class MyHDF5Storage(MinariStorage):
    FORMAT = "my_hdf5"

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__(data_path, observation_space, action_space)
        file_path = self.data_path.joinpath(_MAIN_FILE_NAME)
        if not file_path.exists():
            raise ValueError(f"No data found in data path {self.data_path}")
        self._file_path = file_path

        self.masks_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(observation_space.shape[0], observation_space.shape[1]),
            dtype=np.uint8,
        )

    @classmethod
    def _create(
        cls,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> MinariStorage:
        data_path.joinpath(_MAIN_FILE_NAME).touch(exist_ok=False)
        obj = cls(data_path, observation_space, action_space)
        return obj

    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)

        sentinel = object()
        with h5py.File(self._file_path, "a") as file:
            for metadata, episode_id in zip_longest(
                metadatas, episode_indices, fillvalue=sentinel
            ):
                if sentinel in (metadata, episode_id):
                    raise ValueError(
                        "Metadatas and episode_indices have different lengths"
                    )

                assert isinstance(metadata, dict)
                ep_group = file[f"episode_{episode_id}"]
                ep_group.attrs.update(metadata)

        self.update_metadata({"dataset_size": self.get_size()})

    def get_episode_metadata(self, episode_indices: Iterable[int]) -> Iterable[Dict]:
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                metadata: dict = dict(ep_group.attrs)
                metadata = unflatten_dict(metadata)
                if metadata.get("seed") is not None:
                    metadata["seed"] = int(metadata["seed"])

                yield metadata

    def _decode_space(
        self,
        hdf_ref: Union[h5py.Group, h5py.Dataset, h5py.Datatype],
        space: gym.spaces.Space,
    ) -> Union[Dict, Tuple, List, np.ndarray]:
        assert not isinstance(hdf_ref, h5py.Datatype)

        if isinstance(space, gym.spaces.Tuple):
            assert isinstance(hdf_ref, h5py.Group)
            result = []
            for i in range(len(hdf_ref.keys())):
                result.append(
                    self._decode_space(hdf_ref[f"_index_{i}"], space.spaces[i])
                )
            return tuple(result)
        elif isinstance(space, gym.spaces.Dict):
            assert isinstance(hdf_ref, h5py.Group)
            result = {}
            for key in hdf_ref.keys():
                result[key] = self._decode_space(hdf_ref[key], space.spaces[key])
            return result
        elif isinstance(space, gym.spaces.Text):
            assert isinstance(hdf_ref, h5py.Dataset)
            result = map(lambda string: string.decode("utf-8"), hdf_ref[()])
            return list(result)
        elif is_image_space(space):
            jpeg_bytes_list = hdf_ref[()]
            first_image = np.array(
                Image.open(io.BytesIO(jpeg_bytes_list[0])), dtype=np.uint8
            )
            jpeg_images = np.empty(
                (len(jpeg_bytes_list),) + first_image.shape, dtype=np.uint8
            )
            jpeg_images[0] = first_image
            for i, jpeg_bytes in enumerate(jpeg_bytes_list[1:], start=1):
                image = Image.open(io.BytesIO(jpeg_bytes))
                jpeg_images[i] = np.array(image, dtype=np.uint8)
            return jpeg_images
        else:
            assert isinstance(hdf_ref, h5py.Dataset)
            return hdf_ref[()]

    def get_episodes(self, episode_indices: Iterable[int]) -> Iterable[dict]:
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                infos = None
                if "infos" in ep_group:
                    info_group = ep_group["infos"]
                    assert isinstance(info_group, h5py.Group)
                    infos = _decode_info(info_group)

                observations = self._decode_space(
                    ep_group["observations"], self.observation_space
                )
                masks = np.stack(ep_group["masks"], axis=0)
                ep_dict = {
                    "id": ep_idx,
                    "observations": observations,
                    # "masks": self._decode_space(ep_group["masks"], self.masks_space),
                    "masks": masks,
                    "actions": self._decode_space(
                        ep_group["actions"], self.action_space
                    ),
                    "infos": infos,
                }
                for key in {"rewards", "terminations", "truncations"}:
                    group_value = ep_group[key]
                    assert isinstance(group_value, h5py.Dataset)
                    ep_dict[key] = group_value[:]

                yield ep_dict

    def update_episodes(self, episodes: Iterable[MyEpisodeBuffer]):
        additional_steps = 0
        with h5py.File(self._file_path, "a", track_order=True) as file:
            for eps_buff in episodes:
                total_episodes = len(file.keys())
                episode_id = eps_buff.id if eps_buff.id is not None else total_episodes
                assert (
                    episode_id <= total_episodes
                ), "Invalid episode id; ids must be sequential."
                episode_group = _get_from_h5py(file, f"episode_{episode_id}")
                episode_group.attrs["id"] = episode_id
                if eps_buff.seed is not None:
                    assert "seed" not in episode_group.attrs.keys()
                    episode_group.attrs["seed"] = eps_buff.seed
                if eps_buff.options is not None:
                    assert "options" not in episode_group.attrs.keys()
                    flatten_option = flatten_dict(eps_buff.options, "options")
                    episode_group.attrs.update(flatten_option)

                episode_steps = len(eps_buff.rewards)
                episode_group.attrs["total_steps"] = episode_steps
                additional_steps += episode_steps

                dict_buffer = {
                    "observations": eps_buff.observations,
                    # "masks": eps_buff.masks,
                    "actions": eps_buff.actions,
                    "rewards": eps_buff.rewards,
                    "terminations": eps_buff.terminations,
                    "truncations": eps_buff.truncations,
                    "infos": eps_buff.infos,
                }
                _add_episode_to_group(dict_buffer, episode_group)

                episode_group.create_dataset(
                    "masks", data=eps_buff.masks, dtype=np.uint8, chunks=True
                )

            total_episodes = len(file.keys())

        total_steps = self.total_steps + additional_steps
        self.update_metadata(
            {
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "dataset_size": self.get_size(),
            }
        )
