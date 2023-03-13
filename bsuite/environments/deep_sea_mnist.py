from typing import Optional
import warnings

from bsuite.environments import deep_sea
from bsuite.experiments.deep_sea import sweep
from bsuite.utils import datasets

import collections

import dm_env
from dm_env import specs
import numpy as np

ActionSpace = collections.namedtuple('action_space', ['n'])

class DeepSeaMNIST(deep_sea.DeepSea):
    """MNIST Deep Sea environment to test for deep exploration."""
    
    def __init__(self,
                 size: int,
                 deterministic: bool = True,
                 unscaled_move_cost: float = 0.01,
                 randomize_actions: bool = True,
                 seed: Optional[int] = None,
                 mapping_seed: Optional[int] = None):

        (images, labels), _ = datasets.load_mnist()
        self._obs_shape = (28, 28, 2)
        self.label_to_image_dict = {}
        self.observation_space = {} 

        # TODO: Hacky stuff here to add gym env components.
        self.observation_space['image'] = np.zeros((28, 28, 2))
        self.observation_space['row_image'] = np.zeros((28, 28, 1))
        self.observation_space['col_image'] = np.zeros((28, 28, 1))
        
        self.action_space = ActionSpace(2)

        for label in range(10):
            self.label_to_image_dict[label] = images[
                labels == label][0:2]
        
        super().__init__(size=size,
                         deterministic=deterministic,
                         unscaled_move_cost=unscaled_move_cost,
                         randomize_actions=randomize_actions,
                         seed=seed,
                         mapping_seed=mapping_seed)
           
    def _get_observation(self):
        row_obs = np.zeros(shape=(28, 28), dtype=np.float32)
        obs = np.zeros(shape=self._obs_shape, dtype=np.float32)

        if self._row >= self._size:  # End of episode null observation
            return {'image': obs, 'row_image': row_obs, 'col_image': row_obs, 
                    'row': self._row, 'col': self._column}

        row_image_ind = 0
        col_image_ind = 0

        row_obs = self.label_to_image_dict[self._row][row_image_ind]
        col_obs = self.label_to_image_dict[self._column][col_image_ind]
        
        obs = np.concatenate(
            (row_obs[:, :, None], col_obs[:, :, None]), 
            axis=-1)
        
        return  {'image': obs, 'row_image': row_obs, 'col_image': row_obs, 
                 'row': self._row, 'col': self._column}
    
    def observation_spec(self):
        return specs.Array(
            shape=self._obs_shape, dtype=np.float32, name='observation')
    
    def reset(self) -> dm_env.TimeStep:
        x = super().reset()
        return x.observation

    def step(self, action: int) -> dm_env.TimeStep:
        x = super().step(action)
        done = False
        if self._row == self._size:
            done = True
        return x.observation, x.reward, done, None