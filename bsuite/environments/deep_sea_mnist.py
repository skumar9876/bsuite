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
        self._obs_shape = (28, 28, 4)
        self.label_to_image_dict = {}

        self.observation_space = {} 
        self.observation_space['image'] = np.zeros((28, 28, 4))
        self.observation_space['row_image'] = np.zeros((28, 28, 2))
        self.observation_space['col_image'] = np.zeros((28, 28, 2))
        
        self.action_space = ActionSpace(2)

        for label in range(99):
            a = label // 10
            b = label % 10
            
            first_digit_img = images[labels == a][0]
            second_digit_img = images[labels == b][0]

            img = np.concatenate(
                (first_digit_img[:, :, None], second_digit_img[:, :, None]), 
                axis=-1)

            self.label_to_image_dict[label] = img
        
        super().__init__(size=size,
                         deterministic=deterministic,
                         unscaled_move_cost=unscaled_move_cost,
                         randomize_actions=randomize_actions,
                         seed=seed,
                         mapping_seed=mapping_seed)
           
    def _get_observation(self):
        row_obs = np.zeros(shape=(28, 28, 2), dtype=np.float32)
        obs = np.zeros(shape=self._obs_shape, dtype=np.float32)

        if self._row >= self._size:  # End of episode null observation
            return {'image': obs, 'row_image': row_obs, 'col_image': row_obs, 
                    'row': self._row, 'col': self._column}

        row_obs = self.label_to_image_dict[self._row]
        col_obs = self.label_to_image_dict[self._column]

        obs = np.concatenate((row_obs, col_obs), axis=-1)
        
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
        reward = 0  # x.reward
        
        done = False
        if self._column == 0 and self._row == 4:
            reward = 0.25
            done = True
        elif self._column == 0 and self._row == 8:
           reward = 0.5
           done = True
        elif self._column == 0 and self._row == 12:
            reward = 0.75
            done = True   
        elif self._column == self._size - 1 and self._row == self._size - 1:
            reward = 1
            done = True
        elif self._row == self._size - 1:
            done = True

        # if done:
        #     print(f'RETURN: {reward}')
        #     print()

        return x.observation, reward, done, None