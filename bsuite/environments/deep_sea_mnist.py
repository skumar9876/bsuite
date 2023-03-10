from typing import Optional
import warnings

from bsuite.environments import deep_sea
from bsuite.experiments.deep_sea import sweep
from bsuite.utils import datasets

import dm_env
from dm_env import specs
import numpy as np

class DeepSeaMNIST(deep_sea.DeepSea):
    """MNIST Deep Sea environment to test for deep exploration."""
    
    def __init__(self,
                 size: int,
                 deterministic: bool = True,
                 unscaled_move_cost: float = 0.01,
                 randomize_actions: bool = True,
                 seed: Optional[int] = None,
                 mapping_seed: Optional[int] = None):
        
        super().__init__(size=size,
                         deterministic=deterministic,
                         unscaled_move_cost=unscaled_move_cost,
                         randomize_actions=randomize_actions,
                         seed=seed,
                         mapping_seed=mapping_seed)
        
        (images, labels), _ = datasets.load_mnist()
        self._obs_shape = (28, 28, 2)
        self.label_to_image_dict = {}
        
        for label in range(10):
           self.label_to_image_dict[label] = images[
               labels == label][0:10]
           
    def _get_observation(self):
        obs = np.zeros(shape=self._obs_shape, dtype=np.float32)

        if self._row >= self._size:  # End of episode null observation
            return obs
        
        row_image_ind = np.random.randint(10)
        col_image_ind = np.random.randint(10)
        
        row_obs = self.label_to_image_dict[self._row][row_image_ind]
        col_obs = self.label_to_image_dict[self._col][col_image_ind]
        
        obs = np.concatenate(
            (row_obs[:, :, None], col_obs[:, :, None]), 
            axis=-1)
        
        return obs
    
    def observation_spec(self):
        return specs.Array(
        shape=self._image_shape, dtype=np.float32, name='observation')