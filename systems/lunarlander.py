"""
This module defines a degradable lunar-lander environment derived from OpenAI gym.
"""

import numpy as np
from sklearn.base import BaseEstimator
from gym.envs.box2d import lunar_lander
from gym.envs.box2d import LunarLander as OGLunarLander
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class LunarLanderEnv(OGLunarLander):

    params = ('MAIN_ENGINE_POWER', 'SIDE_ENGINE_POWER')


    def __init__(self, seed=None):
        super().__init__()
        self.default_params = {p: getattr(lunar_lander, p) for p in LunarLanderEnv.params}
        self._seed = seed
        self.seed(seed)


    def set_parameters(self, MAIN_ENGINE_POWER: float=13., SIDE_ENGINE_POWER: float=0.6):
        lunar_lander.MAIN_ENGINE_POWER = MAIN_ENGINE_POWER
        lunar_lander.SIDE_ENGINE_POWER = SIDE_ENGINE_POWER


    def set_state(self, state: np.ndarray):
        raise NotImplementedError

    # pylint: disable=no-member
    def randomize(self):
        return random_lunarlander(self.np_random, env=self)



# pylint: disable=no-member
def random_lunarlander(random: np.random.RandomState=None, env=None):
    env = LunarLanderEnv() if env is None else env
    feature_size = len(LunarLanderEnv.params)
    feature_min = np.asarray([10., 0.5])
    feature_max = np.asarray([16., 0.7])
    feature_min_abs = np.asarray([10., 0.5])

    if isinstance(random, np.random.RandomState):
        features = random.randn(feature_size)
    elif isinstance(random, np.ndarray):
        features = random
    elif isinstance(random, (int, float)):
        random = np.random.RandomState(random)
        features = random.rand(feature_size)
    
    features = np.clip(features, feature_min, feature_max)
    features = np.where(np.abs(features) < feature_min_abs,
                        np.sign(features) * feature_min_abs,
                        features)
    params = {k:v for k, v in zip(env.params, features)}
    env.set_parameters(**params)
    return env



def plot_lunarlander(env, agent):
    pass