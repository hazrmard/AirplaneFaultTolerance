import gym
import numpy as np



class Track1DEnv(gym.Env):


    def __init__(self, seed=None):
        super().__init__(self)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.np_random = np.random.RandomState(seed)
        self.x = None


    def seed(self, seed):
        self.np_random.seed(seed)


    def reset(self):
        self.x = self.observation_space.sample()
        return self.x


    def step(self, action):
        action = action * 2 - 1 # [0, 1] -> [-1, 1]