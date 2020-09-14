"""
This module defines a degradable cartpole environment derived from OpenAI gym.
"""

import numpy as np
from sklearn.base import BaseEstimator
from gym.envs.classic_control import CartPoleEnv as OGCartPole
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class CartPoleEnv(OGCartPole):

    params = ('masscart', 'masspole', 'length', 'force_mag')


    def __init__(self, seed=None):
        super().__init__()
        self.default_params = {p: getattr(self, p) for p in CartPoleEnv.params}
        self._seed = seed
        self.seed(seed)


    def set_parameters(self, masscart: float=1.,
                       masspole: float=0.1, length: float=0.5,
                       force_mag: float=10):
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masscart + self.masspole
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag


    def set_state(self, state: np.ndarray):
        self.state = state

    # pylint: disable=no-member
    def randomize(self):
        return random_cartpole(self.np_random, env=self)


    def plot(self, agent=None, state0=None):
        backup = self.state
        plot_cartpole(self, agent, state0)
        self.state = backup



# pylint: disable=no-member
def random_cartpole(random: np.random.RandomState=None, env=None):
    env = CartPoleEnv() if env is None else env
    feature_size = len(CartPoleEnv.params)
    feature_min = np.asarray([0.75, 0.075, 0.75, 7.5])
    feature_max = np.asarray([1.25, 0.125, 1.25, 12.5])
    feature_min_abs = np.asarray([0.1, 0.01, 0.1, 1.])

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



class CartPoleDataEnv(CartPoleEnv):


    def __init__(self, env, model: BaseEstimator, maxlen: int=200):
        super().__init__()
        self.env = env
        self.model = model
        self.maxlen = maxlen


    def reset(self):
        self._t = 0
        return super().reset()


    def step(self, action):
        self._t += 1
        # print(np.concatenate((self.state, (action,))).reshape(1, -1))
        state = self.model.predict(
            np.concatenate((self.state, (action,))).reshape(1, -1))[0]
        x, _, theta, _ = state
        # x_threshold and theta_threshold_radians are assigned in the base
        # CartPoleEnv class
        done = bool(
            self._t > self.maxlen
            or x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0 if not done else 0.0
        return state, reward, done, {}



def plot_cartpole(env, agent=None, state0=None):
    if agent=='left':
        actor = lambda s: 0
    elif agent=='right':
        actor = lambda s: 1
    elif agent is None:
        actor = lambda s: env.action_space.sample()
    else:
        actor = lambda s: agent.predict(s)[0]

    if state0 is not None:
        env.state = state0
        state = state0
    else:
        state = env.reset()

    done = False
    states = [state]
    actions = []
    while not done:
        action = actor(state)
        actions.append(action)
        state, _, done, _ = env.step(action)
        states.append(state)

    states = np.asarray(states)
    actions = np.asarray(actions)
    x, theta = states[:, 0], states[:, 2]
    xline = plt.plot(x, label='X')[0]
    thetaline = plt.plot(theta, label='Angle /rad')[0]
    plt.legend()
    # pylint: disable=no-member
    im = plt.imshow(actions.reshape(1, -1), aspect='auto', alpha=0.3,
                    extent=(*plt.xlim(), *plt.ylim()), origin='lower',
                    vmin=0, vmax=1, cmap=plt.cm.coolwarm)
    colors = [im.cmap(im.norm(value)) for value in (0, 1)]
    patches = [mpatches.Patch(color=colors[0], label="Left", alpha=0.3),
               mpatches.Patch(color=colors[1], label="Right", alpha=0.3)]
    plt.legend(handles=[xline, thetaline] + patches)
    plt.grid(True)