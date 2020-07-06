"""
This module defines a degradable cartpole environment derived from OpenAI gym.
"""

import numpy as np
from sklearn.base import BaseEstimator
from gym.envs.classic_control import CartPoleEnv as OGCartPole
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class CartPoleEnv(OGCartPole):

    params = ('gravity', 'masscart', 'masspole', 'length', 'force_mag', 'tau')


    def __init__(self):
        super().__init__()
        self.default_params = {p: getattr(self, p) for p in CartPoleEnv.params}


    def set_parameters(self, gravity: float=9.8, masscart: float=1.,
                       masspole: float=0.1, length: float=0.5,
                       force_mag: float=10, tau: float=0.2):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masscart + self.masspole
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = tau



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
    state = env.reset()
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