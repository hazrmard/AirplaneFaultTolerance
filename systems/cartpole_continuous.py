# The MIT License

# Copyright (c) 2020 Ibrahim Ahmed
# Copyright (c) 2016 OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math

from .cartpole import CartPoleEnv
from gym.spaces import Box
from gym import logger
import numpy as np
import matplotlib.pyplot as plt



class CartPoleContinuousEnv(CartPoleEnv):


    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1., high=1., shape=(1,), dtype=np.float32)


    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        # ===
        # This assignment overwrites the discrete behavior of the cartpole.
        # The action is the scale of the force magnitude itself
        force = action[0] * self.force_mag
        # ===
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}



def plot_cartpole_continuous(env, agent=None, state0=None, maxlen=500, legend=True):
    if agent=='left':
        actor = lambda s: np.asarray([0.])
    elif agent=='right':
        actor = lambda s: np.asarray([1.])
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
    t = 0
    while not done and t < maxlen:
        action = actor(state)
        actions.append(action)
        state, _, done, _ = env.step(action)
        states.append(state)
        t += 1

    states = np.asarray(states)
    actions = np.asarray(actions)
    x, theta = states[:, 0], states[:, 2]
    xline = plt.plot(x, 'b-', label='X')[0]
    plt.ylabel('X')
    plt.ylim(bottom=-env.x_threshold, top=env.x_threshold)
    plt.twinx()
    thetaline = plt.plot(theta, 'g:', label='Angle /rad')[0]
    plt.ylabel('Angle /rad')
    plt.ylim(bottom=-env.theta_threshold_radians, top=env.theta_threshold_radians)
    # plt.legend()
    # pylint: disable=no-member
    im = plt.imshow(actions.reshape(1, -1), aspect='auto', alpha=0.3,
                    extent=(*plt.xlim(), *plt.ylim()), origin='lower',
                    vmin=-1, vmax=1, cmap=plt.cm.coolwarm) # vmin/vmax differ from cartpole
    colors = [im.cmap(im.norm(value)) for value in (-1, 1)]
    patches = [mpatches.Patch(color=colors[0], label="Left", alpha=0.3),
               mpatches.Patch(color=colors[1], label="Right", alpha=0.3)]
    plt.grid(True)
    if legend:
        plt.legend(handles=[xline, thetaline] + patches)
    return [xline, thetaline] + patches, ('X', 'Angle /rad', 'Left', 'Right')