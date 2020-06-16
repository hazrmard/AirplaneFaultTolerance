"""
This module defines a degradable cartpole environment derived from OpenAI gym.
"""
from gym.envs.classic_control import CartPoleEnv as OGCartPole



class CartPoleEnv(OGCartPole):

    params = ('gravity', 'masscart', 'masspole', 'length', 'force_mag', 'tau')


    def __init__(self):
        super().__init__()
        self.og_params = {p: getattr(self, p) for p in CartPoleEnv.params}


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


    def degrade(self, factor: float=0.):
        force = self.og_params['force'] * (1. - factor)
        self.set_parameters(force_mag=force)
