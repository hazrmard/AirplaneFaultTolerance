"""
Defines the model for a fuel tank system for an airplane.
"""
from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pystatespace import Trapezoid      # pylint: disable=import-error



class TanksFactory:
    """
    Create a fuel tank system based on parameters provided at initialization.
    The fuel tank system has a number of tanks connected via a shared fuel
    conduit. Each tank has a valve. Opening the valves allows fuel to flow between
    tanks. Additionaly there are fuel pumps on each tank. Fuel is trained via
    the pumps based on engine demand, from innermost pumps to outer pumps.

    The system is defined by its state which is the height level of fuel in
    tanks. The system responds to actions which are the states of valves at
    the bottom of fuel tanks.

    Parameters
    ----------
    n : int, optional
        Number of tanks in the system, by default 6. This is the dimension
        of the state and action vectors.
    e : int, optional
        Number of engines in the system, by default 2
    rho : float, optional
        Density of fuel , by default 1e3
    g : float, optional
        Gravitational potential (m/s^2), by default 10.0
    heights : np.ndarray, optional
        Maximum height of fuel levels in tanks (m), by default np.ones(6)
    cross_section : np.ndarray, optional
        Cross-sectional area of tanks (m^2), by default np.ones(6)
    valves_min : np.ndarray, optional
        Minimum state of valves when closed, by default np.zeros(6)
    valves_max : np.ndarray, optional
        Maximum state of valves when opened, by default np.ones(6)
    resistances : np.ndarray, optional
        Resistance of valves, by default np.ones(6)*1e2
    pumps : np.ndarray, optional
        Maximum capacity of pumps (m^3/s), by default np.ones(6)*0.01
    leaks : np.ndarray, optional
        Fuel leakage from each tank (m^3/s), by default np.zeros(6)
    engines : np.ndarray, optional
        Fuel demand of engines (m^3/s), by default np.ones(2)*0.02

    Attributes
    ----------
    dxdt : Callable
        A function with the signature (t: float, x: np.ndarray, u: np.ndarray) ->
        np.ndarray. It returns the next state in response to current time, state
        and action.
    y : Callable
        A function with the same signature as `dxdt`. Returns the output variables
        of the system. In this case y also returns the state vector.
    """


    def __init__(self, n: int=6, e: int=2, rho: float=1e3, g: float=10.0,
                 heights: np.ndarray=np.ones(6),
                 cross_section: np.ndarray=np.ones(6),
                 valves_min: np.ndarray=np.zeros(6),
                 valves_max: np.ndarray=np.ones(6),
                 resistances: np.ndarray=np.ones(6) * 1e3,
                 leaks: np.ndarray=np.zeros(6),
                 pumps: np.ndarray=np.ones(6) * 0.01,
                 engines: np.ndarray=np.ones(2) * 0.01):
        self.n = n                     # Number of tanks
        self.e = e                     # Number of engines
        self.rho = rho                 # Density of fluid
        self.g = g                  # Gravitational acceleration
        self.heights = np.copy(heights)      # height of tanks
        self.cross_section = np.copy(cross_section)
        self.valves_min = np.copy(valves_min )      # valve capacities on each tank
        self.valves_max = np.copy(valves_max)       # valve capacities on each tank
        self.resistances = np.copy(resistances)  # valve resistances where flowrate = g.height/resistance
        self.leaks = leaks
        self.pumps = np.copy(pumps)        # pumps capacities on each tank
        self.engines = np.copy(engines)     # fuel demand per engine

        # derived params
        median_tank = n % 2 == 1       # Whether there is a central pump
        median_tank_idx = n // 2
        median_engine = e % 2 == 1
        median_engine_idx = e // 2



        def _dxdt(time: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            # print(x)
            dx = np.zeros_like(x)
            dx -= self.leaks
            # Adjust actual valve state
            u = self.valves_min + u * (self.valves_max - self.valves_min)
            # pump fuel to engines
            demand_left, demand_right = 0., 0.
            if median_engine:
                demand_left = self.engines[median_engine_idx] / 2
                demand_right = self.engines[median_engine_idx] / 2
            demand_left += sum(self.engines[:median_engine_idx])
            demand_right += sum(self.engines[median_engine_idx + int(median_engine):])
            # print('dl', demand_left, 'dr', demand_right)

            if median_tank:
                supply_left = min(self.pumps[median_tank_idx] / 2, demand_left)
                supply_right = min(self.pumps[median_tank_idx] / 2, demand_right)
                demand_left -= supply_left
                demand_right -= supply_right
                dx[median_tank_idx] -= (supply_left + supply_right)  \
                                       / self.cross_section[median_tank_idx]
            
            i = median_tank_idx - 1
            while demand_left > 0 and i >= 0:
                supply_left = min(self.pumps[i], x[i], demand_left)
                demand_left -= supply_left
                dx[i] -= supply_left / self.cross_section[i]
                # print('sl', i, supply_left, 'dl', demand_left)
                i -= 1

            i = median_tank_idx + int(median_tank)
            while demand_right > 0 and i < n:
                supply_right = min(self.pumps[i], x[i], demand_right)
                demand_right -= supply_right
                dx[i] -= supply_right / self.cross_section[i]
                # print('sr', i, supply_right, 'dr', demand_right)
                i += 1
            # print(dx)
            # print()
            
            # re-balance tanks
            # Effective pressure at bottom of tanks (if disconnected, then == 0)
            potential_effective = (u > 0.) * (self.g * x)
            # Conduit pressure is governed by the highest fuel level
            potential_conduit = np.max(potential_effective)
            # All tanks that are connected to conduit and that will act as source
            source_tanks_idx = np.arange(n)[(u > 0.) & (potential_effective == potential_conduit)]
            # There is no pressure differential, no re-balancing occurs:
            if len(source_tanks_idx) == n:
                return dx
            # All tanks that are connected to conduit and that will act as sink
            sink_tanks_idx = np.arange(n)[(u > 0.) & (potential_effective < potential_conduit)]
            # print(potential_conduit, potential_effective)
            # Velocity of fluid into sink tanks: change of potential / resistance
            flowrate_sink = (potential_conduit - potential_effective[sink_tanks_idx])  * u[sink_tanks_idx] / self.resistances[sink_tanks_idx]
            # Total flow into sink == total flow from source == flow through conduit
            flowrate_total = sum(flowrate_sink)
            # Flowrate per source tank ~ 1 / resistance
            resistance_inverse = u / self.resistances
            flowrate_source = flowrate_total * resistance_inverse[source_tanks_idx] / sum(resistance_inverse[source_tanks_idx])
            # Change of level
            dx[sink_tanks_idx] += flowrate_sink / self.cross_section[sink_tanks_idx]
            dx[source_tanks_idx] -= flowrate_source / self.cross_section[source_tanks_idx]

            return dx
        


        def _y(time: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
            return x



        self.dxdt = _dxdt
        self.y = _y


class TanksPhysicalEnv(gym.Env):
    """
    An `OpenAI Gym` environment for simulating a fuel system model generated
    by `TanksFactory`.

    Parameters
    ----------
    tanks : TanksFactory
        An instance of the fuel tanks factory class.
    tstep : float, optional
        Time step of each action, by default 1e-1
    
    Attributes
    ----------
    solver: Trapezoid
        A class instance with a predict(time, state, action) method that returns
        the a tuple of next state and output vectors as 2D arrays.
    """
    
    params = ('rho', 'heights', 'cross_section', 'valves_min', 'valves_max',
               'resistances', 'pumps', 'engines')


    def __init__(self, tanks: TanksFactory, tstep: float=1., seed=None):
        super().__init__()
        self.tanks = tanks
        self.tstep = tstep
        self.n = len(tanks.pumps)
        self.solver = Trapezoid(dims=self.n, outputs=self.n, dx=tanks.dxdt,
                                out=tanks.y, tstep=tstep)
        self.action_space = gym.spaces.MultiBinary(self.n)
        self.observation_space = gym.spaces.Box(0, np.inf, (self.n,))

        self.x = None
        self.t = None
        self.np_random = None
        self.og_params = {p: getattr(self.tanks, p) for p in TanksPhysicalEnv.params}
        self.seed(seed)
        self.reset()

        
    def randomize(self):
        for param, value in self.og_params.items():
            if param in ('heights, cross_section'):
                factor = np.clip((1 + 0.2 * self.np_random.randn()), 0, None)
                setattr(self.tanks, param, value * factor)


    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)


    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        
        Returns
        -------
        np.ndarray
            The initial state vector.
        """
        # Tanks
        median = self.n // 2
        self.odd = self.n % 2 != 0
        self.median_idx = median
        self.left_idx = slice(None, median)
        self.right_idx = slice(median + (self.n % 2), None)
        self.left_arm = np.arange(median, 0, -1)
        self.right_arm = np.arange(1, median + 1)
        self.max_arm = abs(max(self.right_arm))
        # Max std dev of fuel weight for reward calculation, when fuel is only in
        # extreme tanks:
        weights = np.zeros(self.n)
        weights[[0, -1]] = self.tanks.heights[[0, -1]] * self.tanks.cross_section[[0, -1]]
        self.max_spread = np.sqrt(sum(self.left_arm**2 * weights[self.left_idx]  \
                          + self.right_arm**2 * weights[self.right_idx]) \
                        / (sum(weights) + 1e-5))
    
        # Engines
        median = len(self.tanks.engines) // 2
        self.odd_e = len(self.tanks.engines) % 2 != 0
        self.median_e_idx = median
        self.left_e_idx = slice(None, median)
        self.right_e_idx = slice(median + (len(self.tanks.engines) % 2), None)

        episode_duration = sum(self.tanks.heights * self.tanks.cross_section) \
                       / min(sum(self.tanks.pumps), sum(self.tanks.engines))
        self.episode_length = int(episode_duration / self.tstep)
        # Time/state-keeping
        self.t = 0.
        self.x = np.copy(self.tanks.heights)
        self.x *= (1 - 0.2 * self.np_random.rand(len(self.x)))
        # self.x = np.clip(self.x, np.zeros_like(self.tanks.heights), self.tanks.heights)
        return self.x


    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Simulate and respond to an action.
        
        Parameters
        ----------
        action : np.ndarray
            A vector indicating the state of each valve.
        
        Returns
        -------
        np.ndarray
            A vector containing the new fuel tank levels.
        """
        self.t += self.tstep
        x_next = self.solver.predict(self.tstep, self.x, [action])[0][-1]

        median_demand = 0.5 * self.tanks.engines[self.median_e_idx] * self.odd_e
        left_demand = sum(self.tanks.engines[self.left_e_idx]) + median_demand
        right_demand = sum(self.tanks.engines[self.right_e_idx]) + median_demand
        median_supply = 0.5 * x_next[self.median_idx] * self.odd
        left_supply = sum(x_next[self.left_idx] + median_supply)
        right_supply = sum(x_next[self.right_idx] + median_supply)
        done = (left_demand > left_supply) or (right_demand > right_supply) or self.t > self.episode_length
        
        reward, components = self.reward(self.t, self.x, action, x_next, done)
        self.x = x_next
        return self.x, reward, done, components


    def reward_components(self, t: float, x: np.ndarray, u: np.ndarray, \
        x_next: np.ndarray, done: bool) -> Tuple[float, float, float, float]:
        """
        Calculate individual reward components based on centre of gravity,
        number of valves open, variance of fuel distribution, and level of fuel.

        Parameters
        ----------
        t : float
             From start of episode.
        x : np.ndarray
             The previous state vector.
        u : np.ndarray
            The action vector.
        x_next : np.ndarray
            The current state vector.
        done : bool
            Whether the current state concludes an episode.
        
        Returns
        -------
        centre: float
            Centre of graviy relative to longitudinal axis.
        activity: float
            Average state of valves.
        spread: float
            Standard deviation of fuel mass distribution.
        level: float
            Average height of fuel in tanks.
        deficit: float
            Fraction of fuel demand from engines not met.
        """
        x_next = np.clip(x_next, 0, None)
        weights = x_next * self.tanks.cross_section
        total_weights = sum(weights)
        moments_counterclockwise = weights[self.left_idx] * self.left_arm
        moments_clockwise = weights[self.right_idx] * self.right_arm

        centre = sum(moments_clockwise - moments_counterclockwise) / (total_weights + 1e-5)
        centre /= self.median_idx
        if total_weights == 0: centre = 0.

        activity = np.mean(np.clip(u, self.tanks.valves_min, self.tanks.valves_max))
    
        spread = np.sqrt(sum((self.left_arm - centre)**2 * weights[self.left_idx]  \
                          + (self.right_arm - centre)**2 * weights[self.right_idx]) \
                        / (total_weights + 1e-5))
        spread /= self.max_spread
    
        level = np.sum(x_next) / np.sum(self.tanks.heights)
    
        fuel_lost = np.sum(x - x_next)
        fuel_required = np.sum(self.tanks.engines) * self.tstep
        deficit = np.clip((fuel_required - fuel_lost) / fuel_required, 0, 1)
        # TODO: No point in penalizing activity if the corresponding tank is empty,
        # penalty should be proportional (?) to fuel level.
        return centre, activity, spread, level, deficit


    def reward(self, t: float, x: np.ndarray, u: np.ndarray, x_next: np.ndarray,\
        done: bool) -> float:
        """
        Calculates the reward for going from state `x` via action `u` to state
        `x_next`.
        
        Parameters
        ----------
        t : float
             From start of episode.
        x : np.ndarray
             The previous state vector.
        u : np.ndarray
            The action vector.
        x_next : np.ndarray
            The current state vector.
        done : bool
            Whether the current state concludes an episode.
        
        Returns
        -------
        float
            The composite reward value.
        """
        centre, activity, spread, level, deficit = self.reward_components(t, x, u, x_next, done)
        # return -abs(centre) + spread
        return (
            (level + (2 * (1 - abs(centre)) + spread) / 2 - 0.25*activity) * (1 - deficit),
            dict(centre=centre, activity=activity, spread=spread, level=level, deficit=deficit)
        )
        # return level + 2 * (1 - abs(centre)) * spread - activity - deficit


    def set_parameters(self, **params):
        """
        Set parameters for the tanks system. For unspecified parameters, they
        are reset to defaults (`self.og_params`).

        Parameters
        ----------
        **params
            Keyword arguments where keywords can be one of TanksPhysicalEnv.params.
        """
        for p in self.params:
            val = params.get(p, self.og_params.get(p))
            if isinstance(val, (np.ndarray, list, tuple)):
                # copy so that object cannot be modified elsewhere
                # or changes to object in simulation to not have side-effects
                val = np.copy(val)
            elif val is None:
                continue
            setattr(self.tanks, p, val)


    def plot(self, agent=None, state0=None, plot='both'):
        backups = (self.x, self.t)
        self.x = state0
        plot_tanks(self, agent, plot)
        self.x, self.t = backups


class TanksDataEnv(TanksPhysicalEnv):
    """
    An `OpenAI Gym` environment which uses a data-driven model of fuel tanks.
    
    Parameters
    ----------
    tanks : TanksFactory
        An instance of the fuel tanks factory class.
    model: BaseEstimator
        A class instance that satisfies Scikit-Learn's Estimator API. It has a
        `predict([x, u]) -> [x]` method. It accepts a 2D array where each row
        is the concatenated state action vector, and returns a 2D array where
        each row is the next state.
    tstep : float, optional
        Time step of each action, by default 1e-1. Not used, only kept for
        compatibility with parent class.
    
    Attributes
    ----------
    solver: BaseEstimator
        The model provided at instantiation. There's no need for an integration/
        explicit solution as the data-model is doing it implicitly.
    """


    def __init__(self, tanks: TanksFactory, model: BaseEstimator, tstep: float=1e-1):
        super().__init__(tanks=tanks, tstep=tstep)
        self.solver = model


    def step(self, action: np.ndarray) -> np.ndarray:
        self.t += 1
        solver_input = np.concatenate((self.x, action)).reshape(1, -1)
        x_next = self.solver.predict(solver_input)[0]

        median_demand = 0.5 * self.tanks.engines[self.median_e_idx] * self.odd_e
        left_demand = sum(self.tanks.engines[self.left_e_idx]) + median_demand
        right_demand = sum(self.tanks.engines[self.right_e_idx]) + median_demand
        median_supply = 0.5 * x_next[self.median_idx] * self.odd
        left_supply = sum(x_next[self.left_idx] + median_supply)
        right_supply = sum(x_next[self.right_idx] + median_supply)
        done = (left_demand > left_supply) or (right_demand > right_supply)

        reward = self.reward(self.t, self.x, action, x_next, done)
        self.x = x_next
        return self.x, reward, done, {}



class TanksDataRecurrentEnv(TanksDataEnv):


    def __init__(self, tanks: TanksFactory, model: BaseEstimator, tstep: float=1e-1):
        super().__init__(tanks=tanks, model=model, tstep=tstep)
        self.h0c0 = None


    def reset(self):
        self.h0c0 = None
        return super().reset()


    def step(self, action):
        self.t += 1
        # pylint: disable=too-many-function-args
        solver_input = np.concatenate((self.x, action)).reshape(1, 1, -1)
        x_next, self.h0c0 = self.solver.predict(solver_input, self.h0c0)
        x_next = np.squeeze(x_next)
        done = sum(x_next) < sum(self.tanks.engines)
        reward = self.reward(self.t, self.x, action, x_next, done)
        self.x = x_next
        return self.x, reward, done, {}



def plot_tanks(env, agent=None, plot='both', columns=2, single_size=(6,4), legend=True):
    n_tanks = len(env.tanks.heights)
    if agent is not None:
        x, u, done = [], [], False
        x.append(env.reset())
        while not done:
            u_, _ = agent.predict(x[-1])
            x_, _, done, _ = env.step(u_)
            x.append(x_)
            u.append(u_)
        x, u = np.asarray(x), np.asarray(u)
        opened = u == 1
        episode_length = len(x)
        episode_duration = episode_length * env.tstep
    else:
        episode_duration = sum(env.tanks.heights * env.tanks.cross_section)\
                           / min(sum(env.tanks.pumps), sum(env.tanks.engines))
        episode_length = int(episode_duration / env.tstep)

    initial = None
    if plot in ('closed', 'both'):
        u_closed = np.zeros((episode_length, n_tanks))
        x_closed = np.zeros_like(u_closed)
        initial = env.reset()
        for i in range(len(u_closed)):
            x_closed[i] = env.step(u_closed[i])[0]
    if plot in ('open', 'both'):
        u_open = np.ones((episode_length, n_tanks))
        x_open = np.zeros_like(u_open)
        env.reset()
        if initial is not None:
            env.x = initial
        for i in range(len(u_open)):
            x_open[i] = env.step(u_open[i])[0]

    width, height = single_size
    rows = n_tanks // columns + (1 if n_tanks % columns else 0)
    figsize = (columns * width, rows * height)
    plt.figure(figsize=figsize)
    patches = None
    for i in range(n_tanks):
        plt.subplot(rows, columns, i+1)
        plt.ylim(0, 1.05 * max(env.tanks.heights))
        if plot in ('open', 'both'):
            plt.plot(x_open[:, i], '--', label='Open' if i==n_tanks-1 else None)
        if plot in ('closed', 'both'):
            plt.plot(x_closed[:, i], ':', label='Closed' if i==n_tanks-1 else None)
        if agent is not None:
            cmap = plt.cm.gray      # pylint: disable=no-member
            im = plt.imshow(opened[:, i].reshape(1, -1), aspect='auto', alpha=0.3,
                            extent=(*plt.xlim(), *plt.ylim()), origin='lower',
                            vmin=0, vmax=1, cmap=cmap)
            if len(np.unique(opened) == 2):
                colors = [ im.cmap(im.norm(value)) for value in (0, 1)]
                patches = [mpatches.Patch(color=colors[0], label="Closed", alpha=0.3),
                           mpatches.Patch(color=colors[1], label="Opened", alpha=0.3),]
            plt.plot(x[:, i], '-', label='RL' if i==n_tanks-1 else None)
        
        if i !=0 and i % columns !=0:
            plt.gca().set_yticklabels([])
        plt.ylabel('Tank ' + str(i + 1))
        if i >= columns * (rows-1) and legend: plt.xlabel('Time /s')
        if (i == n_tanks-2) and patches is not None and legend: plt.legend(handles=patches)
        if i==n_tanks-1 and legend: plt.legend()
        plt.grid(True)
        # plt.tight_layout()