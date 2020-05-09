"""
Defines the model for a fuel tank system for an airplane.
"""


from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator
import gym

from pystatespace import Trapezoid, SS_ODE



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
        Maximum capacity of pumps (m^3/s), by default np.ones(6)*0.1
    engines : np.ndarray, optional
        Fuel demand of engines (m^3/s), by default np.ones(6)*0.1

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
                 resistances: np.ndarray=np.ones(6) * 1e2,
                 pumps: np.ndarray=np.ones(6) * 0.1,
                 engines: np.ndarray=np.ones(6) * 0.1):
        n = n                     # Number of tanks
        e = e                     # Number of engines
        self.rho = rho                 # Density of fluid
        self.g = g                  # Gravitational acceleration
        self.heights = np.copy(heights)      # height of tanks
        self.cross_section = np.copy(cross_section)
        self.valves_min = np.copy(valves_min )      # valve capacities on each tank
        self.valves_max = np.copy(valves_max)       # valve capacities on each tank
        self.resistances = np.copy(resistances)  # valve resistances where flowrate = g.height/resistance
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
                demand_left -= demand_left
                demand_right -= demand_right
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


    def __init__(self, tanks: TanksFactory, tstep: float=1e-1):
        super().__init__()
        self.tanks = tanks
        self.tstep = tstep
        self.n = len(tanks.pumps)
        self.solver = Trapezoid(dims=self.n, outputs=self.n, dx=tanks.dxdt,
                                out=tanks.y, tstep=tstep)
        self.action_space = gym.spaces.MultiBinary(self.n)
        self.observation_space = gym.spaces.Box(0, np.inf, (self.n,), dtype=np.float32)

        median = len(tanks.heights) // 2
        self.left_idx = slice(None, median)
        self.right_idx = slice(median + (len(tanks.heights) % 2), None)
        self.left_arm = np.arange(median, 0, -1)
        self.right_arm = np.arange(1, median + 1)
        self.max_arm = abs(max(self.right_arm))

        self.x = None
        self.t = None
        self.reset()


    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        
        Returns
        -------
        np.ndarray
            The initial state vector.
        """
        self.t = 0.
        self.x = np.copy(self.tanks.heights)
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
        x_next = self.solver.predict(self.tstep, self.x, action)[0][-1]
        done = sum(x_next) < sum(self.tanks.engines)
        reward = self.reward(self.t, self.x, action, x_next, done)
        self.x = x_next
        return self.x, reward, done, {}


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
        """
        x_next = np.clip(x_next, 0, None)
        weights = x_next * self.tanks.cross_section
        total_weights = sum(weights)
        moments_counterclockwise = weights[self.left_idx] * self.left_arm
        moments_clockwise = weights[self.right_idx] * self.right_arm
        centre = sum(moments_clockwise - moments_counterclockwise) / total_weights
        if total_weights == 0: centre = 0.
        activity = np.mean(u)
        spread = np.sqrt(sum((self.left_arm - centre)**2 * weights[self.left_idx]  \
                          + (self.right_arm - centre)**2 * weights[self.right_idx]) \
                        / total_weights)
        level = np.sum(x_next)
        return centre, activity, spread, level


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
        centre, activity, spread, level = self.reward_components(t, x, u, x_next, done)
        # return -abs(centre) + spread
        return (1 - abs(centre) / self.max_arm) * spread - activity



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
        done = sum(x_next) < sum(self.tanks.engines)
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
        solver_input = np.concatenate((self.x, action)).reshape(1, 1, -1)
        x_next, self.h0c0 = self.solver.predict(solver_input, self.h0c0)
        x_next = np.squeeze(x_next)
        done = sum(x_next) < sum(self.tanks.engines)
        reward = self.reward(self.t, self.x, action, x_next, done)
        self.x = x_next
        return self.x, reward, done, {}
