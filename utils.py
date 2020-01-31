"""
Utility functions for data transformation.
"""



from typing import List, Tuple, Any

import gym
import numpy as np



def cache_function(x_cache: list, u_cache: list, d_cache: list, r_cache: list):
    """
    Creates cache of states, actions, rewards, and done signals after a
    `stable-baselines` agent finishes `learn()`-ing. A cache is a list of
    numpy arrays where each array corresponds to results from a single call to
    `agent.learn(timesteps)`. Each element in the array corresponds to an
    interaction with the environment. This returns a callback function that can
    be provided to the agent. The callback has reference to the lists provided
    which it populates with values.

    Parameters
    ----------
    x_cache : list
        Empty list that will be populated with states.
    u_cache : list
        Empty list that will be populated with actions.
    d_cache : list
        Empty list that will be populated with rewards.
    r_cache : list
        Empty list that will be populated with done signals (True -> episode end)

    Returns
    -------
    Callable
        A function that accepts a dictionary of local and global variables. The
        callable returns nothing.
    """
    def cache_experience(local_vars, global_vars):
        x_cache.append(local_vars.get('obs'))
        u_cache.append(local_vars.get('actions'))
        d_cache.append(local_vars.get('masks'))
        r_cache.append(local_vars.get('true_reward'))
    return cache_experience



def cache_to_training_set(x_cache: List[np.ndarray], u_cache: List[np.ndarray], \
    d_cache: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert cached actions and states into (state, action, next state) tuples.

    Parameters
    ----------
    x_cache : List[np.ndarray]
        List of state observations.
    u_cache : List[np.ndarray]
        List of actions taken.
    d_cache : List[np.ndarray]
        List of arrays of booleans indicating whether each time step is end
        of episode.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        An array of concatenated (state, action) tuples and an array of next state
        measurements.
    """
    x_cache = np.concatenate(x_cache, axis=0)
    u_cache = np.concatenate(u_cache, axis=0)
    d_cache = np.concatenate(d_cache, axis=0)

    # Assuming episode ends at last instance in batch,
    # makes it easy to split cached instances into
    # x and x_next.
    d_cache[-1] = True
    non_terminal_idx = np.nonzero(~d_cache)[0]

    # x is made of states except terminal or last state,
    # x_next is x delayed by 1 step
    x = x_cache[non_terminal_idx]
    u = u_cache[non_terminal_idx]
    x_next = x_cache[non_terminal_idx + 1]

    xu = np.concatenate((x, u), axis=1)

    return xu, x_next



def cache_to_episodes(cache: List[np.ndarray], d_cache: List[np.ndarray])\
    -> List[np.ndarray]:
    """
    Converts a cache of rewards to an array of total rewards per episode.

    Parameters
    ----------
    cache : List[np.ndarray]
        A list of arrays where each array element is a measurement per step.
    d_cache : List[np.ndarray]
        A list of arrays where each element is a boolean indicating whether that
        step is the last in an episode.

    Returns
    -------
    List[np.ndarray]
        A list of arrays such that each array corresponds to an episode.
    """
    cache = np.concatenate(cache, axis=0)
    d_cache = np.concatenate(d_cache, axis=0)
    terminal_idx = np.nonzero(d_cache)[0]
    episodic = [cache[:terminal_idx[0]]]
    for i in range(1, len(terminal_idx) - 1):
        episodic.append(cache[terminal_idx[i]: terminal_idx[i+1]])
    return episodic



def cache_to_episodic_rewards(r_cache: List[np.ndarray], d_cache: List[np.ndarray])\
    -> np.ndarray:
    """
    Converts a cache of rewards to an array of total rewards per episode.

    Parameters
    ----------
    r_cache : List[np.ndarray]
        A list of arrays where each element is the reward per step.
    d_cache : List[np.ndarray]
        A list of arrays where each element is a boolean indicating whether that
        step is the last in an episode.

    Returns
    -------
    np.ndarray
        An array where each element is the total reward per episode.
    """
    episodic = cache_to_episodes(r_cache, d_cache)
    return np.asarray([sum(ep) for ep in episodic])



def rewards_from_actions(env: gym.Env, u: List[Any]) -> float:
    """
    Returns the total rewards earned from an environment given a sequence of
    actions. Environment may be reset if episode ends.

    Parameters
    ----------
    env : gym.Env
        An OpenAI gym environment.
    u : List[Any]
        A sequence of actions that `env` accepts.
    
    Returns
    -------
    float
        Total reward gained.
    """
    env.reset()
    rewards = 0.
    for i in range(len(u)):
        _, r, done, _ = env.step(u[i])
        rewards += r
        if done: env.reset()
    return rewards
