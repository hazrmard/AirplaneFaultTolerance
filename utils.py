"""
Utility functions for data transformation and computations.
"""
from typing import List, Tuple, Any, Union, Dict, Iterable
from collections import OrderedDict
import multiprocessing as mp
from copy import deepcopy
import pickle
import pathlib

import gym
import numpy as np
from sklearn.neural_network import MLPRegressor
import torch
from torch.autograd import grad
from pytorchbridge import TorchEstimator



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
    d_cache: List[np.ndarray], mode: str='closed') -> Tuple[np.ndarray, np.ndarray]:
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
    mode: str
        If "open", then values at indices where d_cache is True will be included
        at the end of the previous episode. If "closed", those values will be
        included at the start of the next episode.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        An array of concatenated (state, action) tuples and an array of next state
        measurements.
    """
    x_ep = cache_to_episodes(x_cache, d_cache, mode)
    u_ep = cache_to_episodes(u_cache, d_cache, mode)

    xu, x_next = [], []
    for x, u in zip(x_ep, u_ep):
        if len(x) < 1: continue
        xu_ = np.concatenate((x[:-1], u[:-1]), axis=1)
        x_next_ = x[1:]
        xu.append(xu_)
        x_next.append(x_next_)

    return np.concatenate(xu, axis=0), np.concatenate(x_next, axis=0)



def cache_to_episodes(cache: List[np.ndarray], d_cache: List[np.ndarray],
    mode: str='closed', discard_trailing: bool=True) -> List[np.ndarray]:
    """
    Converts a cache of rewards to an array of total rewards per episode.

    Parameters
    ----------
    cache : List[np.ndarray]
        A list of arrays where each array element is a measurement per step.
    d_cache : List[np.ndarray]
        A list of arrays where each element is a boolean indicating whether that
        step is the last in an episode.
    mode: str
        If "open", then values at indices where d_cache is True will be included
        at the end of the previous episode. If "closed", those values will be
        included at the start of the next episode.
    discard_trailing: bool
        Whether to discard elements at the end of cache corresponding to an unfinished
        episode.

    Returns
    -------
    List[np.ndarray]
        A list of arrays such that each array corresponds to an episode.
    """
    cache = np.concatenate(cache, axis=0)
    d_cache = np.concatenate(d_cache, axis=0)
    if True in d_cache and discard_trailing:
        idx_from_end = np.where(d_cache==True)[0][-1]
        if idx_from_end > 0:
            cache = cache[:-idx_from_end]
            d_cache = d_cache[:-idx_from_end]
    
    terminal_idx = np.nonzero(d_cache)[0]
    if len(terminal_idx) == 0: # no episode end
        return [cache]
    if terminal_idx[-1] != len(cache) - 1:      # Include trailing end of cache
        terminal_idx = np.hstack((terminal_idx, (len(cache) - 1,)))
    if mode == 'open':                         # Include values at indices where
        terminal_idx[terminal_idx != 0] += 1   # d_cache==True in previous episode
    elif mode == 'closed':
        pass
    else:
        raise ValueError('Only "open" and "closed" mode supported.')
    episodic = [cache[:terminal_idx[0]]]
    for i in range(0, len(terminal_idx) - 1):
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




def homogenous_array(arrays: List[Iterable], start_align=True) -> np.ndarray:
    """
    Convert a list of 1D arrays of multiple lengths into a 2D array padded with
    zeros.

    Parameters
    ----------
    arrays : List[Iterable]
        List of 1D iterables.
    start_align : bool, optional
        Whether to align all arrays' start positions, by default True

    Returns
    -------
    np.ndarray
        A 2D array of size len(arrays) x max array length
    """
    maxlen = max(map(len, arrays))
    res = np.zeros((len(arrays), maxlen))
    for i, arr in enumerate(arrays):
        if start_align:
            res[i, :len(arr)] = arr
        else:
            res[i, -len(arr):] = arr
    return res



def copy_tensor(t: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]) \
    -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Make a copy of a tensor or a state_dict such that it is detached from the
    computation graph and does not share underlying data.

    Parameters
    ----------
    t : Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]
        A tensor or a dictionary of [name, torch.Tensor]

    Returns
    -------
    Union[torch.Tensor, Dict[str, torch.Tensor]]
        Same object as t
    """
    if isinstance(t, OrderedDict):
        return OrderedDict([(k, v.clone().detach()) for k, v in t.items()])
    elif isinstance(t, torch.Tensor):
        return t.clone().detach()
    elif isinstance(t, (list, tuple)):
        return [t_.clone().detach() for t_ in t]
    else:
        raise TypeError('Only OrderedDict or Tensor supported')



def copy_mlp_regressor(est: MLPRegressor, **params) -> MLPRegressor:
    if isinstance(est, MLPRegressor):
        est_ = deepcopy(est)
        est_.set_params(**params)
    elif isinstance(est, TorchEstimator):
        est_ = deepcopy(est)
        est_.module.load_state_dict(copy_tensor(est.module.state_dict()))
        # Handle references to parameters in optimizer as well.
        raise NotImplementedError('torch.nn.Module copy not implemented yet.')
    return est_



def vectorize_parameters(p: Union[Dict, Iterable[torch.Tensor]]) -> torch.Tensor:
    """
    Convert a state dict or a list of tensors into a 1D vector.

    Parameters
    ----------
    p : Union[Dict, Iterable[torch.Tensor]]
        The result of a `module.state_dict()` or `module.parameters()` containing
        `torch.Tensor`s.

    Returns
    -------
    torch.Tensor
        A 1D vector of flattened tensors.
    """
    if isinstance(p, (dict, OrderedDict)):
        plist = []
        for _, param in p.items():
            plist.append(param.flatten())
    else:
        plist = list(map(torch.flatten, p))
    return torch.cat(plist)



def get_gradients(params: Iterable[torch.Tensor], wrtparams: Iterable[torch.Tensor])\
    -> Iterable[torch.Tensor]:
    """
    Generate dictionary or list of gradients of provided tensors.

    Parameters
    ----------
    p : Iterable[torch.Tensor]
        The result of `module.parameters()` containing `torch.Tensor`s which
        have gradients.

    Returns
    -------
    Iterable[torch.Tensor]
        The gradients of the input in the same structure.
    """
    gradients = []
    for param, wrt in zip(params, wrtparams):
        g = grad(param.sum(), wrt, allow_unused=True, retain_graph=True)[0]
        gradients.append(g)
    return gradients



def get_difference(pto: Union[Dict, Iterable[torch.Tensor]],
        pfrom: Union[Dict, Iterable[torch.Tensor]]) -> \
        Union[Dict, Iterable[torch.Tensor]]:
    if isinstance(pto, (dict, OrderedDict)):
        ddict = OrderedDict()
        for key in pto:
            ddict[key] = pto[key] - pfrom[key]
        return ddict
    else:
        glist = []
        for paramto, paramfrom in zip(pto, pfrom):
            glist.append(paramto - paramfrom)
        return glist



def sanitize_filename(fname: str) -> str:
    fname = fname.replace(':', '')
    fname = fname.replace("'", '')
    fname = fname.replace(' ', '')
    fname = fname.replace(',', '-')
    fname = fname.replace('{', '')
    fname = fname.replace('}', '')
    fname = fname.replace('(', '')
    fname = fname.replace(')', '')
    fname = fname.replace('[', '_')
    fname = fname.replace(']', '_')
    return fname



def read_pickle(fname: Union[str, pathlib.Path]):
    if isinstance(fname, pathlib.Path):
        if not fname.name.endswith('.pickle'):
            fname = fname.parent / (fname.name + '.pickle')
    elif not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    return res



def write_pickle(data, fname: Union[str, pathlib.Path]):
    if isinstance(fname, pathlib.Path):
        if not fname.name.endswith('.pickle'):
            fname = fname.parent / (fname.name + '.pickle')
    elif not fname.endswith('.pickle'):
        fname += '.pickle'
    with open(fname, 'wb') as f:
        res = pickle.dump(data, f)


def rollmean(arr, window) -> np.ndarray:
    return np.convolve(np.ones(window), arr, 'valid') / window

class higher_dummy_context:


    def __init__(self, model, optimizer, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer

        
    def __enter__(self):
        return self.model, self.optimizer


    def __exit__(self, *args):
        return None