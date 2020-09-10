"""
Meta-learning functions.
"""

from itertools import combinations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from higher import innerloop_ctx
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import jensenshannon
from pytorchbridge import TorchEstimator    # pylint: disable=import-error

from utils import copy_mlp_regressor, copy_tensor, higher_dummy_context, get_gradients
from ppo import PPO, Memory, DEVICE, returns



def meta_update(starting_policy, env, library, memory=None,
                n_inner=1, n_outer=1, alpha_inner=0.01, alpha_outer=0.1,
                interactive_env=True, mode='fomaml', lib_grads=None, rank=-1,
                **ppo_params):
    ppo_params['lr'] = alpha_outer

    # Setup for each meta-update mode
    if mode == 'fomaml':
        pass
    elif mode == 'reptile':
        pass
    elif mode == 'maml':
        pass

    # Outer Loop: Update policy by evaluating library on new environment.
    agent_outer = PPO(env, **ppo_params)
    agent_outer.policy.load_state_dict(copy_tensor(starting_policy))
    lr_scheduler = CosineAnnealingLR(agent_outer.optimizer, T_max=n_outer)

    for _ in trange(n_outer, desc='Outer updates', leave=False):
        agent_outer.optimizer.zero_grad()
        
        # Inner Loop: Iterate over library of parameters to calculate update
        # step. Optionally fine-tune library under new task before calculating
        # gradients/new parameters.
        ppo_params['lr'] = alpha_inner

        if rank > 0:
            rankings = rank_policies(memory, library, **ppo_params)[0][:rank]
            policies = [starting_policy] + \
                       [policy for i, policy in enumerate(library) if i in rankings]
        else:
            policies = [starting_policy] + library # first item is current policy

        for i, params in enumerate(tqdm(policies, desc='Library', leave=False)):
            agent_inner = PPO(env, **ppo_params)
            agent_inner.policy.load_state_dict(copy_tensor(params))
            if i == 0 and memory is not None and interactive_env:
                # Get gradients of current policy (i=0) w.r.t buffered experience on the
                # actual system. Using `memory` passed to function.
                agent_inner.update(agent_inner.policy, memory, epochs=1, optimizer=None)
                baseline_params = copy_tensor(agent_inner.policy.state_dict())
            else:
                # Get gradients of library of policies w.r.t experience on system model,
                # Do updates for n_inner steps. Using `memory_` from model.
                # Iterate for n_inner+1 because the last iteration only accumulates
                # gradients and doesn't update.
                for j in trange(n_inner + 1, desc='Inner updates', leave=False):
                    agent_inner.optimizer.zero_grad()
                    # Create trajectory in new environment for each policy;
                    # If environment is interactive (in case of a data model)
                    # then sample new experiences from the env. Otherwise
                    # reuse buffered states/actions and just calculate their
                    # probabilities under the new policy.
                    memory_ = Memory()
                    if interactive_env:
                        agent_inner.experience(memory_, ppo_params['update_interval'],
                                               env, agent_inner.policy)
                    else:
                        memory_.states = memory.states
                        memory_.actions = memory.actions
                        memory_.rewards = memory.rewards
                        memory_.is_terminals = memory.is_terminals
                        # pylint: disable=not-callable
                        states = torch.tensor(memory.states).float().to(DEVICE).detach()
                        actions = torch.tensor(memory.actions).float().to(DEVICE).detach()
                        memory_.logprobs = agent_inner.policy.evaluate(states, actions)[0]
                    # Populate gradients of policy params by going over trajectory once
                    agent_inner.update(agent_inner.policy, memory_, epochs=1, optimizer=None)
                    # Update library policy for next epoch in meta update (except for last
                    # iteration where the gradients are on the test sample)
                    if j < n_inner - 1:
                        agent_inner.optimizer.step()

            # Get gradients calculated on the current parameters in the library
            for k, (param_outer, param_inner) in enumerate(zip(agent_outer.policy.parameters(),
                                                               agent_inner.policy.parameters())):
                if param_outer.grad is None:
                    param_outer.grad = torch.zeros_like(param_outer) # pylint: disable=no-member
                if mode == 'fomaml':
                    param_outer.grad += param_inner.grad
                elif mode == 'reptile':
                    param_outer.grad += param_inner - param_outer
                elif mode == 'maml':
                    if i > 0:
                        # i==0 is the starting policy. Therefore the gradient
                        # d theta / d theta == 1. We get gradients for policies
                        # in the library i.e. d theta n / d theta to complete
                        # the chain rule:
                        # d Loss/d theta = (d Loss / d theta n) * (d theta n / d theta)
                        param_outer.grad += param_inner.grad * lib_grads[i-1][k]
                    else:
                        param_outer.grad += param_inner.grad

        # Make parameter update after all policy gradients accumulated
        agent_outer.optimizer.step()
        lr_scheduler.step()

    # Evaluate whether adapted parameters outperform standard-RL
    if interactive_env:
        agent_inner.policy.load_state_dict(baseline_params)
        memory_ = Memory()
        agent_inner.experience(memory_, ppo_params['update_interval'],
                               env, agent_inner.policy)
        rewards_baseline = sum(memory_.rewards)
        memory_ = Memory()
        agent_outer.experience(memory_, ppo_params['update_interval'],
                               env, agent_inner.policy)
        rewards = sum(memory_.rewards)
        return agent_outer.policy.state_dict() if rewards > rewards_baseline else baseline_params

    return agent_outer.policy.state_dict()



def maml_initialize(starting_policy, env_fn, n, n_inner, alpha_inner, **ppo_params):
    timesteps = ppo_params.get('update_interval') * max(n_inner, 1)
    ppo_params['lr'] = alpha_inner
    library = []
    gradients = []
    env = env_fn(seed=ppo_params.get('seed'))
    for i in range(n):
        env.randomize()
        env.reset()
        agent = PPO(env, **ppo_params)
        agent.policy.load_state_dict(copy_tensor(starting_policy))
        agent.learn(timesteps, track_higher_gradients=True)
        library.append(agent.policy.state_dict())
        gradients.append(get_gradients(agent.meta_policy.parameters(),
                                       agent.meta_policy.parameters(time=0)))
    return library, gradients



def rank_policies(memory, library, **ppo_params):
    agent = PPO(None, **ppo_params)
    # pylint: disable=not-callable
    returns = torch.tensor(memory.returns).float().to(DEVICE).detach()
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    states = torch.tensor(memory.states).float().to(DEVICE).detach()
    actions = torch.tensor(memory.actions).float().to(DEVICE).detach()
    vals = []
    for params in library:
        agent.policy.load_state_dict(params)
        logp, _, _ = agent.policy.evaluate(states, actions)
        p = torch.exp(logp)
        vals.append(torch.sum(p * returns).item())
    return np.argsort(vals)[::-1], np.asarray(vals)



def learn_env_model(memory, est, verbose=False):
    # Convert episodes to training set (x=[x_t, u_t] -> y=[x_t+1])
    x, y = [], []
    for i, (d, s, a) in enumerate(zip(memory.is_terminals, memory.states, memory.actions)):
        if i == len(memory.is_terminals) - 1: break
        nd = memory.is_terminals[i+1]
        ns = memory.states[i+1]
        if nd: continue
        # action a can be a single number or an array
        x.append(np.concatenate((s, np.asarray(a).reshape(-1))))
        y.append(ns)
    x, y = np.asarray(x), np.asarray(y)
    if isinstance(est, MLPRegressor):
        est_ = copy_mlp_regressor(est, warm_start=True)
        est_.set_params(verbose=verbose)
    est_.fit(x, y);
    return est_



def distance(memory, policy0, policy1, kind='jensenshannon', **ppo_params):
    Policy = ppo_params['policy'] # get the policy class
    policy = Policy(ppo_params['state_dim'], ppo_params['action_dim'],
                    ppo_params['n_latent_var']).to(DEVICE)
    # pylint: disable=not-callable
    s = torch.tensor(memory.states).float().to(DEVICE)
    a = torch.tensor(memory.actions).float().to(DEVICE)

    policy.load_state_dict(policy0)
    logprobs0, _, _ = policy.evaluate(s, a)
    p0 = torch.exp(logprobs0).detach()

    policy.load_state_dict(policy1)
    logprobs1, _, _ = policy.evaluate(s, a)
    p1 = torch.exp(logprobs1).detach()

    if kind=='kldivergence':
        kl_0_1 = torch.sum(p0 * torch.log(p0 / p1)).item()
        kl_1_0 = torch.sum(p1 * torch.log(p1 / p0)).item()
        return kl_0_1, kl_1_0
    elif kind == 'jensenshannon':
        dist = jensenshannon(p0.cpu(), p1.cpu())
        return dist, dist



def prune_library(library, library_size, memory, **ppo_params):
    divmat = np.zeros((len(library), len(library)))
    for p1, p2 in combinations(range(len(library)), 2):
        divmat[p1, p2], divmat[p2, p1] = distance(memory, library[p1], library[p2], **ppo_params)
    size = min(library_size, len(library))  # size of divergence matrix
    keep = np.argsort(np.sum(divmat, axis=1))[::-1][:size]
    idx = np.ix_(keep, keep)  # generate indices to select range of axes
    return [library[k] for k in keep], divmat, idx



def plot_adaption(rs, stds, labels, faults=(0,), avg_window=10):
    for r, std, label in zip(rs, stds, labels):
        r = np.convolve(r, np.ones(avg_window), 'valid') / avg_window
        if std is not None:
            std = np.convolve(std, np.ones(avg_window), 'valid') / avg_window
            minlen = len(r) if std is None else min(len(r), len(std))
            r, std = r[:minlen], std[:minlen]
            plt.fill_between(np.arange(len(r)), r + std, np.clip(r - std, 0, None), alpha=0.3)
        plt.plot(r, label=label)
    
    if faults:
        for f in faults:
            plt.axvline(x=f, c='r')
    
    # TODO: Set proper text coordinates using plt.annotate()
    # plt.text(-30, 74, 'Fault', c='r', rotation=90)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Average episodic rewards after abrupt fault')
    plt.grid(True, 'both')
    plt.legend()