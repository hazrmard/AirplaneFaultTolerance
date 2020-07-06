"""
Meta-learning functions.
"""

from itertools import combinations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

from utils import copy_mlp_regressor, copy_tensor
from ppo import PPO, ActorCriticBinary, Memory, DEVICE



def meta_update(starting_policy, env_, library, memory=None, n_inner=1, n_outer=1, alpha_inner=0.01,
                alpha_outer=0.1, interactive_env=True, **ppo_params):
    ppo_params['lr'] = alpha_outer
    # Outer Loop
    agent_outer = PPO(env_, **ppo_params)
    agent_outer.policy.load_state_dict(copy_tensor(starting_policy))
    lr_scheduler = CosineAnnealingLR(agent_outer.optimizer, T_max=n_outer)
    for _ in trange(n_outer, desc='Outer updates', leave=False):
        agent_outer.optimizer.zero_grad()
        
        # Inner Loop: Update gradents on buffered experience on current policy +
        # modeled experience on library of policies
        ppo_params['lr'] = alpha_inner
        for i, params in enumerate(tqdm([starting_policy] + library,
                                        desc='Library', leave=False)):
            agent_inner = PPO(env_, **ppo_params)
            agent_inner.policy.load_state_dict(copy_tensor(params))
            if i == 0 and memory is not None and interactive_env:
                # Get gradients of current policy w.r.t buffered experience on the
                # actual system. Using `memory` passed to function.
                agent_inner.update(agent_inner.policy, memory, epochs=1, optimizer=None)
                baseline_params = copy_tensor(agent_inner.policy.state_dict())
            else:
                # Get gradients of library of policies w.r.t experience on system model,
                # Do updates for n_inner steps. Using `memory_` from model.
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
                                               env_, agent_inner.policy)
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

            # Accumulate gradients of library of policies, and update main policy
            for param_outer, param_inner in zip(agent_outer.policy.parameters(), agent_inner.policy.parameters()):
                if param_outer.grad is None:
                    param_outer.grad = torch.zeros_like(param_outer) # pylint: disable=no-member
                param_outer.grad += param_inner.grad

        # Make parameter update after all policy gradients accumulated
        agent_outer.optimizer.step()
        lr_scheduler.step()

    # Evaluate whether adapted parameters outperform standard-RL
    if interactive_env:
        agent_inner.policy.load_state_dict(baseline_params)
        memory_ = Memory()
        agent_inner.experience(memory_, ppo_params['update_interval'],
                               env_, agent_inner.policy)
        rewards_baseline = sum(memory_.rewards)
        memory_ = Memory()
        agent_outer.experience(memory_, ppo_params['update_interval'],
                               env_, agent_inner.policy)
        rewards = sum(memory_.rewards)
        return agent_outer.policy.state_dict() if rewards > rewards_baseline else baseline_params

    return agent_outer.policy.state_dict()



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
    est_ = copy_mlp_regressor(est, warm_start=True)
    est_.set_params(verbose=verbose)
    est_.fit(x, y);
    return est_



def kl_div(memory, policy0, policy1, **ppo_params):
    Policy = ppo_params['policy'] # get the policy class
    policy = Policy(ppo_params['state_dim'], ppo_params['action_dim'],
                    ppo_params['n_latent_var']).to(DEVICE)
    s = torch.tensor(memory.states).float().to(DEVICE)
    a = torch.tensor(memory.actions).float().to(DEVICE)

    policy.load_state_dict(policy0)
    logprobs0, _, _ = policy.evaluate(s, a)
    p0 = torch.exp(logprobs0)

    policy.load_state_dict(policy1)
    logprobs1, _, _ = policy.evaluate(s, a)
    p1 = torch.exp(logprobs1)

    kl_0_1 = torch.sum(p0 * torch.log(p0 / p1)).item()
    kl_1_0 = torch.sum(p1 * torch.log(p1 / p0)).item()
    return kl_0_1, kl_1_0



def prune_library(library, library_size, memory, **ppo_params):
    divmat = np.zeros((len(library), len(library)))
    for p1, p2 in combinations(range(len(library)), 2):
        divmat[p1, p2], divmat[p2, p1] = kl_div(memory, library[p1], library[p2], **ppo_params)
    size = min(library_size, len(library))  # size of divergence matrix
    keep = np.argsort(np.sum(divmat, axis=1))[-size:]
    idx = np.ix_(keep, keep)  # generate indices to select range of axes
    return [library[k] for k in keep], divmat[idx]



def plot_adaption(r, r_b=None, std=None, std_b=None, faults=(0,), avg_window=10):
    r = np.convolve(r, np.ones(avg_window), 'valid') / avg_window
    if std is not None:
        std = np.convolve(std, np.ones(avg_window), 'valid') / avg_window
        minlen = len(r) if std is None else min(len(r), len(std))
        r, std = r[:minlen], std[:minlen]
        plt.fill_between(np.arange(len(r)), r + std, np.clip(r - std, 0, None), alpha=0.3)
    plt.plot(r, label='Meta RL')
    
    if r_b is not None:
        r_b = np.convolve(r_b, np.ones(avg_window), 'valid') / avg_window
        if std_b is not None:
            std_b = np.convolve(std_b, np.ones(avg_window), 'valid') / avg_window
            minlen = len(r_b) if std_b is None else min(len(r_b), len(std_b))
            r_b, std_b = r_b[:minlen], std_b[:minlen]
            plt.fill_between(np.arange(len(r_b)), r_b + std_b, np.clip(r_b - std_b, 0, None), alpha=0.3)
        plt.plot(r_b, label='Standard RL')
    
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