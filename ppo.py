"""
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""
from typing import Callable, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Bernoulli, Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from higher import innerloop_ctx
from higher.optim import DifferentiableOptimizer
import gym
import numpy as np
from tqdm.auto import trange

from utils import higher_dummy_context

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:


    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.returns = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.returns[:]
        del self.is_terminals[:]



class Policy(nn.Module):

    dist = Distribution
    dist_kwargs = {}


    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var

        self.base = None

        self.value_layer = nn.Sequential(
                    nn.Linear(state_dim, n_latent_var),
                    nn.Tanh(),
                    nn.Linear(n_latent_var, n_latent_var),
                    nn.Tanh(),
                    nn.Linear(n_latent_var, 1)
                    )
        
        self.action_layer = lambda x: x


    def forward(self, state, action):
        return self.evaluate(state, action)


    def predict(self, state: np.ndarray):
        # pylint: disable=no-member
        state = torch.from_numpy(state).float().to(DEVICE)
        action_probs = self.action_layer(state)  # Discrete
        dist = self.dist(action_probs, **self.dist_kwargs)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu()


    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        action_probs = self.action_layer(state)  # Discrete
        dist = self.dist(action_probs, **self.dist_kwargs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() # TODO, sum entropy over variables
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy



class ActorCriticDiscrete(Policy):

    dist = Categorical
    dist_kwargs = {}

    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__(state_dim=state_dim, action_dim=action_dim, n_latent_var=n_latent_var)

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )


    def predict(self, state):
        action, log_prob = super().predict(state)
        return action.item(), log_prob.item()


class ActorCriticMultiBinary(Policy):

    dist = Bernoulli
    dist_kwargs = {}


    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__(state_dim, action_dim, n_latent_var)

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Sigmoid()
                )


    def predict(self, state):
        action, log_prob = super().predict(state)
        return action.numpy(), log_prob.sum(-1).item()


    def evaluate(self, state, action):
        action_logprobs, state_value, dist_entropy = \
            super().evaluate(state, action)
        return action_logprobs.sum(-1), state_value, dist_entropy



class ActorCriticBox(Policy):

    dist = MultivariateNormal
    dist_kwargs = {}


    def __init__(self, state_dim, action_dim, n_latent_var, action_std=0.01,
            activation=nn.Tanh):
        super().__init__(state_dim=state_dim, action_dim=action_dim, n_latent_var=n_latent_var)
        self.activation = activation
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                self.activation()
                )
        self.dist_kwargs = dict(covariance_matrix=(torch.eye(action_dim) * action_std).to(DEVICE))


    def predict(self, state):
        action, log_prob = super().predict(state)
        # return torch.clamp(action, -1., 1.).numpy(), log_prob.item()
        return action.numpy(), log_prob.item()


        
class PPO:


    def __init__(self, env, policy, state_dim, action_dim, n_latent_var=64, lr=0.02,
                 betas=(0.9, 0.999), gamma=0.99, epochs=5, eps_clip=0.2,
                 update_interval=2000, seed=None, summary: SummaryWriter=None,
                 **policy_kwargs):
        self.env = env
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.update_interval = update_interval
        
        self.policy = policy(state_dim, action_dim, n_latent_var, **policy_kwargs).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.MseLoss = nn.MSELoss()

        self.random = np.random.RandomState(seed)
        self.seed = self.random.rand() if seed is None else seed
        self.summary = summary
        self.meta_policy = None
        if seed is not None: torch.manual_seed(self.seed)

    

    def update(self, policy, memory, epochs: int=1, optimizer=None, summary=None,
        grad_callback=None):

        rewards = returns(memory.rewards, memory.is_terminals, self.gamma)
        truncate = len(rewards)
        # Casting to correct data type and DEVICE
        # pylint: disable=not-callable
        rewards = torch.tensor(rewards[:truncate]).float().to(DEVICE)
        old_states = torch.tensor(memory.states[:truncate]).float().to(DEVICE).detach()
        old_actions = torch.tensor(memory.actions[:truncate]).float().to(DEVICE).detach()
        old_logprobs = torch.tensor(memory.logprobs[:truncate]).float().to(DEVICE).detach()

        # If states/actions are 1D arrays of single number states/actions,
        # convert them to 2D matrix of 1 column where each row is one timestep.
        # This is to make sure the 0th dimension always indexes time, and the
        # last dimension indexes feature.
        if policy.state_dim == 1 and old_states.ndim == 1:
            old_states = old_states.unsqueeze(dim=-1)
        if policy.action_dim == 1 and old_actions.ndim == 1:
            old_actions = old_actions.unsqueeze(dim=-1)

        # Normalizing the rewards:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Optimize policy for K epochs:
        for e in range(epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = policy(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) # policy loss
            loss += 0.5 * self.MseLoss(state_values, rewards) # value loss
            loss -= 0.01 * dist_entropy  # randomness (exploration)
            loss = loss.mean()
            
            # Take gradient step. If optimizer==None, then just backpropagate
            # gradients.
            if optimizer is not None:
                # the 'higher' library wraps optimizers to make parameters
                # differentiable w.r.t earlier versions of parameters. Those
                # optimizers to not need `backward()` and `zero_grad()`
                if not isinstance(optimizer, DifferentiableOptimizer):
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    if grad_callback is not None:
                        apply_grad_callback(policy, grad_callback)
                    optimizer.step()
                else:
                    optimizer.step(loss, grad_callback=grad_callback)
            else:
                loss.backward(retain_graph=True)
                if grad_callback is not None:
                    apply_grad_callback(policy, grad_callback)
        return loss


    def experience(self, memory, timesteps, env, policy, state0=None):
        state = env.reset() if state0 is None else state0
        for t in range(timesteps):
            # Running policy:
            memory.states.append(state)
            action, logprob = policy.predict(state)
            # print('a', action, 'log(p(a))', logprob)
            state, reward, done, _ = env.step(action)
            if done:
                state = env.reset()
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
        memory.returns = returns(memory.rewards, memory.is_terminals, self.gamma)


    def learn(self, timesteps, update_interval=None, track_higher_grads=False,
              lr_scheduler=None, callback=None, reward_aggregation='episodic'):
        if update_interval is None:
            update_interval = self.update_interval
        state = self.env.reset()
        memory = Memory()
        episodic_rewards = [0.]
        interval_rewards = [0.]
        # This context wraps the policy and optimizer to track parameter updates
        # over time such that d Params(time=t) / d Params(time=t-n) can be calculated.
        # If not tracking higher gradients, a dummy context is used which does
        # nothing.
        with innerloop_ctx(self.policy, self.optimizer, track_higher_grads=track_higher_grads,
                           copy_initial_weights=False) as (policy, optimizer):

            for t in trange(1, int(timesteps) + 1, leave=False):
                # Running policy:
                memory.states.append(state)
                action, logprob = policy.predict(state)
                state, reward, done, _ = self.env.step(action)
                episodic_rewards[-1] += reward
                interval_rewards[-1] += reward
                if done:
                    state = self.env.reset()
                    episodic_rewards.append(0.)
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                # update if its time
                if t % update_interval == 0:
                    interval_rewards.append(0.)
                    loss = self.update(policy, memory, self.epochs, optimizer,
                                       self.summary)
                    if callback is not None:
                        callback(locals())
                    if lr_scheduler is not None:
                        lr_scheduler()
                    memory.clear()

        
            self.meta_policy = policy if track_higher_grads else None
        self.policy.load_state_dict(policy.state_dict())

        if reward_aggregation == 'episodic':
            return episodic_rewards[:-1 if len(episodic_rewards) > 1 else None]
        else:
            return interval_rewards


    def predict(self, state):
        return self.policy.predict(state)



def returns(rewards, is_terminals, gamma, truncate=False):
    # Monte Carlo estimate of state rewards:
    # Discarding rewards for incomplete episodes because their returns
    # will be inaccurate. For e.g. if episode rewards = 1,1,1,1,1 but the current
    # batch only has rewards for first 3 steps =1,1,1, then the returns=3,2,1,
    # where they should be 5,4,3
    if truncate:
        if True in is_terminals:
            idx_from_end = is_terminals[::-1].index(True)
            if idx_from_end > 0:
                rewards = rewards[:-idx_from_end]
                is_terminals = is_terminals[:-idx_from_end]
    returns = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)
    return returns



def apply_grad_callback(model: nn.Module, callback: Callable[[List[Tensor]], List[Tensor]]):
    new_grads = callback([p.grad for p in model.parameters()])
    for p, ng in zip(model.parameters(), new_grads):
        p.grad = ng
