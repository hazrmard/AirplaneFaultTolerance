"""
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch.utils.tensorboard import SummaryWriter
import gym
from tqdm.auto import trange

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:


    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class ActorCriticBinary(nn.Module):


    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_latent_var = n_latent_var

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Sigmoid()
                )
        
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )


    def forward(self):
        raise NotImplementedError


    def predict(self, state):
        state = torch.from_numpy(state).float().to(DEVICE)
        action_probs = self.action_layer(state)  # [Multi]Binary
        dist = Bernoulli(action_probs)
        action = dist.sample()
        return action.squeeze().cpu().numpy(), dist.log_prob(action).sum(-1).item()


    def evaluate(self, state, action):
        action_probs = self.action_layer(state)  # [Multi]Binary
        dist = Bernoulli(action_probs)
        
        action_logprobs = dist.log_prob(action).sum(-1)
        dist_entropy = dist.entropy().sum(-1) # TODO, sum entropy over variables
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy



class ActorCriticDiscrete(nn.Module):


    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_latent_var = n_latent_var

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )


    def forward(self):
        raise NotImplementedError


    def predict(self, state):
        # pylint: disable=no-member
        state = torch.from_numpy(state).float().to(DEVICE)
        action_probs = self.action_layer(state)  # Discrete
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.squeeze().cpu().item(), dist.log_prob(action).sum(-1).item()


    def evaluate(self, state, action):
        action_probs = self.action_layer(state)  # Discrete
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action).sum(-1)
        dist_entropy = dist.entropy().sum(-1) # TODO, sum entropy over variables
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

        
class PPO:


    def __init__(self, env, policy, state_dim, action_dim, n_latent_var=64, lr=0.02,
                 betas=(0.9, 0.999), gamma=0.99, epochs=5, eps_clip=0.2,
                 update_interval=2000, summary: SummaryWriter=None):
        self.env = env
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.update_interval = update_interval
        
        self.policy = policy(state_dim, action_dim, n_latent_var).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.MseLoss = nn.MSELoss()

        self.summary = summary
    

    def update(self, policy, memory, epochs: int=1, optimizer=None, summary=None):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Casting to correct data type and DEVICE
        # pylint: disable=not-callable
        rewards = torch.tensor(rewards).float().to(DEVICE)
        old_states = torch.tensor(memory.states).float().to(DEVICE).detach()
        old_actions = torch.tensor(memory.actions).float().to(DEVICE).detach()
        old_logprobs = torch.tensor(memory.logprobs).float().to(DEVICE).detach()

        # Normalizing the rewards:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Optimize policy for K epochs:
        for e in range(epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2)
            loss += 0.5*self.MseLoss(state_values, rewards)
            # loss -= 0.01*dist_entropy
            
            # Take gradient step. If optimizer==None, then just backpropagate
            # gradients.
            if optimizer is not None:
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
            else:
                loss.mean().backward()


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


    def learn(self, timesteps, update_interval=None):
        if update_interval is None:
            update_interval = self.update_interval
        state = self.env.reset()
        memory = Memory()
        episodic_rewards = [0.]
        for t in trange(1, timesteps + 1, leave=False):
            
            # Running policy:
            memory.states.append(state)
            action, logprob = self.policy.predict(state)
            # print('a', action, 'log(p(a))', logprob)
            state, reward, done, _ = self.env.step(action)
            episodic_rewards[-1] += reward
            if done:
                state = self.env.reset()
                episodic_rewards.append(0.)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if t % update_interval == 0:
                self.update(self.policy, memory, self.epochs, self.optimizer, self.summary)
                memory.clear_memory()
        return episodic_rewards[:-1 if len(episodic_rewards) > 1 else None]


    def predict(self, state):
        return self.policy.predict(state)

