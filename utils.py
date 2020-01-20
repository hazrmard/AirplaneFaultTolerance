"""
Utility functions for data transformation.
"""



import numpy as np



def cache_function(x_cache: list, u_cache: list, d_cache: list, r_cache: list):
    def cache_experience(local_vars, global_vars):
        x_cache.append(local_vars.get('obs'))
        u_cache.append(local_vars.get('actions'))
        d_cache.append(local_vars.get('masks'))
        r_cache.append(local_vars.get('true_reward'))
    return cache_experience



def cache_to_training_set(x_cache, u_cache, d_cache):
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



def rewards_from_actions(env, u):
    env.reset()
    rewards = 0.
    for i in range(len(u)):
        _, r, done, _ = env.step(u[i])
        rewards += r
        if done: env.reset()
    return rewards
