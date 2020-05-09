"""
Plotting functions for jupyter notebook.
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def plot_tanks(env, agent=None, plot='both'):
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


    if plot in ('closed', 'both'):
        u_closed = np.zeros((episode_length, n_tanks))
        x_closed = np.zeros_like(u_closed)
        env.reset()
        for i in range(len(u_closed)):
            x_closed[i] = env.step(u_closed[i])[0]
    if plot in ('open', 'both'):
        u_open = np.ones((episode_length, n_tanks))
        x_open = np.zeros_like(u_open)
        env.reset()
        for i in range(len(u_open)):
            x_open[i] = env.step(u_open[i])[0]

    # plt.figure(figsize=(12, 12))
    patches = None
    for i in range(n_tanks):
        plt.subplot(n_tanks // 2, 2, i+1)
        plt.ylim(0, 1.05 * max(env.tanks.heights))
        if plot in ('open', 'both'):
            plt.plot(x_open[:, i], '--', label='Open' if i==n_tanks-1 else None)
        if plot in ('closed', 'both'):
            plt.plot(x_closed[:, i], ':', label='Closed' if i==n_tanks-1 else None)
        if agent is not None:
            cmap = plt.cm.gray
            im = plt.imshow(opened[:, i].reshape(1, -1), aspect='auto', alpha=0.3,
                            extent=(*plt.xlim(), *plt.ylim()), origin='lower',
                            vmin=0, vmax=1, cmap=cmap)
            if len(np.unique(opened) == 2):
                colors = [ im.cmap(im.norm(value)) for value in (0, 1)]
                patches = [mpatches.Patch(color=colors[0], label="Closed", alpha=0.3),
                           mpatches.Patch(color=colors[1], label="Opened", alpha=0.3),]
            plt.plot(x[:, i], '-', label='RL' if i==n_tanks-1 else None)
        plt.ylabel('Tank ' + str(i + 1))
        if i >= 4: plt.xlabel('Time /s')
        if (i == n_tanks-2) and patches is not None: plt.legend(handles=patches)
        if i==n_tanks-1: plt.legend()
        plt.grid(True)
