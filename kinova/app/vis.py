#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np



def ax_plot(axis, data, config):
    X = np.array(range(len(data))) + 1
    #data = data/data.max()

    axis.plot(X, data, config['symbol'], color=config['color'])
    axis.set_xticks(X)
    axis.grid()
    axis.set_xlabel(config['x'])
    axis.set_ylabel(config['y'])



if __name__ == "__main__":
    cum_reward = np.load('data/sceneA-II/cum_reward.npy')
    actor_loss = np.load('data/sceneA-II/actor_loss_evol.npy')
    critic_loss = np.load('data/sceneA-II/critic_loss_evol.npy')

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    ax_plot(ax[0], cum_reward, {'symbol':'-o', 'color':'black', 'x':'Episode', 'y':'Reward'})
    ax_plot(ax[1], actor_loss, {'symbol':'-x', 'color':'red', 'x':'Episode', 'y':'Actor Loss'})
    ax_plot(ax[2], critic_loss, {'symbol':'-+', 'color':'blue', 'x':'Episode', 'y':'Critic Loss'})

    plt.show()
