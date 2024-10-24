import gymnasium as gym
from tqdm import tqdm
from threading import Thread
from PyQt5 import QtWidgets

from wanocchio.buffer import RolloutBuffer
from wanocchio.policy import StochasticActor
from wanocchio.value import Crtic
from wanocchio.apprx import fc_apprx
from wanocchio.agent import AgentPPO

import matplotlib.pyplot as plt



def framework(axis, xdata, ydata, X, Y):
    axis.plot(xdata, ydata, '-o', color='b')
    axis.grid()
    axis.set_xlabel(X)
    axis.set_ylabel(Y)    


reward_ep = []
ac_ep = []
cr_ep = []
done_ep = []

env = gym.make("HalfCheetah-v5", render_mode='human', width=400, height=400)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
size = env.spec.max_episode_steps

N_episodes = 3
config = {'buffer':RolloutBuffer, 'buffer_size':size,
        'apprx':fc_apprx, 
        'actor':StochasticActor, 'actor_layer': 6, 'actor_lr':1e-4,
        'critic':Crtic, 'critic_layer':6, 'critic_lr':1e-2, 
        'epoch':12, 'clip':0.3}
    
agent = AgentPPO(state_dim, action_dim, config)

for i in tqdm(range(N_episodes)):
    state, info = env.reset()
    done = False

    R = 0
    timer = 0

    while not done:
        action, value, logp = agent.policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.buffer.push(state, action, reward, value, logp)
        state = next_state
        
        R += reward
        timer += 1
        
    _, last_value, _ = agent.policy(state)     
    
    if done:
        last_value = None
        
    actor_loss, critic_loss = agent.learn(last_value)
    
    print('Cumulative reward: {}'.format(R))
    print('Actor loss: {}'.format(actor_loss))
    print('Critic loss: {}'.format(critic_loss))
    print('Required steps: {}'.format(timer))
    
    reward_ep.append(R)
    ac_ep.append(actor_loss)
    cr_ep.append(critic_loss)
    done_ep.append(timer)
    
env.close()


fig, axis = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
ep = list(range(1, N_episodes+1))

framework(axis[0][0], ep, reward_ep, 'Steps', 'Cumulative Reward')
framework(axis[0][1], ep, ac_ep, 'Steps', 'Actor Loss')
framework(axis[1][0], ep, cr_ep, 'Steps', 'Critic Loss')
framework(axis[1][1], ep, done_ep, 'Steps', 'Required Steps')

plt.show()
