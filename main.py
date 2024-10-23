import gymnasium as gym

from wanocchio.buffer import RolloutBuffer
from wanocchio.policy import StochasticActor
from wanocchio.value import Crtic
from wanocchio.apprx import fc_apprx
from wanocchio.agent import AgentPPO



env = gym.make("HalfCheetah-v5", render_mode='human', width=1200, height=800)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

N_episodes = 10
size = env.spec.max_episode_steps
config = {'buffer':RolloutBuffer, 'buffer_size':size,
          'apprx':fc_apprx, 
          'actor':StochasticActor, 'actor_layer': 6, 'actor_lr':1e-3,
          'critic':Crtic, 'critic_layer':4, 'critic_lr':1e-3, 
          'epoch':4, 'clip':0.1}
    
agent = AgentPPO(state_dim, action_dim, config)

for i in range(N_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action, value, logp = agent.policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.buffer.push(state, action, reward, value, logp)
        state = next_state
        
    _, last_value, _ = agent.policy(state)     
    
    if done:
        last_value = None
        
    agent.learn(last_value)
    
env.close()
       
# for _ in range(1000):
#     action = env.action_space.sample()  # Sample random actions
#     observation, reward, done, truncated, info = env.step(action)

#     if done or truncated:
#         observation, info = env.reset()

env.close()
