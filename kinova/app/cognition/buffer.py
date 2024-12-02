#!/usr/bin/env python3

import numpy as np
import torch



class ReplayBuffer:
    def __init__(self):
        pass
    
    

class RolloutBuffer:
    # I modify the original code from following: 
    # https://github.com/upb-lea/reinforcement_learning_course_materials/blob/master/exercises/solutions/ex12/DDPG_and_PPO.ipynb
    # Note that GAE is employed to estimate advantage function, see https://arxiv.org/abs/1506.02438, lambda = 1 case.
    def __init__(self, size, state_dim, action_dim):
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32) 
        self.state_buf = np.zeros((size, *state_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size+1, dtype=np.float32)
        self.val_buf = np.zeros(size+1, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.i = 0
        self.size = size

    
    def push(self, state, action, reward, value, logp):
        if state.dtype == 'uint8':
            state = state.astype('float32')/255.0

        self.state_buf[self.i] = state
        self.action_buf[self.i] = action
        self.rew_buf[self.i] = reward
        self.val_buf[self.i] = value
        self.logp_buf[self.i] = logp

        self.i += 1
        
        
    def fetch(self, last_val=None, gamma=0.999):
        if last_val is not None:
            self.rew_buf[self.i] = last_val
            self.val_buf[self.i] = last_val
        
        rewards = self.rew_buf[:self.i+1]
        values = self.val_buf[:self.i+1]
        # create matrix to calc sum of power series
        w = np.arange(rewards.size - 1)
        w = w - w[:, np.newaxis]
        w = np.triu(gamma ** w.clip(min=0)).T
        # estimate TD error at each step and multiply it with the above
        advantages = rewards[:-1] + gamma * values[1:] - values[:-1]
        advantages = (advantages.reshape(-1, 1) * w).sum(axis=0)
        
        rewards_to_go = ((gamma*rewards[:-1]).reshape(-1, 1) * w).sum(axis=0)
        
        advantages = (advantages - advantages.mean()) / np.std(advantages)

        data =  {'state': self.state_buf[:self.i],
                 'action': self.action_buf[:self.i],
                 'rewards_to_go': rewards_to_go.copy(),  # make stride positive
                 'advantages': advantages, 
                 'logp': self.logp_buf[:self.i]}
        
        self.i = 0
        

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
    
    
    

