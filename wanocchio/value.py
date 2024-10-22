import torch
import torch.nn as nn
import numpy as np

from apprx import fc_apprx


class Crtic(nn.Module):
    def __init__(self, func_nn, state_dim, action_dim, N):
        super().__init__()
        # state-action value network
        self.q_value = func_nn(state_dim+action_dim, 1, N, nn.ReLU)
    
    
    def forward(self, state, action):
        x = self.q_value(torch.cat([state, action], dim=-1))
        
        return x
    
    
    
if __name__ == "__main__":
    state = torch.rand(10, 5)
    action = torch.rand(10, 3)
    critic = Crtic(fc_apprx, 5, 3, 2)
    
    val = critic(state, action)
    print(val)
    