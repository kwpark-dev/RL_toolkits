import torch
import torch.nn as nn
import numpy as np

# from apprx import fc_apprx


class Crtic(nn.Module):
    def __init__(self, func_nn, state_dim, N):
        super().__init__()
        # state value network
        self.q_value = func_nn(state_dim, 1, N, nn.Identity)
    
    
    def forward(self, state):
        x = self.q_value(state)
        
        return x
    
    
    
# if __name__ == "__main__":
    # state = torch.rand(10, 5)
    # action = torch.rand(10, 3)
    # critic = Crtic(fc_apprx, 5, 3, 2)
    
    # val = critic(state, action)
    # print(val)
    
