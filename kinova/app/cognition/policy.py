# !/usr/bin/env python3
import torch
import torch.nn as nn
from torch.distributions import Normal



class StochasticActor(nn.Module):
    def __init__(self, model, channel_in, action_dim):
        super().__init__()
        
        self.model = model(channel_in, action_dim)
        self.feat = None
        self.imp = None


    def dist(self, state):
        self.feat, self.imp, res = self.model(state)
        means, log_stds = res.chunk(2, dim=-1)
        
        normal = Normal(means, log_stds.exp())

        return normal


    def forward(self, state, action):
        pi = self.dist(state)
        logp = pi.log_prob(action).sum(axis=-1)

        return pi, logp


    def train(self):
        self.model.train()


    def eval(self):
        self.model.eval()





if __name__ == '__main__':
    from network import ResidualEncoder

    ch = 3
    adim = 5
    batch = 6

    actor = StochasticActor(ResidualEncoder, 3, 5)
    
    for _ in range(3):
        state = torch.rand(1, ch, 128, 128)
        action = torch.rand(1, adim)

        pi, logp = actor(state, action)
        print(pi.mean, logp.shape)
