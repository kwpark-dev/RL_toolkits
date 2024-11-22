# !/usr/bin/env python3
import torch
import torch.nn as nn
from torch.distributions import Normal



class StochasticActor(nn.Module):
    def __init__(self, model, channel_in, action_dim, is_multi_head=False):
        super().__init__()
        
        self.model = model(channel_in, action_dim, is_multi_head)
        self.feat = None
        self.imp = None

        self.is_multi_head = is_multi_head


    def dist(self, state):
        if self.is_multi_head:
            self.feat, self.imp, context, res = self.model(state)
            means, log_stds = res.chunk(2, dim=-1)

            normal = Normal(means, log_stds.exp())

            return normal, context

        else:
            self.feat, self.imp, res = self.model(state)
            means, log_stds = res.chunk(2, dim=-1)
        
            normal = Normal(means, log_stds.exp())

            return normal


    def forward(self, state, action):
        if self.is_multi_head:
            pi, context = self.dist(state)
            logp = pi.log_prob(action).sum(axis=-1)
        
            return pi, logp, context

        else:
            pi, self.dist(state)
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

    actor = StochasticActor(ResidualEncoder, 3, 5, True)
    
    for _ in range(3):
        state = torch.rand(1, ch, 128, 128)
        action = torch.rand(1, adim)

        pi, logp, context = actor(state, action)
        print(pi.mean, logp.shape)
