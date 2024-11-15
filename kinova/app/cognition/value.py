#!/usr/bin/env python3
import torch
import torch.nn as nn


# wrapper to sync format with actor class 
class CumRewardCritic(nn.Module):
    def __init__(self, model, channel_in, output_dim):
        super().__init__()

        self.model = model(channel_in, output_dim)
        self.feat = None
        self.imp = None

    def forward(self, state):
        self.feat, self.imp, res = self.model(state)

        return res


    def train(self):
        self.model.train()


    def eval(self):
        self.model.eval()






if __name__ == '__main__':
    from network import ValueEncoder


    img = torch.rand(1, 3, 128, 128)
    critic = CumRewardCritic(ValueEncoder, 3, 1)

    res = critic(img)
    print(res.shape, critic.feat.shape, critic.imp.shape)
