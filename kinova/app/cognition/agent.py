#!/usr/bin/env python3

import torch
import torch.optim as optim


# config:{
#         buffer:{name, size}
#         actor:{actor, model, lr, epoch}
#         critic:{model, lr, epoch}
#         ppo:{clip}
# }


class AgentPPO:
    def __init__(self, state_dim, action_dim, config):
        ch, w, h = state_dim # in torch order

        cfig_buffer = config['buffer']
        cfig_actor = config['actor']
        cfig_critic = config['critic']
        cfig_ppo = config['ppo']

        self.buffer = cfig_buffer['name'](cfig_buffer['size'], state_dim, action_dim)
        self.actor = cfig_actor['actor'] (cfig_actor['model'], ch, action_dim)
        self.critic = cfig_critic['model'](ch, 1)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfig_actor['lr'])
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfig_critic['lr'])

        self.epoch_actor = cfig_actor['epoch']
        self.epoch_critic = cfig_critic['epoch']

        self.clip = cfig_ppo['clip']
        

    def critic_loss(self, data):
        pass


    def actor_loss(self, data):
        pass


    def learn(self, last_value=None):
        pass


    def policy(self, state):
        pass


if __name__ == '__main__':
    from network import ResidualEncoder, ValueEncoder
    from agent import StochasticActor


    pass
