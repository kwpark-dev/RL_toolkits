#!/usr/bin/env python3

import torch
import torch.optim as optim



# config:{
#         buffer:{name, size}
#         actor:{name, model, lr, epoch}
#         critic:{name, model, lr, epoch}
#         ppo:{clip}}


class AgentPPO:
    def __init__(self, state_dim, action_dim, config):
        ch, _, _ = state_dim
        
        buffer = config['buffer']
        actor = config['actor']
        critic = config['critic']
        ppo = config['ppo']

        self.buffer = buffer['name'](buffer['size'], state_dim, action_dim)
        self.actor = actor['name'](actor['model'], ch, action_dim)
        self.critic = critic['name'](critic['model'], ch, 1)
        
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=actor['lr'])
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=critic['lr'])
        
        self.epoch_actor = actor['epoch']
        self.epoch_critic = critic['epoch']

        self.clip = ppo['clip']
        
        
    def critic_loss(self, data):
        state, rewards_to_go = data['state'], data['rewards_to_go']
        v_loss = ((self.critic(state).squeeze() - rewards_to_go)**2).mean()

        return v_loss
    
    
    def actor_loss(self, data):
        state, action, advantage, logp_old = [
            data[k] for k in ('state', 'action', 'advantages', 'logp')
        ]
        _, logp = self.actor(state, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantage
        loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()
        
        return loss_pi
    
    
    def learn(self, last_value=None):
        self.actor.train()
        self.critic.train()
        
        batch = self.buffer.fetch(last_value)
        
        for _ in range(self.epoch):
            self.optimizer_actor.zero_grad()
            loss_pi = self.actor_loss(batch)
            loss_pi.backward()
            self.optimizer_actor.step()
        
            self.optimizer_critic.zero_grad()
            loss_v = self.critic_loss(batch)
            loss_v.backward()
            self.optimizer_critic.step()
            
        return loss_pi.item(), loss_v.item()
        
        
    def policy(self, state):
        # note that separation btw train and eval mode is required if the model incldues 
        # special layers such as batch norm or layer norm.
        self.actor.eval()
        self.critic.eval()
        
        state = torch.as_tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
        
            pi = self.actor.dist(state.unsqueeze(dim=0))
            
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.critic(state.unsqueeze(dim=0))
            
            print(a.shape)
            print(logp_a.shape)
            print(v.shape)

        return a.squeeze().numpy(), v.squeeze().numpy(), logp_a.squeeze().numpy()
    
    
    
if __name__ == '__main__':
    from network import ResidualEncoder, ValueEncoder
    from policy import StochasticActor
    from value import CumRewardCritic
    from buffer import RolloutBuffer



    config = {}
    config['buffer'] = {'name':RolloutBuffer,
                        'size': 1}
    config['actor'] = {'name':StochasticActor,
                       'model':ResidualEncoder,
                       'lr':1e-4,
                       'epoch':10}
    config['critic'] = {'name':CumRewardCritic,
                        'model':ValueEncoder,
                        'lr':1e-4,
                        'epoch':10}
    config['ppo'] = {'clip':0.3}

    print(config)

    state_dim = (3, 128, 128)
    action_dim = 6

    agent = AgentPPO(state_dim, action_dim, config)
    state = torch.rand(3, 128, 128)

    a, v, logp = agent.policy(state)
    print(a, v, logp)













