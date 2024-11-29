#!/usr/bin/env python3

import torch
import torch.optim as optim
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2


# config:{
#         buffer:{name, size}
#         actor:{name, model, lr}
#         critic:{name, model, lr}
#         epoch
#         ppo:{clip}}


class AgentPPO:
    def __init__(self, state_dim, action_dim, config):
        ch, _, _ = state_dim
        
        buffer = config['buffer']
        actor = config['actor']
        critic = config['critic']
        ppo = config['ppo']

        self.epoch = config['epoch']

        self.buffer = buffer['name'](buffer['size'], state_dim, action_dim, actor['is_multi_head'])
        self.actor = actor['name'](actor['model'], ch, action_dim, actor['is_multi_head'])
        self.critic = critic['name'](critic['model'], ch, 1, critic['is_multi_head'])
        
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=actor['lr'])
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=critic['lr'])
        
        self.clip = ppo['clip']

        self.is_multi_head = actor['is_multi_head']
        
        
    def critic_loss(self, data):
        state, rewards_to_go = data['state'], data['rewards_to_go']
        state = state.permute(0, 3, 1, 2)
        
        v_loss = ((self.critic(state).squeeze() - rewards_to_go)**2).mean()

        return v_loss
    
    
    def actor_loss(self, data):
        state, action, advantage, logp_old, context = [
            data[k] for k in ('state', 'action', 'advantages', 'logp', 'context')
        ]
        state = state.permute(0, 3, 1, 2) # batch, ch, W, H
        
        if self.is_multi_head:
            _, logp, pred = self.actor(state, action)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantage
            loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()
        
            loss_cont = ((pred.sigmoid() - context)**2).mean()

            return loss_pi, loss_cont

        else:
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
            self.opt_actor.zero_grad()
            loss_pi, loss_cont = self.actor_loss(batch)
            loss_ac = loss_pi + loss_cont*0.1
            loss_ac.backward()
            self.opt_actor.step()
        
            self.opt_critic.zero_grad()
            loss_v = self.critic_loss(batch)
            loss_v.backward()
            self.opt_critic.step()
            
            print('policy loss: ', loss_pi.item(), 
                  'context loss: ', loss_cont.item(),
                  'actor loss', loss_ac.item(),
                  'critic loss: ',  loss_v.item())

        return loss_ac.item(), loss_v.item()
        
        
    def policy(self, state, is_eval=False):
        # note that separation btw train and eval mode is required if the model incldues 
        # special layers such as batch norm or layer norm.
        self.actor.eval()
        self.critic.eval()
        
        state = torch.as_tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            pi, cont = self.actor.dist(state.unsqueeze(dim=0))
                
            if is_eval:
                value = self.critic(state).squeeze().numpy()
                mean = pi.mean.squeeze().numpy()
                feat_ac = self.actor.feat.squeeze().permute(1, 2, 0).numpy()
                imp_ac = self.actor.imp.squeeze().numpy()
                feat_cr = self.critic.feat.squeeze().permute(1, 2, 0).numpy()
                imp_cr = self.critic.imp.squeeze().numpy()
                cont = cont.squeeze().sigmoid().numpy()

                return mean, value, feat_ac, imp_ac, feat_cr, imp_cr, cont
            
            else:
                a = pi.sample()

            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.critic(state.unsqueeze(dim=0))

        return a.squeeze().numpy(), v.squeeze().numpy(), logp_a.squeeze().numpy()
    

    def save_model(self, path_to_name_prefix):
        # save DNN models inside of actor and critic
        torch.save(self.actor.model.state_dict(), path_to_name_prefix+'_actor.pth')
        torch.save(self.critic.model.state_dict(), path_to_name_prefix+'_critic.pth')

    
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
                       'is_multi_head':True}
    config['critic'] = {'name':CumRewardCritic,
                        'model':ValueEncoder,
                        'lr':1e-4, 
                        'is_multi_head':False}
    config['epoch'] = 10
    config['ppo'] = {'clip':0.3}

    print(config)

    state_dim = (3, 128, 128)
    action_dim = 6

    agent = AgentPPO(state_dim, action_dim, config)
    state = torch.rand(3, 128, 128)

    #a, v, logp = agent.policy(state)
    a, v, f1, i1, f2, i2, cont = agent.policy(state, True)
    print(a.shape, v, f1.shape, i1.shape, f2.shape, i2.shape, cont.shape)

    #mini_cam = f1@i1
    #cam = ndimage.zoom(mini_cam, 4)
    #print(cam.dtype)

    #activation_map = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    #plt.imshow(activation_map)
    #plt.show()











