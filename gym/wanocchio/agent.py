import torch
import torch.optim as optim



class AgentBase:
    def __init__(self, state_dim, action_dim):
        pass
    
    
    def actor_loss(self, data):
        pass
    
    
    def critic_loss(self, data):
        pass
    
    
    def learn(self):
        pass


    


# basically code copied from the paderborne for PPO, but will find generalized version 
class AgentPPO(AgentBase):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        
        # {buffer:name, buffer_size:int, 
        #  apprx:fc_apprx,
        #  actor:model, actor_layer: int, actor_lr:float,
        #  critic:model, critic_layer:int, critici_lr:float, 
        #  epoch:int, clip:float}
        self.buffer = config['buffer'](config['buffer_size'], state_dim, action_dim)
        self.actor = config['actor'](config['apprx'], state_dim, action_dim, config['actor_layer'])
        self.critic = config['critic'](config['apprx'], state_dim, config['critic_layer'])
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        self.epoch = config['epoch']
        self.clip = config['clip']
        
        
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
        
        
    def policy(self, state, deterministic=False):
        # note that separation btw train and eval mode is required if the model incldues 
        # special layers such as batch norm or layer norm.
        self.actor.eval()
        self.critic.eval()
        
        state = torch.as_tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
        
            pi = self.actor.dist(state.unsqueeze(dim=0))
            if deterministic:
                return pi.mean.numpy()
            
            else:
                a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            
            v = self.critic(state.unsqueeze(dim=0))
    
        return a.squeeze().numpy(), v.squeeze().numpy(), logp_a.squeeze().numpy()
    
    
    

