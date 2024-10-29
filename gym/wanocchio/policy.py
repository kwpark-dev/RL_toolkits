import torch
import torch.nn as nn
from torch.distributions import Normal
# from apprx import fc_apprx



class DeterministicActor(nn.Module):
    def __init__(self):
        super().__init__()
        
        pass



class StochasticActor(nn.Module):
    def __init__(self, func_nn, state_dim, action_dim, N):
        super().__init__()
        # network for mean values of the N > 1 dim'l distribution
        # but multivariate Gaussian returns correlated sequences, which should be avoided
        # so prepare network returning N * (mean, std) which are employed by N Gaussian distribution
        self.dist_net = func_nn(state_dim, action_dim*2, N, nn.Identity)
        # self.log_std = nn.Parameter(torch.rand(1, action_dim))
        
        
    def dist(self, state):
        means, log_stds = self.dist_net(state).chunk(2, dim=1)
        normal = Normal(means, log_stds.exp())
        
        return normal
    
    
    def forward(self, state, action):
        pi = self.dist(state)
        logp = pi.log_prob(action).sum(axis=-1)
        # logp = pi.log_prob(action)
        return pi, logp
    
    
    def train(self):
        self.dist_net.train()
        
        
    def eval(self):
        self.dist_net.eval()
    
    
    
# if __name__ == "__main__":
    # actor = StochasticActor(fc_apprx, 10, 4, 4)
    
    # state = torch.randn(12, 10)
    # action = torch.randn(12, 4)
    
    # pi, logp = actor(state, action)
    # print(pi.shape, logp.shape)
#     means = torch.tensor([0.0, 2.0, -1.0])       # Mean for each distribution
#     std_devs = torch.tensor([1.0, 0.5, 1.5])     # Standard deviation for each distribution

#     # Create independent Normal distributions
#     normal_dist = torch.distributions.Normal(means, std_devs)

#     # Sample one value from each distribution
#     samples = normal_dist.sample()
#     print("Samples:", torch.tanh(samples))