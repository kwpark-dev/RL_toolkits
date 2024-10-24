import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

# from apprx import fc_apprx



class DeterministicActor(nn.Module):
    def __init__(self):
        super().__init__()
        
        pass



class StochasticActor(nn.Module):
    def __init__(self, func_nn, state_dim, action_dim, N):
        super().__init__()
        # network for mean values of the N > 1 dim'l distribution
        self.mean = func_nn(state_dim, action_dim, N)
        # cov matrix for N > 1 dim'l distribution, should be positive def.
        # init with cholesky factorization, LL^T
        A = torch.rand(action_dim, action_dim)
        L = torch.tril(A)
        
        self.cov = nn.Parameter(L@L.T)
        
        
    def dist(self, state):
        
        return MultivariateNormal(self.mean(state), self.cov)
    
    
    def forward(self, state, action):
        pi = self.dist(state)
        logp = pi.log_prob(action).sum(axis=-1)
        # logp = pi.log_prob(action)
        return pi, logp
    
    
    def train(self):
        self.mean.train()
        
        
    def eval(self):
        self.mean.eval()
    
    
    
# if __name__ == "__main__":
    # A = torch.rand(5, 5)
    # L = torch.tril(A)
    # cov = L@L.T
    
    # pi = MultivariateNormal(torch.rand(5), cov)
    # print(pi.sample((1, )))
    # state = torch.rand((5, 5))
    # action = torch.rand((5, 3))
    # N = 4
    # actor = StochasticActor(fc_apprx, 5, 3, N)
    # mvn = actor.dist(state)
    # print(mvn.batch_shape, mvn.event_shape)
    # print(mvn.loc, mvn.covariance_matrix)
    # policy, logp = actor(state, action)
    
    # print(policy, logp)