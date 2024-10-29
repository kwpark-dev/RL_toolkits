#from collections import deque
from torch import nn

import torch
import numpy as np
import gymnasium as gym
import torch.optim as optim



class BaseLearner:
    def __init__(self, env_name, is_render=True, episode=10):
        if is_render:
            self.env = gym.make(env_name, render_mode='human')
        
        else :
            self.env = gym.make(env_name)

        self.episode = episode


    def learn(self):
        for i in range(self.episode):
            obs, info = self.env.reset()

            while True:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    break

        self.env.close()



class TDLearner(BaseLearner):
    def __init__(self, env_name, is_render=True, sample=100, episode=100, lr=1e-4):
        super().__init__(env_name, is_render, episode)
        
        self.samples = sample
        
        self.Qnet = StateActionValueNet(action_dim=1, state_dim=2)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=lr)
        
        # self.trace = [torch.zeros_like(param) for param in self.Qnet.parameters()]


    def optimize(self, state_action, value):    
        self.Qnet.train()
            
        outcome = self.Qnet(state_action) 
        outcome = torch.squeeze(outcome, dim=0)
        
        loss = self.loss_fn(outcome, value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def trace_update(self):
        self.Qnet.eval()
        
        theta = 0
        with torch.no_grad():
            for param in self.Qnet.parameters():
                grad = param.grad
                
                theta += (grad*param).sum()
        
        return theta


    def q_learn(self, gamma=0.8, forget=0.2, lamb=0.9):
        eligibility = 0
        
        for j in range(self.episode):
            current_state, _ = self.env.reset()
            
            R = 0
            
            while True:                                     
                current_q_est, current_action, current_state_action = self.epsilon_greedy(current_state)
                next_state, reward, terminated, truncated, _ = self.env.step(current_action)
                next_q_est, _, next_state_action  = self.epsilon_greedy(next_state, eps=0.)
                R += reward

                delta = reward + gamma*next_q_est - current_q_est
                q_value = current_q_est + forget*delta*eligibility
                
                self.optimize(current_state_action, q_value)
                eligibility = eligibility*gamma*lamb + self.trace_update()
                
                if terminated or truncated:
                    self.optimize(next_state_action, torch.FloatTensor([0.])) # Q value of the terminal state is 0
                    
                    break

                current_state = next_state
                
            print("Q-learning: Episode {} is done. Total reward is {}".format(j+1, R))
            
        self.env.close()


    def sarsa_learn(self, gamma=0.8, forget=0.2, lamb=0.9):
        eligibility = 0
        
        for j in range(self.episode):
            current_state, _ = self.env.reset()
            current_q_est, current_action, current_state_action = self.epsilon_greedy(current_state)

            R = 0
            
            while True:                                     
                next_state, reward, terminated, truncated, _ = self.env.step(current_action)
                next_q_est, next_action, next_state_action  = self.epsilon_greedy(next_state)
                R += reward
                
                delta = reward + gamma*next_q_est - current_q_est
                q_value = current_q_est + forget*delta*eligibility
                
                self.optimize(current_state_action, q_value)
                eligibility = eligibility*gamma*lamb + self.trace_update()
                
                if terminated or truncated:
                    self.optimize(next_state_action, torch.FloatTensor([0.])) # Q value of the terminal state is 0
                    
                    break 
                
                current_state = next_state
                current_action = next_action
                current_state_action = next_state_action
                current_q_est = next_q_est                       

            print("SARSA: Episode {} is done. Total reward is {}".format(j+1, R))
            
        self.env.close()


    def epsilon_greedy(self, state, eps=0.1):
        self.Qnet.eval() # note that batchnorm demands more than 1 data points. 

        if torch.rand(1).item() < eps:
            action = torch.rand(1, 1)*2 - 1
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
    
            state_action = torch.concat((state, action), dim=1)
             
            with torch.no_grad():
                Q = self.Qnet(state_action)
             
            return torch.squeeze(Q, dim=1), torch.squeeze(Q, dim=1), state_action

        else :
            states = torch.FloatTensor(state)*torch.ones((self.samples, 1))
            actions = torch.rand(self.samples, 1)*2 - 1
            states_actions = torch.concat((states, actions), dim=1)
            
            with torch.no_grad():
                Qs = self.Qnet(states_actions)
            
            idx = torch.argmax(Qs)
            
            return Qs[idx], actions[idx], torch.unsqueeze(states_actions[idx], 0)



class StateActionValueNet(nn.Module):
    def __init__(self, action_dim, state_dim, output_dim=1):
        super().__init__()
        self.input_dim = action_dim + state_dim
        self.output_dim = output_dim

        self.layer = nn.Sequential(nn.Linear(self.input_dim, 64),
                                #    nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                #    nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.output_dim))


    def forward(self, x):
        x = self.layer(x)

        return x



if __name__=='__main__':

    env_name = 'MountainCarContinuous-v0'
    episode = 1000

    agent = TDLearner(env_name=env_name, episode=episode)
    agent.sarsa_learn()
    # agent.q_learn()