# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:26:44 2023

@author: Jean-Baptiste Bouvier

Neural networks for DDPG.
The DDPG_agent creates the two critic and the two actor networks.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ddpg_utils import Memory
from torch.autograd import Variable



class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential( nn.Linear(input_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, output_size))

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



class Actor(nn.Module):
    """Each actor produces an action in [-1,1] thanks to the hyperbolic tangent."""
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, output_size), nn.Tanh())

        
    def forward(self, state):
        return self.net(state)
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())




### Class for the DDPG agent containing the 4 neural networks
class DDPG_agent:
    def __init__(self, env, hidden_size=64, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=2**14):
        # Params
        self.num_states = env.state_size
        self.num_actions = env.action_size
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    
    def get_action(self, state):
        """Let DDPG predicts the action to take at a given state."""
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
    
    
    def update(self, batch_size):
        """Training DDPG on memory batch samples."""
        states, actions, rewards, next_states = self.memory.sample(batch_size)        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))