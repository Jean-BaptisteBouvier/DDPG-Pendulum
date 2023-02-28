# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:25:45 2023

@author: Jean-Baptiste Bouvier

Utils functions for the simulation.
RandnNoise is a class to generate exploration noise on the action.
Memory is a class to store previous experiences and replay them as mini-batches.
controller_test is a function testing the predictions of the DDPG controller for the pendulum and plotting the state response.
"""


import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt




class RandnNoise(object):
    """Normal noise instead of the Ornstein-Ulhenbeck noise."""
    def __init__(self, env, sigma=0.01):
        self.sigma = sigma
        self.action_size = env.action_size

    def update_sigma(self, sigma):
        """Change the standard deviation"""
        self.sigma = sigma
    
    def get_action(self, action, t=0):
        """Add normal noise to a given action, with noise decreasing with t"""
        noise = self.sigma * np.random.randn(self.action_size)/np.sqrt(t+1)
        return action + noise



def controller_test(env, controller, N_step):
    """Testing the controller on a trajectory propagation of length N_step.
    Plots the trajectory and displays the fraction of time upwards."""
    state = env.reset()
    
    States = np.zeros((N_step, env.state_size))
    U = np.zeros((N_step, env.action_size))
    reward = 0
    
    ### Trajectory propagation
    for step in range(N_step):
        U[step] = np.array([controller.get_action(state)])
        state, r, metric = env.step(U[step])
        States[step] = state
        reward += r
        
    print(f"Fraction time upright: {metric['fraction_upright']:.2f} \t reward: {reward:.2f}")
    
    time = list(np.arange(start=0., stop=N_step*env.dt, step=env.dt))
    
    plt.plot(time, States[:,0])
    plt.title('Inverted pendulum angle')
    plt.xlabel('time (s)')
    plt.ylabel('theta (rad)')
    plt.show() 
    
    plt.plot(time, States[:,1])
    plt.title('Inverted pendulum angular velocity')
    plt.xlabel('time (s)')
    plt.ylabel('theta dot (rad/s)')
    plt.show() 
    
    plt.plot(time, U[:,0])
    plt.title('Torque input')
    plt.ylabel('u (N)')
    plt.xlabel('time (s)')
    plt.show() 
   
        


# Replay memory to store past experiences and replay them randomly by minibatches
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
        
        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)