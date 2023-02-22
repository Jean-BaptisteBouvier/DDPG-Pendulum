# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:09:16 2023

@author: Jean-Baptiste Bouvier

Simple environment for the inverted pendulum, inspired from the gym environment.
"""


import numpy as np




class PendulumEnv():
    """Inverted pendulum environment modified for RL with continuous action space."""
    
    def __init__(self, init_theta=np.pi, init_thetadot=1):

        self.dt = .05
        self.g = 9.81
        
        self.init_theta = init_theta
        self.init_thetadot = init_thetadot
        self.max_speed = np.inf
        self.max_torque = self.g # needs to be >g/2
        
        self.action_size = 1
        self.action_low = -self.max_torque
        self.action_high = self.max_torque
        
        self.state_size = 2
        
        self.total_time = 0
        self.total_time_upright = 0
        self.upright_tol = 0.1 # tolerance on the angle for the upright position
        self.metric = 'fraction_upright'


    def step(self, u):
        """ Returns: new_state, reward, metric"""
        theta, theta_dot = self.state
        # input u in [-1,1] needs to be scaled by self.max_torque
        u = np.clip(u[0]*self.max_torque, self.action_low, self.action_high)
        ### new state
        new_theta = angle_normalize(theta + theta_dot * self.dt)
        new_theta_dot = theta_dot + ( 3*self.g*np.sin(theta)/2 + 3*u )*self.dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)

        self.state = np.array([new_theta, new_theta_dot])
        self.total_time += 1
        self.total_time_upright += np.abs(new_theta) < self.upright_tol
        metric = {self.metric: self.total_time_upright / self.total_time}
        
        ### reward upright and penalize u, theta and theta_dot
        reward = (np.abs(new_theta) < self.upright_tol) - u**2 - 5*new_theta**2 - new_theta_dot**2
        ### normalized rewards
        reward = np.tanh(reward/10.)
        return self.state, reward, metric


    def reset(self):
        """Resets the pendulum."""
        high = np.array([self.init_theta, self.init_thetadot])
        self.state = np.random.uniform(low=-high, high=high)
        self.total_time = 0
        self.total_time_upright = 0
        return self.state



def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
