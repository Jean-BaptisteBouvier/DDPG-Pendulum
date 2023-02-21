# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:28:18 2023

@author: Jean-Baptiste Bouvier

Self-contained implementation of a Deep Deterministic Policy Gradient (DDPG) algorithm
to learn to stabilize a inverted pendulum.
Only requires numpy, torch and matplotlib, no need for the gym library.
We reward the time upright and penalize the angle, angular velocity and the torque.
The exploration noise is setup to decrease with the epoch.
The hyperparameters are all collated below, except those designing the reward function which are set in PendulumEnv.
"""

import numpy as np
import matplotlib.pyplot as plt
from ddpg_networks import DDPG_agent
from ddpg_utils import RandnNoise, controller_test
from pendulum_env import PendulumEnv



#### Hyperparameters
batch_size = 2**9 # size of the learning mini batches
N_step = 2**7 # numbers of steps per episode
N_episodes = 200 # number of training episodes
hid_sz = 100 # size of the hidden layers in each neural network
actor_lr = 1e-3 # learning rate of the actor networks
critic_lr = 1e-3 # learning rate of the critic networks
len_mem = round(N_step*N_episodes/2**4) # size of the replay memory to store experiences
initial_sigma = 0.1 # exploration noise at episode 0
final_sigma = 0.002 # exploration noise at final episode


#### Setup
env = PendulumEnv()
controller = DDPG_agent(env, hidden_size=hid_sz, actor_learning_rate=actor_lr, critic_learning_rate=critic_lr, max_memory_size=len_mem)
action_noise = RandnNoise(env, sigma=initial_sigma)



### Training the networks
rewards = np.zeros(N_episodes)
time_upright = np.zeros(N_episodes)

num_params = controller.actor.num_params()
print(f"Number of params per neural net: {num_params}   Number of data points: {N_step*N_episodes}\nNumber of data points per parameter: {N_step*N_episodes/num_params:.1f}")


for episode in range(N_episodes):
    state = env.reset()
    
    for step in range(N_step):
        u = controller.get_action(state)
        u = action_noise.get_action(u, step)
        
        new_state, reward, metric = env.step(u) 
        controller.memory.push(state, u, reward, new_state)
        rewards[episode] += reward
        state = new_state
        
        ### Training when enough steps are stored in memory   
        if len(controller.memory) > batch_size:
            controller.update(batch_size)
        
        
    time_upright[episode] = metric['fraction_upright']
    
    ### Decreasing the noise level with the episodes
    sigma = initial_sigma + episode*(final_sigma - initial_sigma)/N_episodes
    action_noise.update_sigma(sigma)
    
    if episode%10 == 0:
        print(f"episode: {episode} \t fraction time upright: {time_upright[episode]:.2f} \t reward: {rewards[episode]:.2f}")
        
    ### Ending training early if performance is good enough
    if episode > 100 and time_upright[episode-20:episode].mean() > 0.95:
        break



    
### Plotting time upright and rewards
plt.plot(time_upright[:episode])
plt.xlabel('Episode')
plt.title('Fraction Time Upright (at +- '+ str(env.upright_tol) +' rad)')
plt.show()

plt.plot(rewards[:episode])
plt.xlabel('Episode')
plt.title('Reward')
plt.show()
  
   
### Testing the trained controller
controller_test(env, controller, N_step)




