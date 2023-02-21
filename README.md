# DDPG-Pendulum

Implementation of a simple, self-contained Deep Deterministic Policy Gradient (DDPG) algorithm to learn to stabilize a inverted pendulum.

The code is as simple as possible to be easily understandable and to be reused in more complex projects.
To get rid of unnecessary complexity this code does NOT require the gym library.


**Prerequisite**
---
The only libraries needed to run this project are `numpy`, `torch` and `matplotlib`.



**Project overview**
---

The rewards are normalized to [-1, 1] with `tanh` to have a stable learning and avoid large gradient swings.




**File Structure**
---
- The file `ddpg_main.py` is the main code launching the training of DDPG.
- The file `ddpg_networks.py` contains the code for the 4 neural networks used by DDPG: 2 actors and 2 critics along with their update mechanism.
- The file `ddpg_utils.py` contains the classes for the exploration noise, the replay memory and a function to test the performance of DDPG at stabilizing the inverted pendulum.
- The file `pendulum_env.py` contains a simple, self-contained environment for the pendulum.


**Running**
---

Run `ddpg_main.py` to train the DDPG agent on the inverted pendulum.


**Results**
---

![Upright time](Plots/time_upright.png "Upright time during training")
![Pendulum angle](Plots/angle.png "Pendulum angle")
![Pendulum angular velocity](Plots/angular_velocity.png "Pendulum angular velocity")
![Pendulum torque](Plots/torque.png "Pendulum torque")
