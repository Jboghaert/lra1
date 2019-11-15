#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.helpers import launch_env, wrap_env, view_results_ipython, force_done
from utils.helpers import SteeringToWheelVelWrapper, ResizeWrapper, ImgWrapper

import numpy as np

import torch
import torch.nn as nn
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Generating Data from a Teacher
# 
# In order to use imitation learning in practice, we need to have _demonstrations_. However, demonstrations need to be gathered; in general, we can collect the demonstrations that we need in one of four ways:
# 
# * Human demonstrator teleoperating the robot
# * Data logs or historical data
# * Learned policy (i.e from reinforcement learning) is rolled out
# * Hard-coded expert is rolled out
# 
# While these trajectories can be gathered on real robots, to speed up collection, we work mainly in simulation. Duckietown has a [vast](https://logs.duckietown.org) collection of logs gathered over years of running programs on Duckiebots, but here, we focus on the last data collection method: a hard-coded expert.
# 
# **<font color='red'>Question 1:</font> What are some pros and cons of each approach? List two pros and two cons for each of the four methods listed above.**
# 
# We first introduce a _pure-pursuit expert_ - often, in robotic imitation learning, we have controllers to control many of our robots and systems; a pure-pursuit expert is about the simplest controller that we can have for a Duckiebot.
# 
# Our expert drives with ground-truth state data; while more complicated controllers incorporate and fuse observational data to estimate a state, we use data that'd a robot would not normally have access to.

# In[51]:


class PurePursuitExpert:
    def __init__(self, env, ref_velocity=0.9999, position_threshold=0.8, gain=15,
                 following_distance=4., max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.gain = gain
        self.position_threshold = position_threshold

    def predict(self, observation):  
        # Our expert drives with "cheating" data, something your implementation will not have access to
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = self.gain * -dot

        return self.ref_velocity, steering


# In[52]:


nsteps = 3500


# In[53]:


local_env = launch_env()
local_env = wrap_env(local_env)
local_env = ResizeWrapper(local_env)
local_env = ImgWrapper(local_env)

local_env.reset()
wrapper = SteeringToWheelVelWrapper()

# Create an demonstrator
expert = PurePursuitExpert(env=local_env)

observations = []
actions = []

# Collect samples

for steps in range(0, nsteps):
    # use our 'expert' to predict the next action.
    action = expert.predict(None)
    action = wrapper.convert(action)
    observation, reward, done, info = local_env.step(action)
    observations.append(observation)
    actions.append(action)

    if done:
        local_env.reset()
        
local_env.close()

print('\nDone!\n')


# In[54]:


view_results_ipython(local_env)


# **<font color='red'>Question 2:</font> When you visualize the results, what are two major issues? Play with the expert's code and the execution code above, and list five changes that you tried, as well as their _qualitative_ effects on performance (i.e cover the most distance). DO NOT RESEED THE ENVIRONMENT**

# # Defining a Model
# 
# While the above expert isn't great, it's a start. What's best is that we now have image `observations` and real-valued `actions` that we can use to train a neural network in Pytorch. Our imitation learner will driver directly from observations, and will be trained with a popular imitation learning loss: Mean Squared Error.

# In[55]:


class Model(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Model, self).__init__()

        # TODO: You'll need to change this!
        flat_size = 31968
        
        ###########################################
        # QUESTION 3. What does the next line do? #
        ###########################################
        self.lr = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.1)

        self.lin1 = nn.Linear(flat_size, 100)
        self.lin2 = nn.Linear(100, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        x = self.lin2(x)
        x = self.max_action * self.tanh(x)
        
        return x


# # Training from the Teacher Data
# 
# We can then write our _training loop_ : the piece of code that implements the process of stochastic gradient descent to minimize the loss between our network's predicted actions and those implemented by our expert.

# In[56]:


nepochs = 350
batchsize = 20

actions = np.array(actions)
observations = np.array(observations)

model = Model(action_dim=2, max_action=1.)
model.train().to(device)

# weight_decay is L2 regularization, helps avoid overfitting
optimizer = optim.SGD(
    model.parameters(),
    lr=0.005,
    weight_decay=1e-3
)

avg_loss = 0
for epoch in range(nepochs):
    optimizer.zero_grad()

    batch_indices = np.random.randint(0, observations.shape[0], (batchsize))
    obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
    act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

    model_actions = model(obs_batch)

    loss = (model_actions - act_batch).norm(2).mean()
    loss.backward()
    optimizer.step()

    loss = loss.data.item()
    avg_loss = avg_loss * 0.995 + loss * 0.005

    print('epoch %d, loss=%.3f' % (epoch, avg_loss))

    # Periodically save the trained model
    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'models/imitate.pt')
        
print('\nDone!\n')


# **<font color='red'>Question 3:</font> Qualitatively explain at least 2 changes you made to both the expert and network (architecture, hyperparameters, episode lengths, number of training episodes / epochs, etc.) (including partial points if we find that you didn't make changes to any part of our code - hyperparameters, network, etc.)**
# 
# 
# **<font color='red'>Question 4:</font> Explain the issues with the imitation learning loop above. Specifically, comment on the loss function and training objective. Explain at least one issue, and propose a way that could help solve the issues you've brought up.**

# In[57]:


force_done(local_env)
local_env = launch_env()
local_env = wrap_env(local_env)
local_env = ResizeWrapper(local_env)
local_env = ImgWrapper(local_env)

obs = local_env.reset()

done = False
rewards = []
nsteps = 500
for steps in range(0, nsteps):
    obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    action = model(obs)
    action = action.squeeze().data.cpu().numpy()
    obs, reward, done, info = local_env.step(action) 
    rewards.append(reward)
    
    if done:
        local_env.reset()
        print("Reset!")

print(info)
        
local_env.close()


print("\nDone!\n")
    


# In[58]:


view_results_ipython(local_env)


# **<font color='red'>Question 5:</font> Copy the value of _info_ , after simulating for 500 steps. If your simulation fails earlier, save the results _before_ the failure (i.e. when the simulation returns `done = True`.  DO NOT RESEED THE ENVIRONMENT** 

# # A nastier environment
# 
# Once your solution is able to pass a curve while staying in the lane, you can try to see what happens if you modify the test environment with respect to the one used to generate the training dataset. 
# 
# To do this, create a new environment called *new_environment* by using the **launch_env()** function as above. This time passing the argument *domain_rand=True*. Basically it randomizes the environment. Once you have the new environment, run again the model without retraining. 
# 
# Then, visualize the results. 
# 
# **<font color='red'>Question 6:</font> Comment the performance of your solution on the new environment, name two reasons that justify the performance.**

# In[61]:


# TODO: Run again the agent in the new randomized environment as explained above

new_env = launch_env(domain_rand=True)
new_env = wrap_env(new_env)
new_env = ResizeWrapper(new_env)
new_env = ImgWrapper(new_env)

obs = new_env.reset()

done = False
rewards = []
nsteps = 300
for steps in range(0, nsteps):
    obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
    action = model(obs)
    action = action.squeeze().data.cpu().numpy()
    obs, reward, done, info = new_env.step(action) 
    rewards.append(reward)
    
    if done:
        new_env.reset()

new_env.close()

print("\nDone!\n")


# In[62]:


# TODO: visualize the results
view_results_ipython(new_env)


# In[ ]:




