#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base tests for demonstrating CP udner covariate shift - 
Experiemntally evaluating the math behind this https://arxiv.org/abs/1904.06019ule 
"""

# %% 
import numpy as np 
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn 
import scipy.stats as stats
from tqdm import tqdm 

from utils import * 

torch.set_default_dtype(torch.float32)
# %% 
def func(x):
    return np.sin(x) + np.cos(2*x)

#Sampling from a normal distribution
def normal_dist(mean, std, N):
    dist = stats.norm(mean, std)
    return dist.rvs(N)


N = 1000 #Datapoints 
x1 = normal_dist(np.pi/2, np.pi/6, N)
x_shift = normal_dist(np.pi, np.pi/6, N) #Covariate shifted

#Visualising the covariate shift
plt.hist(x1, label='Initial')
plt.hist(x_shift, label='Shifted')
plt.legend()
# %%
#Obtaining the outputs 
y1 = func(x1)
y_shift = func(x_shift)

#Visualising the outputs 
plt.plot(np.sort(x1), func(np.sort(x1)), label="Initial")
plt.plot(np.sort(x_shift), func(np.sort(x_shift)), label="Shifted")


# %% 
#Training a GP to model the function 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.001)

train_N = 100
x_train = np.expand_dims(np.linspace(0, 1*np.pi, train_N), -1)
y_train= func(x_train)
gpr.fit(x_train, y_train)
y_mean, y_std = gpr.predict(x_train, return_std=True)
# %%
#Visualising the model performance
viz_N = 500
x_viz = np.expand_dims(np.linspace(0, 2*np.pi, viz_N), -1)
y_viz= func(x_viz)
y_mean, y_std = gpr.predict(x_viz, return_std=True)

plt.plot(x_viz, y_viz, label='Actual')
plt.plot(x_viz, y_mean, label='Pred')
plt.fill_between(x_viz.flatten(), y_mean-y_std, y_mean+y_std, color='gray')
plt.legend()
plt.title("Visualising the Model Performance")

# %% 
#Standard Inductive CP 

N = 1000 #Datapoints 
x1 = normal_dist(np.pi/2, np.pi/6, N)

cal_scores = np.abs(y_pred - y_cal)
