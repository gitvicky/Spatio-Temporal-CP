#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base tests for demonstrating CP und er covariate shift - 
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
plt.title("Visualising the distribution of the initial and the shifted")

# %%
#Obtaining the outputs 
y1 = func(x1)
y_shift = func(x_shift)

#Visualising the outputs 
plt.plot(np.sort(x1), func(np.sort(x1)), label="Initial")
plt.plot(np.sort(x_shift), func(np.sort(x_shift)), label="Shifted")
plt.title("Visualising the function outputs")

# %% 
#Training a GP to model the function 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = RBF(length_scale=1, length_scale_bounds=(1e-2, 1e3)) 
gpr = GaussianProcessRegressor(kernel=kernel)

train_N = 10
x_train = np.expand_dims(np.linspace(0, np.pi, train_N), -1)
y_train= func(x_train)
gpr.fit(x_train, y_train)
y_mean, y_std = gpr.predict(x_train, return_std=True)
# %%
#Visualising the model performance
viz_N = 500
x_viz = np.expand_dims(np.linspace(0, np.pi, viz_N), -1)
y_viz= func(x_viz)
y_mean, y_std = gpr.predict(x_viz, return_std=True)

plt.plot(x_viz, y_viz, label='Actual')
plt.plot(x_viz, y_mean, label='Pred')
plt.fill_between(x_viz.flatten(), y_mean-y_std, y_mean+y_std, color='gray', label='+- 1 std')
plt.legend()
plt.title("Visualising the Model Performance")


# %% 
# Exploring Covariate Shift -- using the known estimations from the distributions that we sample from 

def likelihood_ratio(x, mean_1, std_1, mean_2, std_2):
    pdf1 = stats.norm.pdf(x, mean_1, std_1)
    pdf2 = stats.norm.pdf(x, mean_2, std_2)
    return pdf2 / pdf1 

mean_1, std_1 = 3*np.pi/4, np.pi/4
mean_2, std_2 = np.pi, np.pi/4


x1 = normal_dist(mean_1, std_1, N)
x_shift = normal_dist(mean_2, std_2, N) #Covariate shifted

#Visualising the covariate shift
plt.hist(x1, label='Initial')
plt.hist(x_shift, label='Shifted')
plt.legend()
plt.title("Visualising the distribution of the initial and the shifted")

# %%
N = 1000 #Datapoints 
x_calib = normal_dist(mean_1, std_1, N)
x_shift = normal_dist(mean_2, std_2, N) #Covariate shifted

y_calib = func(x_calib)
y_calib_gp = gpr.predict(np.expand_dims(x_calib, -1))

cal_scores = np.abs(y_calib - y_calib_gp)
# %%
def pi(x_new, x_cal):
    return likelihood_ratio(x_cal, mean_1, std_1, mean_2, std_2) / (np.sum(likelihood_ratio(x_cal, mean_1, std_1, mean_2, std_2)) + likelihood_ratio(x_new, mean_1, std_1, mean_2, std_2))
    
weighted_scores = cal_scores.squeeze() * pi(x_shift, x_calib).squeeze()

# %% 
#Estimating qhat 

alpha = 0.1

def weighted_quantile(data, alpha, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.quantile(np.sort(data), alpha, axis = 0, interpolation='higher')
    
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]

    #d = d[1:]
    #d = np.append(d, np.inf)

    p=1.*w.cumsum()/w.sum()
    y=np.interp(alpha, p, d)

    return y

qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi(x_shift, x_calib).squeeze())
qhat_true = np.quantile(np.sort(weighted_scores), np.ceil((N+1)*(1-alpha))/(N), axis = 0, interpolation='higher')

# %%
y_shift = func(x_shift)
y_shift_gp = gpr.predict(np.expand_dims(x_shift,  -1))

prediction_sets =  [y_shift_gp - qhat, y_shift_gp + qhat]
empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %%
def calibrate_res(alpha):
    qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi(x_shift, x_calib).squeeze())
    prediction_sets = [y_shift_gp - qhat, y_shift_gp + qhat]
    empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_res = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual - weighted' ,ls='-.', color='teal', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
#Using Kernel Density Estimation - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html

