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

# %% 
#Standard Inductive CP using residuals

N_calib = 1000 #Datapoints 
x_calib = normal_dist(np.pi/2, np.pi/6, N_calib)
y_calib = func(x_calib)
y_calib_gp = gpr.predict(np.expand_dims(x_calib, axis=-1))
cal_scores = np.abs(y_calib_gp - y_calib)

alpha = 0.1
n = N_calib
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

x_pred= normal_dist(np.pi/2, np.pi/6, N_calib)
y_pred = func(x_pred)
y_pred_gp = gpr.predict(np.expand_dims(x_pred, axis=-1))

prediction_sets =  [y_pred_gp - qhat, y_pred_gp + qhat]
empirical_coverage = ((y_pred >= prediction_sets[0]) & (y_pred <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")


viz_N = 500
x_viz = np.expand_dims(np.linspace(0, np.pi, viz_N), -1)
y_viz= func(x_viz)
y_mean, y_std = gpr.predict(x_viz, return_std=True)
prediction_sets =  [y_mean - qhat, y_mean + qhat]

plt.plot(x_viz, y_viz, label='Actual')
plt.plot(x_viz, y_mean, label='Pred')
plt.fill_between(x_viz.flatten(), prediction_sets[0], prediction_sets[1], color='gray', label='alpha = ' + str(alpha))
plt.legend()
plt.title("Visualising the Coverage")

# %% 
n = N_calib
y_pred_gp = gpr.predict(np.expand_dims(x_pred, axis=-1))
y_response = y_pred
def calibrate_res(alpha):
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    prediction_sets = [y_pred_gp - qhat, y_pred_gp + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_res = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
#Inductive CP using the GP uncertainty with the same formulation as in the gentle introduction paper 

N_calib = 1000 #Datapoints 
x_calib = normal_dist(np.pi/2, np.pi/6, N_calib)
y_calib = func(x_calib)
y_calib_gp, std_calib_gp = gpr.predict(np.expand_dims(x_calib, axis=-1), return_std=True)

# cal_upper = y_calib_gp + std_calib_gp
# cal_lower =  y_calib_gp + std_calib_gp
# cal_scores = np.maximum(y_calib_gp-cal_upper, cal_lower-y_calib_gp)    

cal_scores = np.abs(y_calib_gp-y_calib)/std_calib_gp

alpha = 0.1
n = N_calib
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

x_pred= normal_dist(np.pi/2, np.pi/6, N_calib)
y_pred = func(x_pred)

y_pred_gp, std_pred_gp = gpr.predict(np.expand_dims(x_pred, axis=-1), return_std=True)


prediction_sets =  [y_pred_gp - std_pred_gp*qhat, y_pred_gp + std_pred_gp*qhat]

empirical_coverage = ((y_pred >= prediction_sets[0]) & (y_pred <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

viz_N = 500
x_viz = np.expand_dims(np.linspace(0, np.pi, viz_N), -1)
y_viz= func(x_viz)
y_mean, y_std = gpr.predict(x_viz, return_std=True)
prediction_sets =  [y_mean - qhat, y_mean + qhat]

plt.plot(x_viz, y_viz, label='Actual')
plt.plot(x_viz, y_mean, label='Pred')
plt.fill_between(x_viz.flatten(), prediction_sets[0], prediction_sets[1], color='gray', label='alpha = ' + str(alpha))
plt.legend()
plt.title("Visualising the Coverage")

# %% 
n = N_calib
y_pred_gp = gpr.predict(np.expand_dims(x_pred, axis=-1))
y_response = y_pred
def calibrate_res(alpha):
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    prediction_sets =  [y_pred_gp - std_pred_gp*qhat, y_pred_gp + std_pred_gp*qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_res = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov_res, label='GP Uncertainty' ,ls='-.', color='teal', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %% 
#Inductive CP using the GP uncertainty with the same formulation as in the gentle introduction paper 

N_calib = 1000 #Datapoints 
x_calib = normal_dist(np.pi/2, np.pi/6, N_calib)
y_calib = func(x_calib)
y_calib_gp, std_calib_gp = gpr.predict(np.expand_dims(x_calib, axis=-1), return_std=True)

cal_upper = y_calib_gp + std_calib_gp
cal_lower =  y_calib_gp + std_calib_gp

cal_scores = np.maximum(y_calib_gp-cal_upper, cal_lower-y_calib_gp)    


alpha = 0.1
n = N_calib
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

x_pred= normal_dist(np.pi/2, np.pi/6, N_calib)
y_pred = func(x_pred)

y_pred_gp, std_pred_gp = gpr.predict(np.expand_dims(x_pred, axis=-1), return_std=True)

pred_upper = y_pred_gp + std_pred_gp
pred_lower = y_pred_gp - std_pred_gp

prediction_sets =  [pred_lower -qhat, pred_upper +qhat]

empirical_coverage = ((y_pred >= prediction_sets[0]) & (y_pred <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

viz_N = 500
x_viz = np.expand_dims(np.linspace(0, np.pi, viz_N), -1)
y_viz= func(x_viz)
y_mean, y_std = gpr.predict(x_viz, return_std=True)
prediction_sets =  [y_mean - qhat, y_mean + qhat]

plt.plot(x_viz, y_viz, label='Actual')
plt.plot(x_viz, y_mean, label='Pred')
plt.fill_between(x_viz.flatten(), prediction_sets[0], prediction_sets[1], color='gray', label='alpha = ' + str(alpha))
plt.legend()
plt.title("Visualising the Coverage")

# %% 
n = N_calib
y_pred_gp = gpr.predict(np.expand_dims(x_pred, axis=-1))
y_response = y_pred
def calibrate_res(alpha):
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    prediction_sets =  [pred_lower -qhat, pred_upper + std_pred_gp+qhat]

    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_res = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov_res, label='GP Uncertainty' ,ls='-.', color='teal', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %% 