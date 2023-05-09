#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 February 2023

@author: vgopakum, agray, lzanisi

U-Net built using PyTorch to model the 2D Wave Equation. 
Dataset buitl by changing by performing a LHS across the x,y pos and amplitude of the initial gaussian distibution
Code for the spectral solver can be found in : https://github.com/farscape-project/PINNs_Benchmark

----------------------------------------------------------------------------------------------------------------------------------------

Experimenting with a range of UQ Methods:
    1. Dropout
    2. Quantile Regression 
    3. NN Ensemble 
    4. Physics-based residual
    5. Deep Kernel Learning

Once UQ methodolgies have been demonstrated on each, we can use Conformal Prediction over a
 multitude of conformal scores to find empirically rigorous coverage. 
"""

# %%

configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Type": 'U-Net',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No',
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 20,    
                 "T_out": 60,
                 "Step": 10,
                 "Width": 32, 
                 "Modes": 'NA',
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'Quantile Loss',
                 "UQ": 'None',
                 "Pinball Gamma": 0.95,
                 "Dropout Rate": 'NA'
                 }


#%% 
import os
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

import time 
from timeit import default_timer
from tqdm import tqdm 

from utils import *

torch.manual_seed(0)
np.random.seed(0)

# %% 
path = os.getcwd()
model_loc = path + '/Models/'
data_loc = path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# load data
# _a -- referes to the input 
# _u -- referes ot the output
################################################################
t1 = default_timer()

data =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_5K.npz')
u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
u = torch.from_numpy(u_sol)
xx, yy = np.meshgrid(x,y)

ntrain = 1000
ncal = 1000
npred = 1000
S = 33 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

# %%
#Chunking the data. 
train_a = u[:ntrain,:T_in,:]
train_u = u[:ntrain,T_in:T+T_in,:]

cal_a = u[ntrain:ntrain+ncal,:T_in, :]
cal_u = u[ntrain:ntrain+ncal,T_in:T+T_in,:]

pred_a = u[ntrain+ncal:ntrain+ncal+npred,:T_in, :]
pred_u = u[ntrain+ncal:ntrain+ncal+npred,T_in:T+T_in,:]

print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)

t2 = default_timer()
print('Data sorting finished, time used:', t2-t1)

# %% 
#Normalisation. 
a_normalizer = MinMax_Normalizer(train_a)
train_a = a_normalizer.encode(train_a)
cal_a = a_normalizer.encode(cal_a)
pred_a = a_normalizer.encode(pred_a)

y_normalizer = MinMax_Normalizer(train_u)
train_u = y_normalizer.encode(train_u)
cal_u = y_normalizer.encode(cal_u)
pred_u = y_normalizer.encode(pred_u)

# %%
#####################################
#Conformalised Quantile Regression
#####################################

#Invoking the trained models 

model_05 = UNet2d(T_in, step, width)
model_05.load_state_dict(torch.load(model_loc + 'Unet_Wave_indigo-reload_QR_05.pth', map_location='cpu'))

model_95 = UNet2d(T_in, step, width)
model_95.load_state_dict(torch.load(model_loc + 'Unet_Wave_round-underwriter_QR_95.pth', map_location='cpu'))

model_50 = UNet2d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_shy-bevel_QR_50.pth', map_location='cpu'))

# %%
t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx_lower = cal_a
    xx_upper = cal_a
    for tt in tqdm(range(0, T, step)):
        pred_lower = model_05(xx_lower)
        pred_upper = model_95(xx_upper)

        if tt == 0:
            cal_lower = pred_lower
            cal_upper = pred_upper

        else:
            cal_lower = torch.cat((cal_lower, pred_lower), 1)       
            cal_upper = torch.cat((cal_upper, pred_upper), 1)       

        xx_lower = torch.cat((xx_lower[:, step:, :, :], pred_lower), dim=1)
        xx_upper = torch.cat((xx_upper[:, step:, :, :], pred_upper), dim=1)


cal_u = cal_u.numpy()
cal_lower = cal_lower.numpy()
cal_upper = cal_upper.numpy()

cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')


# %% 
#Obtaining the Prediction Sets

y_response = pred_u.numpy()

with torch.no_grad():
    xx_lower = pred_a
    xx_upper = pred_a
    xx_mean = pred_a

    for tt in tqdm(range(0, T, step)):
        pred_lower = model_05(xx_lower)
        pred_upper = model_95(xx_upper)
        pred_mean = model_50(xx_mean)

        if tt == 0:
            val_lower = pred_lower
            val_upper = pred_upper
            val_mean = pred_mean
        else:
            val_lower = torch.cat((val_lower, pred_lower), 1)       
            val_upper = torch.cat((val_upper, pred_upper), 1)       
            val_mean = torch.cat((val_mean, pred_mean), 1)       

        xx_lower = torch.cat((xx_lower[:, step:, :, :], pred_lower), dim=1)
        xx_upper = torch.cat((xx_upper[:, step:, :, :], pred_upper), dim=1)
        xx_mean = torch.cat((xx_mean[:, step:, :, :], pred_mean), dim=1)

prediction_sets = [val_lower - qhat, val_upper + qhat]

# %% 
print('Conformal by way QCR')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0].numpy()) & (y_response <= prediction_sets_uncalibrated[1].numpy())).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")

t2 = default_timer()
print('Conformalised Quantile Regression, time used:', t2-t1)


# %%
#Testing calibration across range of Alpha for QCR 

def calibrate(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        xx_lower = cal_a
        xx_upper = cal_a
        for tt in tqdm(range(0, T, step)):
            pred_lower = model_05(xx_lower)
            pred_upper = model_95(xx_upper)

            if tt == 0:
                cal_lower = pred_lower
                cal_upper = pred_upper

            else:
                cal_lower = torch.cat((cal_lower, pred_lower), 1)       
                cal_upper = torch.cat((cal_upper, pred_upper), 1)       

            xx_lower = torch.cat((xx_lower[:, step:, :, :], pred_lower), dim=1)
            xx_upper = torch.cat((xx_upper[:, step:, :, :], pred_upper), dim=1)

    cal_lower = cal_lower.numpy()
    cal_upper = cal_upper.numpy()

    cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)           
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')
    
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()

    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov = []

for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate(alpha_levels[ii]))

# %% 
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#############################################################
# Conformal Prediction Residuals
#############################################################

t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a

    for tt in tqdm(range(0, T, step)):
        pred = model_50(xx)

        if tt == 0:
            cal_mean = pred

        else:
            cal_mean = torch.cat((cal_mean, pred), 1)       

        xx = torch.cat((xx[:, step:, :, :], pred), dim=1)


cal_mean = cal_mean.numpy()
cal_scores = np.abs(cal_u-cal_mean)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()

with torch.no_grad():
    xx_mean = pred_a

    for tt in tqdm(range(0, T, step)):
        pred_mean = model_50(xx_mean)

        if tt == 0:
            val_mean = pred_mean
        else:     
            val_mean = torch.cat((val_mean, pred_mean), 1)       

        xx_mean = torch.cat((xx_mean[:, step:, :, :], pred_mean), dim=1)

prediction_sets = [val_mean - qhat, val_mean + qhat]

# %%
print('Conformal by way Residual')
# Calculate empirical coverage (before and after calibration)
empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

t2 = default_timer()
print('Conformal by Residual, time used:', t2-t1)
# %%
def calibrate(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        xx = cal_a
        for tt in tqdm(range(0, T, step)):
            pred_mean = model_50(xx)

            if tt == 0:
                cal_mean = pred_mean

            else:
                cal_mean = torch.cat((cal_mean, pred_mean), 1)       

            xx = torch.cat((xx[:, step:, :, :], pred_mean), dim=1)

    cal_mean = cal_mean.numpy()

    cal_scores = np.abs(cal_u-cal_mean)           
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    
    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()

    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov = []

for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate(alpha_levels[ii]))

# %% 
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.title('Residuals')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()


# %%
##############################
# Conformal using Dropout 
##############################

model_dropout = UNet2d_dropout(T_in, step, width)
model_dropout.load_state_dict(torch.load(model_loc + 'Unet_Wave_frigid-hill_dropout.pth', map_location='cpu'))

# %%
#Performing the Calibration for Dropout

t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval(model_dropout, xx, step)

        if tt == 0:
            cal_mean = mean
            cal_std = std
        else:
            cal_mean = torch.cat((cal_mean, mean), 1)       
            cal_std = torch.cat((cal_std, std), 1)       

        xx = torch.cat((xx[:, step:, :, :], mean), dim=1)


# cal_mean = cal_mean.numpy()

cal_upper = cal_mean + cal_std
cal_lower = cal_mean - cal_std

cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

# %% 
#Obtaining the Prediction Sets
with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval(model_dropout, xx, step)

        if tt == 0:
            val_mean = mean
            val_std = std
        else:
            val_mean = torch.cat((val_mean, mean), 1)       
            val_std = torch.cat((val_std, std), 1)       

        xx = torch.cat((xx[:, step:, :, :], mean), dim=1)

val_upper = val_mean + val_std
val_lower = val_mean - val_std

prediction_sets_uncalibrated = [val_lower, val_upper]
prediction_sets = [val_lower - qhat, val_upper + qhat]

# %% 
y_response = pred_u.numpy()

print('Conformal by way Dropout')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0].numpy()) & (y_response <= prediction_sets_uncalibrated[1].numpy())).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")

# %% 
def calibrate(alpha):
    with torch.no_grad():
        xx = cal_a

        for tt in tqdm(range(0, T, step)):
            mean, std = Dropout_eval(model_dropout, xx, step)

            if tt == 0:
                cal_mean = mean
                cal_std = std
            else:
                cal_mean = torch.cat((cal_mean, mean), 1)       
                cal_std = torch.cat((cal_std, std), 1)       

            xx = torch.cat((xx[:, step:, :, :], mean), dim=1)


    # cal_mean = cal_mean.numpy()

    cal_upper = cal_mean + cal_std
    cal_lower = cal_mean - cal_std

    cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

        
    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()

    return empirical_coverage

# %% 

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.title('Residuals')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
cal_u = cal_u.numpy()
cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')


y_response = pred_u.numpy()
stacked_x = torch.FloatTensor(pred_a)

with torch.no_grad():
    val_lower = model_05(stacked_x).numpy()
    val_upper = model_95(stacked_x).numpy()
    mean = model_50(stacked_x).numpy()

prediction_sets = [val_lower - qhat, val_upper + qhat]

# %%
print('Conformal by way Dropout')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0]) & (y_response <= prediction_sets_uncalibrated[1])).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")

t2 = default_timer()
print('Conformalised Quantile Regression, time used:', t2-t1)
# %%



# %%
idx = 0
tt = -1
x_id = 16

# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)

fig, ax = plt.subplots()
[plt.fill_between(x_points, pred_sets[i][0][idx, tt][x_id,:], pred_sets[i][1][idx, tt][x_id,:], color = cols[i]) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)

plt.plot(x_points, y_response[idx, tt][x_id, :], linewidth = 4, color = "black", label = "exact")
plt.legend()

