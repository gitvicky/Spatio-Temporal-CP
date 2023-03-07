#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31 October 2022

@author: vgopakum, agray, lzanisi

1D U-Net built using PyTorch to model the 1D Burgers Equation. 
Conformal Prediction using various Conformal Score estimates

"""

# %% 
configuration = {"Case": 'Burgers',
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
                 "Physics Normalisation": 'No',
                 "T_in": 10,    
                 "T_out": 10,
                 "Step": 10,
                 "Width": 32, 
                 "Variables":1, 
                 "Noise":0.0, 
                #  "Loss Function": 'MSE Loss',
                #  "UQ": 'Dropout',
                #  "Pinball Gamma": 'NA',
                #  "Dropout Rate": 1.0
                 }

# %%
#Importing the necessary packages
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

from collections import OrderedDict
from utils import *

torch.manual_seed(0)
np.random.seed(0)
from utils import *


# %% 
#Setting the seeds and the path for the run. 
torch.manual_seed(0)
np.random.seed(0)
path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #positioning the GPUs if they are available. only works for Nvidia at the moment 

model_loc = path + '/Models/'
data_loc = path

# %% 

################################################################
# load data
# _a -- referes to the input 
# _u -- referes ot the output
################################################################
t1 = default_timer()

data =  np.load(data_loc + '/Data/Burgers1d_sliced.npy')
u_sol = data.astype(np.float32)
u = torch.from_numpy(u_sol)
x_range = np.linspace(-1,1,1000)

ntrain = 500
ncal = 480
npred = len(u_sol) - (ntrain + ncal)
S = 1000 #Grid Size

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

cal_a = u[ntrain:-npred,:T_in, :]
cal_u = u[ntrain:-npred,T_in:T+T_in,:]

pred_a = u[-npred:,:T_in, :]
pred_u = u[-npred:,T_in:T+T_in,:]

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



# %%
#####################################
#Conformalised Quantile Regression
#####################################

#Conformalised Quantile Regression

model_05 = UNet1d(T_in, step, width)
model_05.load_state_dict(torch.load(model_loc + 'Unet_Burgers_QR_05.pth', map_location='cpu'))

model_95 = UNet1d(T_in, step, width)
model_95.load_state_dict(torch.load(model_loc + 'Unet_Burgers_QR_95.pth', map_location='cpu'))

model_50 = UNet1d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_Burgers_QR_50.pth', map_location='cpu'))


# %%
t1 = default_timer()

#Performing the Calibration for Quantile Regression
n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    cal_lower = model_05(torch.Tensor(cal_a)).numpy()
    cal_upper = model_95(torch.Tensor(cal_a)).numpy()

# %%
cal_u = cal_u.numpy()
cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

# %% 
#Obtaining the Prediction Sets

y_response = pred_u.numpy()
stacked_x = torch.FloatTensor(pred_a)

with torch.no_grad():
    val_lower = model_05(stacked_x).numpy()
    val_upper = model_95(stacked_x).numpy()
    mean = model_50(stacked_x).numpy()

prediction_sets = [val_lower - qhat, val_upper + qhat]

# %%
print('Conformal by way QCR')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0]) & (y_response <= prediction_sets_uncalibrated[1])).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")

t2 = default_timer()
print('CQR, time used:', t2-t1)

# %% 
idx = 5
t_val = -1 
Y_pred_viz = y_response[idx, t_val]
mean_viz = mean[idx, t_val]
pred_set_0_viz = prediction_sets[0][idx, t_val]
pred_set_1_viz = prediction_sets[1][idx, t_val]
pred_set_uncal_0_viz = prediction_sets_uncalibrated[0][idx, t_val]
pred_set_uncal_1_viz = prediction_sets_uncalibrated[1][idx, t_val]

plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.plot(x_range, Y_pred_viz, label='Analytical', color='black')
plt.plot(x_range, mean_viz, label='Mean', color='firebrick')
plt.plot(x_range, pred_set_0_viz, label='lower-cal', color='teal')
plt.plot(x_range, pred_set_uncal_0_viz, label='lower - uncal', color='darkorange')
plt.plot(x_range, pred_set_1_viz, label='upper-cal', color='navy')
plt.plot(x_range, pred_set_uncal_1_viz, label='upper - uncal', color='gold')
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
# %%
#Testing calibration across range of Alpha for QCR 
def calibrate(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        cal_lower = model_05(torch.Tensor(cal_a)).numpy()
        cal_upper = model_95(torch.Tensor(cal_a)).numpy()

    cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)           
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


# alpha_levels = np.arange(0.05, 0.95, 0.1)
# emp_cov = []

# for ii in tqdm(range(len(alpha_levels))):
#     emp_cov.append(calibrate(alpha_levels[ii]))

# plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
# plt.plot(1-alpha_levels, emp_cov, label='Coverage')
# plt.xlabel('1-alpha')
# plt.ylabel('Empirical Coverage')
# plt.legend()

# %% 
# %%
#####################################
#Conformalising using Residuals (MAE)
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#####################################
t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    cal_mean = model_50(torch.Tensor(cal_a)).numpy()

# cal_u = cal_u.numpy()
cal_scores = np.abs(cal_u-cal_mean)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()
stacked_x = torch.FloatTensor(pred_a)

with torch.no_grad():
    mean = model_50(stacked_x).numpy()

prediction_sets =  [mean - qhat, mean + qhat]


# %%
print('Conformal by way Residual')
# Calculate empirical coverage (before and after calibration)
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

t2 = default_timer()
print('Residuals, time used:', t2-t1)

# %% 
idx =10
t_val = -1 
Y_pred_viz = y_response[idx, t_val]
mean_viz = mean[idx, t_val]
pred_set_0_viz = prediction_sets[0][idx, t_val]
pred_set_1_viz = prediction_sets[1][idx, t_val]


plt.figure()
plt.title(f"Conformal using Residuals, alpha = {alpha}")
plt.plot(x_range, Y_pred_viz, label='Analytical', color='black')
plt.plot(x_range, mean_viz, label='Mean', color='firebrick')
plt.plot(x_range, pred_set_0_viz, label='lower-cal', color='teal')
plt.plot(x_range, pred_set_1_viz, label='upper-cal', color='navy')
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
# %%
def calibrate(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        cal_mean = model_50(torch.Tensor(cal_a)).numpy()
        
    cal_scores = np.abs(cal_u-cal_mean)     
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    prediction_sets =  [mean - qhat, mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


# alpha_levels = np.arange(0.05, 0.95, 0.05)
# emp_cov = []
# for ii in tqdm(range(len(alpha_levels))):
#     emp_cov.append(calibrate(alpha_levels[ii]))

# plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
# plt.plot(1-alpha_levels, emp_cov, label='Coverage')
# plt.xlabel('1-alpha')
# plt.ylabel('Empirical Coverage')
# plt.legend()