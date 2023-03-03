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
                 "T_out": 20,
                 "Step": 20,
                 "Width": 32, 
                 "Variables":1, 
                 "Noise":0.0, 
                 "UQ": 'Dropout',
                 }

#%% 

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
import os 
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

data =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS.npz')
u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
u = torch.from_numpy(u_sol)
xx, yy = np.meshgrid(x,y)

ntrain = 500
ncal = 480
npred = len(u_sol) - (ntrain + ncal)
S = 33 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

# %%
#Chunking the data. 
train_a = u[:ntrain,:T_in,:,:]
train_u = u[:ntrain,T_in:T+T_in,:,:]

cal_a = u[ntrain:-npred,:T_in, :, :]
cal_u = u[ntrain:-npred,T_in:T+T_in,:,:]

pred_a = u[-npred:,:T_in, :, :]
pred_u = u[-npred:,T_in:T+T_in,:,:]

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
#Invoking the trained Models
#####################################

#Conformalised Quantile Regression

model_05 = UNet2d(T_in, step, width)
model_05.load_state_dict(torch.load(model_loc + 'Unet_Wave_QR_05.pth', map_location='cpu'))

model_95 = UNet2d(T_in, step, width)
model_95.load_state_dict(torch.load(model_loc + 'Unet_Wave_QR_95.pth', map_location='cpu'))

model_50 = UNet2d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_QR_50.pth', map_location='cpu'))

# %%
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
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal

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
# %%


# %%
idx = 0
tt = -1
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Surface(z=y_response[idx, tt], opacity=0.9, colorscale='viridis'),
    go.Surface(z=mean[idx, tt], opacity=0.9, colorscale='tealrose'),
    # go.Surface(z=prediction_sets[0][idx, tt], colorscale = 'turbid', showscale=False, opacity=0.6),
    # go.Surface(z=prediction_sets[1][idx, tt], colorscale = 'Electric',showscale=False, opacity=0.3)

])

fig.show()
# %%

def get_prediction_sets(alpha):
    
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    #Obtaining the Prediction Sets
    stacked_x = torch.FloatTensor(pred_a)

    with torch.no_grad():
        mean = model_50(stacked_x).numpy()

    return  [mean - qhat, mean + qhat]


alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alpha_levels)
pred_sets = [get_prediction_sets(a) for a in alpha_levels] 

# %%
x_id = 16

# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)

fig, ax = plt.subplots()
[plt.fill_between(x_points, pred_sets[i][0][idx, tt][x_id,:], pred_sets[i][1][idx, tt][x_id,:], color = cols[i]) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)

plt.plot(x_points, y_response[idx, tt][x_id, :], linewidth = 4, color = "black", label = "exact")
plt.legend()


# %%
