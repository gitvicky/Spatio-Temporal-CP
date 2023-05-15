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
                 "Type": 'Unet',
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
                 "Loss Function": 'Quantile Loss',
                 "UQ": 'None',
                 "Pinball Gamma": 0.95,
                 "Dropout Rate": 'NA',
                 "Spatial Resolution": 200
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
import matplotlib as mpl 
plt.rcParams['text.usetex'] = True

plt.rcParams['grid.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.titlepad'] = 30
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 10.0
plt.rcParams['ytick.minor.size'] = 10.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['lines.linewidth'] = 1

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

data =  np.load(data_loc + '/Data/Conv_Diff_u_2.npz')
u_sol = data['u'].astype(np.float32) [:, ::5, :]
u = torch.from_numpy(u_sol)

x_range =  np.arange(0, 10, 0.05)

ntrain = 3000
ncal = 1000
npred = 1000
S = 200 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

# %%
#Chunking the data. 
u_train = torch.from_numpy(np.load(data_loc + '/Data/Conv_Diff_u_1.npz')['u'].astype(np.float32) [:, ::5, :])

train_a = u_train[:ntrain,:T_in,:]
train_u = u_train[:ntrain,T_in:T+T_in,:]

cal_a = u[:ncal,:T_in, :]
cal_u = u[:ncal,T_in:T+T_in,:]

pred_a = u[ncal:ncal+npred,:T_in, :]
pred_u = u[ncal:ncal+npred,T_in:T+T_in,:]

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
model_05.load_state_dict(torch.load(model_loc + 'Unet_CD_lower_0.05.pth', map_location='cpu'))

model_95 = UNet1d(T_in, step, width)
model_95.load_state_dict(torch.load(model_loc + 'Unet_CD_upper_0.95.pth', map_location='cpu'))

model_50 = UNet1d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_CD_mean_0.5.pth', map_location='cpu'))


# %%
t1 = default_timer()

#Performing the Calibration for Quantile Regression
n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    cal_lower = model_05(torch.Tensor(cal_a)).numpy()
    cal_upper = model_95(torch.Tensor(cal_a)).numpy()

# %%
cal_scores = np.maximum(cal_u.numpy()-cal_upper, cal_lower-cal_u.numpy())           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

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

# %%% 
idx = 12
t_val = -1
Y_pred_viz = y_response[idx, t_val]
mean_viz = mean[idx, t_val]
pred_set_0_viz = prediction_sets[0][idx, t_val]
pred_set_1_viz = prediction_sets[1][idx, t_val]
pred_set_uncal_0_viz = prediction_sets_uncalibrated[0][idx, t_val]
pred_set_uncal_1_viz = prediction_sets_uncalibrated[1][idx, t_val]

plt.figure()
plt.title(rf"CQR, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x_range, Y_pred_viz, label='Analytical', color='black', alpha = 0.7)
plt.plot(x_range, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x_range, pred_set_0_viz, label='Lower - Calibrated', color='teal', alpha = 0.7)
plt.plot(x_range, pred_set_uncal_0_viz, label='Lower - Uncalibrated', color='teal', alpha = 0.5, ls='--')
plt.plot(x_range, pred_set_1_viz, label='Upper - Calibrated', color='navy', alpha = 0.7)
plt.plot(x_range, pred_set_uncal_1_viz, label='Upper - Uncalibrated', color='navy', alpha = 0.5, ls='--')
plt.xlabel(r"\textbf{x}")
plt.ylabel(r"\textbf{u}")
plt.legend()
plt.grid() #Comment out if you dont want grids.

plt.savefig("convdiff_cqr.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()
# %% 
#Testing calibration across range of Alpha for QCR 
def calibrate_cqr(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        cal_lower = model_05(torch.Tensor(cal_a)).numpy()
        cal_upper = model_95(torch.Tensor(cal_a)).numpy()

    cal_scores = np.maximum(cal_u.numpy()-cal_upper, cal_lower-cal_u.numpy())           
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_cqr = []

for ii in tqdm(range(len(alpha_levels))):
    emp_cov_cqr.append(calibrate_cqr(alpha_levels[ii]))


# %% 
# plt.figure()
# plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.75, linewidth=3.0)
# # plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.75, linewidth=3.0)
# # plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.75, linewidth=3.0)
# plt.xlabel(r'1-$\alpha$')
# plt.ylabel('Empirical Coverage')
# plt.legend()
# plt.grid() #Comment out if you dont want grids.

# %% 
# %%
#####################################
#Conformalising using Residuals (MAE)
#Performing the Calibration using Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#####################################

model_50 = UNet1d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_CD_mean_0.5.pth', map_location='cpu'))


t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    cal_mean = model_50(torch.Tensor(cal_a)).numpy()

cal_scores = np.abs(cal_u.numpy()-cal_mean)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()

with torch.no_grad():
    mean = model_50(torch.FloatTensor(pred_a)).numpy()

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

idx = 12 
t_val = -1
Y_pred_viz = y_response[idx, t_val]
mean_viz = mean[idx, t_val]
pred_set_0_viz = prediction_sets[0][idx, t_val]
pred_set_1_viz = prediction_sets[1][idx, t_val]

plt.figure()
# plt.title(f"Residuals, alpha = {alpha}")
plt.title(rf"Residuals, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x_range, Y_pred_viz, label='Analytical', color='black', alpha = 0.7)
plt.plot(x_range, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x_range, pred_set_0_viz, label='lower-cal', color='teal', alpha = 0.7)
plt.plot(x_range, pred_set_1_viz, label='upper-cal', color='navy', alpha = 0.7)
plt.xlabel(r"\textbf{x}")
plt.ylabel(r"\textbf{u}")
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("convdiff_residual.svg", format="svg", bbox_inches='tight',  transparent='True')
plt.show()

# %%
def calibrate_res(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        cal_mean = model_50(torch.Tensor(cal_a)).numpy()
        
    cal_scores = np.abs(cal_u-cal_mean)     
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets =  [mean - qhat, mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res(alpha_levels[ii]))


# %% 

# plt.figure()
# plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75, linewidth=3.0)
# # plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.75, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.75, linewidth=3.0)
# # plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.75, linewidth=3.0)
# plt.xlabel(r'1-$\alpha$')
# plt.ylabel('Empirical Coverage')
# plt.legend()
# plt.grid() #Comment out if you dont want grids.


# %%

##########################################
#Conformal using Dropout
##########################################

model_dropout = UNet1d_dropout(T_in, step, width)
model_dropout.load_state_dict(torch.load(model_loc + 'Unet_CD_dropout_NA.pth', map_location='cpu'))

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

        xx = torch.cat((xx[:, step:, :], mean), dim=1)

# cal_mean = cal_mean.numpy()

cal_upper = cal_mean + cal_std
cal_lower = cal_mean - cal_std

cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

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

        xx = torch.cat((xx[:, step:, :], mean), dim=1)

val_upper = val_mean + val_std
val_lower = val_mean - val_std

val_lower = val_lower.numpy()
val_upper = val_upper.numpy()

prediction_sets_uncalibrated = [val_lower, val_upper]
prediction_sets = [val_lower - qhat, val_upper + qhat]

# %% 
y_response = pred_u.numpy()

print('Conformal by way of Dropout')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0]) & (y_response <= prediction_sets_uncalibrated[1])).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
t2 = default_timer()
print('CQR, time used:', t2-t1)

# %% 

idx = 12
t_val = -1
Y_pred_viz = y_response[idx, t_val]
mean_viz = mean[idx, t_val]
pred_set_0_viz = prediction_sets[0][idx, t_val]
pred_set_1_viz = prediction_sets[1][idx, t_val]
pred_set_uncal_0_viz = prediction_sets_uncalibrated[0][idx, t_val]
pred_set_uncal_1_viz = prediction_sets_uncalibrated[1][idx, t_val]

plt.figure()
# plt.title(f"Conformal by Dropout, alpha = {alpha}")
plt.title(rf"Dropout, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x_range, Y_pred_viz, label='Analytical', color='black', alpha = 0.7)
plt.plot(x_range, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x_range, pred_set_0_viz, label='lower-cal', color='teal', alpha = 0.7)
plt.plot(x_range, pred_set_uncal_0_viz, label='lower - uncal', color='teal', alpha = 0.5, ls='--')
plt.plot(x_range, pred_set_1_viz, label='upper-cal', color='navy', alpha = 0.7)
plt.plot(x_range, pred_set_uncal_1_viz, label='upper - uncal', color='navy', alpha = 0.5, ls='--')
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid() #Comment out if you dont want grids.

plt.savefig("convdiff_dropout.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()
# %%

def calibrate_dropout(alpha):

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

            xx = torch.cat((xx[:, step:, :], mean), dim=1)


    # cal_mean = cal_mean.numpy()

    cal_upper = cal_mean + cal_std
    cal_lower = cal_mean - cal_std

    cal_scores = np.maximum(cal_u-cal_upper, cal_lower-cal_u)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_dropout = []

for ii in tqdm(range(len(alpha_levels))):
    emp_cov_dropout.append(calibrate_dropout(alpha_levels[ii]))

# %% 

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.75, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.75, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.75, linewidth=3.0)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.legend()
plt.grid() #Comment out if you dont want grids.

# %%
plt.rcParams['grid.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 2
mpl.rcParams['lines.linewidth'] = 3
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.75, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.75, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.75, linewidth=3.0)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("convdiff_comparison.svg", format="svg", bbox_inches='tight')
plt.show()
# %%
