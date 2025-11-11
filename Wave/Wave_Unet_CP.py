#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net built using PyTorch to model the 2D Wave Equation. 
Dataset buitl by changing by performing a LHS across the x,y pos and amplitude of the initial gaussian distibution

----------------------------------------------------------------------------------------------------------------------------------------

Experimenting with a range of UQ Methods:
    1. Dropout
    2. Quantile Regression 
    3. NN Ensemble 

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
                 "T_out": 30,
                 "Step": 30,
                 "Width": 32, 
                 "Modes": 'NA',
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'Quantile Loss',
                 "UQ": 'None',
                 "Pinball Gamma": 0.5,
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
import matplotlib as mpl 
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True

plt.rcParams['grid.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,12)
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

from utils import *
from utils_cp import *

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

ntrain = 500
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
model_05.load_state_dict(torch.load(model_loc + 'Unet_Wave_lower.pth', map_location='cpu'))

model_95 = UNet2d(T_in, step, width)
model_95.load_state_dict(torch.load(model_loc + 'Unet_Wave_upper.pth', map_location='cpu'))

model_50 = UNet2d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_mean.pth', map_location='cpu'))

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


cal_lower = cal_lower.numpy()
cal_upper = cal_upper.numpy()


# %%
#Performing the calibration
cal_scores = nonconf_score_lu(cal_u.numpy(), cal_lower, cal_upper)
qhat = calibrate(cal_scores, n, alpha)

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

    val_lower = val_lower.numpy()
    val_upper = val_upper.numpy()

    prediction_sets_calibrated = [val_lower - qhat, val_upper + qhat]


# %%
print('Conformal by way QCR')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = emp_cov(prediction_sets_uncalibrated, y_response)
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage_calibrated = emp_cov(prediction_sets_calibrated, y_response)
print(f"The empirical coverage after calibration is: {empirical_coverage_calibrated}")

t2 = default_timer()
print('CQR, time used:', t2-t1)

tightness_metric = est_tight(prediction_sets_calibrated, y_response)
print(f"Tightness of the coverage : Average of the distance between error bars {tightness_metric}")

# %%
#Testing calibration across range of Alpha for CQR
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_cqr = []

for ii in tqdm(range(len(alpha_levels))):
    qhat = calibrate(cal_scores, n, alpha_levels[ii])
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    emp_cov_cqr.append(emp_cov(prediction_sets, y_response))


# %% 
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()


# %% 
#Plots

def get_prediction_sets(alpha):
    qhat = calibrate(cal_scores, n, alpha)
    # qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return  prediction_sets

alpha_levels = np.arange(0.05, 0.95, 0.1)
coverage_levels = (1 - alpha_levels)
cols = cm.plasma_r(coverage_levels)
pred_sets = [get_prediction_sets(a) for a in alpha_levels] 

# %%
idx = 0
tt = -1
x_id = 16

# x_points = pred_a[idx, tt][x_id, :]
# x_points = np.arange(S)
x_points = np.linspace(-1, 1, 33)

fig, ax = plt.subplots()
plt.title("CQR", fontsize=72)
[plt.fill_between(x_points, pred_sets[i][0][idx, tt][x_id,:], pred_sets[i][1][idx, tt][x_id,:], color = cols[i], alpha=0.7) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma_r"), ax=ax)
plt.plot(x_points, y_response[idx, tt][x_id, :], linewidth = 1, color = "black", label = "exact", marker='o', ms=2, mec = 'white')
plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{u}")
plt.legend()
plt.savefig("wave_unet_cqr.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()
# %%
#Performing the Calibration using Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#############################################################
# Conformal Prediction Residuals
#############################################################
model_50 = UNet2d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_mean.pth', map_location='cpu'))

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

cal_scores = nonconf_score_abs(cal_u.numpy(), cal_mean.numpy())
qhat = calibrate(cal_scores, n, alpha)
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

val_mean = val_mean.numpy()
prediction_sets = [val_mean - qhat, val_mean + qhat]

# %% 
print('Conformal by way Residual')
# Calculate empirical coverage (before and after calibration)
empirical_coverage = emp_cov(prediction_sets, y_response) 
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

t2 = default_timer()
print('Residuals, time used:', t2-t1)

#Estimating the tightness of fit
tightness_metric = est_tight(prediction_sets, y_response)
print(f"Tightness of the coverage : Average of the distance between error bars {tightness_metric}")

# %% 
#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for ii in tqdm(range(len(alpha_levels))):
    qhat = calibrate(cal_scores, n, alpha_levels[ii])
    prediction_sets =  [pred_mean.numpy() - qhat, pred_mean.numpy() + qhat]
    emp_cov_res.append(emp_cov(prediction_sets, y_response))

# %% 
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 
#PLots

def get_prediction_sets(alpha):
    qhat = calibrate(cal_scores, n, alpha)
    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return  prediction_sets


alpha_levels = np.arange(0.05, 0.95, 0.1)
coverage_levels = (1 - alpha_levels)
cols = cm.plasma_r(coverage_levels)
pred_sets = [get_prediction_sets(a) for a in alpha_levels] 
# %%
idx = 0
tt = -1
x_id = 16

# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)
x_points = np.linspace(-1, 1, 33)

fig, ax = plt.subplots()
plt.title("Residuals", fontsize=72)
[plt.fill_between(x_points, pred_sets[i][0][idx, tt][x_id,:], pred_sets[i][1][idx, tt][x_id,:], color = cols[i], alpha=0.7) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma_r"), ax=ax)
plt.plot(x_points, y_response[idx, tt][x_id, :], linewidth = 1, color = "black", label = "exact", marker='o', ms=2, mec = 'white')
plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{u}")
plt.legend()
plt.savefig("wave_unet_residual.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()
# %%
##############################
# Conformal using Dropout 
##############################

model_dropout = UNet2d_dropout(T_in, step, width)
# model_dropout.load_state_dict(torch.load(model_loc + 'Unet_Wave_frigid-hill_dropout.pth', map_location='cpu'))
# model_dropout.load_state_dict(torch.load(model_loc + 'Unet_Wave_dropout.pth', map_location='cpu'))
model_dropout.load_state_dict(torch.load(model_loc + 'Unet_Wave_piercing-body.pth', map_location='cpu'))

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

cal_upper = cal_upper.numpy()
cal_lower = cal_lower.numpy()

cal_scores = nonconf_score_lu(cal_u.numpy(), cal_lower, cal_upper)
qhat = calibrate(cal_scores, n, alpha)

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

val_lower = val_lower.numpy()
val_upper = val_upper.numpy()

prediction_sets_uncalibrated = [val_lower, val_upper]
prediction_sets_calibrated = [val_lower - qhat, val_upper + qhat]


# %% 
y_response = pred_u.numpy()

print('Conformal by way of Dropout')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = emp_cov(prediction_sets_uncalibrated, y_response)
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = emp_cov(prediction_sets_calibrated, y_response)
print(f"The empirical coverage after calibration is: {empirical_coverage}")
t2 = default_timer()
print('Dropout, time used:', t2-t1)

#Estimating the tightness of fit
tightness_metric = est_tight(prediction_sets_calibrated, y_response)
print(f"Tightness of the coverage : Average of the distance between error bars {tightness_metric}")

# %% 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_dropout = []
for ii in tqdm(range(len(alpha_levels))):
    qhat = calibrate(cal_scores, n, alpha_levels[ii])
    prediction_sets =  [val_lower - qhat, val_upper + qhat]
    emp_cov_dropout.append(emp_cov(prediction_sets, y_response))

# %% 
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
#PLots

def get_prediction_sets(alpha):
    qhat = calibrate(cal_scores, n, alpha)
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return  prediction_sets


alpha_levels = np.arange(0.05, 0.95, 0.1)
coverage_levels = (1 - alpha_levels)
cols = cm.plasma_r(coverage_levels)
pred_sets = [get_prediction_sets(a) for a in alpha_levels] 
# %%
idx = 0
tt = -1
x_id = 16

# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)
x_points = np.linspace(-1, 1, 33)

fig, ax = plt.subplots()
plt.title("Dropout", fontsize=72)
[plt.fill_between(x_points, pred_sets[i][0][idx, tt][x_id,:], pred_sets[i][1][idx, tt][x_id,:], color = cols[i], alpha=0.7) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma_r"), ax=ax)
plt.plot(x_points, y_response[idx, tt][x_id, :], linewidth = 1, color = "black", label = "exact", marker='o', ms=2, mec = 'white')
plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{u}")
plt.legend()
plt.savefig("wave_unet_dropout.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()

# %%
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 10.0
plt.rcParams['ytick.minor.size'] = 10.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75)
plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.75)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.75)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.75)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title("Wave", fontsize=72)
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("wave_unet_comparison.svg", format="svg", bbox_inches='tight')
plt.show()



# %%

idx = np.random.randint(0, ncal) 


cal_u_decoded = y_normalizer.decode(cal_u)
cal_mean_decoded = y_normalizer.decode(torch.Tensor(cal_mean))

u_field = cal_u[idx]

v_min_1 = torch.min(u_field[0])
v_max_1 = torch.max(u_field[0])

v_min_2 = torch.min(u_field[int(T/2)])
v_max_2 = torch.max(u_field[int(T/2)])

v_min_3 = torch.min(u_field[-1])
v_max_3 = torch.max(u_field[-1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in))
ax.set_ylabel('Solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int((T/2+T_in))))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[-1], cmap=cm.coolwarm,  extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(T+T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

u_field = cal_mean[idx]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[int(T/2)], cmap=cm.coolwarm,  extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[-1], cmap=cm.coolwarm,  extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 10.0
plt.rcParams['ytick.minor.size'] = 10.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'

#Plotting the cell-wise CP estimation

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

idx = 10
t_idx = -1

x_len = 5
y_len = 5
x_slice = int(y_response.shape[2] / x_len)
y_slice = x_slice

y_response_slice = y_response[idx, t_idx, ::x_slice, ::x_slice]
mean_slice = mean[idx, t_idx, ::x_slice, ::x_slice]
uncalib_lb_slice = prediction_sets_uncalibrated[0][idx, t_idx, ::x_slice, ::x_slice]
uncalib_ub_slice = prediction_sets_uncalibrated[1][idx, t_idx, ::x_slice, ::x_slice]
calib_lb_slice = prediction_sets[0][idx, t_idx, ::x_slice, ::x_slice]
calib_ub_slice = prediction_sets[1][idx, t_idx, ::x_slice, ::x_slice]

# Create a t_len x x_len grid of cells using gridspec
plt.figure()
gs = gridspec.GridSpec(x_len, y_len, wspace=0, hspace=0, width_ratios=list(np.ones((x_len))), height_ratios=list(np.ones((x_len))))

y_max = np.max(calib_ub_slice)
y_min = np.min(calib_lb_slice)

for aa in range(x_len):
    for bb in range(y_len):
        ax = plt.subplot(gs[aa, bb])
        # ax.scatter(x[::x_slice][bb], mean_slice[aa, bb], color='navy', alpha=0.8, marker='o')
        # ax.errorbar(x[::x_slice][bb], mean_slice[aa, bb].flatten(), yerr=(uncalib_ub_slice[aa, bb] - uncalib_lb_slice[aa, bb]).flatten(), label='Prediction', color='navy', fmt='o', alpha=0.8) #Uncalibrated
        ax.errorbar(x[::x_slice][bb], mean_slice[aa, bb].flatten(), yerr=(calib_ub_slice[aa, bb] - calib_lb_slice[aa, bb]).flatten(), label='Prediction', color='navy', fmt='o', alpha=0.8, ecolor='firebrick', ms= 2, elinewidth=2) #Calibrated 
        ax.set_ylim(bottom=y_min, top=y_max)

        ax.set(xticks=[], yticks=[])

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# # show only the outside spines
# for ax in fig.get_axes():
#     ss = ax.get_subplotspec()
#     ax.spines.top.set_visible(ss.is_first_row())
#     ax.spines.bottom.set_visible(ss.is_last_row())
#     ax.spines.left.set_visible(ss.is_first_col())
#     ax.spines.right.set_visible(ss.is_last_col())

plt.tight_layout()


# plt.savefig('Plots/wave_unet_marginal_cells_calibrated.svg', format="svg", bbox_inches='tight', transparent='True')

# %%
# %%
plt.figure()
plt.imshow(mean_slice, cmap='plasma_r')
# plt.imshow( val_mean[idx, t_idx], cmap=cm.coolwarm,)
# plt.imshow()
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
# plt.savefig('Plots/wave_unet_marginal_cells_slice.svg', format="svg", bbox_inches='tight', transparent='True')
# %%
