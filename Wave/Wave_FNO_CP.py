#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FNO built using PyTorch to model the 2D Wave Equation. 
Dataset buitl by changing by performing a LHS across the x,y pos and amplitude of the initial gaussian distibution
Code for the spectral solver can be found in : https://github.com/farscape-project/PINNs_Benchmark

----------------------------------------------------------------------------------------------------------------------------------------

Experimenting with a range of UQ Methods:
    1. Dropout
    2. Quantile Regression 
    3. NN Ensemble 

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Type": 'FNO',
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
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'LP',
                 "UQ": 'Dropout', #None, Dropout
                 "Pinball Gamma": 'NA',
                 "Dropout Rate": 0.1
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
u = u.permute(0, 2, 3, 1)
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
modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']


# %%
#Chunking the data. 
train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

cal_a = u[ntrain:ntrain+ncal,:,:,:T_in]
cal_u = u[ntrain:ntrain+ncal,:,:,T_in:T+T_in]

pred_a = u[ntrain+ncal:ntrain+ncal+npred,:,:,:T_in]
pred_u = u[ntrain+ncal:ntrain+ncal+npred,:,:,T_in:T+T_in]


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
#Performing the Calibration using Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#############################################################
# Conformal Prediction Residuals
#############################################################

model_50 = FNO2d(modes, modes, width, T_in, step, x, y)
model_50.load_state_dict(torch.load(model_loc + 'FNO_Wave_fno.pth', map_location='cpu'))

# %% 
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
            cal_mean = torch.cat((cal_mean, pred), -1)       

        xx = torch.cat((xx[..., step:], pred), dim=-1)

cal_scores = nonconf_score_abs(cal_u.numpy(), cal_mean.numpy())
qhat = calibrate(cal_scores, n, alpha)
# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()

with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        pred_mean = model_50(xx)

        if tt == 0:
            val_mean = pred_mean
        else:     
            val_mean = torch.cat((val_mean, pred_mean), -1)       

        xx = torch.cat((xx[..., step:], pred_mean), dim=-1)

val_mean = val_mean.numpy()
prediction_sets = [val_mean - qhat, val_mean + qhat]

pred_sets = get_prediction_sets([cal_mean, cal_u], val_mean, alpha, nonconf_score='abs')
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
    prediction_sets =  [val_mean - qhat, val_mean + qhat]
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

#Plotting the evolution of solution, fno output and the cp error 
plt.rcdefaults()
fig = plt.figure(figsize=(10.5, 8))

idx = np.random.randint(0, npred) 

idx = 20

# if configuration['Log Normalisation'] == 'Yes':
#     test_u = torch.exp(test_u)
#     pred_set = torch.exp(pred_set)

pred_u_decoded = y_normalizer.decode(pred_u)
val_mean_decoded = y_normalizer.decode(torch.Tensor(val_mean))
calibrated_decoded = y_normalizer.decode(torch.Tensor(prediction_sets[1] - prediction_sets[0]))

u_field = pred_u_decoded[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(T/2)])
v_max_2 = torch.max(u_field[:, :, int(T/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

c_min_1 = torch.min(calibrated_decoded[idx][:, :, 0])
c_max_1 = torch.max(calibrated_decoded[idx][:, :, 0])

c_min_2 = torch.min(calibrated_decoded[idx][:, :, int(T/2)])
c_max_2 = torch.max(calibrated_decoded[idx][:, :, int(T/2)])

c_min_3 = torch.min(calibrated_decoded[idx][:, :, -1])
c_max_3 = torch.max(calibrated_decoded[idx][:, :, -1])

# fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(3,3,1)
pcm =ax.imshow(u_field[:, :,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
# ax.title.set_text('Initial')
ax.set_title('t='+ str(T_in), fontsize=20)
ax.set_ylabel('Solution', fontsize=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,2)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.set_title('t='+ str(int(T_in + (T/2))), fontsize=20)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.set_title('t='+ str(T+T_in), fontsize=20)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

u_field = val_mean_decoded[idx]

ax = fig.add_subplot(3,3,4)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO', fontsize=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,5)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

# u_field = uncalibrated_decoded[idx]

# ax = fig.add_subplot(4,3,7)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_1, vmax=c_max_1)
# ax.set_ylabel('Uncalibrated Error')

# fig.colorbar(pcm, pad=0.05)

# ax = fig.add_subplot(4,3,8)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_2, vmax=c_max_2)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(4,3,9)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_3, vmax=c_max_3)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)

u_field = calibrated_decoded[idx]

ax = fig.add_subplot(3,3,7)
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_1, vmax=c_max_1)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax.set_ylabel('Calibrated Error', fontsize=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,8)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_2, vmax=c_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,9)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_3, vmax=c_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)
fig.tight_layout()

plt.savefig('Plots/wave_fno_heatmaps.svg', format="svg", bbox_inches='tight', transparent='True')
plt.savefig('Plots/wave_fno_heatmaps.pdf', format="pdf", bbox_inches='tight', transparent='True')

# %%
# ##############################
# # Conformal using Dropout 
# ##############################

model_dropout = FNO2d_dropout(modes, modes, width, T_in, step, x, y)
model_dropout.load_state_dict(torch.load(model_loc + 'FNO_Wave_plastic-serval_dropout.pth', map_location='cpu'))


# %%
#Performing the Calibration for Dropout

t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval_fno(model_dropout, xx, step)

        if tt == 0:
            cal_mean = mean
            cal_std = std
        else:
            cal_mean = torch.cat((cal_mean, mean), -1)       
            cal_std = torch.cat((cal_std, std), -1)       

        xx = torch.cat((xx[..., step:], mean), dim=-1)


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
        mean, std = Dropout_eval_fno(model_dropout, xx, step)

        if tt == 0:
            val_mean = mean
            val_std = std
        else:
            val_mean = torch.cat((val_mean, mean), -1)       
            val_std = torch.cat((val_std, std), -1)       

        xx = torch.cat((xx[..., step:], mean), dim=-1)

val_upper = val_mean + val_std
val_lower = val_mean - val_std

val_lower = val_lower.numpy()
val_upper = val_upper.numpy()

prediction_sets_uncalibrated = [val_lower, val_upper]
prediction_sets_calibrated = [val_lower - qhat, val_upper + qhat]

pred_sets = get_prediction_sets([cal_mean, cal_std, cal_u], [val_mean, val_std], alpha, nonconf_score='std')
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
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20


# %%

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20


# %%

# def get_prediction_sets(alpha):
#     qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
#     prediction_sets = [val_mean - qhat, val_mean + qhat]

# cols = cm.plasma(alpha_levels)
# pred_sets = [get_prediction_sets(a) for a in alpha_levels] 


# idx = 0
# tt = -1
# x_id = 16

# # x_points = pred_a[idx, tt][x_id, :]
# x_points = np.arange(S)
# pred_sets = prediction_sets
# fig, ax = plt.subplots()
# [plt.fill_between(x_points, pred_sets[i][0][idx,:, :, tt][x_id,:], pred_sets[i][1][idx,:,:, tt][x_id,:], color = cols[i]) for i in range(len(alpha_levels))]
# fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)

# plt.plot(x_points, y_response[idx, tt][x_id, :], linewidth = 4, color = "black", label = "exact")
# plt.legend()




#Plotting the evolution of solution, fno output and the cp error 
mpl.rcParams['figure.figsize']=(16, 16)

idx = np.random.randint(0, npred) 

idx = 0

# if configuration['Log Normalisation'] == 'Yes':
#     test_u = torch.exp(test_u)
#     pred_set = torch.exp(pred_set)

pred_u_decoded = y_normalizer.decode(pred_u)
val_mean_decoded = y_normalizer.decode(torch.Tensor(val_mean))
uncalibrated_decoded = y_normalizer.decode(torch.Tensor(prediction_sets_uncalibrated[1] - prediction_sets_uncalibrated[0]))
calibrated_decoded = y_normalizer.decode(torch.Tensor(prediction_sets[1] - prediction_sets[0]))

u_field = pred_u_decoded[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(T/2)])
v_max_2 = torch.max(u_field[:, :, int(T/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

c_min_1 = torch.min(calibrated_decoded[idx][:, :, 0])
c_max_1 = torch.max(calibrated_decoded[idx][:, :, 0])

c_min_2 = torch.min(calibrated_decoded[idx][:, :, int(T/2)])
c_max_2 = torch.max(calibrated_decoded[idx][:, :, int(T/2)])

c_min_3 = torch.min(calibrated_decoded[idx][:, :, -1])
c_max_3 = torch.max(calibrated_decoded[idx][:, :, -1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(4,3,1)
pcm =ax.imshow(u_field[:, :,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)
 

ax = fig.add_subplot(4,3,2)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int(T_in + (T/2))))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(4,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(T+T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


u_field = val_mean_decoded[idx]

ax = fig.add_subplot(4,3,4)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(4,3,5)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(4,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = uncalibrated_decoded[idx]

ax = fig.add_subplot(4,3,7)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_1, vmax=c_max_1)
ax.set_ylabel('Uncalibrated Error')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(4,3,8)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_2, vmax=c_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(4,3,9)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_3, vmax=c_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = calibrated_decoded[idx]

ax = fig.add_subplot(4,3,10)
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_1, vmax=c_max_1)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax.set_ylabel('Calibrated Error')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(4,3,11)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_2, vmax=c_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(4,3,12)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_3, vmax=c_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

# plt.savefig('Plots/Plots/wave_fno_heatmaps.svg', format="svg", bbox_inches='tight', transparent='True')


# %% 
################################################################################################################
#Testing on the data  with half speed
################################################################################################################

model_50 = FNO2d(modes, modes, width, T_in, step, x, y)
model_50.load_state_dict(torch.load(model_loc + 'FNO_Wave_fno.pth', map_location='cpu'))

# %% 
t1 = default_timer()

data =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_5K.npz')
data_halfspeed =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_halfspeed.npz')
u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)
u_hs =  torch.from_numpy(data_halfspeed['u'].astype(np.float32))
u_hs = u_hs.permute(0, 2, 3, 1)
xx, yy = np.meshgrid(x,y)

ntrain = 500
ncal = 500
npred = 500
S = 33 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']


# %%    
#Chunking the data. 
train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

cal_a = u_hs[:ncal,:,:,:T_in]
cal_u = u_hs[:ncal,:,:,T_in:T+T_in]

pred_a = u_hs[ncal:ncal+npred,:,:,:T_in]
pred_u = u_hs[ncal:ncal+npred,:,:,T_in:T+T_in]

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
            cal_mean = torch.cat((cal_mean, pred), -1)       

        xx = torch.cat((xx[..., step:], pred), dim=-1)


cal_scores = nonconf_score_abs(cal_u.numpy(), cal_mean.numpy())
qhat = calibrate(cal_scores, n, alpha)

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()

with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        pred_mean = model_50(xx)

        if tt == 0:
            val_mean = pred_mean
        else:     
            val_mean = torch.cat((val_mean, pred_mean), -1)       

        xx = torch.cat((xx[..., step:], pred_mean), dim=-1)

# val_mean = val_mean.numpy()
prediction_sets = [val_mean.numpy() - qhat, val_mean.numpy() + qhat]

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
emp_cov_res_hs = []
pred_sets_res_hs = []
for ii in tqdm(range(len(alpha_levels))):
    qhat = calibrate(cal_scores, n, alpha_levels[ii])
    prediction_sets =  [val_mean.numpy() - qhat, val_mean.numpy() + qhat]
    pred_sets_res_hs.append(prediction_sets)
    emp_cov_res_hs.append(emp_cov(prediction_sets, y_response))



# %% 
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Normal Speed' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res_hs, label='Half Speed' ,ls='--', color='firebrick', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5 
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20

# %% 
alpha_levels = np.arange(0.05, 0.95, 0.1)
cols = cm.plasma(alpha_levels)

idx = 23
tt = -1
x_id = 16

i = 2
# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)

fig, ax = plt.subplots()
# [plt.fill_between(x_points, pred_sets_res_hs[i][0][idx, :,:,tt][x_id,:], pred_sets_res_hs[i][1][idx, :,:, tt][x_id,:], color = cols[i], alpha=0.5) for i in range(len(alpha_levels))]
plt.fill_between(x_points, pred_sets_res_hs[i][0][idx, :,:,tt][x_id,:], pred_sets_res_hs[i][1][idx, :,:, tt][x_id,:], color = cols[i], alpha=0.5)
# fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)

plt.plot(x_points, y_response[idx,:,:, tt][x_id, :], linewidth = 4, color = "black", label = "exact")
plt.legend()
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20

# %% 

#Plotting the evolution of solution, fno output and the cp error 
plt.rcdefaults()
fig = plt.figure(figsize=(10.5, 8))

idx = np.random.randint(0, npred) 

idx = 10

# if configuration['Log Normalisation'] == 'Yes':
#     test_u = torch.exp(test_u)
#     pred_set = torch.exp(pred_set)

pred_u_decoded = y_normalizer.decode(pred_u)
val_mean_decoded = y_normalizer.decode(torch.Tensor(val_mean))
calibrated_decoded = y_normalizer.decode(torch.Tensor(prediction_sets[1] - prediction_sets[0]))

u_field = pred_u_decoded[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(T/2)])
v_max_2 = torch.max(u_field[:, :, int(T/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

c_min_1 = torch.min(calibrated_decoded[idx][:, :, 0])
c_max_1 = torch.max(calibrated_decoded[idx][:, :, 0])

c_min_2 = torch.min(calibrated_decoded[idx][:, :, int(T/2)])
c_max_2 = torch.max(calibrated_decoded[idx][:, :, int(T/2)])

c_min_3 = torch.min(calibrated_decoded[idx][:, :, -1])
c_max_3 = torch.max(calibrated_decoded[idx][:, :, -1])

# fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(3,3,1)
pcm =ax.imshow(u_field[:, :,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
# ax.title.set_text('Initial')
ax.set_title('t='+ str(T_in), fontsize=20)
ax.set_ylabel('Solution', fontsize=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,2)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.set_title('t='+ str(int(T_in + (T/2))), fontsize=20)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.set_title('t='+ str(T+T_in), fontsize=20)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

u_field = val_mean_decoded[idx]

ax = fig.add_subplot(3,3,4)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO', fontsize=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,5)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

# u_field = uncalibrated_decoded[idx]

# ax = fig.add_subplot(4,3,7)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_1, vmax=c_max_1)
# ax.set_ylabel('Uncalibrated Error')

# fig.colorbar(pcm, pad=0.05)

# ax = fig.add_subplot(4,3,8)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_2, vmax=c_max_2)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(4,3,9)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_3, vmax=c_max_3)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)

u_field = calibrated_decoded[idx]

ax = fig.add_subplot(3,3,7)
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_1, vmax=c_max_1)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax.set_ylabel('Calibrated Error', fontsize=20)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,8)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_2, vmax=c_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = fig.add_subplot(3,3,9)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=c_min_3, vmax=c_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=12)
fig.tight_layout()

plt.savefig('Plots/wave_fno_hs_heatmaps.svg', format="svg", bbox_inches='tight', transparent='True')
plt.savefig('Plots/wave_fno_hs_heatmaps.pdf', format="pdf", bbox_inches='tight', transparent='True')

# %% 
#With Dropout on Half Speed
model_dropout = FNO2d_dropout(modes, modes, width, T_in, step, x, y)
model_dropout.load_state_dict(torch.load(model_loc + 'FNO_Wave_plastic-serval_dropout.pth', map_location='cpu'))

# %% 
#Performing the Calibration for Dropout
t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval_fno(model_dropout, xx, step)

        if tt == 0:
            cal_mean = mean
            cal_std = std
        else:
            cal_mean = torch.cat((cal_mean, mean), -1)       
            cal_std = torch.cat((cal_std, std), -1)       

        xx = torch.cat((xx[..., step:], mean), dim=-1)

cal_upper = cal_mean + cal_std
cal_lower = cal_mean - cal_std

cal_scores = nonconf_score_lu(cal_mean.numpy(), cal_lower.numpy(), cal_upper.numpy())
qhat = calibrate(cal_scores, n, alpha)
# %% 
#Obtaining the Prediction Sets
with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval_fno(model_dropout, xx, step)

        if tt == 0:
            val_mean = mean
            val_std = std
        else:
            val_mean = torch.cat((val_mean, mean), -1)       
            val_std = torch.cat((val_std, std), -1)       

        xx = torch.cat((xx[..., step:], mean), dim=-1)

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

# # %% 
# def calibrate_dropout(alpha):
#     with torch.no_grad():
#         xx = cal_a

#         for tt in tqdm(range(0, T, step)):
#             mean, std = Dropout_eval_fno(model_dropout, xx, step)

#             if tt == 0:
#                 cal_mean = mean
#                 cal_std = std
#             else:
#                 cal_mean = torch.cat((cal_mean, mean), -1)       
#                 cal_std = torch.cat((cal_std, std), -1)       

#             xx = torch.cat((xx[..., step:], mean), dim=-1)


#     # cal_mean = cal_mean.numpy()

#     cal_upper = cal_mean + cal_std
#     cal_lower = cal_mean - cal_std

#     cal_scores = np.maximum(cal_u.numpy()-cal_upper.numpy(), cal_lower.numpy()-cal_u.numpy())
#     qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

        
#     prediction_sets = [val_mean - qhat, val_mean + qhat]
#     empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()

#     return prediction_sets, empirical_coverage

# %%
#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_dropout_hs = []
pred_sets_dropout_hs = []
for ii in tqdm(range(len(alpha_levels))):
    qhat = calibrate(cal_scores, n, alpha_levels[ii])
    prediction_sets =  [val_mean.numpy() - qhat, val_mean.numpy() + qhat]
    pred_sets_dropout_hs.append(prediction_sets)
    emp_cov_dropout_hs.append(emp_cov(prediction_sets, y_response))


# %% 

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout_hs, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20

# %% 

mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 0.5
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

lw = 0.5
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75, linewidth=lw)
plt.plot(1-alpha_levels, emp_cov_res, label='Normal Speed - Residual' ,ls='-', color='chocolate', alpha=0.75, linewidth=lw)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Normal Speed - Dropout',  color='firebrick', ls='-',  alpha=0.75, linewidth=lw)
plt.plot(1-alpha_levels, emp_cov_res_hs, label='Half Speed - Residual' ,ls='-.', color='navy', alpha=0.75, linewidth=lw, marker='^')
plt.plot(1-alpha_levels, emp_cov_dropout_hs, label='Half Speed Dropout',  color='teal', ls='--',  alpha=0.75, linewidth=lw, marker='>')
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title("Wave - FNO", fontsize=72)
plt.grid()
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
plt.savefig("wave_hs_validation.svg", format="svg", bbox_inches='tight')
plt.show()
# %% 

alpha_levels = np.arange(0.05, 0.95, 0.1)
cols = cm.plasma(alpha_levels)

idx = 23
tt = -1
x_id = 16

i = 2
# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)

fig, ax = plt.subplots()
# [plt.fill_between(x_points, pred_sets_dropout_hs[i][0][idx, :,:,tt][x_id,:], pred_sets_res_hs[i][1][idx, :,:, tt][x_id,:], color = cols[i], alpha=0.8) for i in range(len(alpha_levels))]
plt.fill_between(x_points, pred_sets_res_hs[i][0][idx, :,:,tt][x_id,:], pred_sets_res_hs[i][1][idx, :,:, tt][x_id,:], color = cols[4], alpha=0.7, label='Residual')
plt.fill_between(x_points, pred_sets_dropout_hs[i][0][idx, :,:,tt][x_id,:], pred_sets_res_hs[i][1][idx, :,:, tt][x_id,:], color = cols[-1], alpha=1.0, label='Dropout')

# fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)

plt.plot(x_points, y_response[idx,:,:, tt][x_id, :], linewidth = 1, color = "black", label = "exact",  marker='o', ms=2, mec = 'white')
plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{u}")
plt.legend()
# plt.grid()
plt.savefig("wave_hs_comparison.pdf", format="pdf", bbox_inches='tight')
plt.show()


# %%

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %% 
idx = 5


cal_u_decoded = y_normalizer.decode(cal_u)
cal_mean_decoded = y_normalizer.decode(torch.Tensor(cal_mean))

u_field = cal_u_decoded[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(T/2)])
v_max_2 = torch.max(u_field[:, :, int(T/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in))
ax.set_ylabel('Solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int((T/2+T_in))))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(T+T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

u_field = cal_mean_decoded[idx]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm,  extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[-1.0, 1.0, -1.0, 1.0], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

plt.show()
# %%
#Plotting the cell-wise CP estimation. 


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


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

idx = 10
t_idx = 1

x_len = 5
y_len = 5
x_slice = int(y_response.shape[2] / x_len)
y_slice = x_slice

y_response_slice = y_response[idx, ::x_slice, ::x_slice, t_idx]
mean_slice = val_mean[idx, ::x_slice, ::x_slice, t_idx]
# uncalib_lb_slice = prediction_sets_uncalibrated[0][idx, ::x_slice, ::x_slice, t_idx]
# uncalib_ub_slice = prediction_sets_uncalibrated[1][idx, ::x_slice, ::x_slice, t_idx]
calib_lb_slice = prediction_sets[0][idx, ::x_slice, ::x_slice, t_idx]
calib_ub_slice = prediction_sets[1][idx, ::x_slice, ::x_slice, t_idx]

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

# %%
plt.figure()
# plt.imshow(mean_slice, cmap='plasma_r')
plt.imshow(mean_slice, cmap=cm.coolwarm)
# plt.imshow(y_response[idx,:,:, t_idx], cmap=cm.coolwarm)
# plt.imshow()
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
# plt.savefig('Plots/wave_unet_marginal_cells_slice.svg', format="svg", bbox_inches='tight', transparent='True')
# %%
