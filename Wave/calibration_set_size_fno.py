#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave Equation 
Conformal Prediction using various Conformal Score estimates

Studying the influence of calibration dataset size on the CP performance. 
"""

# %% 
#Using FNO 

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

# %%
#Importing the necessary 
import os 
import numpy as np 
import math
from tqdm import tqdm 
from timeit import default_timer
import matplotlib as mpl 
from matplotlib import pyplot as plt 

from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True

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

import torch 
import torch.nn as nn

from pyDOE import lhs
from utils import *


# %% 
#Setting the seeds and the path for the run. 
torch.manual_seed(0)
np.random.seed(0)
path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #positioning the GPUs if they are available. only works for Nvidia at the moment 

if not os.path.exists(path + '/Models'):
    os.mkdir(path  + '/Models')

model_loc = path + '/Models/'
data_loc = path

# %% 
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

ntrain = 500
ncal = 1000
npred = 1000
S = 33 #Grid Size


# %%

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
#Using Residuals 
model_50 = FNO2d(modes, modes, width, T_in, step, x, y)
model_50.load_state_dict(torch.load(model_loc + 'FNO_Wave_fno.pth', map_location='cpu'))
# %% 

# def conf_metric_res(x_cal, y_cal): 
#     with torch.no_grad():
#         xx = x_cal
#         for tt in tqdm(range(0, T, step)):
#             pred = model_50(xx)
#             if tt == 0:
#                 cal_mean = pred
#             else:
#                 cal_mean = torch.cat((cal_mean, pred), -1)       
#             xx = torch.cat((xx[..., step:], pred), dim=-1)

#     return np.abs(cal_mean.numpy() - y_cal.numpy())

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


def conf_metric_res(x_cal, y_cal):
    with torch.no_grad():
        xx = x_cal

        for tt in tqdm(range(0, T, step)):
            pred = model_50(xx)

            if tt == 0:
                cal_mean = pred

            else:
                cal_mean = torch.cat((cal_mean, pred), -1)       

            xx = torch.cat((xx[..., step:], pred), dim=-1)


    cal_mean = cal_mean.numpy()
    cal_scores = np.abs(y_cal.numpy()-cal_mean)
    return cal_scores

def calibrate_res(x_cal, y_cal, alpha):
    n = cal_split
    cal_scores = conf_metric_res(x_cal, y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

y_response = pred_u.numpy()


alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_res = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = cal_a[:cal_split], cal_u[:cal_split], 
    
    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_res(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_res.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)
# %% 
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[0], label='250', color='teal', ls='--',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[1], label='500' ,ls='-.', color='maroon', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[2], label='750',  color='mediumblue', ls=':',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[3], label='1000',  color='purple', ls='--',  alpha=0.8)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title('Residual')
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("Plots/Wave_FNO_cal_size_residual.svg", format="svg", bbox_inches='tight')
plt.show()
# %%
#Using Dropout
model_dropout = FNO2d_dropout(modes, modes, width, T_in, step, x, y)
model_dropout.load_state_dict(torch.load(model_loc + 'FNO_Wave_plastic-serval_dropout.pth', map_location='cpu'))

# %% 
y_response = pred_u.numpy()

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


val_lower = val_mean - val_std
val_upper = val_mean + val_std

# val_lower = val_lower.numpy()
# val_upper = val_upper.numpy()


def calibrate_dropout(x_cal, y_cal, alpha):
    n = cal_split
    with torch.no_grad():
        xx = x_cal

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

    cal_scores = np.maximum(y_cal-cal_upper, cal_lower-y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_lower.numpy() - qhat, val_upper.numpy() + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_res = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = cal_a[:cal_split], cal_u[:cal_split], 
    
    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_dropout(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_res.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)
# %% 
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[0], label='250', color='teal', ls='--',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[1], label='500' ,ls='-.', color='maroon', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[2], label='750',  color='mediumblue', ls=':',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_res[3], label='1000',  color='purple', ls='--',  alpha=0.8)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title('Dropout')
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("Plots/Wave_FNO_cal_size_dropout.svg", format="svg", bbox_inches='tight')
plt.show()
# %%
