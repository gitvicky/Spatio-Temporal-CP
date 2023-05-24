#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Wave Equation 
Conformal Prediction using various Conformal Score estimates

Studying the influence of calibration dataset size on the CP performance. 
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

#############################################################
# Conformalised Quantile Regression 
#############################################################

# #Loading the Trained Model

#Invoking the trained models 
model_05 = UNet2d(T_in, step, width)
# model_05.load_state_dict(torch.load(model_loc + 'Unet_Wave_indigo-reload_QR_05.pth', map_location='cpu'))
model_05.load_state_dict(torch.load(model_loc + 'Unet_Wave_lower.pth', map_location='cpu'))

model_95 = UNet2d(T_in, step, width)
# model_95.load_state_dict(torch.load(model_loc + 'Unet_Wave_round-underwriter_QR_95.pth', map_location='cpu'))
model_95.load_state_dict(torch.load(model_loc + 'Unet_Wave_upper.pth', map_location='cpu'))

model_50 = UNet2d(T_in, step, width)
# model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_shy-bevel_QR_50.pth', map_location='cpu'))
model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_mean.pth', map_location='cpu'))

# %%
#Getting the Coverage
def calibrate_cqr(x_cal, y_cal, alpha):
    n = len(x_cal)

    with torch.no_grad():
        cal_lower = model_05(torch.Tensor(x_cal)).numpy()
        cal_upper = model_95(torch.Tensor(x_cal)).numpy()

    cal_scores = np.maximum(y_cal.numpy()-cal_upper, cal_lower-y_cal.numpy())
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

# %% 

with torch.no_grad():
    val_lower = model_05(torch.FloatTensor(pred_a)).numpy()
    val_upper = model_95(torch.FloatTensor(pred_a)).numpy()

y_response = pred_u.numpy()


alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_cqr = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = cal_a[:cal_split], cal_u[:cal_split]

    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_cqr(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_cqr.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_cqr[0], label='250', color='teal', ls='--',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_cqr[1], label='500' ,ls='-.', color='maroon', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_cqr[2], label='750',  color='mediumblue', ls=':',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_cqr[3], label='1000',  color='purple', ls='--',  alpha=0.8)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title('CQR')
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("Wave_Unet_cal_size_cqr.svg", format="svg", bbox_inches='tight')
plt.show()

# %%
#############################################################
# Conformal using Residuals
#############################################################
#Loading the Trained Model

model_50 = UNet2d(T_in, step, width)
model_50.load_state_dict(torch.load(model_loc + 'Unet_Wave_mean.pth', map_location='cpu'))

# %% 
def conf_metric_res(x_cal, y_cal): 

    with torch.no_grad():
        median =model_50(torch.FloatTensor(x_cal)).numpy()
    return np.abs(y_cal - median)

def calibrate_res(x_cal, y_cal, alpha):
    n = cal_split

    cal_scores = conf_metric_res(x_cal, y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')


    prediction_sets = [prediction - qhat, prediction + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

with torch.no_grad():
    prediction = model_50(torch.FloatTensor(pred_a)).numpy()
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
plt.savefig("Wave_Unet_cal_size_residual.svg", format="svg", bbox_inches='tight')
plt.show()


# %% 
#############################################################
# Conformal using dropout
#############################################################
#Loading the Trained Model
model_dropout = UNet2d_dropout(T_in, step, width)
model_dropout.load_state_dict(torch.load(model_loc + 'Unet_Wave_dropout.pth', map_location='cpu'))


 # %% 
def calibrate_dropout(x_cal, y_cal, alpha):
    n = cal_split
    
    with torch.no_grad():
        cal_mean, cal_std = Dropout_eval(model_dropout, x_cal, step)

    cal_upper = cal_mean + cal_std
    cal_lower = cal_mean - cal_std

    cal_scores = np.maximum(y_cal-cal_upper, cal_lower-y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_lower.numpy()- qhat, val_upper.numpy() + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


with torch.no_grad():
    val_mean, val_std = Dropout_eval(model_dropout, pred_a, step)

val_upper = val_mean + val_std
val_lower = val_mean - val_std
y_response = pred_u.numpy()

alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_dropout = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = cal_a[:cal_split], cal_u[:cal_split], 
    
    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_dropout(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_dropout.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)

# %% 

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_dropout[0], label='250', color='teal', ls='--',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_dropout[1], label='500' ,ls='-.', color='maroon', alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_dropout[2], label='750',  color='mediumblue', ls=':',  alpha=0.8)
plt.plot(1-alpha_levels, emp_cov_dropout[3], label='1000',  color='purple', ls='--',  alpha=0.8)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title('Dropout')
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("Wave_Unet_cal_size_dropout.svg", format="svg", bbox_inches='tight')
plt.show()
# %%