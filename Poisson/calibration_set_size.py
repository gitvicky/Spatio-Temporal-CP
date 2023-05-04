#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31 October 2022

@author: vgopakum, agray, lzanisi

Neural Network (MLP) built using PyTorch to model the 1D Poisson Equation mapping a 
scalar field to a steady state solution
Conformal Prediction using various Conformal Score estimates

Studying the influence of calibration scores. 
"""
# %%
#Importing the necessary 
import os 
import numpy as np 
import math
from tqdm import tqdm 
from timeit import default_timer
import matplotlib as mpl 
from matplotlib import pyplot as plt 
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

data =  np.load(data_loc + '/Data/poisson_1d.npz')
X = data['x'].astype(np.float32)
Y = data['y'].astype(np.float32)
train_split = 5000
cal_split = 1000
pred_split = 1000

##Training data from the same distribution 
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[:train_split], Y[:train_split]), batch_size=batch_size, shuffle=True)

# ##Training data from another distribution 
X_train = torch.FloatTensor(X[:train_split])
Y_train = torch.FloatTensor(Y[:train_split])
batch_size = 100

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

X_pred, Y_pred = X[train_split+cal_split:train_split+cal_split+pred_split], Y[train_split+cal_split:train_split+cal_split+pred_split]

# %% 

#############################################################
# Conformalised Quantile Regression 
#############################################################

#Loading the Trained Model
nn_lower = MLP(32, 32, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_lower = nn_lower.to(device)
nn_lower.load_state_dict(torch.load(model_loc + 'poisson_nn_lower_1.pth', map_location='cpu'))

#Loading the Trained Model
nn_upper = MLP(32, 32, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_upper =  nn_upper.to(device)
nn_upper.load_state_dict(torch.load(model_loc + 'poisson_nn_upper_1.pth', map_location='cpu'))

#Loading the Trained Model
nn_mean = MLP(32, 32, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_mean = nn_mean.to(device)
nn_mean.load_state_dict(torch.load(model_loc + 'poisson_nn_mean_1.pth', map_location='cpu'))

# %%
#Getting the Coverage
def calibrate_cqr(x_cal, y_cal, alpha):
    n = len(x_cal)

    with torch.no_grad():
        cal_lower = nn_lower(torch.Tensor(x_cal)).numpy()
        cal_upper = nn_upper(torch.Tensor(x_cal)).numpy()

    cal_scores = np.maximum(y_cal-cal_upper, cal_lower-y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

# %% 

with torch.no_grad():
    val_lower = nn_lower(torch.FloatTensor(X_pred)).numpy()
    val_upper = nn_upper(torch.FloatTensor(X_pred)).numpy()

y_response = Y_pred


alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_cqr = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = X[train_split:train_split+cal_split], Y[train_split:train_split+cal_split]
    
    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_cqr(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_cqr.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)
# %% 
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_cqr[0], label='250', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_cqr[1], label='500' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_cqr[2], label='750',  color='mediumblue', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_cqr[3], label='1000',  color='red', ls='dotted',  alpha=0.8, linewidth=3.0)
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
#############################################################
# Conformal using Residuals
#############################################################

#Loading the Trained Model
nn_mean = MLP(32, 32, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_mean = nn_mean.to(device)
nn_mean.load_state_dict(torch.load(model_loc + 'poisson_nn_mean_1.pth', map_location='cpu'))

# %% 
def conf_metric_res(x_cal, y_cal): 

    with torch.no_grad():
        mean = nn_mean(torch.FloatTensor(x_cal)).numpy()
    return np.abs(y_cal - mean)

def calibrate_res(x_cal, y_cal, alpha):
    n = cal_split

    cal_scores = conf_metric_res(x_cal, y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')


    prediction_sets = [prediction - qhat, prediction + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

with torch.no_grad():
    prediction = nn_mean(torch.FloatTensor(X_pred)).numpy()
y_response = Y_pred


alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_res = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = X[train_split:train_split+cal_split], Y[train_split:train_split+cal_split]
    
    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_res(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_res.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)
# %% 
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res[0], label='250', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res[1], label='500' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res[2], label='750',  color='mediumblue', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res[3], label='1000',  color='red', ls='dotted',  alpha=0.8, linewidth=3.0)
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
#############################################################
# Conformal using dropout
#############################################################
#Loading the Trained Model
nn_dropout = MLP_dropout(32, 32, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_dropout = nn_dropout.to(device)
nn_dropout.load_state_dict(torch.load(model_loc + 'poisson_nn_dropout.pth', map_location='cpu'))
 

 # %% 
def calibrate_dropout(x_cal, y_cal, alpha):
    n = cal_split
    
    with torch.no_grad():
        cal_mean, cal_std = MLP_dropout_eval(nn_dropout, torch.Tensor(X_cal))

    cal_upper = cal_mean + cal_std
    cal_lower = cal_mean - cal_std

    cal_scores = np.maximum(y_cal-cal_upper, cal_lower-y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


with torch.no_grad():
    val_mean, val_std = MLP_dropout_eval(nn_dropout, torch.FloatTensor(X_pred))

val_upper = val_mean + val_std
val_lower = val_mean - val_std
y_response = Y_pred

alpha_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
cal_sizes = [250, 500, 750, 1000]
emp_cov_dropout = []
for cal_split in cal_sizes:
    #Preppring the Calibration Datasets
    X_cal, Y_cal = X[train_split:train_split+cal_split], Y[train_split:train_split+cal_split]
    
    emp_cov = []
    for ii in tqdm(range(len(alpha_levels))):
        emp_cov.append(calibrate_dropout(X_cal, Y_cal, alpha_levels[ii]))
    emp_cov_dropout.append(emp_cov)

alpha_levels = np.asarray(alpha_levels)

# %% 

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout[0], label='250', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout[1], label='500' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout[2], label='750',  color='mediumblue', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout[3], label='1000',  color='red', ls='dotted',  alpha=0.8, linewidth=3.0)
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
