#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
FNO modelled over the MHD data built using JOREK for multi-blob diffusion. Conformal Prediction over it
"""
# %%
#Training Conditions
################################################################

configuration = {"Case": 'Multi-Blobs', #Specifying the Simulation Scenario
                 "Field": 'rho', #Variable we are modelling - Phi, rho, T
                 "Type": '2D Time', #FNO Architecture
                 "Epochs": 500, 
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No', #Layerwise Normalisation
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes', #Normalising the Variable 
                 "T_in": 10, #Input time steps
                 "T_out": 10, #Max simulation time
                 "Step": 10, #Time steps output in each forward call
                 "Modes":32, #Number of Fourier Modes
                 "Width": 64, #Features of the Convolutional Kernel
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'LP Loss' #Choice of Loss Fucnction
                 }


# %%
#Importing the necessary packages. 
################################################################
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 

import time 
from timeit import default_timer
from tqdm import tqdm 

from utils import * 

torch.manual_seed(0)
np.random.seed(0)

# %% 
#Setting up the directories - data location, model location and plots. 
################################################################
import os 
path = os.getcwd()
model_loc = path + '/Models/'
data_loc = path
# %%
#Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# Loading Data 
################################################################

# %%
data = data_loc + '/Data/MHD_multiblob_rho.npz'

# %%
field = configuration['Field']

u_sol = np.load(data)['u'].astype(np.float32)   / 1e20

if configuration['Log Normalisation'] == 'Yes':
    u_sol = np.log(u_sol)

u_sol = np.nan_to_num(u_sol)

x_grid = np.load(data)['x'].astype(np.float32)
y_grid = np.load(data)['y'].astype(np.float32)
t_grid = np.load(data)['t'].astype(np.float32)

# %% 
#Extracting hyperparameters from the config dict
################################################################

S = 106 #Grid Size 

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']
T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

t1 = default_timer()

np.random.shuffle(u_sol)
u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

#At this stage the data needs to be [Batch_Size, X, Y, T]
ntrain = 100
ncal = 100
npred = 78

# %%
#Chunking the data. 
################################################################

train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

cal_a = u[ntrain:ntrain+ncal,:,:,:T_in]
cal_u = u[ntrain:ntrain+ncal,:,:,T_in:T+T_in]

pred_a = u[ntrain+ncal:ntrain+ncal+npred,:,:,:T_in]
pred_u = u[ntrain+ncal:ntrain+ncal+npred,:,:,T_in:T+T_in]

print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)


# %%
#Normalising the train and test datasets with the preferred normalisation. 
################################################################

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    a_normalizer = MinMax_Normalizer(train_a)
    y_normalizer = MinMax_Normalizer(train_u)

if norm_strategy == 'Range':
    a_normalizer = RangeNormalizer(train_a)
    y_normalizer = RangeNormalizer(train_u)

if norm_strategy == 'Min-Max':
    a_normalizer = GaussianNormalizer(train_a)
    y_normalizer = GaussianNormalizer(train_u)


train_a = a_normalizer.encode(train_a)
cal_a = a_normalizer.encode(cal_a)
pred_a = a_normalizer.encode(pred_a)

train_u = y_normalizer.encode(train_u)
cal_u = y_normalizer.encode(cal_u)
pred_u = y_normalizer.encode(pred_u)


t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
# Loading the trained model
################################################################
#Instantiating the Model. 
model = FNO2d(modes, modes, width, T_in, step, x_grid, y_grid)
model.load_state_dict(torch.load(model_loc + 'FNO_density.pth', map_location='cpu'))
model.to(device)
print("Number of model params : " + str(model.count_params()))

if torch.cuda.is_available():
    y_normalizer.cuda()


# %%
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
################################################################

t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    cal_mean = model(torch.Tensor(cal_a)).numpy()

# cal_u = cal_u.numpy()
cal_scores = np.abs(cal_u-cal_mean)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()
stacked_x = torch.FloatTensor(pred_a)

with torch.no_grad():
    mean = model(stacked_x).numpy()

prediction_sets =  [mean - qhat, mean + qhat]


# %%
print('Conformal by way Residual')
# Calculate empirical coverage (before and after calibration)
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
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
        cal_mean = model(torch.Tensor(cal_a)).numpy()
        
    cal_scores = np.abs(cal_u-cal_mean)     
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    prediction_sets =  [mean - qhat, mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate(alpha_levels[ii]))

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()


# %%
