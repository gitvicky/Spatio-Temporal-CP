#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Neural Network (MLP) built using PyTorch to model the 1D Poisson Equation mapping a 
scalar field to a steady state solution
Conformal Prediction using various Conformal Score estimates

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

from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True

plt.rcParams['grid.linewidth'] = 0.5
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
x_range = np.linspace(0, 1, 32)
##Training data from the same distribution 
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[:train_split], Y[:train_split]), batch_size=batch_size, shuffle=True)

# ##Training data from another distribution 
X_train = torch.FloatTensor(X[:train_split])
Y_train = torch.FloatTensor(Y[:train_split])
batch_size = 100

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

#Preppring the Calibration Datasets
X_cal, Y_cal = X[train_split:train_split+cal_split], Y[train_split:train_split+cal_split]

#Prepping the Prediction Datasets
X_pred, Y_pred = X[train_split+cal_split:train_split+cal_split+pred_split], Y[train_split+cal_split:train_split+cal_split+pred_split]



# torch.save(nn_mean.state_dict(), path + '/Models/poisson_nn_mean.pth')

#Loading the Trained Model
nn_mean = MLP(32, 32, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_mean = nn_mean.to(device)
nn_mean.load_state_dict(torch.load(model_loc + 'poisson_nn_mean_1.pth', map_location='cpu'))


# %%
alpha = 0.1

idx = 23
x_viz = X_pred[idx]

X_pred_viz = torch.FloatTensor(x_viz)
Y_pred_viz = Y_pred[idx]

stacked_x_viz = X_pred_viz

with torch.no_grad():
    mean_viz = nn_mean(stacked_x_viz).numpy()

y_response = Y_pred

# %% 

# %% 
# from matplotlib import cm 

# x_true = x_viz
# y_true = Y_pred_viz 
# alpha_levels = np.arange(0.05, 0.95, 0.05)
# cols = cm.plasma(alpha_levels)
# pred_sets = [get_prediction_sets(x_true.squeeze().reshape(-1,32).astype(np.float32), a) for a in alpha_levels] 

# fig, ax = plt.subplots()
# [plt.fill_between(x_true, pred_sets[i][0].squeeze(), pred_sets[i][1].squeeze(), color = cols[i]) for i in range(len(alpha_levels))]
# fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
# plt.plot(x_true, y_true, '--', label='function', alpha=1, linewidth = 2, color = 'darkblue')

# %%
#############################################################
# Conformal Prediction using Residuals
#############################################################


# Using one network with residuals
# https://www.stat.cmu.edu/~larry/=sml/Conformal

def conf_metric(X_cal, Y_cal): 

    stacked_x = torch.FloatTensor(X_cal)
    with torch.no_grad():
        mean = nn_mean(stacked_x).numpy()
    return np.abs(Y_cal - mean)

cal_scores = conf_metric(X_cal, Y_cal)

stacked_x = torch.FloatTensor(X_pred)
with torch.no_grad():
    prediction = nn_mean(stacked_x).numpy()

n = len(cal_scores)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

prediction_sets =  [prediction - qhat, prediction + qhat]

empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %% 
# plt.figure()
# plt.hist(cal_scores, 50)
# plt.xlabel("Calibration scores")
# plt.ylabel("Frequency")

# %%
# Plot residuals conformal predictor

alpha = 0.1

stacked_x = torch.FloatTensor(X_pred_viz)
with torch.no_grad():
    prediction = nn_mean(stacked_x).numpy()
n = len(cal_scores)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

prediction_sets =  [prediction - qhat, prediction + qhat]

pred_residual = prediction
prediction_sets_residual = prediction_sets

# plt.figure()
# plt.title(rf"Residual, $\alpha$ = {alpha}")
# plt.plot(Y_pred_viz, label='Exact', color='black')
# plt.plot(prediction, label='Mean', color='firebrick')
# plt.plot(prediction_sets[0], label='lower-cal', color='teal')
# plt.plot(prediction_sets[1], label='upper-cal', color='navy')
# plt.xlabel("x")
# plt.ylabel("u")
# plt.legend()

# %%
plt.figure()
plt.title(rf"Residual, $\alpha$ = {alpha}")
plt.errorbar(x_range, prediction.flatten(), yerr=(prediction_sets[1] - prediction_sets[0]).flatten(), label='Prediction', color='teal', fmt='o', alpha=0.5)
plt.scatter(x_range, Y_pred_viz, label = 'Exact', color='black', alpha=0.8)
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("poisson_residual.svg", format="svg", bbox_inches='tight')
plt.show()
# %%
# plt.figure()
# plt.title(rf"Residual, $\alpha$ = {alpha}")
# plt.scatter(x_range, prediction.flatten(), label='Prediction', color='firebrick', s=10)
# plt.scatter(x_range, Y_pred_viz, label = 'Exact', color='black', s=10)
# plt.plot(prediction_sets[0], label='lower-cal', color='teal')
# plt.plot(prediction_sets[1], label='upper-cal', color='navy')
# plt.xlabel("x")
# plt.ylabel("u")
# plt.legend()
# %%

def calibrate_res(alpha):
    n = cal_split

    cal_scores = conf_metric(X_cal, Y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    with torch.no_grad():
        prediction = nn_mean(stacked_x).numpy()

    prediction_sets = [prediction - qhat, prediction + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_res = []
stacked_x = torch.FloatTensor(X_pred)
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res(alpha_levels[ii]))


plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
plt.show()

##
#% Multivariate residual CP

## Plot all calibration data

Y_mean = np.mean(Y_cal, axis =0)
Y_cal_std = np.std(Y_cal, axis = 0)

plt.plot(x_range, Y_cal.T, color ="red", alpha = 0.1)
plt.plot(x_range, Y_mean, color = "blue")
plt.plot(x_range, Y_cal_std, color = "blue")
plt.show()


### 

def conf_metric(X_cal, Y_cal): 

    stacked_x = torch.FloatTensor(X_cal)
    with torch.no_grad():
        mean = nn_mean(stacked_x).numpy()
    # return np.max(np.abs((Y_cal - mean)/Y_cal_std),axis =1)
    return np.max(np.abs((Y_cal - mean)), axis =1)

cal_scores = conf_metric(X_cal, Y_cal)

stacked_x = torch.FloatTensor(X_pred)
with torch.no_grad():
    prediction = nn_mean(stacked_x).numpy()

alpha = 0.1
n = len(cal_scores)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

prediction_sets =  [prediction - qhat, prediction + qhat]

plt.plot(x_range, prediction_sets[0])
plt.plot(x_range, prediction_sets[1])
plt.plot(x_range, prediction)
plt.show()

empirical_coverage = ((y_response >= prediction_sets[0]).all(axis = 1) & (y_response <= prediction_sets[1]).all(axis = 1)).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")


###
#   Check empirical coverage
###

def calibrate_res_MV(alpha):
    n = cal_split

    cal_scores = conf_metric(X_cal, Y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    with torch.no_grad():
        prediction = nn_mean(stacked_x).numpy()

    prediction_sets = [prediction - qhat, prediction + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]).all(axis = 1) & (y_response <= prediction_sets[1]).all(axis = 1)).mean()
    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_res = []
stacked_x = torch.FloatTensor(X_pred)
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_res_MV(alpha_levels[ii]))


plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.savefig("empirical_coverage_multivariate.png")
plt.legend()

plt.show()



### Plot prediction sets


from matplotlib import cm 

idx = 100

alpha = 0.1

# alpha_levels = np.arange(0.05, 0.95, 0.05)

qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

# prediction_sets =  [prediction - qhat*Y_cal_std, prediction + qhat*Y_cal_std]
prediction_sets =  [prediction - qhat, prediction + qhat]

plt.plot(x_range, prediction_sets[0][idx], color = "red")
plt.plot(x_range, prediction_sets[1][idx], color = "red", label = f"alpha = {alpha}")

plt.plot(x_range, prediction[idx], color = "Black", linewidth = 1, label = "data")
plt.legend(fontsize = 20)
plt.savefig("c_prediction_NN.png")

plt.show()


###
from matplotlib import cm 

idx = 23
alphas = np.linspace(0.1, 0.9, 10)

# alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alphas)
n = len(cal_scores)

for (i, alpha) in enumerate(alphas):
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    # prediction_sets =  [prediction - qhat*Y_cal_std, prediction + qhat*Y_cal_std]
    prediction_sets =  [prediction - qhat, prediction + qhat]

    plt.plot(x_range, prediction_sets[0][idx], color = cols[i])
    plt.plot(x_range, prediction_sets[1][idx], color = cols[i], label = alpha)

    empirical_coverage = ((y_response >= prediction_sets[0]).all(axis = 1) & (y_response <= prediction_sets[1]).all(axis = 1)).mean()
    print(f"The empirical coverage after calibration is: {empirical_coverage}")
    print(f"alpha is: {alpha}")
    print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# plt.plot(x_range, prediction[idx], color = "Black", linewidth = 3, label = "true")
plt.legend(fontsize = 10)
plt.show()