#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Neural Network (MLP) built using PyTorch to model the 1D Poisson Equation mapping a 
scalar field to a steady state solution
Conformal Prediction using various Conformal Score estimates

This script performs multivariate conformal prediction, predicting simultaneous 
error bounds over entire spatio-temporal domain, using method method outlined in

Diquigiovanni, J., Fontana, M., & Vantini, S. (2021). "The importance of being a band: 
Finite-sample exact distribution-free prediction sets for functional data."
arXiv preprint arXiv:2102.06746.

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

from bokeh.layouts import column, row
from bokeh.models import Slider
from bokeh.plotting import ColumnDataSource, figure, curdoc


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


# %% predict all calibration dataset 
# and prediction sets using ML model

stacked_x = torch.FloatTensor(X_cal)
with torch.no_grad():
    mean_cal = nn_mean(stacked_x).numpy()

stacked_x = torch.FloatTensor(X_pred)
with torch.no_grad():
    prediction = nn_mean(stacked_x).numpy()


# %%
#######################################################################
# Multivariate Conformal Prediction using surface's max displacement
#######################################################################

# Using one network with residuals
# https://www.stat.cmu.edu/~larry/=sml/Conformal

def conf_metric(X_mean, Y_cal): 
    return np.max(np.abs(Y_cal - X_mean), axis =1)

cal_scores = conf_metric(mean_cal, Y_cal)

# %% 
# plt.figure()
# plt.hist(cal_scores, 50)
# plt.xlabel("Calibration scores")
# plt.ylabel("Frequency")


##
#% Multivariate residual CP with varying width.
# "Modulation function" is the std of the data
#
#   r(X,Y) = |NN(X) -| / std(Y)

## Plot all calibration data

Y_mean = np.mean(Y_cal, axis = 0)
modulation = np.std(Y_cal, axis = 0)
# Y_cal_std = np.std(Y_cal - mean_cal, axis = 0)

### 

def conf_metric(X_mean, Y_cal): 
    return np.max(np.abs((Y_cal - X_mean)/modulation), axis =1)

cal_scores_1 = conf_metric(mean_cal, Y_cal)


##
#% Multivariate residual CP with varying width.
# "Modulation function" is the std of the error
#
#   r(X,Y) = |NN(X) - Y| / std(NN(X) - Y)

## Plot all calibration data

Y_mean = np.mean(Y_cal, axis = 0)
modulation_err = np.std(Y_cal - mean_cal, axis = 0)
# modulation_err = np.mean(np.abs(Y_cal - mean_cal), axis = 0)

### 

def conf_metric(X_mean, Y_cal): 
    return np.max(np.abs((Y_cal - X_mean)/modulation_err), axis =1)

cal_scores_2 = conf_metric(mean_cal, Y_cal)



##########
## Bokeh plot paramaters

n = len(cal_scores_1)

bounds = np.array([(0, 4), (0, 1)])

default_parameters = np.mean(bounds, axis = 1)

with torch.no_grad():
    X_test = np.ones(X_cal.shape[1]) * default_parameters[0]
    stacked_x = torch.FloatTensor(X_test)
    mean_1 = nn_mean(stacked_x).numpy()

alpha_test = default_parameters[1]

qhat = np.quantile(cal_scores_1, np.ceil((n+1)*(1-alpha_test))/n, axis = 0, method='higher')

prediction_sets =  [mean_1 - qhat*modulation, mean_1 + qhat*modulation]

source = ColumnDataSource(data=dict(grid=x_range, mean=mean_1, lower=prediction_sets[0], upper=prediction_sets[1]))

plot = figure(y_range = (- 1.2, 1) ,x_range=(0, 1), width=400, height=400)

plot.line('grid', 'mean', source=source, line_width=3, line_color='black', line_alpha=0.8)
plot.line('grid', 'lower', source=source, line_width=2, line_color='black', line_dash='dashed', line_alpha=0.5)
plot.line('grid', 'upper', source=source, line_width=2, line_color='black', line_dash='dashed', line_alpha=0.5)

Input_slider = Slider(start=bounds[0,0], end=bounds[0,1], value=default_parameters[0], step=.01, title="input")
Alpha_slider = Slider(start=bounds[1,0], end=bounds[1,1], value=default_parameters[1], step=.01, title="alpha")


def callback(attr, old, new):
    
    in_X = Input_slider.value
    alpha = Alpha_slider.value
    
    with torch.no_grad():
        X_test = np.ones(X_cal.shape[1]) * in_X
        stacked_x = torch.FloatTensor(X_test)
        mean_x = nn_mean(stacked_x).numpy()

    qhat = np.quantile(cal_scores_1, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets =  [mean_x - qhat*modulation, mean_x + qhat*modulation]
    
    source.data = {'grid': x_range, 'mean': mean_x, 'lower': prediction_sets[0], 'upper': prediction_sets[1]}

Input_slider.on_change('value_throttled', callback)
Alpha_slider.on_change('value_throttled', callback)


layout = row(
    plot,
    column(Input_slider, Alpha_slider),
)

curdoc().add_root(layout)

from bokeh.io import show
show(plot)




# band = Band(base='x', lower='lower', upper='upper', source=source, 
#             level='underlay', fill_alpha=1.0, line_width=1, line_color='black')

