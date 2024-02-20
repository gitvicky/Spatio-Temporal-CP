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

alpha = 0.2
n = len(cal_scores)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

prediction_sets =  [prediction - qhat, prediction + qhat]

empirical_coverage = ((Y_pred >= prediction_sets[0]).all(axis = 1) & (Y_pred <= prediction_sets[1]).all(axis = 1)).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

### Plot coverage plot

def calibrate_res(alpha):

    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [prediction - qhat, prediction + qhat]
    empirical_coverage = ((Y_pred >= prediction_sets[0]).all(axis = 1) & (Y_pred <= prediction_sets[1]).all(axis = 1)).mean()
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

# %%
# Plot some prediction sets

from matplotlib import cm 

idx_s = [30, 50, 352]

alphas = np.linspace(0.1, 0.9, 10)

# alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alphas)
n = len(cal_scores)
for idx in idx_s:

    fig, ax = plt.subplots()
    for (i, alpha) in enumerate(alphas):
        qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
        prediction_sets =  [prediction - qhat, prediction + qhat]

        plt.fill_between(x_range, prediction_sets[0][idx].squeeze(), prediction_sets[1][idx].squeeze(), color = cols[i])


    fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
    plt.plot(x_range, Y_pred[idx], linewidth = 2, color = "black", label = "exact")
    # plt.plot(x_range, prediction[idx], color = "Black", linewidth = 3, label = "true")
    plt.legend(fontsize = 10)

    plt.savefig(f"poisson_CP_abs_{idx}.png")
    plt.show()


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

plt.plot(x_range, Y_cal.T, color ="red", alpha = 0.1)
plt.plot(x_range, Y_mean, color = "blue", label = "mean")
plt.plot(x_range, modulation, color = "black", label = "STD")
plt.legend()
plt.savefig("all_data.png")
plt.show()


### 

def conf_metric(X_mean, Y_cal): 
    return np.max(np.abs((Y_cal - X_mean)/modulation), axis =1)

cal_scores_1 = conf_metric(mean_cal, Y_cal)

alpha = 0.1
n = len(cal_scores_1)
qhat = np.quantile(cal_scores_1, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

prediction_sets =  [prediction - qhat*modulation, prediction + qhat*modulation]

# plt.plot(x_range, prediction_sets[0])
# plt.plot(x_range, prediction_sets[1])
# plt.plot(x_range, prediction)
# plt.show()

empirical_coverage = ((Y_pred >= prediction_sets[0]).all(axis = 1) & (Y_pred <= prediction_sets[1]).all(axis = 1)).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")


###
#   Check empirical coverage
###

def calibrate_res_MV(alpha):

    qhat = np.quantile(cal_scores_1, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [prediction - qhat*modulation, prediction + qhat*modulation]
    empirical_coverage = ((Y_pred >= prediction_sets[0]).all(axis = 1) & (Y_pred <= prediction_sets[1]).all(axis = 1)).mean()

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
plt.savefig("empirical_coverage_data_std_multivariate.png")
plt.legend()

plt.show()


### Plot prediction sets

# from matplotlib import cm 

# idx = 100

# alpha = 0.8

# # alpha_levels = np.arange(0.05, 0.95, 0.05)

# qhat = np.quantile(cal_scores_1, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

# prediction_sets =  [prediction - qhat*Y_cal_std, prediction + qhat*Y_cal_std]
# # prediction_sets =  [prediction - qhat, prediction + qhat]

# plt.plot(x_range, prediction_sets[0][idx], color = "red")
# plt.plot(x_range, prediction_sets[1][idx], color = "red", label = f"alpha = {alpha}")

# plt.plot(x_range, Y_pred[idx], color = "Black", linewidth = 1, label = "data")
# plt.legend(fontsize = 20)
# plt.savefig("c_prediction_NN.png")

# plt.show()

# %%
# Plot some prediction sets


from matplotlib import cm 

idx_s = [30, 50, 352]

alphas = np.linspace(0.1, 0.9, 10)

# alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alphas)
n = len(cal_scores_1)
for idx in idx_s:
    fig, ax = plt.subplots()
    for (i, alpha) in enumerate(alphas):

        qhat = np.quantile(cal_scores_1, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
        prediction_sets =  [prediction - qhat*modulation, prediction + qhat*modulation]

        plt.fill_between(x_range, prediction_sets[0][idx].squeeze(), prediction_sets[1][idx].squeeze(), color = cols[i])


    fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
    plt.plot(x_range, Y_pred[idx], linewidth = 3, color = "black", label = "exact")
    # plt.plot(x_range, prediction[idx], color = "Black", linewidth = 3, label = "true")
    plt.legend(fontsize = 10)

    plt.savefig(f"poisson_CP_data_std_{idx}.png")
    plt.show()






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

alpha = 0.1
n = len(cal_scores_2)
qhat = np.quantile(cal_scores_2, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

prediction_sets =  [prediction - qhat*modulation_err, prediction + qhat*modulation_err]

# plt.plot(x_range, prediction_sets[0])
# plt.plot(x_range, prediction_sets[1])
# plt.plot(x_range, prediction)
# plt.show()

empirical_coverage = ((Y_pred >= prediction_sets[0]).all(axis = 1) & (Y_pred <= prediction_sets[1]).all(axis = 1)).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")


###
#   Check empirical coverage
###

def calibrate_res_MV(alpha):

    qhat = np.quantile(cal_scores_2, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [prediction - qhat*modulation_err, prediction + qhat*modulation_err]
    empirical_coverage = ((Y_pred >= prediction_sets[0]).all(axis = 1) & (Y_pred <= prediction_sets[1]).all(axis = 1)).mean()

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
plt.savefig("empirical_coverage_multivariate_error_std.png")
plt.legend()

plt.show()


# %%
# Plot some prediction sets


from matplotlib import cm 

idx_s = [30, 50, 352]

alphas = np.linspace(0.1, 0.9, 10)

# alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alphas)
n = len(cal_scores_2)
for idx in idx_s:
    fig, ax = plt.subplots()
    for (i, alpha) in enumerate(alphas):

        qhat = np.quantile(cal_scores_2, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
        prediction_sets =  [prediction - qhat*modulation_err, prediction + qhat*modulation_err]

        plt.fill_between(x_range, prediction_sets[0][idx].squeeze(), prediction_sets[1][idx].squeeze(), color = cols[i])


    fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
    plt.plot(x_range, Y_pred[idx], linewidth = 3, color = "black", label = "exact")
    # plt.plot(x_range, prediction[idx], color = "Black", linewidth = 3, label = "true")
    plt.legend(fontsize = 10)

    plt.savefig(f"poisson_CP_error_std_{idx}.png")
    plt.show()


####
# Plot width of errors with X

cal_scores_vec = [cal_scores, cal_scores_1, cal_scores_2]
modulations = [1, modulation, modulation_err]

names = ["Abs_error", "data_std", "error_std"]

alphas = np.linspace(0.1, 0.9, 10)

# alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alphas)
n = len(cal_scores_2)
for (j,cal_scores) in enumerate(cal_scores_vec):
    fig, ax = plt.subplots()
    for (i, alpha) in enumerate(alphas):

        qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
        prediction_sets =  [-qhat*modulations[j], qhat*modulations[j]]

        plt.fill_between(x_range, prediction_sets[0].squeeze(), prediction_sets[1].squeeze(), color = cols[i])


    fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
    # plt.plot(x_range, prediction[idx], color = "Black", linewidth = 3, label = "true")
    plt.title(names[j])
    plt.legend(fontsize = 10)

    plt.savefig(f"width_{names[j]}.png")
    plt.show()



# %%
from sklearn.decomposition import TruncatedSVD

errors = Y_cal - mean_cal

plt.plot(mean_cal.T, color = "red")

N_dim=5

a1 = TruncatedSVD(n_components=N_dim)
a1.fit(mean_cal)

var_decomp = np.cumsum(a1.explained_variance_ratio_)

print(np.sum(a1.explained_variance_ratio_))

trans_NN = a1.transform(mean_cal)
trans_Cal = a1.transform(Y_cal)

diff = trans_Cal - trans_NN

# plt.figure()
# plt.hist(np.vstack(trans_data[:, 0]), bins = 100)

plt.figure()
plt.scatter(trans_NN[:, 0], trans_NN[:,1])
plt.scatter(trans_Cal[:, 0], trans_Cal[:,1])
plt.scatter(diff[:, 0], diff[:,1])

plt.figure()
plt.scatter(trans_NN[:, 2], trans_NN[:,3])
plt.scatter(trans_Cal[:, 2], trans_Cal[:,3])
plt.scatter(diff[:, 2], diff[:,3])

plt.figure()
plt.scatter(trans_NN[:, 3], trans_NN[:,4])
plt.scatter(trans_Cal[:, 3], trans_Cal[:,4])
plt.scatter(diff[:, 3], diff[:,4])


plt.figure()
[plt.hist(diff[:,i], 100) for i in range(0, N_dim)] 

##
# plt.scatter(trans_data[:,2], trans_data[:,3])
# plt.scatter(trans_data[:,3], trans_data[:,4])
# plt.scatter(trans_data[:,4], trans_data[:,5])
# plt.scatter(trans_data[:,5], trans_data[:,6])


Back_prop = a1.inverse_transform(diff)

plt.figure()
plt.plot(x_range, errors.T, color ="red", alpha = 0.1)
plt.plot(x_range, Back_prop.T, color ="blue", alpha = 0.1)
# plt.plot(x_range, Y_mean, color = "blue", label = "mean")
# plt.plot(x_range, modulation, color = "black", label = "STD")
# plt.legend()
plt.show()

# %%
plt.figure()

cal_scores = cal_scores_2

for (i, alpha) in enumerate(alphas):

    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')
    prediction_sets =  [-qhat*modulations[j], qhat*modulations[j]]

    plt.fill_between(x_range, prediction_sets[0].squeeze(), prediction_sets[1].squeeze(), color = cols[i])


fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)

plt.plot(x_range, errors.T, color ="red", alpha = 0.1)
# plt.plot(x_range, Back_prop.T, color ="blue", alpha = 0.1)
# plt.plot(x_range, Y_mean, color = "blue", label = "mean")
# plt.plot(x_range, modulation, color = "black", label = "STD")
# plt.legend()
plt.show()


## %
# Get errors distance to origin

scores = np.sqrt(np.sum(diff * diff, axis = 1))     # Euclidian distance. Makes a circle of equal scores

plt.figure()
plt.hist(scores, 100)

plt.figure()
plt.step(np.sort(scores), range(0,len(scores)))

## %
from scipy.spatial import ConvexHull
from itertools import product

# Bounding using convex hull
CH = ConvexHull(diff)
verts = diff[CH.vertices]

# Bounding using hyper-rectangle
Data_min = np.min(diff, axis = 0)
Data_max = np.max(diff, axis = 0)

my_list = [(Data_min), (Data_max)]

X_ = np.vstack([Data_min, Data_max])
box_vertex = np.asarray(list(product(*X_.T)), dtype=float)


plt.figure()
plt.scatter(diff[:, 0], diff[:,1], label = "errors")
plt.scatter(verts[:, 0], verts[:,1], label = "CH vertex")
plt.scatter(box_vertex[:, 0], box_vertex[:,1], label = "Hyper-rectangle vert")
plt.legend()

plt.figure()
plt.scatter(diff[:, 2], diff[:,3], label = "errors")
plt.scatter(verts[:, 2], verts[:,3], label = "CH vertex")
plt.scatter(box_vertex[:, 2], box_vertex[:,3], label = "Hyper-rectangle vert")
plt.legend()

plt.figure()
plt.scatter(diff[:, 3], diff[:,4], label = "errors")
plt.scatter(verts[:, 3], verts[:,4], label = "CH vertex")
plt.scatter(box_vertex[:, 3], box_vertex[:,4], label = "Hyper-rectangle vert")
plt.legend()


back_verts = a1.inverse_transform(verts)
bounds_back_max = np.max(back_verts, axis = 0)
bounds_back_min = np.min(back_verts, axis = 0)

box_bounds_back = a1.inverse_transform(box_vertex)
bounds_back_box_max = np.max(box_bounds_back, axis = 0)
bounds_back_box_min = np.min(box_bounds_back, axis = 0)


plt.figure()
# plt.plot(x_range, errors.T, color ="red", alpha = 0.1)
plt.plot(x_range, Back_prop.T, color ="red", alpha = 0.1)
plt.plot(x_range, bounds_back_max, color ="blue",linewidth = 2)
plt.plot(x_range, bounds_back_min, color ="blue",linewidth = 2)
plt.plot(x_range, bounds_back_box_max, color ="purple",linewidth = 2)
plt.plot(x_range, bounds_back_box_min, color ="purple",linewidth = 2)
# plt.plot(x_range, Back_prop.T, color ="blue", alpha = 0.1)
# plt.plot(x_range, Y_mean, color = "blue", label = "mean")
# plt.plot(x_range, modulation, color = "black", label = "STD")
# plt.legend()
plt.show()

# %% Marco's functions
import numpy

def minimal_enclosing_hyperbox(x):
    """
    Inputs
    x: (nxd) array, where n is the size of the data, and d is the number of dimensions of the box

    Outputs
    active_scenarios: list[int], a list of integers pointing to the corresponding active scenarios in the dataset
    box: 2xd array, an enclosing box of dimension d.
    """
    x = numpy.asarray(x,dtype=float) # coherce any iterable to numpy array
    n,d = x.shape
    index,box = [],[]
    for j in range(d):
        s = numpy.argsort(x[:,j])
        index += [s[0],s[-1]]
        box.append([x[s[0],j], x[s[-1],j]])
    active_scenarios = list(set(index)) # list of indices corresponding to the active scenarios
    return active_scenarios, numpy.asarray(box,dtype=float)

# def is_inside_box(x,abox):
#     """
#     x:    (nxd) array
#     abox: (dx2) array, or list[list[2 float]] ex.: [[0,1],[3,8],[1,9]] <- a.k.a. interval iterable
#     """
#     n,d = x.shape
#     box = numpy.asarray(abox, dtype=float)
#     box_lo = box[:,0]
#     box_hi = box[:,1]
#     inside = (matlib.repmat(box_lo,n,1) <= x) & (x <= matlib.repmat(box_hi,n,1))
#     return numpy.all(inside,axis=1)

def width(x:numpy.ndarray):
    """
    IN
    x: An interval or interval iterable, i.e. an (dx2) array

    OUT
    w: the width of the intervals
    """
    x = numpy.asarray(x,dtype=float)
    if len(x.shape)==1: return x[1]-x[0]
    else: return x[:,1]-x[:,0]

def index_to_mask(x,n=100):  
    """
    Converts the list of indices x into a (nx1) boolean mask, true at those particular indices
    """
    nrange=numpy.arange(n)
    boole = numpy.zeros(nrange.shape, dtype=bool)
    for xi in x: boole += nrange==xi
    return boole

def make_hashtable(x):
    h = {}
    for i,xi in enumerate(x):
        h[tuple(xi)]=i
    return h

def sanity_check(a,b,tol=1e-4): # checks that the smallest set has enough observation to make a decsion.
    b_last = b[-1] # last enclosing set in the sequence
    for interval in b_last:
        if width(interval)<tol:
            return a[:-1],b[:-1] # returns sequences stripped of the last element
    return a,b


def data_peeling_algorithm(X,tol=1e-4): # should also input the shape, rectangle, circle, etc.
    """
    X: (nxd) data set of iid observations
    tol: a tolerance parameter determining the minimal size of an allowed enclosing box

    sequence_of_indices: a list (number of levels long) of sets (or list) of indices of heterogeneous size
    sequence_of_boxes: a list (number of levels long) of (dx2) boxes
    """
    H = make_hashtable(X) # turns dataset in to hashtable
    n,d = X.shape
    sequence_of_indices,sequence_of_boxes = [],[]
    index_num = 0 # number of support vectors per level
    j=0 # number of levels
    while index_num < n:
        if j==0:
            active_indices, box  = minimal_enclosing_hyperbox(X) # solve the optimization
            active_mask = index_to_mask(active_indices, n=n)
            nonactive_mask = active_mask==False
            X_strip = X[active_mask,:].copy()
            collect_indices =[]
            for xstrip in X_strip: 
                collect_indices.append(H[tuple(xstrip)])
                index_num+=1
            X_new = X[nonactive_mask,:].copy() # strip active scenarios from the original dataset
        else:
            active_indices, box  = minimal_enclosing_hyperbox(X_new)
            active_mask = index_to_mask(active_indices, n=X_new.shape[0])
            nonactive_mask = active_mask==False
            X_strip = X_new[active_mask,:].copy()
            X_new = X_new[nonactive_mask,:].copy()
            collect_indices = []
            for xstrip in X_strip: 
                collect_indices.append(H[tuple(xstrip)])
                index_num+=1
        sequence_of_indices.append(collect_indices)
        sequence_of_boxes.append(box)
        j+=1
    # return sequence_of_indices, sequence_of_boxes
    a,b = sanity_check(sequence_of_indices, sequence_of_boxes, tol=tol) # this checks that the algorithm terminates as it is supposed to.
    return a,b

active_s, box = minimal_enclosing_hyperbox(diff)
a, b = data_peeling_algorithm(diff)


# %%
# s, v, d = np.linalg.svd(errors.T)

# plt.plot(v)
# plt.imshow(s)
# plt.imshow(d)

# T_trunc = 2

# s_ = s[:, T_trunc]

# Y_mean = np.mean(Y_cal, axis = 0)
# modulation_err = np.std(Y_cal - mean_cal, axis = 0)
# # modulation_err = np.mean(np.abs(Y_cal - mean_cal), axis = 0)

# ### 

# def conf_metric(X_mean, Y_cal): 
#     return np.max(np.abs((Y_cal - X_mean)/modulation_err), axis =1)

# cal_scores_2 = conf_metric(mean_cal, Y_cal)

# alpha = 0.1
# n = len(cal_scores_2)
# qhat = np.quantile(cal_scores_2, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

# prediction_sets =  [prediction - qhat*modulation_err, prediction + qhat*modulation_err]


# ##
# alpha_levels = np.arange(0.05, 0.95, 0.05)
# cols = cm.plasma(alpha_levels)
# pred_sets = [get_prediction_sets(x_true.squeeze().reshape(-1,1).astype(np.float32), a) for a in alpha_levels] 

# fig, ax = plt.subplots()
# [plt.fill_between(x_true, pred_sets[i][0].squeeze(), pred_sets[i][1].squeeze(), color = cols[i]) for i in range(len(alpha_levels))]
# cbar = fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
# plt.plot(x_true, y_true, '--', label='function', alpha=1, linewidth = 2)

# cbar.ax.set_ylabel('alpha', rotation=270)
# ##