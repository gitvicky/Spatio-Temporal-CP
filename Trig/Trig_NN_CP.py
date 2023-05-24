#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network (MLP) built using PyTorch to model the 1D regression function sin(2.pi.x) + cos(2.pi.x)
Conformal Prediction using various Conformal Score estimates

"""
# %%
#Importing the necessary 
import os 
import numpy as np 
import math
from tqdm import tqdm 
from timeit import default_timer
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
# %% 
#Generating Dataset
num_points = 5000
batch_size = int(num_points/10)

def LHS(lb, ub, N): #Latin Hypercube Sampling for each the angles and the length. 
    return lb + (ub-lb)*lhs(1, N)

lb = np.asarray([-np.pi]) #Lower Bound of the input space
ub = np.asarray([np.pi]) #Upper Bound of the input space

x = LHS(lb, ub, num_points) 

# True function is sin(2*pi*x) + cos(2*pi*x) with Gaussian noise
f_x = lambda x:  np.sin(x[:,0]) + np.cos(2*x[:,0]) #+ np.random.rand(x.shape[0]) * math.sqrt(0.04)

# %%


# %%
X =torch.FloatTensor(x)
Y = torch.FloatTensor(f_x(x)).unsqueeze(dim=-1)

cal_split = 1000
pred_split = 1000
train_split = num_points - cal_split - pred_split

##Training data from the same distribution 
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[:train_split], Y[:train_split]), batch_size=batch_size, shuffle=True)

# ##Training data from another distribution 
X_train = torch.FloatTensor(np.linspace(lb, ub, train_split)).unsqueeze(dim=-1)

Y_train = torch.FloatTensor(f_x(X_train.numpy())).unsqueeze(dim=-1)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

#Prepping the Prediction Datasets
X_pred, Y_pred = X[-pred_split:].numpy(), Y[-pred_split:].numpy()
#Preppring the Calibration Datasets
X_cal, Y_cal = X[train_split:-pred_split].numpy(), Y[train_split:-pred_split].numpy()

# %% 

nn_lower = MLP(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_lower = nn_lower.to(device)
#Training the Model -- Comment out this entire cell if you are loading a pre-trained model. 

optimizer = torch.optim.Adam(nn_lower.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
# loss_func = torch.nn.MSELoss()
loss_func = quantile_loss

it=0
epochs = 1000
loss_list = []

start_time = default_timer()

while it < epochs :
    t1 = default_timer()
    loss_val = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        out = nn_lower(xx)
        loss = loss_func(out, yy, gamma=0.05).pow(2).mean()
        loss_val += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_list.append(loss_val.mean().item())
    scheduler.step()

    it += 1
    t2 = default_timer()
    print('It: %d, Time %.3e, Loss: %.3e' % (it, t2 - t1, loss.item()))


train_time = default_timer() - start_time
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('L2 Error')
# plt.xscale('log')
plt.yscale('log')
plt.title('Loss plot of Lower')

# torch.save(nn_lower.state_dict(), path + '/Models/sin_nn_lower.pth')

# #Loading the Trained Model
# nn_lower = MLP(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
# nn_lower = nn_lower.to(device)
# nn_lower.load_state_dict(torch.load(model_loc + 'sin_nn_lower.pth', map_location='cpu'))

# %% 

nn_upper = MLP(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_upper = nn_upper.to(device)
#Training the Model -- Comment out this entire cell if you are loading a pre-trained model. 
optimizer = torch.optim.Adam(nn_upper.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
# loss_func = torch.nn.MSELoss()
loss_func = quantile_loss

it=0
loss_list = []

start_time = default_timer()

while it < epochs :
    t1 = default_timer()
    loss_val = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        out = nn_upper(xx)
        loss = loss_func(out, yy, gamma=0.95).pow(2).mean()
        loss_val += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_list.append(loss_val.mean().item())
    scheduler.step()

    it += 1
    t2 = default_timer()
    print('It: %d, Time %.3e, Loss: %.3e' % (it, t2 - t1, loss.item()))


train_time = default_timer() - start_time
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('L2 Error')
# plt.xscale('log')
plt.yscale('log')
plt.title('Loss plot of Upper')


if not os.path.exists(path + '/Models'):
    os.mkdir(path  + '/Models')

torch.save(nn_upper.state_dict(), path + '/Models/sin_nn_upper.pth')

# #Loading the Trained Model
# nn_upper = MLP(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
# nn_upper =  nn_upper.to(device)
# nn_upper.load_state_dict(torch.load(model_loc + 'sin_nn_upper.pth', map_location='cpu'))

# %% 

nn_mean = MLP(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_mean = nn_mean.to(device)
#Training the Model -- Comment out this entire cell if you are loading a pre-trained model. 
optimizer = torch.optim.Adam(nn_mean.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
# loss_func = torch.nn.MSELoss()
loss_func = quantile_loss

it=0
loss_list = []

start_time = default_timer()

while it < epochs :
    t1 = default_timer()
    loss_val = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)

        out = nn_mean(xx)
        loss = loss_func(out, yy, gamma=0.5).pow(2).mean()

        loss_val += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_list.append(loss_val.mean().item())
    scheduler.step()

    it += 1
    t2 = default_timer()
    print('It: %d, Time %.3e, Loss: %.3e' % (it, t2 - t1, loss.item()))


train_time = default_timer() - start_time
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('L2 Error')
# plt.xscale('log')
plt.yscale('log')
plt.title('Loss plot of mean')


torch.save(nn_mean.state_dict(), path + '/Models/sin_nn_mean.pth')

# #Loading the Trained Model
# nn_mean = MLP(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
# nn_mean = nn_mean.to(device)
# nn_mean.load_state_dict(torch.load(model_loc + 'sin_nn_mean.pth', map_location='cpu'))

# %% 
#Performing the Calibration
n = cal_split
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    cal_lower = nn_lower(torch.Tensor(X_cal)).numpy()
    cal_upper = nn_upper(torch.Tensor(X_cal)).numpy()

cal_scores = np.maximum(Y_cal-cal_upper, cal_lower-Y_cal)           

qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

plt.figure()
plt.hist(cal_scores, 50)
plt.xlabel("Calibration scores")
plt.ylabel("Frequency")

# %%
def get_prediction_sets(x, alpha = 0.1):

    X_pred = x
    stacked_x = torch.FloatTensor(X_pred)

    with torch.no_grad():
        val_lower = nn_lower(stacked_x).numpy()
        val_upper = nn_upper(stacked_x).numpy()

    n = len(cal_scores)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    return [val_lower - qhat, val_upper + qhat]


# %%
alpha = 0.1

num_viz_points = 50

x_viz = np.linspace(lb, ub, num_viz_points)

X_pred_viz = torch.FloatTensor(x_viz).unsqueeze(dim=-1)
Y_pred_viz = f_x(X_pred_viz)

stacked_x_viz = torch.FloatTensor(X_pred_viz)

with torch.no_grad():
    val_lower_viz = nn_lower(stacked_x_viz).numpy()
    val_upper_viz = nn_upper(stacked_x_viz).numpy()
    mean_viz = nn_mean(stacked_x_viz).numpy()

prediction_sets = get_prediction_sets(stacked_x_viz)
prediction_sets_uncalibrated = [val_lower_viz, val_upper_viz]

plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.plot(X_pred_viz, Y_pred_viz, label='Analytical', color='black')
plt.plot(X_pred_viz, mean_viz, label='Mean', color='firebrick')
plt.plot(X_pred_viz, prediction_sets[0], label='lower-cal', color='teal')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[0], label='lower - uncal', color='darkorange')
plt.plot(X_pred_viz, prediction_sets[1], label='upper-cal', color='navy')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[1], label='upper - uncal', color='gold')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# %% 
plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.scatter(X_pred_viz, Y_pred_viz, label = 'Analytical', color='black', s=10)
plt.scatter(X_pred_viz, mean_viz.flatten(), label='Prediction', color='firebrick', s=10)
plt.plot(X_pred_viz, prediction_sets[0], label='lower-cal', color='teal')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[0], label='lower - uncal', color='darkorange')
plt.plot(X_pred_viz, prediction_sets[1], label='upper-cal', color='navy')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[1], label='upper - uncal', color='gold')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# %% 
plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.errorbar(X_pred_viz, mean_viz.flatten(), yerr=(prediction_sets[1] - prediction_sets[0]).flatten(), label='Prediction', color='firebrick', fmt='o', alpha=0.5)
plt.scatter(X_pred_viz, Y_pred_viz, label = 'Analytical')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# %% 
# Calculate empirical coverage (before and after calibration)

stacked_x = torch.FloatTensor(X_pred)

with torch.no_grad():
    val_lower = nn_lower(stacked_x).numpy()
    val_upper = nn_upper(stacked_x).numpy()

y_response = Y_pred
prediction_sets = get_prediction_sets(X_pred, alpha)

prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0]) & (y_response <= prediction_sets_uncalibrated[1])).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %%
def calibrate(alpha):
    n = cal_split

    with torch.no_grad():
        cal_lower = nn_lower(torch.Tensor(X_cal)).numpy()
        cal_upper = nn_upper(torch.Tensor(X_cal)).numpy()

    cal_scores = np.maximum(Y_cal-cal_upper, cal_lower-Y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_cqr = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_cqr.append(calibrate(alpha_levels[ii]))

# %%
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov_cqr, label='Coverage')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
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
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

prediction_sets =  [prediction - qhat, prediction + qhat]

empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %% 
plt.figure()
plt.hist(cal_scores, 50)
plt.xlabel("Calibration scores")
plt.ylabel("Frequency")

# %%
# Plot residuals conformal predictor

alpha = 0.1

stacked_x = torch.FloatTensor(X_pred_viz)
with torch.no_grad():
    prediction = nn_mean(stacked_x).numpy()
n = len(cal_scores)
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

prediction_sets =  [prediction - qhat, prediction + qhat]

plt.figure()
plt.title(f"Residual conformal, alpha = {alpha}")
plt.plot(X_pred_viz, Y_pred_viz, label='Analytical', color='black')
plt.plot(X_pred_viz, prediction, label='Mean', color='firebrick')
plt.plot(X_pred_viz, prediction_sets[0], label='lower-cal', color='teal')
plt.plot(X_pred_viz, prediction_sets[1], label='upper-cal', color='navy')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# %%
plt.figure()
plt.title(f"Residual conformal, alpha = {alpha}")
plt.errorbar(X_pred_viz, prediction.flatten(), yerr=(prediction_sets[1] - prediction_sets[0]).flatten(), label='Prediction', color='firebrick', fmt='o', alpha=0.5)
plt.scatter(X_pred_viz, Y_pred_viz, label = 'Analytical')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# %%
plt.figure()
plt.title(f"Residual conformal, alpha = {alpha}")
plt.scatter(X_pred_viz, prediction.flatten(), label='Prediction', color='firebrick', s=10)
plt.scatter(X_pred_viz, Y_pred_viz, label = 'Analytical', color='black', s=10)
plt.plot(X_pred_viz,prediction_sets[0], label='lower-cal', color='teal')
plt.plot(X_pred_viz,prediction_sets[1], label='upper-cal', color='navy')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# %%

def calibrate_res(alpha):
    n = cal_split

    cal_scores = conf_metric(X_cal, Y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

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

# %%
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov_res, label='Coverage')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
# Using one network with dropout

nn_dropout = MLP_dropout(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
nn_dropout = nn_dropout.to(device)
#Training the Model -- Comment out this entire cell if you are loading a pre-trained model. 

optimizer = torch.optim.Adam(nn_dropout.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
loss_func = torch.nn.MSELoss()
# loss_func = quantile_loss

it=0
epochs = 1000
loss_list = []

start_time = default_timer()

while it < epochs :
    t1 = default_timer()
    loss_val = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        out = nn_dropout(xx)
        loss = loss_func(out, yy).pow(2).mean()
        loss_val += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_list.append(loss_val.mean().item())
    scheduler.step()

    it += 1
    t2 = default_timer()
    print('It: %d, Time %.3e, Loss: %.3e' % (it, t2 - t1, loss.item()))


train_time = default_timer() - start_time
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('L2 Error')
# plt.xscale('log')
plt.yscale('log')
plt.title('Loss plot of NN with Dropout')

torch.save(nn_dropout.state_dict(), path + '/Models/sin_nn_dropout.pth')

# #Loading the Trained Model
# nn_dropout = MLP_dropout(1, 1, 3, 64) #Input Features, Output Features, Number of Layers, Number of Neurons
# nn_dropout = nn_dropout.to(device)
# nn_dropout.load_state_dict(torch.load(model_loc + 'sin_nn_dropout.pth', map_location='cpu'))

# %% 
#Performing the Calibration
n = cal_split
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    mean_cal, std_cal = MLP_dropout_eval(nn_dropout, torch.FloatTensor(X_cal))

cal_upper = mean_cal + std_cal
cal_lower = mean_cal - std_cal

cal_scores = np.maximum(Y_cal-cal_upper, cal_lower-Y_cal)           

qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

plt.figure()
plt.hist(cal_scores, 50)
plt.xlabel("Calibration scores")
plt.ylabel("Frequency")

# %%
def get_prediction_sets(x, alpha = 0.1):

    X_pred = x
    stacked_x = torch.FloatTensor(X_pred)

    with torch.no_grad():
        val_mean, val_std = MLP_dropout_eval(nn_dropout, stacked_x)

    val_upper = val_mean + val_std
    val_lower = val_mean - val_std

    n = len(cal_scores)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    return [val_lower - qhat, val_upper + qhat]

# %%
alpha = 0.1

num_viz_points = 50

x_viz = np.linspace(lb, ub, num_viz_points)

X_pred_viz = torch.FloatTensor(x_viz).unsqueeze(dim=-1)
Y_pred_viz = f_x(X_pred_viz)

stacked_x_viz = torch.FloatTensor(X_pred_viz)

with torch.no_grad():
    val_mean_viz, val_std_viz = MLP_dropout_eval(nn_dropout, stacked_x_viz)

val_upper_viz = val_mean_viz + val_std_viz
val_lower_viz = val_mean_viz - val_std_viz

prediction_sets = get_prediction_sets(stacked_x_viz)
prediction_sets_uncalibrated = [val_lower_viz, val_upper_viz]

#%%

plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.plot(X_pred_viz, Y_pred_viz, label='Analytical', color='black')
plt.plot(X_pred_viz, val_mean_viz, label='Mean', color='firebrick')
plt.plot(X_pred_viz, prediction_sets[0], label='lower-cal', color='teal')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[0], label='lower - uncal', color='darkorange')
plt.plot(X_pred_viz, prediction_sets[1], label='upper-cal', color='navy')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[1], label='upper - uncal', color='gold')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# %% 
plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.scatter(X_pred_viz, Y_pred_viz, label = 'Analytical', color='black', s=10)
plt.scatter(X_pred_viz, val_mean_viz.flatten(), label='Prediction', color='firebrick', s=10)
plt.plot(X_pred_viz, prediction_sets[0], label='lower-cal', color='teal')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[0], label='lower - uncal', color='darkorange')
plt.plot(X_pred_viz, prediction_sets[1], label='upper-cal', color='navy')
plt.plot(X_pred_viz, prediction_sets_uncalibrated[1], label='upper - uncal', color='gold')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# %% 
plt.figure()
plt.title(f"Conformalised Quantile Regression, alpha = {alpha}")
plt.errorbar(X_pred_viz, val_mean_viz.flatten(), yerr=(prediction_sets[1] - prediction_sets[0]).flatten(), label='Prediction', color='firebrick', fmt='o', alpha=0.5)
plt.scatter(X_pred_viz, Y_pred_viz, label = 'Analytical')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# %% 
# Calculate empirical coverage (before and after calibration)

stacked_x = torch.FloatTensor(X_pred)


with torch.no_grad():
    val_mean, val_std = MLP_dropout_eval(nn_dropout, stacked_x)

val_upper = val_mean + val_std
val_lower = val_mean - val_std

y_response = Y_pred
prediction_sets = get_prediction_sets(X_pred, alpha)

prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0]) & (y_response <= prediction_sets_uncalibrated[1])).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %%
def calibrate(alpha):
    n = cal_split
    
    with torch.no_grad():
        cal_mean, cal_std = MLP_dropout_eval(nn_dropout, torch.Tensor(X_cal))

    cal_upper = cal_mean + cal_std
    cal_lower = cal_mean - cal_std

    cal_scores = np.maximum(Y_cal-cal_upper, cal_lower-Y_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov_dropout = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_dropout.append(calibrate(alpha_levels[ii]))

# %%
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov_dropout, label='Coverage')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
    # %%

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', lw=2.5)
plt.plot(1-alpha_levels, emp_cov_cqr, label='Coverage - Quantile Regression', ls='--')
plt.plot(1-alpha_levels, emp_cov_res, label='Coverage - Residual' ,ls='-')
plt.plot(1-alpha_levels, emp_cov_dropout, label='Coverage - Dropout',  ls='-.')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
from matplotlib import cm 

x_true = x_viz
y_true = Y_pred_viz 
alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alpha_levels)
pred_sets = [get_prediction_sets(x_true.squeeze().reshape(-1,1).astype(np.float32), a) for a in alpha_levels] 

fig, ax = plt.subplots()
[plt.fill_between(x_true, pred_sets[i][0].squeeze(), pred_sets[i][1].squeeze(), color = cols[i]) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
plt.plot(x_true, y_true, '--', label='function', alpha=1, linewidth = 2, color = 'darkblue')

# %%
