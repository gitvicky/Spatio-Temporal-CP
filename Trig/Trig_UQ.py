#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 February 2023

@author: vgopakum, agray, lzanisi

Simple conformal example for y =  sin(x) + cos(2x)
"""
#%% Import required
import numpy as np
from tqdm import tqdm 
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib import cm 

from pyDOE import lhs

import time 
from timeit import default_timer
from tqdm import tqdm 

from utils import *

torch.manual_seed(0)
np.random.seed(0)

# True function is sin(x) + cos(2x)
f_x = lambda x:  np.sin(x) + np.cos(2*x)

# %% Make data sets

N_train = 10
N_cal = 40
N_val = 100

def LHS(lb, ub, N): #Latin Hypercube Sampling for each the angles and the length. 
    return lb + (ub-lb)*lhs(1, N)

x_train = LHS(0, 4, N_train).squeeze().reshape(-1, 1)
x_cal = LHS(0, 4, N_cal).squeeze().reshape(-1, 1)
x_val = LHS(0, 4, N_val).squeeze().reshape(-1, 1)

y_train = f_x(x_train).reshape(-1, 1)
y_cal = f_x(x_cal).reshape(-1, 1)
y_val = f_x(x_val).reshape(-1, 1)

x_train = x_train.astype(np.float32)
x_cal = x_cal.astype(np.float32)
x_val = x_val.astype(np.float32)

y_train = y_train.astype(np.float32)
y_cal = y_cal.astype(np.float32)
y_val = y_val.astype(np.float32)


# # # create dummy data for training
# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# x_train = x_train.reshape(-1, 1)

# y_values = [2*i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)

# %% Define linear Regression Model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    
# %% Model instantiation

inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 100


model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# %% Train model
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        #inputs = torch.FloatTensor(torch.from_numpy(x_train).cuda())
        #labels = torch.FloatTensor(torch.from_numpy(y_train).cuda())
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        # inputs = torch.FloatTensor(torch.from_numpy(x_train))
        # labels = torch.FloatTensor(torch.from_numpy(y_train))
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


# %% Predict
with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

N_true = 1000
x_true = np.linspace(0, 4, N_true)
y_true = f_x(x_true)
plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.plot(x_true, y_true, '--', label='function', alpha=0.5)
plt.legend(loc='best')
plt.show()

# %% Conformal


with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        Y_predicted = model(Variable(torch.from_numpy(x_cal).cuda())).cpu().data.numpy()
    else:
        Y_predicted = model(Variable(torch.from_numpy(x_cal))).data.numpy()

cal_scores = np.abs(Y_predicted-y_cal)

# %%
def get_prediction_sets(x, alpha = 0.1):
    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            Y_predicted = model(Variable(torch.from_numpy(x).cuda())).cpu().data.numpy()
        else:
            Y_predicted = model(Variable(torch.from_numpy(x))).data.numpy()

    qhat = np.quantile(cal_scores, np.ceil((N_cal+1)*(1-alpha))/N_cal, interpolation='higher')
    return [Y_predicted - qhat, Y_predicted + qhat]


# %%
alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov = []
for ii in tqdm(range(len(alpha_levels))):
    sets = get_prediction_sets(x_val, alpha_levels[ii])
    empirical_coverage = ((y_val >= sets[0]) & (y_val <= sets[1])).mean()
    emp_cov.append(empirical_coverage)


plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%

alpha = 0.1

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

N_true = 1000
x_true = np.linspace(0, 4, N_true)
y_true = f_x(x_true)

pred_sets = get_prediction_sets(x_true.squeeze().reshape(-1,1).astype(np.float32), 0.1)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.plot(x_true, pred_sets[0], '--', label='Conf_lower', alpha=0.5)
plt.plot(x_true, pred_sets[1], '--', label='Conf_upper', alpha=0.5)
plt.plot(x_true, y_true, '--', label='function', alpha=0.5)
plt.legend(loc='best')
plt.show()


# %% Alphas plot

alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alpha_levels)
pred_sets = [get_prediction_sets(x_true.squeeze().reshape(-1,1).astype(np.float32), a) for a in alpha_levels] 

fig, ax = plt.subplots()
[plt.fill_between(x_true, pred_sets[i][0].squeeze(), pred_sets[i][1].squeeze(), color = cols[i]) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)


# [plt.plot(x_grid, inf.(ints[i,:]), color = "black", linewidth = 0.7) for i in 1:length(ps)]
# [plt.plot(x_grid, sup.(ints[i,:]), color = "black", linewidth = 0.7) for i in 1:length(ps)]
