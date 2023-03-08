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

import random
import sys


# seed = random.randrange(2**32 - 1)
# print("Seed was:", seed)

seed = 3993374104

torch.manual_seed(seed)
np.random.seed(seed)

# True function is sin(x) + cos(2x)
f_x = lambda x:  np.sin(x) + np.cos(2*x)

# %% Make data sets

N_train = 10
N_cal = 40
N_val = 1000

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

plt.figure()
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
cbar = fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
plt.plot(x_true, y_true, '--', label='function', alpha=1, linewidth = 2)

cbar.ax.set_ylabel('alpha', rotation=270)


# %% Polynomial regression

mymodel = np.poly1d(np.polyfit(x_train.squeeze(), y_train.squeeze(), 3))
cal_scores_poly = np.abs(mymodel(x_cal.squeeze()) - y_cal.squeeze())

alpha = 0.1
alpha_cut = np.ceil((N_cal+1)*(1-alpha))/(N_cal)
qhat = np.quantile(np.sort(cal_scores_poly), alpha_cut, axis = 0,interpolation='higher')

# %% Histogram of calibration scores
plt.figure()
plt.hist(cal_scores)
plt.show()


# %% cdf of calibration scores, with evaluation of inverse

fig, ax = plt.subplots()
plt.step(np.sort(cal_scores_poly), np.linspace(0, 1, N_cal+1)[:-1])

ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()

ax.annotate('', xy= (qhat, alpha_cut), xytext=(0, alpha_cut), arrowprops=dict(arrowstyle="-", linestyle="--"), fontsize = 15)

ax.annotate('', xy= (qhat, ymin), xytext=(qhat, alpha_cut), arrowprops=dict(arrowstyle="->", linestyle="--"), fontsize = 15)

ax.annotate(r'$1 - \alpha$', xy= (xmin, alpha_cut), xytext=(-0.09, 0.7), arrowprops=dict(arrowstyle="->", color="red",connectionstyle="angle3,angleA=-70,angleB=0"), fontsize = 15, color = "red")

ax.text(qhat, ymin - 0.09, r'$\hat{q}$', fontsize = 20, color = "red")

plt.xlabel("s(x,y)", fontsize = 18)
plt.ylabel("cdf", fontsize = 18)
plt.savefig("figures/cal_score_distribution.png", dpi = 600)
plt.show()

# %% Plot of regression with confidence band

def get_prediction_sets_poly(x, alpha = 0.1):

    Y_predicted = mymodel(x)

    qhat = np.quantile(np.sort(cal_scores_poly), np.ceil((N_cal+1)*(1-alpha))/(N_cal), axis = 0,interpolation='higher')

    return [Y_predicted - qhat, Y_predicted + qhat]

alpha = 0.1

[Y_lo, Y_hi] = get_prediction_sets_poly(x_true, alpha)

X_anotate = 2.8

[Y_lo_anon, Y_hi_anon] = get_prediction_sets_poly(X_anotate, alpha)

fig, ax = plt.subplots()
plt.scatter(x_train, y_train, label = "Training points")
plt.plot(x_true, mymodel(x_true), '--', label='Regressor', alpha=0.5,  color = 'tab:orange')
plt.plot(x_true, Y_lo, '--', alpha=0.5, color = 'green')
plt.plot(x_true, Y_hi, '--', label=r'$(1-\alpha)$ confidence band', alpha=0.5, color = 'green')
plt.plot(x_true, y_true, '--', label='True function', alpha=1, linewidth = 2,  color = 'tab:cyan')

ax.annotate('', xy= (X_anotate, Y_hi_anon), xytext=(X_anotate, mymodel(X_anotate)), arrowprops=dict(arrowstyle="->"), fontsize = 15)

ax.annotate('', xy= (X_anotate, Y_lo_anon), xytext=(X_anotate, mymodel(X_anotate)), arrowprops=dict(arrowstyle="->"), fontsize = 15)

ax.text(2.9, 0.5, r'$\hat{q}$', fontsize = 15)

plt.xlabel("X", fontsize = 18)
plt.ylabel("Y", fontsize = 18)
plt.legend()
plt.savefig("figures/Regressor_90.png", dpi = 600)
plt.show()

# %% Compute empirical coverage

[Y_lo_val, Y_hi_val] = get_prediction_sets_poly(x_val, alpha)

empirical_coverage = ((y_val >= Y_lo_val) & (y_val <= Y_hi_val)).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"1 - Alpha <=  empirical_coverage : 1 - {alpha} <= {empirical_coverage} is {1 - alpha <= empirical_coverage}")

# %% Plot all alpha levels

alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alpha_levels)
pred_sets = [get_prediction_sets_poly(x_true, a) for a in alpha_levels] 

fig, ax = plt.subplots()
[plt.fill_between(x_true, pred_sets[i][0].squeeze(), pred_sets[i][1].squeeze(), color = cols[i]) for i in range(len(alpha_levels))]
cbar = fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
plt.plot(x_true, y_true, '--', label='function', alpha=1, linewidth = 2, color = 'tab:cyan')
cbar.ax.set_ylabel('alpha', rotation=270)
plt.xlabel("X", fontsize = 18)
plt.ylabel("Y", fontsize = 18)
plt.savefig("figures/Calibration_all_levels.png", dpi = 600)
plt.show()


# %% Plot empirical coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov = []
for ii in tqdm(range(len(alpha_levels))):
    sets = get_prediction_sets_poly(x_val, alpha_levels[ii])
    empirical_coverage = ((y_val >= sets[0]) & (y_val <= sets[1])).mean()
    emp_cov.append(empirical_coverage)

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.xlabel('1-alpha', fontsize = 18)
plt.ylabel('Empirical Coverage', fontsize = 18)
plt.legend()
plt.savefig("figures/Empirical_coverage.png", dpi = 600)
plt.show()
# %% Plot explaining polynomical conformal regression

N_cal_show = 10     
N_annotate = 9

Y_cal_model = mymodel(x_cal)

fig, ax = plt.subplots()
ax.scatter(x_train, y_train, label = 'Training points')
ax.scatter(x_cal[:N_cal_show], y_cal[:N_cal_show], label = 'Calibration points')
[ax.plot([x_cal[i], x_cal[i]], [Y_cal_model[i], y_cal[i]], alpha = 0.5, color = "red") for i in range(N_cal_show)]
ax.plot(x_true, mymodel(x_true), '--', label='Regressor', alpha = 0.5, color = 'tab:orange')
ax.plot(x_true, y_true, '--', label='True function', alpha = 1, linewidth = 2, color = 'tab:cyan')

ax.annotate('s(x,y)', xy=(x_cal[N_annotate], (y_cal[N_annotate] + Y_cal_model[N_annotate])/2), xytext=(1.5, 1.0), arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=-90,angleB=0"), fontsize = 15)

plt.xlabel("X", fontsize = 18)
plt.ylabel("Y", fontsize = 18)

plt.legend()
plt.savefig("figures/Calibration_example.png", dpi = 600)
plt.show()




# %%
