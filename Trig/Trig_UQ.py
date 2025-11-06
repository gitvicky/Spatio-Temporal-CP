#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
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
#Better plots for the talks 

import matplotlib.pyplot as plt
import numpy as np


# Set up the plot with high-quality defaults
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Configuration
N_cal_show = 10     
N_annotate = 9

# Assuming you have your data variables: x_train, y_train, x_cal, y_cal, x_true, y_true, mymodel
# Y_cal_model = mymodel(x_cal)

fig, ax = plt.subplots(figsize=(10, 8))

# Enhanced color scheme
colors = {
    'training': '#2E86AB',      # Deep blue
    'calibration': '#F24236',   # Coral red
    'regressor': '#F18F01',     # Orange
    'true_function': '#00B4A6', # Teal
    'residuals': '#E63946'      # Red for residual lines
}




# Plot with enhanced styling
ax.scatter(x_train, y_train, 
          label='Training points', 
          color=colors['training'], 
          s=60, 
          alpha=0.8, 
          edgecolors='white', 
          linewidth=1.5,
          zorder=5)

ax.scatter(x_cal[:N_cal_show], y_cal[:N_cal_show], 
          label='Calibration points', 
          color=colors['calibration'], 
          s=60, 
          alpha=0.9, 
          edgecolors='white', 
          linewidth=1.5,
          zorder=5)

# Residual lines with better styling
for i in range(N_cal_show):
    ax.plot([x_cal[i], x_cal[i]], [Y_cal_model[i], y_cal[i]], 
           alpha=0.7, 
           color=colors['residuals'], 
           linewidth=2,
           zorder=3)

# Enhanced line plots
ax.plot(x_true, mymodel(x_true), 
       '--', 
       label='Neural Network', 
       alpha=0.8, 
       color=colors['regressor'],
       linewidth=2.5,
       zorder=4)

ax.plot(x_true, y_true, 
       '-', 
       label='True function', 
       alpha=1, 
       linewidth=3, 
       color=colors['true_function'],
       zorder=4)

# Enhanced annotation
ax.annotate('s(x,y)', 
           xy=(x_cal[N_annotate], (y_cal[N_annotate] + Y_cal_model[N_annotate])/2), 
           xytext=(1.5, 1.0), 
           arrowprops=dict(arrowstyle="->", 
                          connectionstyle="angle3,angleA=-90,angleB=0",
                          color='black',
                          lw=1.5), 
           fontsize=16,
           fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', 
                    edgecolor='black',
                    alpha=0.9))

# Enhanced labels and styling
ax.set_xlabel("X", fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel("Y", fontsize=18, fontweight='bold', labelpad=10)

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=14, colors='black')

# Enhanced legend
legend = ax.legend(loc='lower left', 
                  fontsize=20, 
                  frameon=True, 
                  fancybox=True, 
                  shadow=True,
                  framealpha=0.95,
                  edgecolor='black',
                  facecolor='white')

# Set legend marker sizes
for handle in legend.legendHandles:
    if hasattr(handle, 'set_sizes'):
        handle.set_sizes([80])

# Fine-tune the plot appearance
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-1.1, 1.3)

# Set custom tick intervals
ax.set_xticks(np.arange(0, 5, 1))  # X ticks every 1 unit
ax.set_yticks(np.arange(-1, 1.5, 0.5))  # Y ticks every 0.5 units

# Add subtle background color
ax.set_facecolor('#fafafa')

# Tight layout for better spacing
plt.tight_layout()

# Save with high resolution and quality
plt.savefig("figures/Calibration_example_enhanced.png", 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           transparent=False)

# Also save as PDF for vector graphics (best for presentations)
plt.savefig("figures/Calibration_example_enhanced.pdf", 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           transparent=False)

plt.show()

# Reset matplotlib parameters to defaults (optional)
plt.rcParams.update(plt.rcParamsDefault)


# %% 
def get_prediction_sets_poly(x, alpha = 0.1):
    Y_predicted = mymodel(x)
    qhat = np.quantile(np.sort(cal_scores_poly), np.ceil((N_cal+1)*(1-alpha))/(N_cal), axis = 0,interpolation='higher')
    return [Y_predicted - qhat, Y_predicted + qhat]

# Set up the plot with high-quality defaults (matching your enhanced style)
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

alpha = 0.1
[Y_lo, Y_hi] = get_prediction_sets_poly(x_true, alpha)
X_anotate = 2.8
[Y_lo_anon, Y_hi_anon] = get_prediction_sets_poly(X_anotate, alpha)

# Enhanced color scheme (matching your first plot)
colors = {
    'training': '#2E86AB',      # Deep blue
    'regressor': '#F18F01',     # Orange
    'confidence': '#E63946',    # Teal (for confidence bands)
    'true_function': '#00B4A6', # Teal
    'annotation': '#E63946'     # Red for annotations
}

fig, ax = plt.subplots(figsize=(10, 8))

# Enhanced training points
ax.scatter(x_train, y_train, 
          label='Training points', 
          color=colors['training'], 
          s=60, 
          alpha=0.8, 
          edgecolors='white', 
          linewidth=1.5,
          zorder=5)

# Enhanced regressor line
ax.plot(x_true, mymodel(x_true), 
       '--', 
       label='Neural Network', 
       alpha=0.8, 
       color=colors['regressor'],
       linewidth=2.5,
       zorder=4)

# Enhanced confidence bands
ax.plot(x_true, Y_lo, 
       '--', 
       alpha=0.8, 
       color=colors['confidence'],
       linewidth=2.5,
       zorder=3)

ax.plot(x_true, Y_hi, 
       '--', 
       label=r'$(1-\alpha)$ confidence band', 
       alpha=0.8, 
       color=colors['confidence'],
       linewidth=2.5,
       zorder=3)

# Enhanced true function line
ax.plot(x_true, y_true, 
       '-', 
       label='True function', 
       alpha=1, 
       linewidth=3, 
       color=colors['true_function'],
       zorder=4)

# Enhanced annotations with better styling
ax.annotate('', 
           xy=(X_anotate, Y_hi_anon), 
           xytext=(X_anotate, mymodel(X_anotate)), 
           arrowprops=dict(arrowstyle="->",
                          color=colors['annotation'],
                          lw=1.5), 
           fontsize=15)

ax.annotate('', 
           xy=(X_anotate, Y_lo_anon), 
           xytext=(X_anotate, mymodel(X_anotate)), 
           arrowprops=dict(arrowstyle="->",
                          color=colors['annotation'],
                          lw=1.5), 
           fontsize=15)

# Enhanced text annotation
ax.text(2.9, 0.5, r'$\hat{q}$', 
       fontsize=16,
       fontweight='bold',
       bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', 
                edgecolor='black',
                alpha=0.9))

# Enhanced labels and styling
ax.set_xlabel("X", fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel("Y", fontsize=18, fontweight='bold', labelpad=10)

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=14, colors='black')

# Enhanced legend
legend = ax.legend(loc='lower left', 
                  fontsize=20, 
                  frameon=True, 
                  fancybox=True, 
                  shadow=True,
                  framealpha=0.95,
                  edgecolor='black',
                  facecolor='white')

# Set legend marker sizes
for handle in legend.legendHandles:
    if hasattr(handle, 'set_sizes'):
        handle.set_sizes([80])

# Fine-tune the plot appearance (adjust limits as needed for your data)
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-1.1, 1.3)

# Set custom tick intervals
ax.set_xticks(np.arange(0, 5, 1))  # X ticks every 1 unit
ax.set_yticks(np.arange(-1, 1.5, 0.5))  # Y ticks every 0.5 units

# Add subtle background color
ax.set_facecolor('#fafafa')

# Tight layout for better spacing
plt.tight_layout()

# Save with high resolution and quality
plt.savefig("figures/Regressor_90_enhanced.png", 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           transparent=False)

# Also save as PDF for vector graphics (best for presentations)
plt.savefig("figures/Regressor_90_enhanced.pdf", 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           transparent=False)

plt.show()

# Reset matplotlib parameters to defaults (optional)
plt.rcParams.update(plt.rcParamsDefault)

# %% 
# Set up the plot with high-quality defaults (matching your enhanced style)
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Enhanced color scheme (matching your other plots)
colors = {
    'cdf_line': '#2E86AB',      # Deep blue for CDF line
    'annotation': '#E63946',    # Red for annotations
    'grid_lines': '#00B4A6',    # Teal for grid lines
    'text': '#E63946'           # Red for text
}

fig, ax = plt.subplots(figsize=(10, 8))

# Enhanced step plot for CDF
ax.step(np.sort(cal_scores_poly), 
        np.linspace(0, 1, N_cal+1)[:-1],
        color=colors['cdf_line'],
        linewidth=3,
        alpha=0.9,
        where='post',
        zorder=5)

ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()

# Enhanced horizontal dashed line
ax.annotate('', 
           xy=(qhat, alpha_cut), 
           xytext=(0, alpha_cut), 
           arrowprops=dict(arrowstyle="-", 
                          linestyle="--",
                          color=colors['grid_lines'],
                          lw=2,
                          alpha=0.8), 
           fontsize=15)

# Enhanced vertical dashed line with arrow
ax.annotate('', 
           xy=(qhat, ymin), 
           xytext=(qhat, alpha_cut), 
           arrowprops=dict(arrowstyle="->", 
                          linestyle="--",
                          color=colors['grid_lines'],
                          lw=2,
                          alpha=0.8), 
           fontsize=15)

# Enhanced annotation for 1-alpha
ax.annotate(r'$1 - \alpha$', 
           xy=(xmin, alpha_cut), 
           xytext=(-0.09, 0.7), 
           arrowprops=dict(arrowstyle="->", 
                          color=colors['annotation'],
                          connectionstyle="angle3,angleA=-70,angleB=0",
                          lw=1.5), 
           fontsize=18, 
           fontweight='bold',
           color=colors['text'],
           bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', 
                    edgecolor=colors['annotation'],
                    alpha=0.9))

# Enhanced text annotation for q-hat
ax.text(qhat, ymin - 0.09, r'$\hat{q}$', 
       fontsize=20, 
       fontweight='bold',
       color=colors['text'],
       ha='center',
       bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', 
                edgecolor=colors['annotation'],
                alpha=0.9))

# Enhanced labels and styling
ax.set_xlabel("s(x,y)", fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel("CDF", fontsize=18, fontweight='bold', labelpad=10)

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=14, colors='black')

# Fine-tune the plot appearance
ax.set_ylim(ymin - 0.05, ymax + 0.05)  # Add some padding
ax.set_xlim(xmin - 0.02, xmax + 0.02)  # Add some padding

# Add subtle background color
ax.set_facecolor('#fafafa')

# Enhance the grid
ax.grid(True, alpha=0.3, linewidth=0.8, color='gray')

# Add minor ticks for better precision
ax.minorticks_on()

# Tight layout for better spacing
plt.tight_layout()

# Save with high resolution and quality
plt.savefig("figures/cal_score_distribution_enhanced.png", 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           transparent=False)

# Also save as PDF for vector graphics (best for presentations)
plt.savefig("figures/cal_score_distribution_enhanced.pdf", 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           transparent=False)

plt.show()

# Reset matplotlib parameters to defaults (optional)
plt.rcParams.update(plt.rcParamsDefault)
# %% Conformal with covariate shift

###
#   Useful if we want to predict on distribution other than the calibration distribution, without having to re-run the entire conformal algorithm
###

import scipy.stats as stats

# Paramaters of the new distribution, truncated normal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
myclip_a = 0; myclip_b = 4      
my_mean = 3; my_std = 0.2
a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

def X_dist(x):
    return stats.uniform(0, 4).pdf(x)

# # Truncated normal
def X_new(x):
    return stats.truncnorm.pdf(x, a, b, loc = my_mean, scale = my_std)

# Using LHS 
def X_new_sample(N):
    ps = LHS(0, 1, N)
    return stats.truncnorm.ppf(ps, a, b, loc = my_mean, scale = my_std)

# def X_new(x):
#     return stats.uniform(0, 4).pdf(x)

# def X_new_sample(N):
#     return LHS(0, 4, N)

plt.plot(np.linspace(0,4, 100), X_new(np.linspace(0,4, 100)))

def like_ratio(x):
    return X_new(x) / X_dist(x)

N_new = 1000
X_new_samps = X_new_sample(N_new).squeeze()
Y_new_samps = f_x(X_new_samps).reshape(-1, 1).squeeze()

cal_scores_poly_2 = np.abs(mymodel(X_new_samps.squeeze()) - Y_new_samps.squeeze())

# %%
# def pi(x, X_i):
#     return like_ratio(X_i)/ (np.sum(like_ratio(x_cal)) + like_ratio(x))

def pi(x, x_cal):
    return like_ratio(x_cal)/ (np.sum(like_ratio(x_cal)) + like_ratio(x))
    

x_predict = 1

wighted_scores = cal_scores_poly.squeeze() * pi(x_predict, x_cal).squeeze()

fig, ax = plt.subplots()
plt.step(np.sort(cal_scores_poly), np.linspace(0, 1, N_cal+1)[:-1])
plt.step(np.sort(cal_scores_poly_2), np.linspace(0, 1, N_new+1)[:-1])
#plt.step(np.sort(wighted_scores.squeeze()), np.linspace(0, 1, N_cal+1)[:-1])

plt.xlabel("s(x,y)", fontsize = 18)
plt.ylabel("cdf", fontsize = 18)
#plt.show()

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sample = cal_scores_poly.squeeze()
weights = pi(x_predict, x_cal).squeeze()
df = pd.DataFrame(np.vstack((sample, weights)).T, columns = ['sample', 'weights'])
sns.ecdfplot(data = df, x = 'sample', weights = 'weights', stat = 'proportion', legend = True)

# %%

alpha = 0.1

def weighted_quantile(data, alpha, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.quantile(np.sort(data), alpha, axis = 0, interpolation='higher')
    
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]

    #d = d[1:]
    #d = np.append(d, np.inf)

    p=1.*w.cumsum()/w.sum()
    y=np.interp(alpha, p, d)

    return y

qhat = weighted_quantile(cal_scores_poly, np.ceil((N_cal+1)*(1-alpha))/(N_cal), pi(x_predict, x_cal).squeeze())
qhat_true = np.quantile(np.sort(cal_scores_poly_2), np.ceil((N_new+1)*(1-alpha))/(N_new), axis = 0, interpolation='higher')

# %%
def get_prediction_sets_poly_shift(x, alpha = 0.1):
    Y_lo = []; Y_hi = []

    for x_in in x:
        Y_predicted = mymodel(x_in)
        qhat = weighted_quantile(cal_scores_poly, np.ceil((N_cal+1)*(1-alpha))/(N_cal), pi(x_in, x_cal).squeeze())

        Y_lo.append(Y_predicted - qhat)
        Y_hi.append(Y_predicted + qhat)
    
    Y_lo = np.array(Y_lo).squeeze()
    Y_hi = np.array(Y_hi).squeeze()

    return [Y_lo, Y_hi]


alpha = 0.1

[Y_lo_val, Y_hi_val] = get_prediction_sets_poly_shift(X_new_samps, alpha)

empirical_coverage = ((Y_new_samps >= Y_lo_val) & (Y_new_samps <= Y_hi_val)).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"1 - Alpha <=  empirical_coverage : 1 - {alpha} <= {empirical_coverage} is {1 - alpha <= empirical_coverage}")

# %%
alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov = []
for ii in tqdm(range(len(alpha_levels))):
    sets = get_prediction_sets_poly_shift(X_new_samps, alpha_levels[ii])
    empirical_coverage = ((Y_new_samps >= sets[0]) & (Y_new_samps <= sets[1])).mean()
    emp_cov.append(empirical_coverage)

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal')
plt.plot(1-alpha_levels, emp_cov, label='Coverage')
plt.xlabel('1-alpha', fontsize = 18)
plt.ylabel('Empirical Coverage', fontsize = 18)
plt.legend()
plt.show()

# %%

[Y_lo_shift, Y_hi_shift] = get_prediction_sets_poly_shift(x_true, 0.1)
[Y_lo, Y_hi] = get_prediction_sets_poly(x_true, 0.1)

fig, ax = plt.subplots()
ax.plot(x_true, mymodel(x_true), '--', label='Regressor', alpha = 0.5, color = 'tab:orange')
ax.plot(x_true, y_true, '--', label='True function', alpha = 1, linewidth = 2, color = 'tab:cyan')

plt.scatter(X_new_samps, Y_new_samps)

plt.plot(x_true, Y_lo, '--', alpha=0.5, color = 'green')
plt.plot(x_true, Y_hi, '--', label=r'Origional', alpha=0.5, color = 'green')

plt.plot(x_true, Y_lo_shift, '--', alpha=0.5, color = 'red')
plt.plot(x_true, Y_hi_shift, '--', label=r'Shift', alpha=0.5, color = 'red')


plt.xlabel("X", fontsize = 18)
plt.ylabel("Y", fontsize = 18)

plt.legend()
plt.show()

# %% Alphas plot

alpha_levels = np.arange(0.05, 0.95, 0.05)
cols = cm.plasma(alpha_levels)
pred_sets = [get_prediction_sets_poly_shift(x_true.squeeze().reshape(-1,1).astype(np.float32), a) for a in alpha_levels] 

fig, ax = plt.subplots()
[plt.fill_between(x_true, pred_sets[i][0].squeeze(), pred_sets[i][1].squeeze(), color = cols[i]) for i in range(len(alpha_levels))]
cbar = fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax)
plt.plot(x_true, y_true, '--', label='function', alpha=1, linewidth = 2)

cbar.ax.set_ylabel('alpha', rotation=270)

# %% Evaluation of the output distribution
##
# Evaluation of the output confidence distribution
##

x_val = 2.3
y_true = f_x(x_val)

alpha_levels = np.arange(0.05, 0.95, 0.01)
pred_sets = [get_prediction_sets_poly(x_val, a) for a in alpha_levels] 

pred_sets_shift = [get_prediction_sets_poly_shift([x_val], a) for a in alpha_levels] 

lows = [pred_sets[i][0] for i in range(len(alpha_levels))]
his = [pred_sets[i][1] for i in range(len(alpha_levels))]

lows_shift = [pred_sets_shift[i][0] for i in range(len(alpha_levels))]
his_shift = [pred_sets_shift[i][1] for i in range(len(alpha_levels))]

plt.figure()

plt.plot(lows, alpha_levels, color = "blue")
plt.plot(his, alpha_levels, color = "blue")

plt.plot(lows_shift, alpha_levels, color = "orange")
plt.plot(his_shift, alpha_levels, color = "orange")

plt.plot([y_true, y_true], [0, 1])
plt.show()

# %%    Running Monte Carlo through the Polynomial regression

N_prop = 10000
X_prop = LHS(0, 4, N_prop)

Y_prop_true = f_x(X_prop).squeeze()
alpha_levels = np.arange(0.05, 0.95, 0.05)

pred_sets = [[get_prediction_sets_poly(x_val, a) for x_val in X_prop]  for a in alpha_levels]

plt.figure()

cols = cm.plasma(alpha_levels)

for j in range(len(alpha_levels)):
    pred_lows = np.array([pred_sets[j][i][0] for i in range(N_prop)]).squeeze()
    pred_his = np.array([pred_sets[j][i][1] for i in range(N_prop)]).squeeze()
    plt.fill_betweenx(np.linspace(0, 1, N_prop+1)[:-1], np.sort(pred_his), np.sort(pred_lows), color = cols[j])

plt.step(np.sort(Y_prop_true), np.linspace(0, 1, N_prop+1)[:-1])
plt.title("Confidence bands on the output distribution")
plt.xlabel("Y")
plt.ylabel("cdf")
plt.show()

# %% Confidence distribution on expectation value

Ex_true = np.mean(Y_prop_true)
Ex_conf_left = [np.mean([pred_sets[j][i][0] for i in range(N_prop)])for j in range(len(alpha_levels))]
Ex_conf_right = [np.mean([pred_sets[j][i][1] for i in range(N_prop)])for j in range(len(alpha_levels))]

plt.figure()
plt.plot(Ex_conf_left, alpha_levels, color = "blue")
plt.plot(Ex_conf_right, alpha_levels, color = "blue")
plt.plot([Ex_true, Ex_true], [0, 1], color = "red")
plt.title("Confidence on the expected value")
plt.xlabel(r"$P(Y)$")
plt.ylabel(r"$\alpha$")
plt.show()


# %%    Evaluating Monte Carlo with a different (covariate shift) distribution

X_new_MC = X_new_sample(N_prop)
Y_new_MC = f_x(X_new_MC).squeeze()

alpha_levels = np.arange(0.05, 0.95, 0.05)
pred_sets_shift = [[get_prediction_sets_poly_shift(x_val, a) for x_val in X_new_MC]  for a in alpha_levels]


plt.figure()

cols = cm.plasma(alpha_levels)

for j in range(len(alpha_levels)):
    pred_lows = np.array([pred_sets_shift[j][i][0] for i in range(N_prop)]).squeeze()
    pred_his = np.array([pred_sets_shift[j][i][1] for i in range(N_prop)]).squeeze()
    plt.fill_betweenx(np.linspace(0, 1, N_prop+1)[:-1], np.sort(pred_his), np.sort(pred_lows), color = cols[j])

plt.step(np.sort(Y_new_MC), np.linspace(0, 1, N_prop+1)[:-1])
plt.title("Confidence bands on the output on shifted distribution")
plt.xlabel("Y")
plt.ylabel("cdf")
plt.show()

# %%

# %% Confidence distribution on expectation value

Ex_true = np.mean(Y_new_MC)
Ex_conf_left = [np.mean([pred_sets_shift[j][i][0] for i in range(N_prop)])for j in range(len(alpha_levels))]
Ex_conf_right = [np.mean([pred_sets_shift[j][i][1] for i in range(N_prop)])for j in range(len(alpha_levels))]

plt.figure()
plt.plot(Ex_conf_left, alpha_levels, color = "blue")
plt.plot(Ex_conf_right, alpha_levels, color = "blue")
plt.plot([Ex_true, Ex_true], [0, 1], color = "red")
plt.title("Confidence on the expected value of shifted distribution")
plt.xlabel(r"$P(Y)$")
plt.ylabel(r"$\alpha$")
plt.show()

# %%
