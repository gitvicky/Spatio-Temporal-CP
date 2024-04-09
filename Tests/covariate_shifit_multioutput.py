#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base tests for demonstrating CP under covariate shift - Multivariate setting 2-in, 3-out, KDE over the 2 input dimensions. 
Experiemntally evaluating the math behind this https://arxiv.org/abs/1904.06019ule 

Ignore
"""

# %% 
import numpy as np 
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn 
import scipy.stats as stats
from tqdm import tqdm 

from utils import * 

torch.set_default_dtype(torch.float32)
# %% 
def func(x1, x2):
    return np.hstack((2*np.sin(x1),  np.cos(x2), np.cos(2*x1 + x2)))

#Sampling from a normal distribution
def normal_dist(mean, std, N):
    dist = stats.norm(mean, std)
    return np.expand_dims(dist.rvs(N), -1)

N = 10000 #Datapoints 

mean_1, std_1 = np.pi/4, np.pi/4
mean_2, std_2 = 2*np.pi, np.pi/8

x1 = normal_dist(mean_1, std_1, N)
x2 = normal_dist(mean_1, std_1, N)

x1_shift = normal_dist(mean_2, std_2, N) #Covariate shifted
x2_shift = normal_dist(mean_2, std_2, N) #Covariate shifted

#Visualising the covariate shift

plt.hist(x1, label='Initial')
plt.hist(x1_shift, label='Shifted')
plt.legend()
plt.title("Visualising the distribution of the initial and the shifted")

# %%
#Obtaining the outputs 
y1 = func(x1, x2)
y_shift = func(x1_shift, x2_shift)

# %% 
#Training a NN to model the output 

model = MLP(2, 3, 5, 64)
train_N = 1000
x_lin = np.expand_dims(np.linspace(0, np.pi, train_N), -1)
x_train = torch.tensor(np.hstack((x_lin, x_lin)), dtype=torch.float32)
y_train = torch.tensor(func(x_lin, x_lin), dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()

epochs = 500

for ii in tqdm(range(epochs)):    
    optimizer.zero_grad()
    y_out = model(x_train)
    loss = loss_func(y_train, y_out)
    loss.backward()
    optimizer.step()
    
# %% 
#Visualising the model performance
viz_N = 500
x_viz = torch.tensor(np.expand_dims(np.linspace(0, np.pi, viz_N), -1), dtype=torch.float32)
y_viz= func(x_viz, x_viz)
y_mean = model(torch.hstack((x_viz, x_viz)))

plt.plot(x_viz, y_viz, label='Actual')
plt.plot(x_viz, y_mean.detach().numpy(), label='Pred')
plt.legend()
plt.title("Visualising the Model Performance")

# %% 
##############################################################################################################################################
#Using Kernel Density Estimation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
##############################################################################################################################################

def likelihood_ratio_KDE(x, kde1, kde2):
    pdf1 = kde1.pdf(x)
    pdf2 = kde2.pdf(x)
    return pdf2 / pdf1 

x1 = normal_dist(mean_1, std_1, N)
x_shift = normal_dist(mean_2, std_2, N) #Covariate shifted

#Visualising the covariate shift
plt.hist(x1, label='Initial')
plt.hist(x_shift, label='Shifted')
plt.legend()
plt.title("Visualising the distribution of the initial and the shifted")

# %%
N = 1000 #Datapoints 
x_calib_1 = normal_dist(mean_1, std_1, N)
x_calib_2 = normal_dist(mean_1, std_1, N)
x_shift_1 = normal_dist(mean_2, std_2, N)#Covariate shifted
x_shift_2 = normal_dist(mean_2, std_2, N)#Covariate shifted

y_calib = func(x_calib_1, x_calib_2)

X_calib = np.hstack((normal_dist(mean_1, std_1, N), normal_dist(mean_1, std_1, N)))
X_shift = np.hstack((normal_dist(mean_2, std_2, N), normal_dist(mean_2, std_2, N)))

y_calib_nn = model(torch.tensor(X_calib, dtype=torch.float32)).detach().numpy()

#Performing the calibration
cal_scores = np.abs(y_calib - y_calib_nn) #Marginal

# modulation =  np.std(y_calib - y_calib_nn, axis = 0)#Joint
# cal_scores = np.max(np.abs((y_calib - y_calib_nn)/modulation),  axis = (1))#Joint


# %%

kde1 = stats.gaussian_kde(X_calib.T)
kde2 = stats.gaussian_kde(X_shift.T)

# %%
def pi_kde(x_new, x_cal):
    return likelihood_ratio_KDE(x_cal, kde1, kde2) / (np.sum(likelihood_ratio_KDE(x_cal, kde1, kde2)) + likelihood_ratio_KDE(x_new, kde1, kde2))
    
# weighted_scores = cal_scores * pi_kde(X_shift.T, X_calib.T)

# %% 
#Estimating qhat 

alpha = 0.1

def weighted_quantile(data, alpha, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.quantile(np.sort(data), alpha, axis = 0, interpolation='higher')
    
    ind=np.argsort(data, axis=0)
    d=data[ind]
    w=weights[ind]

    p=1.*w.cumsum()/w.sum()
    y=np.interp(alpha, p, d)

    return y

#Multivariate marginal
qhat = []
pi = pi_kde(X_shift.T, X_calib.T)

for ii in range(3):
    qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi))
qhat = np.asarray(qhat)

# qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi_kde(X_shift.T, X_calib.T).squeeze())#Normal method without going cell-wise. 
# qhat_true = np.quantile(np.sort(weighted_scores), np.ceil((N+1)*(1-alpha))/(N), axis = 0, interpolation='higher')

# %%
y_shift = func(x_shift_1, x_shift_2)
y_shift_nn = model(torch.tensor(X_shift, dtype=torch.float32)).detach().numpy()


prediction_sets =  [y_shift_nn - qhat, y_shift_nn + qhat]#Marginal
# prediction_sets =  [y_shift_nn - qhat*modulation, y_shift_nn + qhat*modulation]#Joint

empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %%
def calibrate_res(alpha):
    qhat = []
    for ii in range(3):
     qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi))
    qhat = np.asarray(qhat)

    # qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi_kde(X_shift.T, X_calib.T).squeeze())
    prediction_sets = [y_shift_nn - qhat, y_shift_nn + qhat]
    empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_kde = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_kde.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov_kde, label='Residual - weighted - KDE' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

