#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base tests for demonstrating CP under covariate shift - 
Multivariate setting for 5000-in, 5000-out, PCA over 5000 input dimensions and then KDE over the reduced dimensions. 
Experiemntally evaluating the math behind this https://arxiv.org/abs/1904.06019ule 
"""

# %% 
import numpy as np 
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn 
import scipy.stats as stats
from tqdm import tqdm 

from utils import * 

torch.set_default_dtype(torch.float32   )
# %% 
def func(x):
    return (np.sin(2*x))

#Sampling from a normal distribution
def normal_dist(mean, std, N):
    dist = stats.norm(mean, std)
    return dist.rvs((N, input_size))

N_viz = 1000 #Datapoints a
input_size = output_size = 100

mean_1, std_1 = np.pi/2, np.pi/4
mean_2, std_2 = np.pi/4, np.pi/8

x = normal_dist(mean_1, std_1, N_viz)
x_shift = normal_dist(mean_2, std_2, N_viz) #Covariate shifted

# %% 
#Visualising the covariate shift

plt.hist(x[:, 0], label='Initial')
plt.hist(x_shift[:, 0], label='Shifted')
plt.legend()
plt.title("Visualising the distribution of the initial and the shifted")

# %%
#Obtaining the outputs 
y = func(x)
y_shift = func(x_shift)

# %% 
#Training a NN to model the output 

model = MLP(input_size, output_size, 5, 512)
train_N = 1000
x_lin = normal_dist(mean_1, std_1, train_N)
x_train = torch.tensor(x_lin, dtype=torch.float32)
y_train = torch.tensor(func(x_lin), dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()

epochs = 1000

for ii in tqdm(range(epochs)):    
    optimizer.zero_grad()
    y_out = model(x_train)
    loss = loss_func(y_train, y_out)
    loss.backward()
    optimizer.step()
    
# %% 
#Visualising the model performance on the training data
viz_N = input_size
x_viz = torch.tensor(np.linspace(np.pi/6, np.pi/2, viz_N), dtype=torch.float32)
x_viz = x_viz.repeat(1000, 1)
y_viz= func(x_viz)
y_mean = model(x_viz)


plt.plot(y_train[-1], label='Actual')
plt.plot(y_out[-1].detach().numpy(), label='Pred')
plt.legend()
plt.title("Visualising the Model Performance")


# %%
#Obtaining the Calibration Scores. 
N = 100000 #Datapoints 
X_calib = normal_dist(mean_1, std_1, N)
X_shift = normal_dist(mean_2, std_2, N)#Covariate shifted

y_calib = func(X_calib)

y_calib_nn = model(torch.tensor(X_calib, dtype=torch.float32)).detach().numpy()

#Performing the calibration
cal_scores = np.abs(y_calib - y_calib_nn) #Marginal

# modulation =  np.std(y_calib - y_calib_nn, axis = 0)#Joint
# cal_scores = np.max(np.abs((y_calib - y_calib_nn)/modulation),  axis = (1))#Joint

# %% 
#Using the Known PDFs
def likelihood_ratio(x):
    pdf1 = stats.norm.pdf(x, mean_1, std_1)
    pdf2 = stats.norm.pdf(x, mean_2, std_2)
    return (pdf2 / pdf1)

# def likelihood_ratio(x):
#     return stats.norm.pdf(x, mean_2, std_2)/stats.norm.pdf(x, mean_1, std_1)

def pi(x_new, x_cal):
    return likelihood_ratio(x_cal) / (np.sum(likelihood_ratio(x_cal)) + likelihood_ratio(x_new))
    
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
# %% 
#Multivariate marginal 
pi_vals = pi(X_shift, X_calib)
# %%
qhat = []
for ii in range(output_size):
    qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals[:, ii]))
qhat = np.asarray(qhat)
# %%
y_shift = func(X_shift)
y_shift_nn = model(torch.tensor(X_shift, dtype=torch.float32)).detach().numpy()

prediction_sets =  [y_shift_nn - qhat, y_shift_nn + qhat]#Marginal
# prediction_sets =  [y_shift_nn - qhat*modulation, y_shift_nn + qhat*modulation]#Joint

empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %% 

# %%
idces = np.argsort(X_shift[-1])
plt.plot(X_shift[-1][idces], y_shift[-1][idces], label='Actual')
plt.plot(X_shift[-1][idces], y_shift_nn[-1][idces], label='Pred')
plt.fill_between(X_shift[-1][idces], prediction_sets[0][-1][idces], prediction_sets[1][-1][idces], alpha=0.2)
plt.plot
plt.legend()
plt.title("Visualising the prediction intervals - known pdfs")

# %%
def calibrate_res(alpha):
    qhat = []
    for ii in range(output_size):
     qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals[:, ii]))
    qhat = np.asarray(qhat)

    # qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi_kde(X_shift.T, X_calib.T).squeeze())
    prediction_sets = [y_shift_nn - qhat, y_shift_nn + qhat]
    empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov= []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov, label='Residual - weighted - known pdf' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()


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
plt.hist(x1[:, -1], label='Initial')
plt.hist(x_shift[:, -1], label='Shifted')
plt.legend()
plt.title("Visualising the distribution of the initial and the shifted")


# %%
#KD Estimation. 
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
# %% 
#Multivariate marginal - KDE over the full input space without dim. red.
# pi_vals = pi_kde(X_shift.T, X_calib.T)

# %%
#with dim reduction
#KD Estimation. 
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X_calib)

kde1 = stats.gaussian_kde(pca.transform(X_shift).T)
kde2 = stats.gaussian_kde(pca.transform(X_calib).T)

def pi_kde(x_new, x_cal):
    return likelihood_ratio_KDE(x_cal, kde1, kde2) / (np.sum(likelihood_ratio_KDE(x_cal, kde1, kde2)) + likelihood_ratio_KDE(x_new, kde1, kde2))
   
pi_vals = pi_kde(pca.transform(X_shift).T, pca.transform(X_calib).T)

# %% 
qhat = []
for ii in range(output_size):
    qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals))
qhat = np.asarray(qhat)

# qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi_kde(X_shift.T, X_calib.T).squeeze())#Normal method without going cell-wise. 
# qhat_true = np.quantile(np.sort(weighted_scores), np.ceil((N+1)*(1-alpha))/(N), axis = 0, interpolation='higher')

# %%
y_shift = func(X_shift)
y_shift_nn = model(torch.tensor(X_shift, dtype=torch.float32)).detach().numpy()

prediction_sets =  [y_shift_nn - qhat, y_shift_nn + qhat]#Marginal
# prediction_sets =  [y_shift_nn - qhat*modulation, y_shift_nn + qhat*modulation]#Joint

empirical_coverage = ((y_shift >= prediction_sets[0]) & (y_shift <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# %% 
idces = np.argsort(X_shift[-1])
plt.plot(X_shift[-1][idces], y_shift[-1][idces], label='Actual')
plt.plot(X_shift[-1][idces], y_shift_nn[-1][idces], label='Pred')
plt.fill_between(X_shift[-1][idces], prediction_sets[0][-1][idces], prediction_sets[1][-1][idces], alpha=0.2)
plt.plot
plt.legend()
plt.title("Visualising the prediction intervals - PCA-KDE")
# %%
def calibrate_res(alpha):
    qhat = []
    for ii in range(output_size):
     qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals))
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
plt.plot(1-alpha_levels, emp_cov_kde, label='Residual - weighted - PCA-KDE' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov, label='Residual - weighted - Known' ,ls='-.', color='blue', alpha=0.8, linewidth=1.0)

plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
