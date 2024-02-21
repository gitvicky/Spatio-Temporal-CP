#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FNO built using PyTorch to model the 2D Wave Equation. 
Dataset buitl by changing by performing a LHS across the x,y pos and amplitude of the initial gaussian distibution
Code for the spectral solver can be found in : https://github.com/farscape-project/PINNs_Benchmark

----------------------------------------------------------------------------------------------------------------------------------------
Experimenting with prob classification for DRE over the parameterised PDE. 

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Type": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No',
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 20,    
                 "T_out": 60,
                 "Step": 10,
                 "Width": 32, 
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'LP',
                 "UQ": 'Dropout', #None, Dropout
                 "Pinball Gamma": 'NA',
                 "Dropout Rate": 0.1
                 }
#%% 
import os
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 
import matplotlib as mpl 
plt.rcParams['text.usetex'] = True

plt.rcParams['grid.linewidth'] = 1.0
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

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

import time 
from timeit import default_timer
from tqdm import tqdm 

from utils import *

torch.manual_seed(0)
np.random.seed(0)

# %% 
path = os.getcwd()
model_loc = path + '/Models/'
data_loc = path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# load data
# _a -- referes to the input 
# _u -- referes ot the output
################################################################
t1 = default_timer()

data =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_5K.npz')
u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)
xx, yy = np.meshgrid(x,y)

ntrain = 1000
ncal = 1000
npred = 1000
S = 33 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']
modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']


# %%
#Chunking the data. 
train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

cal_a = u[ntrain:ntrain+ncal,:,:,:T_in]
cal_u = u[ntrain:ntrain+ncal,:,:,T_in:T+T_in]

pred_a = u[ntrain+ncal:ntrain+ncal+npred,:,:,:T_in]
pred_u = u[ntrain+ncal:ntrain+ncal+npred,:,:,T_in:T+T_in]


print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)

t2 = default_timer()
print('Data sorting finished, time used:', t2-t1)

# %% 
#Normalisation. 
a_normalizer = MinMax_Normalizer(train_a)
train_a = a_normalizer.encode(train_a)
cal_a = a_normalizer.encode(cal_a)
pred_a = a_normalizer.encode(pred_a)

y_normalizer = MinMax_Normalizer(train_u)
train_u = y_normalizer.encode(train_u)
cal_u = y_normalizer.encode(cal_u)
pred_u = y_normalizer.encode(pred_u)


# %%
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#############################################################
# Conformal Prediction Residuals
#############################################################

model_50 = FNO2d(modes, modes, width, T_in, step, x, y)
model_50.load_state_dict(torch.load(model_loc + 'FNO_Wave_fno.pth', map_location='cpu'))

# %% 
t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a

    for tt in tqdm(range(0, T, step)):
        pred = model_50(xx)

        if tt == 0:
            cal_mean = pred

        else:
            cal_mean = torch.cat((cal_mean, pred), -1)       

        xx = torch.cat((xx[..., step:], pred), dim=-1)

cal_mean = cal_mean.numpy()
cal_scores = np.abs(cal_u.numpy()-cal_mean)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()

with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        pred_mean = model_50(xx)

        if tt == 0:
            val_mean = pred_mean
        else:     
            val_mean = torch.cat((val_mean, pred_mean), -1)       

        xx = torch.cat((xx[..., step:], pred_mean), dim=-1)

val_mean = val_mean.numpy()
prediction_sets = [val_mean - qhat, val_mean + qhat]

# %%
print('Conformal by way Residual')
# Calculate empirical coverage (before and after calibration)
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

t2 = default_timer()
print('Conformal by Residual, time used:', t2-t1)

#Estimating the tightness of fit
cov = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1]))
cov_idx = cov.nonzero()

tightness_metric = ((prediction_sets[1][cov_idx]  - y_response[cov_idx]) +  (y_response[cov_idx] - prediction_sets[0][cov_idx])).mean()

print(f"Tightness of the coverage : Average of the distance between error bars {tightness_metric}")

# %%
def calibrate_residual(alpha):

    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()

    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []

for ii in tqdm(range(len(alpha_levels))):
    emp_cov_res.append(calibrate_residual(alpha_levels[ii]))

# %% 
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5 
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20

# %% 
#Generating new data at normal speed but with shifted distributions. 
#Sampling the PDE parameters from known distributions. 
#Three parameters of interest currently: amplitude, x-pos, y-pos
from Spectral_Wave_Data_Gen import *
import scipy.stats as stats
from utils import *

def data_generation(ampl, x_pos, y_pos):
    list_u = []
    for ii in tqdm(range(n_sims)):
                x, y, t, u = wave_solution(ampl[ii], x_pos[ii], y_pos[ii])
                list_u.append(u)
    return np.asarray(list_u)

#Sampling from a normal distribution
def normal_dist(mean, std, N):
    dist = stats.norm(mean, std)
    return dist.rvs((N, 1))

n_sims = 1000

# %%
#Generating calibration data 
ampl_cal = normal_dist(20, 5, n_sims)
x_pos_cal = normal_dist(0.25, 0.1, n_sims) #Covariate shifted
y_pos_cal = normal_dist(0.25, 0.1, n_sims) #Covariate shifted

u_cal = data_generation(ampl_cal, x_pos_cal, y_pos_cal)
u_cal = torch.tensor(u_cal, dtype=torch.float32)
u_cal = u_cal.permute(0, 2, 3, 1)
# %% 
#Generating shifted data 
ampl_shift = normal_dist(35, 5, n_sims)
x_pos_shift = normal_dist(0.35, 0.1, n_sims) #Covariate shifted
y_pos_shift = normal_dist(0.35, 0.1, n_sims) #Covariate shifted

u_shift = data_generation(ampl_shift, x_pos_shift, y_pos_shift)
u_shift = torch.tensor(u_shift, dtype=torch.float32)
u_shift = u_shift.permute(0, 2, 3, 1)
# %% 
#Obtaining the calibration scores

u_cal_a = u_cal[...,:T_in]
u_cal_u = u_cal[...,T_in:T+T_in]

with torch.no_grad():
    xx = u_cal_a

    for tt in tqdm(range(0, T, step)):
        pred = model_50(xx)

        if tt == 0:
            cal_mean = pred

        else:
            cal_mean = torch.cat((cal_mean, pred), -1)       

        xx = torch.cat((xx[..., step:], pred), dim=-1)

cal_mean = cal_mean.numpy()
cal_scores = np.abs(u_cal_u.numpy()-cal_mean)

# %% 
#Building an MLP Classifier 
input_size = 3 #PDE parameters
output_size = 1 #classifier output
classifier = MLP(input_size, output_size, 5, 128, activation=torch.nn.ReLU())

loss_func = torch.nn.BCEWithLogitsLoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)


# %%
#Data_prep for the classifier
X_class = np.vstack((np.hstack((ampl_cal, x_pos_cal, y_pos_cal)), np.hstack((ampl_shift, x_pos_shift, y_pos_shift))))
Y_class = np.vstack((np.expand_dims(np.zeros(n_sims), -1), np.expand_dims(np.ones(n_sims) ,-1)))

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_class, dtype=torch.float32), torch.tensor(Y_class, dtype=torch.float32)), batch_size=100, shuffle=True)
# %% 
#Training the classifier. 
epochs = 1000
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = classifier(xx)
        loss = loss_func(y_out, yy)
        loss.backward()
        optimizer.step()

# %%
    
#Classifier performance. - within the training data itself. 
y_pred = torch.sigmoid(classifier(torch.tensor(X_class, dtype=torch.float32))).detach().numpy()
y_true = Y_class

for ii in range(len(y_pred)):
    if y_pred[ii] < 0.5:
        y_pred[ii] =0 
    else: 
        y_pred[ii] = 1.0

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
confusion_matrix(y_true, y_pred)


# %% 
#Estimating the likeloihood ratio
def likelihood_ratio_classifier(X):
    y_pred = torch.sigmoid(classifier(torch.tensor(X, dtype=torch.float32))).detach().numpy()

#Avoiding numerical instabilities in likelihood ratio estimation 
    for ii in range(len(y_pred)): 
        if y_pred[ii] < 0.01:
            y_pred[ii] = 0.01
        elif y_pred[ii] >= 0.99:
            y_pred[ii] = 0.99

    return (y_pred/(1-y_pred))

# %%
#Estimating the pi values. 
def pi_classifer(x_new, x_cal):
    return likelihood_ratio_classifier(np.expand_dims(x_cal, 1)) / (np.sum(likelihood_ratio_classifier(np.expand_dims(x_cal, 1))) + likelihood_ratio_classifier(np.expand_dims(x_new,1)))
    
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

X_calib_params = np.hstack((ampl_cal, x_pos_cal, y_pos_cal))
X_shift_params = np.hstack((ampl_shift, x_pos_shift, y_pos_shift))

pi_vals = pi_classifer(X_shift_params, X_calib_params)

# %% 
#Currently we are doing the qhat estimation for each different variable separately. 
#Need to parallelise this operation. 
N = n_sims
qhat = []
output_size = len(cal_scores.flatten())
for ii in range(output_size):
    qhat.append(weighted_quantile(cal_scores.flatten[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals))
qhat = np.asarray(qhat)

# %%
#Obtaining the Prediction Sets
u_shift_a = u_shift[...,:T_in]
u_shift_u = u_shift[...,T_in:T+T_in]

pred_a = u_shift_a
y_response = u_shift_u.numpy()

with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        pred_mean = model_50(xx)

        if tt == 0:
            val_mean = pred_mean
        else:     
            val_mean = torch.cat((val_mean, pred_mean), -1)       

        xx = torch.cat((xx[..., step:], pred_mean), dim=-1)

val_mean = val_mean.numpy()
prediction_sets = [val_mean - qhat, val_mean + qhat]

empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")


# %%
def calibrate_res(alpha):
    qhat = []
    for ii in range(output_size):
     qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals[:, ii]))
    qhat = np.asarray(qhat)

    # qhat = weighted_quantile(cal_scores, np.ceil((N+1)*(1-alpha))/(N), pi_kde(X_shift.T, X_calib.T).squeeze())
    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov= []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov, label='Residual - weighted - prob. class.' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 
#Setting up the Convolutional classifier.

# 1D convolutional classifier. 
class classifier_2D(nn.Module):
    def __init__(self, in_features, activation=torch.nn.ReLU()):
        super(classifier_2D, self).__init__()

        self.in_features = in_features

        #Convolutional Layers
        self.conv1 = nn.Conv2d(self.in_features, 32, 2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 2, stride=2)
        self.maxpool = nn.MaxPool2d(2,2)

        #Dense Layers
        self.dense1 = nn.Linear(int(64*4*4), 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense_out = nn.Linear(64, 1)

        self.act_func = activation

        self.layers = [self.dense1, self.dense2, self.dense3]

    def forward(self, x):
        x = self.act_func(self.maxpool(self.conv1(x)))
        x = self.act_func(self.conv2(x))

        x = x.view(x.shape[0], -1)

        for dense in self.layers:
            x = self.act_func(dense(x))
        x = self.dense_out(x)
        return x

# %%
in_features = configuration['T_in'] = 20

classifier = classifier_2D(in_features)

loss_func = torch.nn.BCEWithLogitsLoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# %% 
#Prepping the data. 
X_class = torch.vstack((u_cal_a, u_shift_a)).permute(0,3,1,2)
Y_class = np.vstack((np.expand_dims(np.zeros(n_sims), -1), np.expand_dims(np.ones(n_sims) ,-1)))

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_class, torch.tensor(Y_class, dtype=torch.float32)), batch_size=100, shuffle=True)

# %% 
#Training the classifier. 
epochs = 1000
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = classifier(xx)
        loss = loss_func(y_out, yy)
        loss.backward()
        optimizer.step()

# %%
#Classifier performance. - within the training data itself. 
y_pred = torch.sigmoid(classifier(X_class)).detach().numpy()
y_true = Y_class

for ii in range(len(y_pred)):
    if y_pred[ii] < 0.5:
        y_pred[ii] =0 
    else: 
        y_pred[ii] = 1.0

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

# %% 
# # %%
# y_shift = func(X_shift)
# y_shift_nn = model(torch.tensor(X_shift, dtype=torch.float32)).detach().numpy()

# prediction_sets =  [y_shift_nn - qhat, y_shift_nn + qhat]#Marginal
# # prediction_sets =  [y_shift_nn - qhat*modulation, y_shift_nn + qhat*modulation]#Joint


# # %% 



# def LHS_Sampling():
#     #Simulation Data Built using LHS sampling
#     from pyDOE import lhs
    
#     lb = np.asarray([10, 0.10, 0.10]) #Lambda, a, b 
#     ub = np.asarray([50, 0.50, 0.50]) #Lambda, a, b    
    
#     N = 1000
    
#     param_lhs = lb + (ub-lb)*lhs(3, N)
    
#     list_u = []
    
#     for ii in tqdm(range(N)):
#                 x, y, t, u = wave_solution(param_lhs[ii, 0], param_lhs[ii, 1], param_lhs[ii, 2])
                
#                 list_u.append(u)
        
#     ic = param_lhs
#     u = np.asarray(list_u)
    
#     # np.savez('Spectral_Wave_data_LHS.npz', x=x, y=y,t=t, u=u, ic=ic)



# data_halfspeed =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_halfspeed.npz')
# u_sol_hs = data['u'].astype(np.float32)
# x = data['x'].astype(np.float32)
# y = data['y'].astype(np.float32)
# t = data['t'].astype(np.float32)
# u_hs = torch.from_numpy(u_sol_hs)
# u_hs =  torch.from_numpy(data_halfspeed['u'].astype(np.float32))
# u_hs = u_hs.permute(0, 2, 3, 1)
# xx, yy = np.meshgrid(x,y)

# # %%
# npred = ncal
# #Chunking the data. 
# pred_a = u_hs[:ntrain,:,:,:T_in]
# pred_u = u_hs[:ntrain,:,:,T_in:T+T_in]

# #Normalsingin the prediction inputs and outputs with the same normalizer used for calibration. 
# pred_a = a_normalizer.encode(pred_a)
# pred_u = y_normalizer.encode(pred_u)

# # %% 
# #Getting the prediction
# with torch.no_grad():
#     xx = pred_a

#     for tt in tqdm(range(0, T, step)):
#         pred = model_50(xx)

#         if tt == 0:
#             pred_mean = pred

#         else:
#             pred_mean = torch.cat((pred_mean, pred), -1)       

#         xx = torch.cat((xx[..., step:], pred), dim=-1)

# # %% 

#Using a Classifier
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# #Estimating the KDE over the input space.
# #Initially reducing the dimensionality of the input space using PCA

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# cal_a_flatten = cal_a.reshape(ncal, int(33*33*20))
# pred_a_flatten = pred_a.reshape(npred, int(33*33*20))
# pca = PCA(n_components=5)
# pca.fit(cal_a_flatten.numpy())
# cal_a_pca = pca.transform(cal_a_flatten.numpy())
# pred_a_pca = pca.transform(pred_a_flatten.numpy())

# # %%
# # #Using t-SNE instead but not changing the variable names with pca

# # cal_a_pca = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(cal_a_flatten.numpy())
# # pred_a_pca = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(pred_a_flatten.numpy())


# # %% 
# #Estimating KDE over the reduced dimensions .
# import scipy.stats as stats

# def likelihood_ratio_KDE(x, kde1, kde2):
#     pdf1 = kde1.pdf(x)
#     pdf2 = kde2.pdf(x)
#     return pdf2 / pdf1 

# kde1 = stats.gaussian_kde(cal_a_pca.T)
# kde2 = stats.gaussian_kde(pred_a_pca.T)

# # %%
# def pi_kde(x_new, x_cal):
#     return likelihood_ratio_KDE(x_cal, kde1, kde2) / (np.sum(likelihood_ratio_KDE(x_cal, kde1, kde2)) + likelihood_ratio_KDE(x_new, kde1, kde2))
    
# # weighted_scores = cal_scores * pi_kde(cal_point_a, pred_point_a)

# # %% 
# #Estimating qhat 

# alpha = 0.1
# N = ncal 

# def weighted_quantile(data, alpha, weights=None):
#     ''' percents in units of 1%
#         weights specifies the frequency (count) of data.
#     '''
#     if weights is None:
#         return np.quantile(np.sort(data), alpha, axis = 0, interpolation='higher')
    
#     ind=np.argsort(data)
#     d=data[ind]
#     w=weights[ind]

#     p=1.*w.cumsum()/w.sum()
#     y=np.interp(alpha, p, d)

#     return y

# #Multivariate marginal
# qhat = []
# pi = pi_kde(pred_a_pca.T, cal_a_pca.T)
# cal_scores_flatten = cal_scores.reshape(cal_scores.shape[0], int(cal_scores.shape[1]*cal_scores.shape[2]*cal_scores.shape[3]))

# for ii in tqdm(range(cal_scores_flatten.shape[1])):
#     qhat.append(weighted_quantile(cal_scores_flatten[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi))
# qhat = np.asarray(qhat)
# qhat = qhat.reshape(pred_u.shape[1], pred_u.shape[2], pred_u.shape[3])
# # %%

# prediction_sets =  [pred_mean - qhat, pred_mean + qhat]
# empirical_coverage = ((pred_u.numpy() >= prediction_sets[0].numpy()) & (pred_u.numpy() <= prediction_sets[1].numpy())).mean()

# print(f"The empirical coverage after calibration is: {empirical_coverage}")
# print(f"alpha is: {alpha}")
# print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

# # %%
# def calibrate_res(alpha):
#     qhat = []
#     for ii in tqdm(range(cal_scores_flatten.shape[1])):
#         qhat.append(weighted_quantile(cal_scores_flatten[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi))
#     qhat = np.asarray(qhat)
#     qhat = qhat.reshape(pred_u.shape[1], pred_u.shape[2], pred_u.shape[3])

#     prediction_sets =  [pred_mean - qhat, pred_mean + qhat]
#     empirical_coverage = ((pred_u.numpy() >= prediction_sets[0].numpy()) & (pred_u.numpy() <= prediction_sets[1].numpy())).mean()

#     return empirical_coverage

# alpha_levels = np.arange(0.05, 0.95, 0.1)
# emp_cov_kde = []
# for ii in tqdm(range(len(alpha_levels))):
#     emp_cov_kde.append(calibrate_res(alpha_levels[ii]))

# plt.figure()
# plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
# plt.plot(1-alpha_levels, emp_cov_kde, label='Residual - weighted - PCA - KDE' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0)
# plt.xlabel('1-alpha')
# plt.ylabel('Empirical Coverage')
# plt.legend()

# # %% 
# %%
