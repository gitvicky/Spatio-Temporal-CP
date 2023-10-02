#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FNO built using PyTorch to model the 2D Navier-Stokes Equation
Dataset taken from Zongyi's original FNO paper
Training Data : v = 1e-4
Calibration and Prediction Data : v = 1e-3

FNO trained to model the fixed mapping evolution (no AR rollouts) of vorticity
----------------------------------------------------------------------------------------------------------------------------------------

Experimenting with a range of UQ Methods:
    1̶.̶ D̶r̶o̶p̶o̶u̶t̶
    2. Residuals 

Once UQ methodolgies have been demonstrated on each, we can use Conformal Prediction over a
 multitude of conformal scores to find empirically rigorous coverage. 
"""

# %%
configuration = {"Case": 'Turbulent',
                 "Field": 'Vorticity',
                 "viscosity": 1e-4,
                 "Type": '2D Time',
                 "Epochs": 500,
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100 ,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Batch Normalisation": 'No',
                 "T_in": 10,    
                 "T_out": 10,
                 "Step": 10,
                 "Modes":8,
                 "Width": 16,
                 "Variables":1, 
                 "Noise":0.0, 
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
data_loc = path + '/Data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# load data
# _a -- referes to the input 
# _u -- referes ot the output
################################################################
t1 = default_timer()

data_loc = os.getcwd() + '/Data'

# ntrain = 800
# ntest = 200 
field = configuration['Field']

S = 64 #Grid Size
sub = 1 

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']

batch_size = configuration['Batch Size']
batch_size2 = batch_size


T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

# u = torch.tensor(np.load(data_loc + '/NS_v1e-4.npy'))
u = torch.tensor(np.load(data_loc + '/NS_v1e-3.npy'))

# train_a = u[:ntrain,::sub,::sub,:T_in]
# train_u = u[:ntrain,::sub,::sub,T_in:T+T_in]

# test_a =u[-ntest:,::sub,::sub,:T_in]
# test_u = u[-ntest:,::sub,::sub,T_in:T+T_in]

# print(train_u.shape)
# print(test_u.shape)
# %% 

ntrain = 0
ncal = 500
npred = 500

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']


# %%
#Chunking the data. 

# train_a = u[:ntrain,:,:,:T_in]
# train_u = u[:ntrain,:,:,T_in:T+T_in]

cal_a = u[ntrain:ntrain+ncal,:,:,:T_in]
cal_u = u[ntrain:ntrain+ncal,:,:,T_in:T+T_in]

pred_a = u[ntrain+ncal:ntrain+ncal+npred,:,:,:T_in]
pred_u = u[ntrain+ncal:ntrain+ncal+npred,:,:,T_in:T+T_in]


# print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)

t2 = default_timer()
print('Data sorting finished, time used:', t2-t1)

# %% 
#Normalisation. 

#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

a_normalizer = MinMax_Normalizer(-2.4416, 2.4357) #min, max taken from the Laminar data. 
y_normalizer = MinMax_Normalizer(-3.7168, 3.7718)

# train_a = a_normalizer.encode(train_a)
cal_a = a_normalizer.encode(cal_a)
pred_a = a_normalizer.encode(pred_a)

# train_u = y_normalizer.encode(train_u)
cal_u = y_normalizer.encode(cal_u)
pred_u = y_normalizer.encode(pred_u)



# %%
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
#############################################################
# Conformal Prediction Residuals
#############################################################

model_50 = FNO2d(modes, modes, width, T_in, step)
model_50.load_state_dict(torch.load(model_loc + 'FNO_NavierStokes_glum-muntin.pth', map_location='cpu'))



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
prediction_sets_residual = prediction_sets
# %% 
#Estimating the tightness of fit
cov = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1]))
cov_idx = cov.nonzero()

tightness_metric = ((prediction_sets[1][cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]]  - y_response[cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]]) +  (y_response[cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]] - prediction_sets[0][cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]])).mean()
print(f"Tightness of the coverage : Average of the distance between error bars {tightness_metric}")

# %% 
def get_prediction_sets(alpha):
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_mean - qhat, val_mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    print(empirical_coverage)
    return  prediction_sets

alpha_levels = np.arange(0.05, 0.95, 0.1)
coverage_levels = (1 - alpha_levels)
cols = cm.plasma_r(coverage_levels)
pred_sets = [get_prediction_sets(a) for a in alpha_levels] 

idx = 20
tt = 1
x_id = 10

# x_points = pred_a[idx, tt][x_id, :]
x_points = np.arange(S)
x = np.linspace(0, 1, 64)
alpha_levels = np.arange(0.05, 0.95, 0.1)

fig, ax = plt.subplots()
plt.title("Residuals", fontsize=72)
[plt.fill_between(x, pred_sets[i][0][idx, x_id, :, tt], pred_sets[i][1][idx, x_id, :, tt], color = cols[i], alpha=0.7) for i in range(len(alpha_levels))]
fig.colorbar(cm.ScalarMappable(cmap="plasma_r"), ax=ax)
plt.plot(x, y_response[idx, x_id, :, tt], linewidth = 1, color = "black", label = "exact", marker='o', ms=2, mec = 'white')
plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{u}")
plt.legend()
# plt.savefig("wave_unet_residual.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()
# %%
def calibrate_residual(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        xx = cal_a
        for tt in tqdm(range(0, T, step)):
            pred_mean = model_50(xx)

            if tt == 0:
                cal_mean = pred_mean

            else:
                cal_mean = torch.cat((cal_mean, pred_mean), -1)       

            xx = torch.cat((xx[..., step:], pred_mean), dim=-1)

    cal_mean = cal_mean.numpy()

    cal_scores = np.abs(cal_u.numpy()-cal_mean)           
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
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
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
#Plotting and comparing the solution from the FNO and the ground truth 

idx = np.random.randint(0, npred) 

# if configuration['Log Normalisation'] == 'Yes':
#     test_u = torch.exp(test_u)
#     pred_set = torch.exp(pred_set)

u_field = pred_u[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(T/2)])
v_max_2 = torch.max(u_field[:, :, int(T/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm,  vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm,  vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int(T_in + (T_in/2))))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,   vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(T+T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


u_field = val_mean[idx]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[:,:,0], cmap=cm.coolwarm,  vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm,   vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,   vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

# %% 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 0.5
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 10.0
plt.rcParams['ytick.minor.size'] = 10.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '-'


idx = 10
t_idx = -1

x_len = 8
y_len = 8
x_slice = int(y_response.shape[2] / x_len)
y_slice = x_slice

y_response_slice = y_response[idx, ::x_slice, ::x_slice, t_idx]
mean_slice = val_mean[idx, ::x_slice, ::x_slice, t_idx]
# uncalib_lb_slice = prediction_sets_uncalibrated[0][idx, t_idx, ::x_slice, ::x_slice]
# uncalib_ub_slice = prediction_sets_uncalibrated[1][idx, t_idx, ::x_slice, ::x_slice]
calib_lb_slice = prediction_sets[0][idx, ::x_slice, ::x_slice,  t_idx]
calib_ub_slice = prediction_sets[1][idx, ::x_slice, ::x_slice,  t_idx]

# Create a t_len x x_len grid of cells using gridspec
plt.figure()
gs = gridspec.GridSpec(x_len, y_len, wspace=0, hspace=0, width_ratios=list(np.ones((x_len))), height_ratios=list(np.ones((x_len))))

y_max = np.max(calib_ub_slice)
y_min = np.min(calib_lb_slice)

for aa in range(x_len):
    for bb in range(y_len):
        ax = plt.subplot(gs[aa, bb])
        # ax.scatter(x[::x_slice][bb], y_response_slice[aa, bb], color='darkgreen', alpha=0.8, marker='o')
        # ax.errorbar(x[::x_slice][bb], mean_slice[aa, bb].flatten(), yerr=(uncalib_ub_slice[aa, bb] - uncalib_lb_slice[aa, bb]).flatten(), label='Prediction', color='darkgreen', fmt='o', alpha=1.0, ms =5, ecolor='firebrick', elinewidth=0.5) #Uncalibrated
        ax.errorbar(x[::x_slice][bb], mean_slice[aa, bb].flatten(), yerr=(calib_ub_slice[aa, bb] - calib_lb_slice[aa, bb]).flatten(), label='Prediction', color='darkgreen', fmt='o', alpha=1.0, ecolor='firebrick', ms= 5, elinewidth=0.5) #Calibrated 
        ax.set_ylim(bottom=y_min, top=y_max)

        ax.set(xticks=[], yticks=[])

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()


plt.savefig('wave_unet_hs_cells_calibrated.svg', format="svg", bbox_inches='tight', transparent='True')
# %% 

# %%
# ##############################
# # Conformal using Dropout 
# ##############################


class FNO2d_dropout(nn.Module):
    def __init__(self, modes1, modes2, width, dropout_rate=0.1):
        super(FNO2d_dropout, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(T_in+2, self.width)
        # input channel is 12: the solution of the previous T_in timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)


        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = nn.Identity()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, step)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1+x2
        x = F.gelu(x)
        x = self.dropout(x) #Dropout

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1+x2
        x = F.gelu(x)
        x = self.dropout(x) #Dropout

        x1 = self.norm(self.conv2(self.norm(x)))
        x2 = self.w2(x)
        x = x1+x2
        x = F.gelu(x)
        x = self.dropout(x) #Dropout

        x1 = self.norm(self.conv3(self.norm(x)))
        x2 = self.w3(x)
        x = x1+x2
        x = self.dropout(x) #Dropout

        x1 = self.norm(self.conv4(self.norm(x)))
        x2 = self.w4(x)
        x = x1+x2
        x = self.dropout(x) #Dropout

        x1 = self.norm(self.conv5(self.norm(x)))
        x2 = self.w5(x)
        x = x1+x2
        x = self.dropout(x) #Dropout

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# #Using x and y values from the simulation discretisation 
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = gridx = torch.tensor(x_grid, dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(y_grid, dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

# Arbitrary grid discretisation 
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

    
    def enable_dropout(self):
            """Function to enable the dropout layers during test-time"""
            self.dropout.train()
            # for m in self.layers:
            #     if m.__class__.__name__.startswith("Dropout"):
            #         m.train() 


def Dropout_eval(net, x, step, Nrepeat=10):
    net.eval()
    net.enable_dropout()
    preds = torch.zeros(Nrepeat,x.shape[0], x.shape[1], x.shape[2], step)
    for i in range(Nrepeat):    
        preds[i] = net(x)
    return torch.mean(preds, axis=0), torch.std(preds, axis=0)

model_dropout = FNO2d_dropout(modes, modes, width)
model_dropout.load_state_dict(torch.load(model_loc + 'FNO_NavierStokes_exothermic-em.pth', map_location='cpu'))


# %%
#Performing the Calibration for Dropout

t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval(model_dropout, xx, step)

        if tt == 0:
            cal_mean = mean
            cal_std = std
        else:
            cal_mean = torch.cat((cal_mean, mean), -1)       
            cal_std = torch.cat((cal_std, std), -1)       

        xx = torch.cat((xx[..., step:], mean), dim=-1)


# cal_mean = cal_mean.numpy()

cal_upper = cal_mean + cal_std
cal_lower = cal_mean - cal_std

cal_scores = np.maximum(cal_u.numpy()-cal_upper.numpy(), cal_lower.numpy()-cal_u.numpy())
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

# %% 
#Obtaining the Prediction Sets
with torch.no_grad():
    xx = pred_a

    for tt in tqdm(range(0, T, step)):
        mean, std = Dropout_eval(model_dropout, xx, step)

        if tt == 0:
            val_mean = mean
            val_std = std
        else:
            val_mean = torch.cat((val_mean, mean), -1)       
            val_std = torch.cat((val_std, std), -1)       

        xx = torch.cat((xx[..., step:], mean), dim=-1)

val_upper = val_mean + val_std
val_lower = val_mean - val_std

val_lower = val_lower.numpy()
val_upper = val_upper.numpy()

prediction_sets_uncalibrated = [val_lower, val_upper]
prediction_sets = [val_lower - qhat, val_upper + qhat]

    # %% 
y_response = pred_u.numpy()

print('Conformal by way Dropout')
# Calculate empirical coverage (before and after calibration)
prediction_sets_uncalibrated = [val_lower, val_upper]
empirical_coverage_uncalibrated = ((y_response >= prediction_sets_uncalibrated[0]) & (y_response <= prediction_sets_uncalibrated[1])).mean()
print(f"The empirical coverage before calibration is: {empirical_coverage_uncalibrated}")
empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
t2 = default_timer()
print('Conformal using Dropout, time used:', t2-t1)

prediction_sets_dropout = prediction_sets
# %% 
#Estimating the tightness of fit
cov = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1]))
cov_idx = cov.nonzero()

tightness_metric = ((prediction_sets[1][cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]]  - y_response[cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]]) +  (y_response[cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]] - prediction_sets[0][cov_idx[0], cov_idx[1], cov_idx[2], cov_idx[3]])).mean()
print(f"Tightness of the coverage : Average of the distance between error bars {tightness_metric}")

# %% 
def calibrate_dropout(alpha):
    with torch.no_grad():
        xx = cal_a

        for tt in tqdm(range(0, T, step)):
            mean, std = Dropout_eval(model_dropout, xx, step)

            if tt == 0:
                cal_mean = mean
                cal_std = std
            else:
                cal_mean = torch.cat((cal_mean, mean), -1)       
                cal_std = torch.cat((cal_std, std), -1)       

            xx = torch.cat((xx[..., step:], mean), dim=-1)


    # cal_mean = cal_mean.numpy()

    cal_upper = cal_mean + cal_std
    cal_lower = cal_mean - cal_std

    cal_scores = np.maximum(cal_u.numpy()-cal_upper.numpy(), cal_lower.numpy()-cal_u.numpy())
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

    prediction_sets = [val_lower - qhat, val_upper + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0]) & (y_response <= prediction_sets[1])).mean()
    return empirical_coverage

# def calibrate_dropout(alpha):
#     with torch.no_grad():
#         xx = cal_a

#         for tt in tqdm(range(0, T, step)):
#             mean, std = Dropout_eval(model_dropout, xx, step)

#             if tt == 0:
#                 cal_mean = mean
#                 cal_std = std
#             else:
#                 cal_mean = torch.cat((cal_mean, mean), -1)       
#                 cal_std = torch.cat((cal_std, std), -1)       

#             xx = torch.cat((xx[..., step:], mean), dim=-1)


#     # cal_mean = cal_mean.numpy()

#     cal_upper = cal_mean + cal_std
#     cal_lower = cal_mean - cal_std

#     cal_scores = np.maximum(cal_u.numpy()-cal_upper.numpy(), cal_lower.numpy()-cal_u.numpy())
#     qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, method='higher')

#     prediction_sets = [val_mean - qhat, val_mean + qhat]
#     empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()

#     return empirical_coverage

# %%
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_dropout = []

for ii in tqdm(range(len(alpha_levels))):
    emp_cov_dropout.append(calibrate_dropout(alpha_levels[ii]))

# %% 

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_cqr, label='CQR', color='maroon', ls='--',  alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.8, linewidth=3.0)
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

mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 10.0
plt.rcParams['ytick.minor.size'] = 10.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'

plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.75)
plt.plot(1-alpha_levels, emp_cov_dropout, label='Dropout',  color='navy', ls='dotted',  alpha=0.75)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title("Navier-Stokes", fontsize=72)
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("NS_coverage.svg", format="svg", bbox_inches='tight')
plt.show()

# %%
#Slice plots along the x axis. 
idx = 12 
t_val = -1
y_pos = 0
Y_pred_viz = y_response[idx,:, y_pos, t_val]
mean_viz = mean[idx,:, y_pos, t_val]
pred_set_0_viz = prediction_sets_residual[0][idx,:, y_pos, t_val]
pred_set_1_viz = prediction_sets_residual[1][idx,:, y_pos, t_val]

plt.figure()
# plt.title(f"Residuals, alpha = {alpha}")
plt.title(rf"Residuals, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x, Y_pred_viz, label='Exact', color='black', alpha = 0.7)
plt.plot(x, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x, pred_set_0_viz, label='lower-cal', color='teal', alpha = 0.7)
plt.plot(x, pred_set_1_viz, label='upper-cal', color='navy', alpha = 0.7)
plt.xlabel(r"\textbf{x}")
plt.ylabel(r"\textbf{$\nu$}")
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("NS_residual_x.svg", format="svg", bbox_inches='tight')

# %%
#Slice plots along the x axis. 
idx = 12 
t_val = -1
y_pos = 0
Y_pred_viz = y_response[idx,:, y_pos, t_val]
mean_viz = mean[idx,:, y_pos, t_val]
pred_set_0_viz = prediction_sets_dropout[0][idx,:, y_pos, t_val]
pred_set_1_viz = prediction_sets_dropout[1][idx,:, y_pos, t_val]

plt.figure()
# plt.title(f"Residuals, alpha = {alpha}")
plt.title(rf"Dropout, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x, Y_pred_viz, label='Exact', color='black', alpha = 0.7)
plt.plot(x, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x, pred_set_0_viz, label='lower-cal', color='teal', alpha = 0.7)
plt.plot(x, pred_set_1_viz, label='upper-cal', color='navy', alpha = 0.7)

plt.xlabel(r"\textbf{x}")
plt.ylabel(r"\textbf{$\nu$}")
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("NS_dropout_x.svg", format="svg", bbox_inches='tight')

# %% 

#Slice plots along the y axis. 
idx = 12 
t_val = -1
y_pos = 32
Y_pred_viz = y_response[idx,:, y_pos, t_val]
mean_viz = mean[idx,:, y_pos, t_val]
pred_set_0_viz = prediction_sets_residual[0][idx,:, y_pos, t_val]
pred_set_1_viz = prediction_sets_residual[1][idx,:, y_pos, t_val]

plt.figure()
# plt.title(f"Residuals, alpha = {alpha}")
plt.title(rf"Residuals, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x, Y_pred_viz, label='Exact', color='black', alpha = 0.7)
plt.plot(x, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x, pred_set_0_viz, label='lower-cal', color='teal', alpha = 0.7)
plt.plot(x, pred_set_1_viz, label='upper-cal', color='navy', alpha = 0.7)
plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{$\nu$}")
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("NS_residual_y.svg", format="svg", bbox_inches='tight')

# %%
#Slice plots along the y axis. 
idx = 12 
t_val = -1
x_pos = 32
Y_pred_viz = y_response[idx,x_pos, :, t_val]
mean_viz = mean[idx,x_pos, :, t_val]
pred_set_0_viz = prediction_sets_dropout[0][idx,x_pos, :, t_val]
pred_set_1_viz = prediction_sets_dropout[1][idx,x_pos, :, t_val]

plt.figure()
# plt.title(f"Residuals, alpha = {alpha}")
plt.title(rf"Dropout, $\alpha$ = {alpha}", fontsize=72)
plt.plot(x, Y_pred_viz, label='Exact', color='black', alpha = 0.7)
plt.plot(x, mean_viz, label='Mean', color='firebrick', alpha = 0.7)
plt.plot(x, pred_set_0_viz, label='lower-cal', color='teal', alpha = 0.7)
plt.plot(x, pred_set_1_viz, label='upper-cal', color='navy', alpha = 0.7)

plt.xlabel(r"\textbf{y}")
plt.ylabel(r"\textbf{$\nu$}")
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("NS_dropout_y.svg", format="svg", bbox_inches='tight')


# %%
