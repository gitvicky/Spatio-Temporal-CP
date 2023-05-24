#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

U-FNO modelled over the 2D Wave Equation. 
Code inspired from this paper : https://sciencedirect.com/science/article/abs/pii/S0010482519301520?via%3Dihub

Trained Models are utilised for Conformal Prediction over the dataset. 
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
                 "Loss Function": 'MSE',
                 "UQ": 'Dropout', #None, Dropout
                 "Pinball Gamma": 'NA',
                 "Dropout Rate": 0.1
                 }

# %%
from simvue import Run
run = Run()
run.init(folder="/Conformal_Prediction", tags=['Conformal Prediction', 'Wave', 'FNO'], metadata=configuration)

# %% 

import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

import time 
from timeit import default_timer
from tqdm import tqdm 

import platform 
torch.manual_seed(0)
np.random.seed(0)

from utils import *
# %% 
import os 
path = os.getcwd()
# data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# model_loc = os.path.dirname(os.path.dirname(os.getcwd()))
file_loc = os.getcwd()

if platform.processor() == 'x86_64':
    data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))

if platform.processor() == 'arm':
    data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# %%
#################################################
#
# Utilities
#
#################################################

#Defining Quantile Loss
def quantile_loss(pred, label, gamma):
    return torch.where(label > pred, (label-pred)*gamma, (pred-label)*(1-gamma))




# %%

################################################################
# Loading Data 
################################################################

# %%
data =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_5K.npz')


u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

# %% 
ntrain = 1000
ntest = 50
S = 33 #Grid Size

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']

batch_size = configuration['Batch Size']
batch_size2 = batch_size


t1 = default_timer()


T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']
################################################################
# load data
################################################################

train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

test_a = u[-ntest:,:,:,:T_in]
test_u = u[-ntest:,:,:,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)


# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    a_normalizer = MinMax_Normalizer(train_a)
    y_normalizer = MinMax_Normalizer(train_u)

if norm_strategy == 'Range':
    a_normalizer = RangeNormalizer(train_a)
    y_normalizer = RangeNormalizer(train_u)

if norm_strategy == 'Gaussian':
    a_normalizer = GaussianNormalizer(train_a)
    y_normalizer = GaussianNormalizer(train_u)



train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = y_normalizer.encode(train_u)
test_u_encoded = y_normalizer.encode(test_u)

# %%

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# training and evaluation
################################################################

# model = FNO2d(modes, modes, width, T_in, step, x, y)
model = FNO2d_dropout(modes, modes, width, T_in, step, x, y)

model.to(device)

# wandb.watch(model, log='all')
run.update_metadata({'Number of Params': int(model.count_params())})


print("Number of model params : " + str(model.count_params()))

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

# myloss = LpLoss(size_average=False)
myloss = torch.nn.MSELoss()
gamma = configuration['Pinball Gamma']
# myloss = quantile_loss

# gridx = gridx.to(device)
# gridy = gridy.to(device)

# %%
epochs = configuration['Epochs']
if torch.cuda.is_available():
    y_normalizer.cuda()


# %%
#Training Loop
start_time = time.time()
#for ep in tqdm(range(epochs)): #Training Loop - Epochwise
for ep in range(epochs): #Training Loop - Epochwise
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader: #Training Loop - Batchwise
        optimizer.zero_grad()
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step): #Training Loop - Time rollouts. 
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1)) 
            # loss +=  quantile_loss(im.reshape(batch_size, -1), y.reshape(batch_size, -1), gamma=gamma).pow(2).mean()

            #Storing the rolled out outputs. 
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            #Preparing the autoregressive input for the next time step. 
            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        # l2_full = quantile_loss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1), gamma=gamma).pow(2).mean()
        train_l2_full += l2_full.item()

        loss.backward()
        optimizer.step()

#Validation Loop
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                # loss += quantile_loss(im.reshape(batch_size, -1), y.reshape(batch_size, -1), gamma=gamma).pow(2).mean()

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)


            test_l2_step += loss.item()
            test_l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
            # test_l2_full += quantile_loss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1), gamma=gamma).pow(2).mean().item()

    t2 = default_timer()
    scheduler.step()

    train_loss = train_l2_full / ntrain
    test_loss = test_l2_full / ntest

    print('Epochs: %d, Time: %.2f, Train Loss per step: %.3e, Train Loss: %.3e, Test Loss per step: %.3e, Test Loss: %.3e' % (ep, t2 - t1, train_l2_step / ntrain / (T / step), train_loss, test_l2_step / ntest / (T / step), test_loss))

    run.log_metrics({'Train Loss': train_loss, 
                   'Test Loss': test_loss})

train_time = time.time() - start_time


# %%

model_loc = file_loc + '/Models/FNO_Wave_' + run.name + '.pth'
torch.save(model.state_dict(),  model_loc)

# %%

# %%
#Testing 
batch_size = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=1, shuffle=False)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_al, test_ul_encoded), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    #for xx, yy in tqdm(test_loader):
    for xx, yy in test_loader:
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        t1 = default_timer()
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            out = model(xx)
            loss += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            # loss += quantile_loss(out.reshape(batch_size, -1), y.reshape(batch_size, -1), gamma=gamma).pow(2).mean()

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        pred_set[index]=pred
        index += 1
        print(t2-t1, loss)

# %%
#Logging Metrics 
MSE_error = (pred_set - test_u_encoded).pow(2).mean()
MAE_error = torch.abs(pred_set - test_u_encoded).mean()
LP_error = loss / (ntest*T/step)

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))
print('(LP) Testing Error: %.3e' % (LP_error))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(MSE_error),
                     'MAE Test Error': float(MAE_error),
                     'LP Test Error': float(LP_error)
                    })

pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

# %%
#Plotting the comparison plots

idx = np.random.randint(0,ntest) 
idx = 5

# %%
u_field = test_u[idx]

v_min_1 = torch.min(u_field[0,:,:])
v_max_1 = torch.max(u_field[0,:,:])

v_min_2 = torch.min(u_field[int(T/2), :, :])
v_max_2 = torch.max(u_field[int(T/2), :, :])

v_min_3 = torch.min(u_field[-1, :, :])
v_max_3 = torch.max(u_field[-1, :, :])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[0,:,:], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[int(T/2),:,:], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int((T+T_in)/2)))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[-1,:,:], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(T+T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


u_field = pred_set[idx]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[0,:,:], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[int(T/2),:,:], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[-1,:,:], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


output_plot = (file_loc + '/Plots/_FNO_CP_' + run.name + '.png')
plt.savefig(output_plot)


# %%

CODE = ['Wave_FNO.py']
INPUTS = []
OUTPUTS = [model_loc, output_plot]

# Save code files
for code_file in CODE:
    if os.path.isfile(code_file):
        run.save(code_file, 'code')
    elif os.path.isdir(code_file):
        run.save_directory(code_file, 'code', 'text/plain', preserve_path=True)
    else:
        print('ERROR: code file %s does not exist' % code_file)


# Save input files
for input_file in INPUTS:
    if os.path.isfile(input_file):
        run.save(input_file, 'input')
    elif os.path.isdir(input_file):
        run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
    else:
        print('ERROR: input file %s does not exist' % input_file)


# Save output files
for output_file in OUTPUTS:
    if os.path.isfile(output_file):
        run.save(output_file, 'output')
    elif os.path.isdir(output_file):
        run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)   
    else:
        print('ERROR: output file %s does not exist' % output_file)

run.close()
