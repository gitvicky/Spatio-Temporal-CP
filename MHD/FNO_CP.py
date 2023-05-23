#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
FNO modelled over the MHD data built using JOREK for multi-blob diffusion. Conformal Prediction over it
"""
# %%
#Training Conditions
################################################################

configuration = {"Case": 'Multi-Blobs',
                 "Field": 'rho, Phi, T',
                 "Field_Mixing": 'Channel',
                 "Type": '2D Time',
                 "Epochs": 500,
                 "Batch Size": 10,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No',
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 10,    
                 "T_out": 40,
                 "Step": 5,
                 "Modes":16,
                 "Width_time":32, #FNO
                 "Width_vars": 0, #U-Net
                 "Variables":3, 
                 "Noise":0.0, 
                 "Loss Function": 'LP Loss',
                 "Spatial Resolution": 1,
                 "Temporal Resolution": 1,
                 }


# %%
#Importing the necessary packages. 
################################################################
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib import cm 
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

import time 
from timeit import default_timer
from tqdm import tqdm 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict


torch.manual_seed(0)
np.random.seed(0)

# %% 
#Setting up the directories - data location, model location and plots. 
################################################################
import os 
path = os.getcwd()
model_loc = path + '/Models/'
data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# %%
#Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# Loading Data 
################################################################

# %%
data = data_loc + '/Data/MHD_multi_blobs.npz'

# %%
field = configuration['Field']
dims = ['rho', 'Phi', 'T']
num_vars = configuration['Variables']

u_sol = np.load(data)['rho'].astype(np.float32)  / 1e20
v_sol = np.load(data)['Phi'].astype(np.float32)  / 1e5
p_sol = np.load(data)['T'].astype(np.float32)    / 1e6

u_sol = np.nan_to_num(u_sol)
v_sol = np.nan_to_num(v_sol)
p_sol = np.nan_to_num(p_sol)

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

v = torch.from_numpy(v_sol)
v = v.permute(0, 2, 3, 1)

p = torch.from_numpy(p_sol)
p = p.permute(0, 2, 3, 1)

t_res = configuration['Temporal Resolution']
x_res = configuration['Spatial Resolution']
uvp = torch.stack((u,v,p), dim=1)[:,::t_res]


x_grid = np.load(data)['Rgrid'][0,:].astype(np.float32)
y_grid = np.load(data)['Zgrid'][:,0].astype(np.float32)
t_grid = np.load(data)['time'].astype(np.float32)

# %% 
#Extracting hyperparameters from the config dict
################################################################

S = 106 #Grid Size 

modes = configuration['Modes']
width_time = configuration['Width_time']
width_vars = configuration['Width_vars']
output_size = configuration['Step']
batch_size = configuration['Batch Size']
T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

t1 = default_timer()

#At this stage the data needs to be [Batch_Size, X, Y, T]
ntrain = 100
ncal = 100
npred = 78

# %%
#Chunking the data. 
################################################################

train_a = uvp[:ntrain,:,:,:,:T_in]
train_u = uvp[:ntrain,:,:,:,T_in:T+T_in]

cal_a = uvp[ntrain:ntrain+ncal,:,:,:,:T_in]
cal_u = uvp[ntrain:ntrain+ncal,:,:,:,T_in:T+T_in]

pred_a = uvp[ntrain+ncal:ntrain+ncal+npred,:,:,:,:T_in]
pred_u = uvp[ntrain+ncal:ntrain+ncal+npred,:,:,:,T_in:T+T_in]

print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)


# %%

#Normalising the train and test datasets with the preferred normalisation. 
################################################################

#normalization, rangewise but single value. 
class MinMax_Normalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        min_u = torch.min(x[:,0,:,:,:])
        max_u = torch.max(x[:,0,:,:,:])

        self.a_u = (high - low)/(max_u - min_u)
        self.b_u = -self.a_u*max_u + high

        min_v = torch.min(x[:,1,:,:,:])
        max_v = torch.max(x[:,1,:,:,:])

        self.a_v = (high - low)/(max_v - min_v)
        self.b_v = -self.a_v*max_v + high

        min_p = torch.min(x[:,2,:,:,:])
        max_p = torch.max(x[:,2,:,:,:])

        self.a_p = (high - low)/(max_p - min_p)
        self.b_p = -self.a_p*max_p + high
        

    def encode(self, x):
        s = x.size()

        u = x[:,0,:,:,:]
        u = self.a_u*u + self.b_u

        v = x[:,1,:,:,:]
        v = self.a_v*v + self.b_v

        p = x[:,2,:,:,:]
        p = self.a_p*p + self.b_p
        
        x = torch.stack((u,v,p), dim=1)

        return x

    def decode(self, x):
        s = x.size()

        u = x[:,0,:,:,:]
        u = (u - self.b_u)/self.a_u
        
        v = x[:,1,:,:,:]
        v = (v - self.b_v)/self.a_v

        p = x[:,2,:,:,:]
        p = (p - self.b_p)/self.a_p


        x = torch.stack((u,v,p), dim=1)

        return x

    def cuda(self):
        self.a_u = self.a_u.cuda()
        self.b_u = self.b_u.cuda()
        
        self.a_v = self.a_v.cuda()
        self.b_v = self.b_v.cuda() 

        self.a_p = self.a_p.cuda()
        self.b_p = self.b_p.cuda()


    def cpu(self):
        self.a_u = self.a_u.cpu()
        self.b_u = self.b_u.cpu()
        
        self.a_v = self.a_v.cpu()
        self.b_v = self.b_v.cpu()

        self.a_p = self.a_p.cpu()
        self.b_p = self.b_p.cpu()


norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    a_normalizer = MinMax_Normalizer(train_a)
    y_normalizer = MinMax_Normalizer(train_u)

# if norm_strategy == 'Range':
#     a_normalizer = RangeNormalizer(train_a)
#     y_normalizer = RangeNormalizer(train_u)

# if norm_strategy == 'Min-Max':
#     a_normalizer = GaussianNormalizer(train_a)
#     y_normalizer = GaussianNormalizer(train_u)


train_a = a_normalizer.encode(train_a)
cal_a = a_normalizer.encode(cal_a)
pred_a = a_normalizer.encode(pred_a)

train_u = y_normalizer.encode(train_u)
cal_u = y_normalizer.encode(cal_u)
pred_u = y_normalizer.encode(pred_u)


t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, num_vars, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, num_vars, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivxy,iovxy->bovxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes 
        out_ft = torch.zeros(batchsize, self.out_channels, num_vars,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w = nn.Conv3d(self.width, self.width, 1)
    
    
    def forward(self, x):

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1+x2
        x = F.gelu(x)
        return x 



class FNO_multi(nn.Module):
    def __init__(self, modes1, modes2, width_vars, width_time):
        super(FNO_multi, self).__init__()

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
        self.width_vars = width_vars
        self.width_time = width_time

        self.fc0_time  = nn.Linear(T_in+2, self.width_time)

        # self.padding = 8 # pad the domain if input is non-periodic

        self.f0 = FNO2d(self.modes1, self.modes2, self.width_time)
        self.f1 = FNO2d(self.modes1, self.modes2, self.width_time)
        self.f2 = FNO2d(self.modes1, self.modes2, self.width_time)
        self.f3 = FNO2d(self.modes1, self.modes2, self.width_time)
        self.f4 = FNO2d(self.modes1, self.modes2, self.width_time)
        self.f5 = FNO2d(self.modes1, self.modes2, self.width_time)

        # self.dropout = nn.Dropout(p=0.1)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = nn.Identity()


        self.fc1_time = nn.Linear(self.width_time, 128)
        self.fc2_time = nn.Linear(128, step)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)


        x = self.fc0_time(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = self.dropout(x)

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x0 = self.f0(x)
        x = self.f1(x0)
        x = self.f2(x) + x0 
        # x = self.dropout(x)
        x1 = self.f3(x)
        x = self.f4(x1)
        x = self.f5(x) + x1 
        # x = self.dropout(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic

        x = x.permute(0, 2, 3, 4, 1)
        x = x 

        x = self.fc1_time(x)
        x = F.gelu(x)
        # x = self.dropout(x)
        x = self.fc2_time(x)
        
        return x

#Using x and y values from the simulation discretisation 
    def get_grid(self, shape, device):
        batchsize, num_vars, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridx = gridx = torch.tensor(x_grid, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, num_vars, 1, size_y, 1])
        gridy = torch.tensor(y_grid, dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, num_vars, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

## Arbitrary grid discretisation 
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1).to(device)


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# Loading the trained model
################################################################
#Instantiating the Model. 

model = FNO_multi(modes, modes, width_vars, width_time)
model.load_state_dict(torch.load(model_loc + '/FNO_multi_blobs_claret-inventory.pth', map_location='cpu'))
model.to(device)
print("Number of model params : " + str(model.count_params()))

if torch.cuda.is_available():
    y_normalizer.cuda()


# %%
#Performing the Calibration usign Residuals: https://www.stat.cmu.edu/~larry/=sml/Conformal
################################################################

t1 = default_timer()

n = ncal
alpha = 0.1 #Coverage will be 1- alpha 

with torch.no_grad():
    xx = cal_a
    for tt in tqdm(range(0, T, step)):
        out = model(xx)
        if tt == 0:
            pred = out
        else:
            pred = torch.cat((pred, out), -1)       

        xx = torch.cat((xx[..., step:], out), dim=-1)

cal_mean = pred

# cal_u = cal_u.numpy()
cal_scores = np.abs(cal_u-cal_mean)           
qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

# %% 
#Obtaining the Prediction Sets
y_response = pred_u.numpy()
stacked_x = torch.FloatTensor(pred_a)

with torch.no_grad():
    xx = pred_a
    for tt in tqdm(range(0, T, step)):
        out = model(xx)
        if tt == 0:
            pred = out
        else:
            pred = torch.cat((pred, out), -1)       

        xx = torch.cat((xx[..., step:], out), dim=-1)
    mean = pred

prediction_sets =  [mean - qhat, mean + qhat]


# %%
print('Conformal by way of Residual')
# Calculate empirical coverage (before and after calibration)
empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()
print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")

t2 = default_timer()
print('Conformal by Residual, time used:', t2-t1)

# %% 
soln_vals = y_normalizer.decode(torch.Tensor(y_response)) * 1e20 
mean_vals = y_normalizer.decode(torch.Tensor(mean)) * 1e20 
lower_vals = y_normalizer.decode(torch.Tensor(prediction_sets[0])) * 1e20 
upper_vals = y_normalizer.decode(torch.Tensor(prediction_sets[1])) * 1e20 

# %%
idx = 10
var = 0 
time = 0
levels = 2
X, Y = np.meshgrid(x_grid, y_grid)
plt.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()
mean_plot = ax.contour(X, Y, mean_vals[idx, var, :, :, time], levels, colors='k')  
lower_plot = ax.contour(X, Y, lower_vals[idx, var, :, :, time], levels, colors='red') 
upper_plot = ax.contour(X, Y, upper_vals[idx, var, :, :, time], levels, colors='blue') 

ax.set_title('Single color - negative contours solid')

# %% 
plt.imshow(mean_vals[idx, 0, :, :, 0])
plt.colorbar()
# %%
def calibrate(alpha):
    n = ncal
    y_response = pred_u.numpy()

    with torch.no_grad():
        xx = cal_a
        for tt in tqdm(range(0, T, step)):
            out = model(xx)
            if tt == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       

            xx = torch.cat((xx[..., step:], out), dim=-1)
        cal_mean = pred.numpy()
        
    cal_scores = np.abs(cal_u-cal_mean)     
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, axis = 0, interpolation='higher')

    prediction_sets =  [mean - qhat, mean + qhat]
    empirical_coverage = ((y_response >= prediction_sets[0].numpy()) & (y_response <= prediction_sets[1].numpy())).mean()
    return empirical_coverage


alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate(alpha_levels[ii]))

# %% 
import matplotlib as mpl
mpl.rcParams['figure.figsize']=(16,16)
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75)
plt.plot(1-alpha_levels, emp_cov, label='Residual' ,ls='-.', color='teal', alpha=0.75)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
plt.title("MHD", fontsize=72)
plt.legend()
plt.grid() #Comment out if you dont want grids.
plt.savefig("MHD_comparison.svg", format="svg", bbox_inches='tight')
plt.show()
# mpl.rcParams['xtick.minor.visible']=True
# mpl.rcParams['font.size']=45
# mpl.rcParams['figure.figsize']=(16,16)
# mpl.rcParams['xtick.minor.visible']=True
# mpl.rcParams['axes.linewidth']= 3
# mpl.rcParams['axes.titlepad'] = 20
# plt.rcParams['xtick.major.size'] =15
# plt.rcParams['ytick.major.size'] =15
# plt.rcParams['xtick.minor.size'] =10
# plt.rcParams['ytick.minor.size'] =10
# plt.rcParams['xtick.major.width'] =5
# plt.rcParams['ytick.major.width'] =5
# plt.rcParams['xtick.minor.width'] =5
# plt.rcParams['ytick.minor.width'] =5
# mpl.rcParams['axes.titlepad'] = 20

plt.savefig('CP_coverage.png')
# %% 
idx = 10
x_pos = 70
time = 10
var = 0 
plt.figure()
plt.plot(y_grid,mean_vals[idx, var, x_pos, :, time], label='Prediction', alpha=0.8,  color = 'firebrick')
plt.plot(y_grid,lower_vals[idx, var, x_pos, :, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
plt.plot(y_grid, upper_vals[idx, var, x_pos, :, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
plt.plot(y_grid, soln_vals[idx, var, x_pos, :, time], label='Solution', alpha=0.8,  color = 'black')
plt.legend()
plt.xlabel(r'\textbf{$Z$}')
plt.ylabel(r'\textbf{$\rho$}')
plt.grid() #Comment out if you dont want grids.

plt.savefig("rho_Z.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()
# %%
idx = 10
y_pos = 20
time = 10
var = 0 
plt.figure()
plt.plot(x_grid, mean_vals[idx, var, :, y_pos, time], label='Prediction', alpha=0.8,  color = 'firebrick')
plt.plot(x_grid,lower_vals[idx, var, :, y_pos, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
plt.plot(x_grid, upper_vals[idx, var,:, y_pos, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
plt.plot(x_grid, soln_vals[idx, var,:, y_pos, time], label='Solution', alpha=0.8,  color = 'black')
plt.legend()
plt.xlabel(r'\textbf{$R$}')
plt.ylabel(r'\textbf{$\rho$}')
plt.grid() #Comment out if you dont want grids.

plt.savefig("rho_R.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()

plt.figure()
plt.imshow(soln_vals[idx, var, :, :, time], cmap=cm.coolwarm, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
plt.xlabel('R-Axis')
plt.ylabel('Z-Axis')
plt.title('Plasma Density at t=10')
plt.plot(x_grid, np.ones(106)*y_grid[y_pos], linewidth = 4, color='black')
# %% 
var = 1
plt.figure()
plt.plot(y_grid,mean_vals[idx, var, x_pos, :, time], label='Prediction', alpha=0.8,  color = 'firebrick')
plt.plot(y_grid,lower_vals[idx, var, x_pos, :, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
plt.plot(y_grid, upper_vals[idx, var, x_pos, :, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
plt.plot(y_grid, soln_vals[idx, var, x_pos, :, time], label='Solution', alpha=0.8,  color = 'black')
plt.legend()
plt.xlabel(r'\textbf{$Z$}')
plt.ylabel(r'\textbf{$\Phi$}')
plt.grid() #Comment out if you dont want grids.

plt.savefig("phi_Z.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()


plt.figure()
plt.plot(x_grid,mean_vals[idx, var, :, y_pos, time], label='Prediction', alpha=0.8,  color = 'firebrick')
plt.plot(x_grid,lower_vals[idx, var, :, y_pos, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
plt.plot(x_grid, upper_vals[idx, var,:, y_pos, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
plt.plot(x_grid, soln_vals[idx, var,:, y_pos, time], label='Solution', alpha=0.8,  color = 'black')
plt.legend()
plt.xlabel(r'\textbf{$R$}')
plt.ylabel(r'\textbf{$\Phi$}')
plt.grid() #Comment out if you dont want grids.

plt.savefig("phi_R.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()


# %% 

var = 2
plt.figure()
plt.plot(y_grid,mean_vals[idx, var, x_pos, :, time], label='Prediction', alpha=0.8,  color = 'firebrick')
plt.plot(y_grid,lower_vals[idx, var, x_pos, :, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
plt.plot(y_grid, upper_vals[idx, var, x_pos, :, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
plt.plot(y_grid, soln_vals[idx, var, x_pos, :, time], label='Solution', alpha=0.8,  color = 'black')
plt.legend()
plt.xlabel(r'\textbf{$Z$}')
plt.ylabel(r'\textbf{$T$}')
plt.grid() #Comment out if you dont want grids.

plt.savefig("T_Z.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()


plt.figure()
plt.plot(x_grid,mean_vals[idx, var, :, y_pos, time], label='Prediction', alpha=0.8,  color = 'firebrick')
plt.plot(x_grid,lower_vals[idx, var, :, y_pos, time], label='Lower', alpha=0.8,  color = 'teal', ls='--')
plt.plot(x_grid, upper_vals[idx, var,:, y_pos, time], label='Upper', alpha=0.8,  color = 'navy', ls='--')
plt.plot(x_grid, soln_vals[idx, var,:, y_pos, time], label='Solution', alpha=0.8,  color = 'black')
plt.legend()
plt.xlabel(r'\textbf{$R$}')
plt.ylabel(r'\textbf{$T$}')
plt.grid() #Comment out if you dont want grids.

plt.savefig("T_R.svg", format="svg", bbox_inches='tight', transparent='True')
plt.show()



# %%
# %% 
idx = 10
var = 0 
time = 0
levels = 2
X, Y = np.meshgrid(x_grid, y_grid)
plt.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()
mean_plot = ax.contour(X, Y, mean_vals[idx, var, :, :, time], levels, colors='k')  
lower_plot = ax.contour(X, Y, lower_vals[idx, var, :, :, time], levels, colors='red') 
upper_plot = ax.contour(X, Y, upper_vals[idx, var, :, :, time], levels, colors='blue') 



# %% 


# # %% 
# #3D Plotly contour plots 
# import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = "notebook_connected"

# idx = 25
# var = 0
# time = 24

# fig = go.Figure(data=[
#     go.Surface(x = x_grid, y = y_grid, z= mean_vals[idx, var, :, :, time], opacity=0.9, colorscale='tealrose'),
#     go.Surface(x = x_grid, y = y_grid, z=lower_vals[idx, var, :, :, time], colorscale = 'turbid', showscale=False, opacity=0.6),
#     go.Surface(x = x_grid, y = y_grid, z=upper_vals[idx, var, :, :, time], colorscale = 'Electric',showscale=False, opacity=0.3)

# ])
# fig.update_traces(showscale=False)

# # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
# #                                   highlightcolor="limegreen", project_z=True))
# fig.show()

# %%


# plt.plot(np.ones(106)*x_grid[x_pos], y_grid, linewidth = 4, color='black')
# %%

# %%
