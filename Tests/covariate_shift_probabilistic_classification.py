#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base tests for demonstrating CP under covariate shift - Known Distributions and Density Ratio Estimations. 
Multivariate setting for 500-in, 500-out, Density Ratio Estimation using Probability Classification. 
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
input_size = output_size = 500

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
N = 1000 #Datapoints 
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
#Density Ratio Estimation using Probabilistic Classification

#Setting up the classifier.

# Fully Connected Network or a Multi-Layer Perceptron
class classifier_1D(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(classifier_1D, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.conv1 = nn.Conv1d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv1d(32, 64, 3, stride=2)
        self.maxpool1 = nn.MaxPool1d()
        # x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        self.layer1 = nn.Linear()

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp



classifier = MLP(input_size, 1, 5, 256) #Sigmoid at the output is evaluated outside the model definition. 
loss_func = torch.nn.BCEWithLogitsLoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# %%
#Prepping the data. 
X_class = np.vstack((X_calib, X_shift))
Y_class = np.vstack((np.expand_dims(np.zeros(len(X_calib)), -1), np.expand_dims(np.ones(len(X_shift)) ,-1)))

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_class), torch.tensor(Y_class)), batch_size=100, shuffle=True)
# %% 
#Training the classifier. 
epochs = 1000
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = classifier(xx)
        loss = loss_func(xx, yy)
        loss.backward()
        optimizer.step()

# %%
