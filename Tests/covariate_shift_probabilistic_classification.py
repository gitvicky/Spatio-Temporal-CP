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
from scipy.special import logsumexp

from utils import * 

torch.set_default_dtype(torch.float32)

N_viz = 1000 #Datapoints a
input_size = 5
output_size = 2

# %% 
def func(x):
    # return np.array([np.expand_dims(np.sin(2*np.sum(x, axis = -1)), -1) for i in range(output_size)])
    return np.array([np.sin(2*np.sum(x, axis = -1)) for i in range(output_size)]).T

#Sampling from a normal distribution
def normal_dist(mean, std, N):
    dist = stats.norm(mean, std)
    return dist.rvs((N, input_size))


mean_1, std_1 = np.pi/2, np.pi/4
# mean_2, std_2 = np.pi/4, np.pi/8
mean_2, std_2 = np.pi/2, np.pi/8        # Make the problem easier by just changing var

x = normal_dist(mean_1, std_1, N_viz)
x_shift = normal_dist(mean_2, std_2, N_viz) #Covariate shifted

# %% 
#Visualising the covariate shift
plt.figure()
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

model = MLP(input_size, output_size, 2, 10)
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
# viz_N = input_size
# x_viz = torch.tensor(np.linspace(np.pi/6, np.pi/2, viz_N), dtype=torch.float32)
# x_viz = x_viz.repeat(1000, 1)
# y_viz= func(x_viz)
# y_mean = model(x_viz)


# plt.plot(y_train[-1], label='Actual')
# plt.plot(y_out[-1].detach().numpy(), label='Pred')
# plt.legend()
# plt.title("Visualising the Model Performance")


# %%
#Obtaining the Calibration Scores.  
###
# Ander: I've modified below this line
###

N_cal = 1001 #Datapoints 
N_shift = 1000

X_calib = normal_dist(mean_1, std_1, N_cal)
X_shift = normal_dist(mean_2, std_2, N_shift)#Covariate shifted

y_calib = func(X_calib)

y_calib_nn = model(torch.tensor(X_calib, dtype=torch.float32)).detach().numpy()

#Performing the calibration
cal_scores = np.abs(y_calib - y_calib_nn) #Marginal

# modulation =  np.std(y_calib - y_calib_nn, axis = 0)#Joint
# cal_scores = np.max(np.abs((y_calib - y_calib_nn)/modulation),  axis = (1))#Joint

# %% 
#Using the Known PDFs
def likelihood_ratio(x):
    pdf1 = np.prod(stats.norm.pdf(x, mean_1, std_1))
    pdf2 = np.prod(stats.norm.pdf(x, mean_2, std_2))
    return (pdf2 / pdf1)

def loglikelihood_ratio(x):
    pdf1 = np.sum(stats.norm.logpdf(x, mean_1, std_1))
    pdf2 = np.sum(stats.norm.logpdf(x, mean_2, std_2))
    return pdf2 - pdf1

# def likelihood_ratio(x):
#     return stats.norm.pdf(x, mean_2, std_2)/stats.norm.pdf(x, mean_1, std_1)

# These values are the same for every pi_log evaluation
log_like_ratio_xcal = np.array([loglikelihood_ratio(X_calib[i,:]) for i in range(N_cal)])
log_like_ratio_xcal_sum = logsumexp(log_like_ratio_xcal)

def pi_log(x_new):
    return log_like_ratio_xcal - (logsumexp([loglikelihood_ratio(x_new), log_like_ratio_xcal_sum]))

def pi_log_inf(x_new):
    return loglikelihood_ratio(x_new) - (logsumexp([loglikelihood_ratio(x_new), log_like_ratio_xcal_sum]))

def pi(x_new, x_cal):
    return likelihood_ratio(x_cal) / (np.sum(likelihood_ratio(x_cal)) + likelihood_ratio(x_new))

def pi_nplus1(x_new, x_cal):
    return likelihood_ratio(x_new) / (np.sum(likelihood_ratio(x_cal)) + likelihood_ratio(x_new))


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

#Inspired from https://www.pnas.org/doi/abs/10.1073/pnas.2204569119
def get_weighted_quantile(scores, quantile, weights):
    
    if weights.ndim == 1:
        weights = weights[:, None]
        scores = scores[:, None]

    #Normalise weights
    p_weights = weights / np.sum(weights, axis=0)

    #Sort the scores and the weights 
    args_sorted_scores = np.argsort(scores, axis=0)
    sortedscores= np.take_along_axis(scores, args_sorted_scores, axis=0)
    sortedp_weights = np.take_along_axis(p_weights, args_sorted_scores, axis=0)

    # locate quantiles of weighted scores per y
    cdf_pweights = np.cumsum(sortedp_weights, axis=0)
    qidx_y = np.sum(cdf_pweights < quantile, axis=0)  # equivalent to [np.searchsorted(cdf_n1, q) for cdf_n1 in cdf_n1xy]
    q_y = sortedscores[(qidx_y, range(qidx_y.size))]
    return q_y
# %% 
#Multivariate marginal
pi_vals = pi(X_shift, X_calib) # Our Implementation
pi_vals = pi(X_shift, X_calib) +  pi_nplus1(X_shift, X_calib) #Including the n+1 as well 

## Using weighted log samples
import pandas as pd
import seaborn as sns 

## Get true scores for testing
Y_shift_true = func(X_shift)
y_shift_nn = model(torch.tensor(X_shift, dtype=torch.float32)).detach().numpy()
true_shift_scores = np.abs(Y_shift_true - y_shift_nn) #Marginal

## Test for input sample index and output dimension
test_index = 100
test_dimension = 0

# likes_vals = np.array([likelihood_ratio(X_shift[i,:]) for i in range(N_shift)] )
# log_likes_vals = np.array([loglikelihood_ratio(X_shift[i,:]) for i in range(N_shift)] )

pi_logs = np.array([pi_log(X_shift[i, :]) for i in range(N_shift)])
pi_log_infs = np.array([pi_log_inf(X_shift[i, :]) for i in range(N_shift)])

pi_logs_1 = pi_logs[test_index, :]
pi_logs_1 = np.append(pi_logs_1, pi_log_infs[test_index])

log_shift = 10
pi_logs_1_adjust = np.exp(pi_logs_1 + log_shift)
ws = pi_logs_1_adjust / np.sum(pi_logs_1_adjust)

ESS = np.sum(pi_logs_1_adjust) **2 / np.sum(pi_logs_1_adjust**2)    ## Effective sample size

this_dim_true = true_shift_scores[:, test_dimension]
this_dim_shift = cal_scores[:, test_dimension]
this_dim_shift = np.append(this_dim_shift, np.inf)

plt.figure()
plt.step(np.sort(this_dim_true), np.linspace(0, 1, N_shift+1)[:-1], label="true non-conformity scores")
df = pd.DataFrame(np.vstack((this_dim_shift, ws)).T, columns = ['sample', 'weights'])

sns.ecdfplot(data = df, x = 'sample', weights = 'weights', stat = 'proportion', legend = True,  label="predicted scores")
plt.xlabel("s(x,y)", fontsize = 18)
plt.ylabel("cdf", fontsize = 18)
plt.title(f"Effective SS: {ESS}")
plt.legend()
plt.show()

# %% 
#   Check empirical coverage of shifted distribution for one alpha
#
# Add infinity to pi_logs and cal_scores
alpha = 0.1

pi_log_with_inf = np.zeros((pi_logs.shape[0], pi_logs.shape[1] + 1))
pi_log_with_inf[:, :-1] = pi_logs
pi_log_with_inf[:, -1] = pi_log_infs

log_shift = 0 # Use if the weights are tiny, scale them to a non-zero amount
pi_logs_exp = np.exp(pi_log_with_inf + log_shift)
pi_normalised =  pi_logs_exp.T/np.sum(pi_logs_exp, axis =1)
pi_normalised = pi_normalised.T

cal_scores_with_inf = np.ones((cal_scores.shape[0] + 1, cal_scores.shape[1]))
cal_scores_with_inf[:-1, :] = cal_scores
cal_scores_with_inf[-1, :] = np.inf

# qhat_weighted = np.array([get_weighted_quantile(cal_scores_with_inf[:,i], np.ceil((N_cal+1)*(1-alpha))/(N_cal), pi_normalised[i, :]) for i in range(input_size)])
qhat_weighted = np.array([[get_weighted_quantile(cal_scores_with_inf[:,i], np.ceil((N_cal+1)*(1-alpha))/(N_cal), pi_normalised[j, :]) for i in range(output_size)] for j in range(N_shift)])

qhat_weighted = np.array(qhat_weighted)

prediction_sets =  [y_shift_nn - qhat_weighted[:,:,0], y_shift_nn + qhat_weighted[:,:,0]]
# prediction_sets =  [y_shift_nn - qhat*modulation, y_shift_nn + qhat*modulation]#Joint

empirical_coverage = [((y_shift[:, i] >= prediction_sets[0][:, i]) & (y_shift[:, i] <= prediction_sets[1][:, i])).mean() for i in range(output_size)]
empirical_coverage = np.array(empirical_coverage)

print(f"The empirical coverage after calibration is: {empirical_coverage}")
print(f"alpha is: {alpha}")
print(f"1 - alpha <= empirical coverage is {(1-alpha <= empirical_coverage)}")


# %%
# idces = np.argsort(X_shift[-1])
# plt.plot(X_shift[-1][idces], y_shift[-1][idces], label='Actual')
# plt.plot(X_shift[-1][idces], y_shift_nn[-1][idces], label='Pred')
# plt.fill_between(X_shift[-1][idces], prediction_sets[0][-1][idces], prediction_sets[1][-1][idces], alpha=0.2)
# plt.plot
# plt.legend()
# plt.title("Visualising the prediction intervals - known pdfs")

# %%
#
def calibrate_res(alpha):

    qhat_weighted = np.array([[get_weighted_quantile(cal_scores_with_inf[:,i], np.ceil((N_cal+1)*(1-alpha))/(N_cal), pi_normalised[j, :]) for i in range(output_size)] for j in range(N_shift)])
    
    prediction_sets =  [y_shift_nn - qhat_weighted[:,:,0], y_shift_nn + qhat_weighted[:,:,0]]#Marginal

    empirical_coverage = [((y_shift[:, i] >= prediction_sets[0][:, i]) & (y_shift[:, i] <= prediction_sets[1][:, i])).mean() for i in range(output_size)]
    empirical_coverage = np.array(empirical_coverage)

    return empirical_coverage

alpha_levels = np.arange(0.05, 0.95, 0.05)
emp_cov= []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov.append(calibrate_res(alpha_levels[ii]))

emp_cov = np.array(emp_cov)

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
[plt.plot(1-alpha_levels, emp_cov[:,i], label='Residual - weighted - known pdf' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0) for i in range(output_size)]
# plt.plot(1-alpha_levels, this, label='get_weighted' ,ls='-.', color='blue', alpha=0.8, linewidth=1.0)

plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
# plt.legend()
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% 
#Density Ratio Estimation using Probabilistic Classification

#Setting up the classifier.

# 1D convolutional classifier. 
class classifier_1D(nn.Module):
    def __init__(self, in_features, activation=torch.nn.ReLU()):
        super(classifier_1D, self).__init__()

        self.in_features = in_features

        #Convolutional Layers
        self.conv1 = nn.Conv1d(1, 32, 2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, 2, stride=2)
        self.maxpool = nn.MaxPool1d(2)

        #Dense Layers
        self.dense1 = nn.Linear(int(64*62), 1024)
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense_out = nn.Linear(64, 1)

        self.act_func = activation

        self.layers = [self.dense1, self.dense2, self.dense3]

    def forward(self, x):
        x = self.act_func(self.maxpool(self.conv1(x)))
        x = self.act_func(self.conv2(x))

        x = x.view(-1, x.shape[1]*x.shape[2])

        for dense in self.layers:
            x = self.act_func(dense(x))
        x = self.dense_out(x)
        return x

# %%
classifier = classifier_1D(in_features=1) #Sigmoid at the output is evaluated outside the model definition. 
loss_func = torch.nn.BCEWithLogitsLoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# %%
#Prepping the data. 
X_class = np.expand_dims(np.vstack((X_calib, X_shift)), axis=1)
Y_class = np.vstack((np.expand_dims(np.zeros(len(X_calib)), -1), np.expand_dims(np.ones(len(X_shift)) ,-1)))

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

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

# %%
#Classifier Performance over a test data sampled from the same known distributions. 

X_calib_test = normal_dist(mean_1, std_1, N)
X_shift_test = normal_dist(mean_2, std_2, N)#Covariate shifted

X_class_test = np.expand_dims(np.vstack((X_calib_test, X_shift_test)), axis=1)
Y_class_test = np.vstack((np.expand_dims(np.zeros(len(X_calib_test)), -1), np.expand_dims(np.ones(len(X_shift_test)) ,-1)))

y_pred = torch.sigmoid(classifier(torch.tensor(X_class_test, dtype=torch.float32))).detach().numpy()
y_true = Y_class_test

for ii in range(len(y_pred)):
    if y_pred[ii] < 0.5:
        y_pred[ii] =0 

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

#Estimating the likelihood ratio
w = (y_pred)/(1-y_pred)

# %%

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
pi_vals = pi_classifer(X_shift, X_calib)

# %% 
qhat = []
for ii in range(output_size):
    qhat.append(weighted_quantile(cal_scores[:, ii], np.ceil((N+1)*(1-alpha))/(N),  pi_vals))
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
idces = np.argsort(X_shift[-1])
plt.plot(X_shift[-1][idces], y_shift[-1][idces], label='Actual')
plt.plot(X_shift[-1][idces], y_shift_nn[-1][idces], label='Pred')
plt.fill_between(X_shift[-1][idces], prediction_sets[0][-1][idces], prediction_sets[1][-1][idces], alpha=0.2)
plt.plot
plt.legend()
plt.title("Visualising the prediction intervals - Prob. Classifier")
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
emp_cov_prob = []
for ii in tqdm(range(len(alpha_levels))):
    emp_cov_prob.append(calibrate_res(alpha_levels[ii]))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov_prob, label='Residual - weighted - Prob. Classifier' ,ls='-.', color='maroon', alpha=0.8, linewidth=1.0)
plt.plot(1-alpha_levels, emp_cov, label='Residual - weighted - Known' ,ls='-.', color='blue', alpha=0.8, linewidth=1.0)

plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
