#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2 November 2022


Utilities for working with uncertainty in machine learning models

"""

from scipy.stats import beta
import numpy as np
import torch 
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

# Fully Connected Network or a Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

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


# Fully Connected Network or a Multi-Layer Perceptron with dropout
class MLP_dropout(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh, dropout_rate=0.1):
        super(MLP_dropout, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c 

    def enable_dropout(self):
            """Function to enable the dropout layers during test-time"""
            for m in self.layers:
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()        

#Estimating the output uncertainties using Dropout. 
def MLP_dropout_eval(net, x, Nrepeat=100):
    net.eval()
    net.enable_dropout()
    preds = []
    for i in range(Nrepeat):
        preds.append(net(x).detach().numpy())
    return np.mean(preds, axis=0), np.std(preds, axis=0)


#Defining Quantile Loss
def quantile_loss(pred, label, gamma):
    return torch.where(label > pred, (label-pred)*gamma, (pred-label)*(1-gamma))


#Defining Quantile Loss
def quantile_loss(pred, label, gamma):
    return torch.where(label > pred, (label-pred)*gamma, (pred-label)*(1-gamma))


###
#   Clopper-Pearson confidence interval for binomial rate.
#   Gives any confidence interval on a probability observed k out of times: 
#   https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
#
#   Note, confidence level is 1 - alpha, which is standard practice
#
#   Examples: 
#           k_out_n(5, 10, 0.05) -> [0.187, 0.813] is the 95% conf interval for 5/10 samples
#           k_out_n(5, 10, 0.6) ->  [0.372, 0.627] is the 40% conf interval for 5/10 samples
#           k_out_n(50, 100, 0.05) ->  [0.398, 0.601] is the 95% conf interval for 50/100 samples
###
def k_out_n(k, n, alpha = 0.05):
    if k == 0:
        return [0, beta(k + 1, n - k).ppf(1 - alpha/2)]

    if n == k:
        return [beta(k, n - k + 1).ppf(alpha/2), 1]

    return [beta(k, n - k + 1).ppf(alpha/2), beta(k + 1, n - k).ppf(1 - alpha/2)]

###
#   Constructs prediction sets from 1D datasets, for specified alpha levels.
#   Reliability takes into account the sample size used to generate the prediction sets.
#   If there is a lack of data, your prediction sets will be right at least 'reliability' about of times
#   on long run repeated trails. Becomes less and less relevant as samples size increases
###
def prediction_sets(data, alphas, reliability = 0.95):
    Nsamples = len(data)
    data_sort = np.sort(data)

    if (Nsamples % 2 == 0):     # is even
        mid = int(Nsamples/2)
        ints_lo = data_sort[:mid]
        ints_hi = np.flip(data_sort[mid+1:])
    else:                       # is odd
        mid = int((Nsamples-1)/2)
        ints_lo = data_sort[:mid]
        ints_hi = np.flip(data_sort[mid+2:])
    
    ks = [np.sum(np.logical_and(data >= ints_lo[i], data <= ints_hi[i])) for i in range(len(ints_hi))]  # ks of each set
    confs = [k_out_n(k, Nsamples, 1 - reliability) for k in ks]                                         # gets confidence level for each set

    confs_lo = np.array([c[0] for c in confs])

    out_low = np.zeros(len(alphas))
    out_hi = np.zeros(len(alphas))

    for (i,p) in enumerate(alphas):

        if p > confs_lo[0]:
            print(f"Requested confidence level not available. Requested: {p}, largest available: {confs_lo[0]}. Returning {confs_lo[0]}")
            ind = 0
            alphas[i] = confs_lo[0]
        else:
            trues = p <= confs_lo
            ind = np.where(trues)[0][-1]    # gets the index of last true

        out_low[i] = ints_lo[ind]
        out_hi[i] = ints_hi[ind]
    
    return out_low, out_hi, alphas


