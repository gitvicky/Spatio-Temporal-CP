#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base tests for demonstrating CP under covariate shift - Density Estimation using a VAE
VAE code inspired from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
Multivariate setting for 5000-in, 5000-out
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

torch.set_default_dtype(torch.float32)

# %
"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.enc_1 = nn.Linear(input_dim, hidden_dim)
        self.enc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.latent_mean  = nn.Linear(hidden_dim, latent_dim)
        self.latent_var   = nn.Linear (hidden_dim, latent_dim)
    
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h = self.LeakyReLU(self.enc_1(x))
        h = self.LeakyReLU(self.enc_2(h))
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)                     # encoder produces mean and log of variance 
                                                       #    (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.dec_1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.dec_1(x))
        h = self.LeakyReLU(self.dec_2(h))
        
        x_hat = self.FC_output(h)
        return x_hat
    
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)#.to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

# %% 
    from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


# optimizer = torch.optim.Adam(model.parameters(), lr=lr)