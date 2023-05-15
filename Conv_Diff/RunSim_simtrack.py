#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:04:51 2022

@author: vgopakum
"""

# %% 
import os 
import numpy as np 
from matplotlib import pyplot as plt
from time import time
from sympy import *

# from simtrack import Simtrack

plt.ioff()

# %%

def run_sim(ii, D_damp, c, mu, sigma):
    
    start_time = time()
    # run = Simtrack()   

    # run_name = 'Simrun_tests_no_alert'+str(ii)
    run_params = {'D_damp': D_damp, 'c': c, 'mu': mu, 'sigma': sigma}

    #     # Specify a run name, metadata (dict), tags (list), description, folder
    # run.init(run_name,
    #         run_params, 
    #         ['FTCS'],  # Tags
    #         'Convection Diffusion 1D Simulation',   # Description
    #         '/ConvDiff/Mark_1')    # Folder full path
    
    #Upload the code
    # run.save('RunSim_simtrack.py', 'code')
    
    
        
    dx = 0.05
    x = np.arange(0, 10, dx)
    x_length = len(x)
    dt = 0.0005
    

    c = run_params['c'] #Convction Velocity 
    D_damp = run_params['D_damp'] #Damping the variance of the Diffusion Coefficient
    
    x_sym = Symbol('x')
    D_sym = sin(x_sym/D_damp)
    D_func = lambdify(x_sym, D_sym, "numpy") 
    D = D_func(x)
    dD_dx_sym = diff(D_sym, x_sym)
    dD_dx_func = lambdify(x_sym, dD_dx_sym, "numpy") 
    dD_dx = dD_dx_func(x)
    
    #Setting up the Initial Conditions :  Gaussian Distribution
    mu = run_params['mu'] # Location of the Gaussian Mean
    sigma = run_params['sigma'] #Estimation of the Gaussian Variance
    u_0 = np.exp(-(x-mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    
    # np.save(os.getcwd() + '/Inputs/' + run_name + '_u0.npy', u_0)
    # run.save(os.getcwd() + '/Inputs/' + run_name + '_u0.npy', 'input')


    
    #Implementation of FTCS and Implicit scheme to build the test dataset No flux boundary conditions
    n_itim = 5000
    u_dataset = np.zeros((n_itim+1, x_length))
    u_dataset[0] = u_0
    u = u_0.copy()    

    t = np.arange(0, n_itim*dt +dt, dt)
    t_max = np.amax(t)
    t_min = np.amin(t)
    t_norm = (t - t_min) / (t_max - t_min)


    u = u_0

    alpha_diff = (D*dt)/dx**2
    alpha_conv = c*dt/dx

    # plot = plt.figure()
    for ii in range(n_itim):
        u[1:-1] = (u[1:-1]*(1 - 2*alpha_diff[1:-1]) 
                + u[2:]*(alpha_diff[2:] + dD_dx[2:]*(dt/2*dx))
                + u[:-2]*(alpha_diff[:-2] - dD_dx[:-2]*(dt/2*dx))
                - 0.5*alpha_conv*(u[2:] - u[:-2])
                )    
        u[0] = u[1]
        u[-1] = u[-2]
        
        u_mid = u[100]
        # run.log({'u_mid': float(u_mid)})

        # #Alerting a failed simulation
        # if u_mid > 1e6:
        #     run.add_alert('Simulation Unstable', # Name
        #             'is above', # Type
        #             'u_interface',           # Metric
        #             1,                # Frequency
        #             1,                # Window
        #             threshold=1e6)     # Threshold

        u_dataset[ii+1] = u
        
            
    #     if ii% 200 == 0:
    #         plt.plot(u, label=ii)
    # plt.title("Convection + Diffusion : " + run_name)
    # plt.legend()
    # plt.savefig(os.getcwd() + '/Images/' + run_name + '.png')
    # run.save(os.getcwd() + '/Images/' + run_name + '.png', 'output')

    u_dataset = u_dataset[::50]
    
    # np.save(os.getcwd() + '/Outputs/' + run_name + '_u.npy', u_dataset)
    # run.save(os.getcwd() + '/Outputs/' + run_name + '_u.npy', 'output')
    
    end_time = time()
    # run.set_status('completed')

    return u_dataset

