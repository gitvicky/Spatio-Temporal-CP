#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Data Generation for 1D Burgers Surrogate
---------------------------------------------------------------------------------------

Code inspired from from Steve Brunton's FFT Example videos :
https://www.youtube.com/watch?v=hDeARtZdq-U&t=639s

Equation:
u_t + u*u_x =  nu*u_xx on [-1,1] x [-1,1]

Boundary Conditions: 
Periodic (implicit via FFT)

Initial Distribution :
u(x,y,t=0) = sin(alpha*pi*x) + np.cos(beta*np.pi*-x) + 1/np.cosh(gamma*np.pi*x) 
where alpha, beta, gamma all lie within the domain [-3,3]^3 sampled using a hypercube. 

Initial Velocity Condition :
u_t(x,y,t=0) = 0

"""
# %% 
#Importing the required packages
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from pyDOE import lhs 
from Burgers_fft import *
# %%
n_sims = 5000 #Total Number of simulation datapoints to be generated. 

#Grabbing the simulation parameters from the specified domain. 
 #alpha, beta, gamma
lb = np.asarray([-3, -3, -3]) # Lower Bound of the parameter domain
ub = np.asarray([3, 3, 3]) # Upper bound of the parameter domain

params = lb + (ub - lb) * lhs(3, n_sims)

# %% 
from simvue import Run

configuration = {'viscosity': 0.002,
                 'domain length': 2.0,
                 'discretisation': 1000,
                 'dt': 0.0025,
                 'iterations': 500,
                 }
# %%
if __name__ == "__main__":
    u_list = []
    for sim in tqdm(range(n_sims)):
        run = Run(mode='disabled')
        configuration['alpha'] = params[sim, 0]
        configuration['beta'] = params[sim, 1]
        configuration['gamma'] = params[sim, 2]
        run.init(folder="/Burgers", tags=['Burgers1D', 'Spectral'], metadata=configuration)
        u_sol = solve_burgers(run, configuration) #Running the simulation with the specified configuration
        u_list.append(u_sol)
        #Passing the simvue run object as well. 

        run.save(u_sol, 'output', name='u_field') #Saving the solution as a numpy array to simvue

        #Generating the spatio-temporal plot
        fig = plt.figure()
        plt.imshow(np.flipud(u_sol), aspect=.8)
        plt.axis('off')
        plt.set_cmap('plasma')

        # run.save(fig, 'output', name='kymograph') #Saving the solution as a numpy array to simvue

        #Simvue Artifact storage
        run.save('Data_Gen.py', 'code', name='Data_Gen') #Saving the data generation script to simvue
        run.save('Burgers_fft.py', 'code', name='Burgers_fft') #Saving the data generation script to simvue

        run.close()

# %%
#[BS, time, space]
u = np.asarray(u_list)
u = u[:,::10, ::5]
np.save('Burgers1d_sliced_5K.npy', u)
# %%
