#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Generation of the Convection-Diffusion PDE Solutions
"""

# %%
import os
import numpy as np
from pyDOE import lhs 
from tqdm import tqdm 
from time import time

from matplotlib import pyplot as plt 

# from SimRun import run_sim
from RunSim_simtrack import run_sim
# from RunSim_wandb import run_sim


# %%
start_time = time()

n_sims = 2000

# lb = np.asarray([np.pi, 0.1, 1.0, 0.25]) #D, c, mu, sigma
# ub = np.asarray([2*np.pi, 0.5, 8.0, 0.75])


lb = np.asarray([2*np.pi, 0.5, 1.0, 0.25]) #D, c, mu, sigma
ub = np.asarray([4*np.pi, 1.0, 8.0, 0.75])

params = lb + (ub - lb) * lhs(4, n_sims)

# %%

u_dataset = []
for ii in tqdm(range(n_sims)):
    u_dataset.append(run_sim(ii, params[ii,0], params[ii,1], params[ii,2], params[ii,3]))

u_dataset = np.asarray(u_dataset)

# %%
# np.savez(os.getcwd() + '/Data/' + 'ConvDiff_u.npz', u = u_dataset, params=params)

end_time = time()
print("Total Time : " + str(end_time - start_time))
# %%
