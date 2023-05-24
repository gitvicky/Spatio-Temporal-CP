# -*- coding: utf-8 -*-
"""Py-PDE.ipynb

Automatically generated by Colaboratory.

Original file is located at
"""

import numpy as np 
import matplotlib.pyplot as plt

!pip install py-pde
import pde

"""1D Poisson Equation """

from pde import CartesianGrid, ScalarField, solve_poisson_equation

grid = CartesianGrid([[0, 1]], 32, periodic=False)
field = ScalarField(grid, 1.0)
result = solve_poisson_equation(field, bc=[{"value": 0}, {"derivative": 1}])

result.plot()

n_sims = 10000
lb = 0
ub = 4
params = lb + (ub - lb) * np.random.uniform(size=n_sims)

inps = []
outs = []
for ii in range(n_sims):
  grid = CartesianGrid([[0, 1]], 32, periodic=False)
  field = ScalarField(grid, params[ii])
  result = solve_poisson_equation(field, bc=[{"value": 0}, {"derivative": 1}])
  inps.append(field.data)
  outs.append(result.data)

X = np.asarray(inps)
Y = np.asarray(outs)

np.savez('poisson_1d.npz', x=X, y=Y)

X.shape

