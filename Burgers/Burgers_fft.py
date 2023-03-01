#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1st March 2023

@author: vgopakum 

---------------------------------------------------------------------------------------
1D Burgers' Equation using the spectral method in Python 
---------------------------------------------------------------------------------------

Code inspired from from Steve Brunton's FFT Example videos :
https://www.youtube.com/watch?v=hDeARtZdq-U&t=639s

Equation:
u_t + u*u_x =  nu*u_xx on [-1,1] x [-1,1]

Boundary Conditions: 
Periodic (implicit via FFT)

Initial Distribution :
u(x,y,t=0) = sin(alpha*pi*x) + np.cos(beta*np.pi*-x) + 1/np.cosh(gamma*np.pi*x) 
where alpha, beta, gamma all lie within the domain [-3,3]^3

Initial Velocity Condition :
u_t(x,y,t=0) = 0

"""
# %%
#Importing the necessary packages. 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from simvue import run 

def solve_burgers(configuration):

    nu = configuration['viscosity']
    L = configuration['domain length']
    N = configuration['discretisation']
    dt = configuration['dt']
    iterations = configuration['iterations']


    dx = L/N 
    x = np.arange(0, L, dx) #Define the X Domain 
    midpoint = int(N/2)

    #Define the discrete wavenumbers 
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

    #Initial Condition
    alpha = configuration['alpha']
    beta = configuration['beta']
    gamma = configuration['gamma']
    u0 = np.sin(alpha*np.pi*x) + np.cos(beta*np.pi*-x) + 1/np.cosh(gamma*np.pi*x) 

    #Simulate in Fourier Freq domain. 
    t= np.arange(0, iterations*dt, dt)

    def rhsBurgers(u, t, kappa, nu):
        uhat = np.fft.fft(u)
        d_uhat = (1j)*kappa*uhat
        dd_uhat = -np.power(kappa, 2)*uhat
        d_u = np.fft.ifft(d_uhat)
        dd_u = np.fft.ifft(dd_uhat)
        du_dt = -u*d_u + nu*dd_u

        run.log_metrics({'u_mid': uhat.real})
                        
        return du_dt.real
    
    u = odeint(rhsBurgers, u0, t, args=(kappa, nu))
    
    return u


# %%
