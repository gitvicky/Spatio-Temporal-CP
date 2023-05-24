#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D wave equation via FFT 

u_tt = c^2 * (u_xx + u_yy)

on [-1, 1]x[-1, 1], t > 0 and Dirichlet BC u=0


Source : http://people.bu.edu/andasari/courses/numericalpython/python.html
"""
# %%
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  
from tqdm import tqdm 
 
def wave_solution(Lambda, a, b):
        
    N = 30 # Mesh Discretesiation 
    x0 = -1.0 # Minimum value of x
    xf = 1.0 # maximum value of x
    y0 = -1.0 # Minimum value of y 
    yf = 1.0 # Minimum value of y
    tend = 1
    
    #Initialisation 
    
    k = np.arange(N + 1)
    x = np.cos(k*np.pi/N) #Creating the x and y discretisations
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    
    dt = 6/N**2 # dont know why this is taken as dt 
    plotgap = round((1/3)/dt) 
    dt = (1/3)/plotgap
    
    c = 1.0 # Wave Speed <=1.0
    
    
    #Initial Conditions 
    
    vv = np.exp(-Lambda*((xx-a)**2 + (yy-b)**2))
    vvold = vv.copy()
    
    
    
    # Solve and Animate 
    
#    fig = plt.figure()
    
#    ax = fig.add_subplot(111, projection='3d')
    
    tc = 0
    nstep = round(3*plotgap+1) * tend
#    wframe = None
    
    t = np.arange(0,tend+dt,dt)
    data_list = []
    while tc < nstep:
#        if wframe:
#            ax.collections.remove(wframe)
#            
        xxx = np.arange(x0, xf+1/16, 1/16)
        yyy = np.arange(y0, yf+1/16, 1/16)
        vvv = interpolate.interp2d(x, y, vv, kind='cubic')
        Z = vvv(xxx, yyy)
#        
#        xxf, yyf = np.meshgrid(np.arange(x0,xf+1/16,1/16), np.arange(y0,yf+1/16,1/16))
#            
#        wframe = ax.plot_surface(xxf, yyf, Z, cmap=cm.coolwarm, linewidth=0, 
#                antialiased=False)
#        
#        ax.set_xlim3d(x0, xf)
#        ax.set_ylim3d(y0, yf)
#        ax.set_zlim3d(-0.15, 1)
#        
#        ax.set_xlabel("x")
#        ax.set_ylabel("y")
#        ax.set_zlabel("U")
#        
#        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
#        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
#        
#        fig.suptitle("Time = %1.3f" % (tc/(3*plotgap-1)-dt))
#        #plt.tight_layout()
#        ax.view_init(elev=30., azim=-110)
#        plt.pause(0.01)
            
        uxx = np.zeros((N+1, N+1))
        uyy = np.zeros((N+1, N+1))
        ii = np.arange(1, N)
        
        for i in range(1, N):
            v = vv[i,:]
            V = np.hstack((v, np.flipud(v[ii])))
            U = np.fft.fft(V)
            U = U.real
            
            r1 = np.arange(N)
            r2 = 1j*np.hstack((r1, 0, -r1[:0:-1]))*U
            W1 = np.fft.ifft(r2)
            W1 = W1.real
            s1 = np.arange(N+1)
            s2 = np.hstack((s1, -s1[N-1:0:-1]))
            s3 = -s2**2*U
            W2 = np.fft.ifft(s3)
            W2 = W2.real
            
            uxx[i,ii] = W2[ii]/(1-x[ii]**2) - x[ii]*W1[ii]/(1-x[ii]**2)**(3/2)
            
        for j in range(1, N):
            v = vv[:,j]
            V = np.hstack((v, np.flipud(v[ii])))
            U = np.fft.fft(V)
            U = U.real
            
            r1 = np.arange(N)
            r2 = 1j*np.hstack((r1, 0, -r1[:0:-1]))*U
            W1 = np.fft.ifft(r2)
            W1 = W1.real
            s1 = np.arange(N+1)
            s2 = np.hstack((s1, -s1[N-1:0:-1]))
            s3 = -s2**2*U
            W2 = np.fft.ifft(s3)
            W2 = W2.real
            
            uyy[ii,j] = W2[ii]/(1-y[ii]**2) - y[ii]*W1[ii]/(1-y[ii]**2)**(3/2)
            
        vvnew = 2*vv - vvold + c**2*dt**2*(uxx+uyy)
        vvold = vv.copy()
        vv = vvnew.copy()
        tc += 1
        
        data_list.append(Z)
        
    data_array = np.asarray(data_list)
    return xxx, yyy, t, data_array

# %%       
def Parameter_Scan ():
    #Simulation Data built via parameter scans
    Lambda_range = np.arange(10,51,5)
    a_range = np.arange(0.1, 0.51, 0.1)
    b_range = np.arange(0.1, 0.51, 0.1)
    
    #IC = np.exp(-Lambda*((xx-a)**2 + (yy-b)**2))
    
    list_u = []
    list_ic = []
    
    for ii in tqdm(Lambda_range):
        for jj in a_range:
            for kk in b_range:
                x, y, t, u = wave_solution(ii, jj, kk)
                
                list_ic.append([ii, jj, kk])
                list_u.append(u)
        
    ic = np.asarray(list_ic)
    u = np.asarray(list_u)
    
    np.savez('Spectral_Wave_data_Parameter_Scan.npz', x=x, y=y,t=t, u=u, ic=ic)
    
    # data = np.load('Spectral_Wave_data.npz')
    # file_names = data.files

# %%
def LHS_Sampling():
    #Simulation Data Built using LHS sampling
    from pyDOE import lhs
    
    lb = np.asarray([10, 0.10, 0.10]) #Lambda, a, b 
    ub = np.asarray([50, 0.50, 0.50]) #Lambda, a, b 
    
    N = 1000
    
    param_lhs = lb + (ub-lb)*lhs(3, N)
    
    list_u = []
    
    for ii in tqdm(range(N)):
                x, y, t, u = wave_solution(param_lhs[ii, 0], param_lhs[ii, 1], param_lhs[ii, 2])
                
                list_u.append(u)
        
    ic = param_lhs
    u = np.asarray(list_u)
    
    np.savez('Spectral_Wave_data_LHS.npz', x=x, y=y,t=t, u=u, ic=ic)
